# developed for embedded vision
# 17/11/2023
# binary input images, and binary Win

from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path
import networkx as nx
import argparse
import time
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat
#数据设置

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")


def tanh(x):
    tanh =( torch.exp(x)-torch.exp(-x) )/ (torch.exp(x) + torch.exp(-x))
    return tanh

#one_hot
def to_one_hot(label,demension):
    result = np.zeros((len(label),demension))
    for i,label in enumerate(label):
        result[i,label] = 1.
    return result



parser = argparse.ArgumentParser(description='PyTorch Reservoir Network for Image Classification')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=304, help='Random seed')

# model
parser.add_argument('--inSize', type=int, default=784, help='The number input nodes')
parser.add_argument('--outSize', type=int, default=10, help='Reservoir output size')
parser.add_argument('--resSize', type=int, default=2000, help='Reservoir capacity')
parser.add_argument('--alpha', type=float, default=0.7, help='Leaky rate')
parser.add_argument('--sigma', type=float, default=0.5, help='Leaky rate')
parser.add_argument('--trainLen', type=int, default=10000, help='Number of data for training, 60,000 in total')
parser.add_argument('--init', type=int, default=5000, help='Number of data for initing')
parser.add_argument('--testlen', type=int, default=10000, help='Number of data for testing')

# dir
parser.add_argument('--save_dir', type=str, default='./result/train/weights/')
parser.add_argument('--img_save_dir', type=str, default='./result/train/')
parser.add_argument('--load_dir', type=str, default='./result/train/weights/')

args = parser.parse_args()
print(args)

myseed = args.seed
np.random.seed(myseed)
torch.manual_seed(myseed)
print("Current Seed: ", myseed)

print("Initializing...")
inSize = args.inSize
outSize = args.outSize
resSize = args.resSize
a = args.alpha
trainLen = args.trainLen
init = args.init
testlen = args.testlen
device = torch.device(args.device)
sigma = args.sigma

# Win = np.random.uniform(-sigma,sigma, (resSize, inSize + 1))  # 生成一个 n*m+1的随机矩阵，元素在sigma内均匀分布
# Win = (torch.rand([resSize, 1 + inSize]) - 0.5)

# random -1 or 1
Win = torch.randint(0, 2, (resSize, 1 + inSize)).float() * 2 - 1
W = (torch.rand([resSize,resSize])-0.5)*2.4
W[abs(W) > 0.6] = 0
rhoW = max(abs(torch.linalg.eig(W)[0]))
W *=(1.25 / rhoW)
reg = 1e-3
bias = 1
BATCH_SIZE = 1000
enc = OneHotEncoder(sparse = False)

print(f"total number of parameters: {inSize*resSize + resSize**2 + resSize*outSize }")
print("dataset...")

def show_images(images, num_images=4):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 4))
    for i in range(num_images):
        ax = axes[i]
        img = images[i].cpu().numpy().squeeze()  # Convert to numpy array and remove channel dimension
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()

def binary_threshold(tensor, threshold=0.5):
    return (tensor > threshold).float()

transform = transforms.Compose([transforms.ToTensor(),
                                lambda x: binary_threshold(x, threshold=0.5),  # Apply binary threshold
                                transforms.Normalize(mean=[0.5], std=[0.5])])

trainData = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
testData = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)


print("training......")
processBar = tqdm(trainDataLoader,unit = 'step')
i = 0
Yt = np.zeros([10,trainLen - init])
yt = np.zeros([1,trainLen - init])
X = np.zeros([1 + inSize + resSize,trainLen-init])
x = torch.zeros((resSize, 1))
for step,(trainImgs,labels) in enumerate(processBar):
    flattened_data = trainImgs.view(BATCH_SIZE,-1).to(device)
    labels = labels.to(device)
    label = to_one_hot(labels,10)
    #labels = labels.view(1,-1).to(device)
    ini_step = int(init/BATCH_SIZE)
    total_step = int(trainLen/BATCH_SIZE)
    if i == 0:
        show_images(trainImgs)

    if step < ini_step:
        for batch in range(BATCH_SIZE):
            u =flattened_data[batch:batch+1,...].T
            x = (1 - a) * x + a * tanh(np.matmul(Win, np.vstack((bias, u))) + np.matmul(W, x))

    elif step >= ini_step and step < total_step:
        yt[:,(step-ini_step)*BATCH_SIZE:(step-ini_step+1)*BATCH_SIZE] = labels
        Yt[:,(step-ini_step)*BATCH_SIZE:(step-ini_step+1)*BATCH_SIZE] = label.T
        print("training ......")
        for batch in range(BATCH_SIZE):
                u = flattened_data[batch:batch+1,...].T
                x = (1 - a) * x + a * tanh(np.matmul(Win, np.vstack((bias, u))) + np.matmul(W, x))
                X[:, i] = np.vstack((bias, u, x))[:, 0]
                i += 1

print("train  complete")

Wout = np.matmul(np.matmul(Yt, X.T),np.linalg.inv(
                                np.matmul(X, X.T) + reg * np.eye(1 + inSize + resSize)))
outputs = Wout @ X
predictions = np.argmax(outputs,axis=0)
C = yt - predictions
c = trainLen - init - np.count_nonzero(C)
accuracy = c/(trainLen-init)

print("train accuracy is ",accuracy)
# accuracy = torch.sum(predictions == labels)/labels.shape[0]
# processBar.set_description("[%d/%d] ACC:%.4f" % ( EPOCHS, accuracy.item()))
#testing
i = 0
target = np.zeros([1, testlen])
out = np.zeros([10, testlen])
x = torch.zeros((resSize, 1))
print("test start")

processBar = tqdm(testDataLoader,unit = 'step')
print("test......")
for step,(testImgs,labels) in enumerate(processBar):
    testImgs = testImgs.view(BATCH_SIZE,-1).to(device)
    labels = labels.to(device)
    target[:, step * BATCH_SIZE:(step + 1) * BATCH_SIZE] = labels
    for batch in range(BATCH_SIZE):
        u = testImgs[batch:batch + 1, ...].T
        x = (1 - a) * x + a * tanh(np.matmul(Win, np.vstack((bias, u))) + np.matmul(W, x))

        y = np.matmul(Wout, np.vstack((bias, u, x)))
        out[:, i] = np.squeeze(np.asarray(y))
        i += 1

pre = np.argmax(out,axis=0)
C = target - pre
c = testlen - np.count_nonzero(C)
accuracy = c/testlen
print("accuracy is :",accuracy)
print("test  complete")