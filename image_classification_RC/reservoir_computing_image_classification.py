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

def tanh(x):
    tanh =( torch.exp(x)-torch.exp(-x) )/ (torch.exp(x) + torch.exp(-x))
    return tanh

#one_hot
def to_one_hot(label,demension):
    result = np.zeros((len(label),demension))
    for i,label in enumerate(label):
        result[i,label] = 1.
    return result



parser = argparse.ArgumentParser(description='PyTorch Siamese Reservoir Network')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=304, help='Random seed')

# model
parser.add_argument('--inSize', type=int, default=784, help='The number input nodes')
parser.add_argument('--outSize', type=int, default=10, help='Reservoir output size')
parser.add_argument('--resSize', type=int, default=2000, help='Reservoir capacity')
parser.add_argument('--alpha', type=float, default=0.5, help='Leaky rate')
parser.add_argument('--sigma', type=float, default=0.5, help='Leaky rate')
parser.add_argument('--trainLen', type=int, default=9000, help='Number of data for training')
parser.add_argument('--init', type=int, default=500, help='Number of data for initing')
parser.add_argument('--testlen', type=int, default=1000, help='Number of data for testing')

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
Win = (torch.rand([resSize, 1 + inSize]) - 0.5)
W = (torch.rand([resSize,resSize])-0.5)*2.4
W[abs(W) > 0.6] = 0

rhoW = max(abs(torch.linalg.eig(W)[0]))
W *=(1.25 / rhoW)
reg = 1e-3
bias = 1
BATCH_SIZE = 250
enc = OneHotEncoder(sparse = False)
print("dataset...")

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
trainData = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
testData = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
train_dataset, _ = random_split(trainData, [2000, len(trainData) - 2000])
test_dataset,_ = random_split(testData, [1000, len(testData) - 1000])

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
    if step < 2:
        for batch in range(BATCH_SIZE):
                    u =flattened_data[batch:batch+1,...].T
                    x = (1 - a) * x + a * tanh(np.matmul(Win, np.vstack((bias, u))) + np.matmul(W, x))

    elif step >= 2 and step<36:
        yt[:,(step-2)*250:(step-1)*250] = labels
        Yt[:,(step-2)*250:(step-1)*250] = label.T
        print("training ......")
        for batch in range(BATCH_SIZE):
                u = flattened_data[batch:batch+1,...].T
                x = (1 - a) * x + a * tanh(np.matmul(Win, np.vstack((bias, u))) + np.matmul(W, x))
                X[:, i] = np.vstack((bias, u, x))[:, 0]
                i += 1
    else:
        break
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
    if step < 4:
        target[:, step * 250:(step + 1) * 250] = labels
        for batch in range(BATCH_SIZE):
            u = testImgs[batch:batch + 1, ...].T
            x = (1 - a) * x + a * tanh(np.matmul(Win, np.vstack((bias, u))) + np.matmul(W, x))

            y = np.matmul(Wout, np.vstack((bias, u, x)))
            out[:, i] = np.squeeze(np.asarray(y))
            i += 1
    else:
            break
pre = np.argmax(out,axis=0)
C = target - pre
c = testlen - np.count_nonzero(C)
accuracy = c/testlen
print("accuracy is :",accuracy)
print("test  complete")








