# Created by Yanan Liu on 14:33 22/11/2023
# Location: Your Location
# Log: Your Log Information
# Version: Your Version Information

import numpy as np

# 假设一个光强的对数值L的图像，这里我们用一个简单的梯度图像来模拟
L = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])

# 计算空间梯度
grad_L = np.gradient(L)

# 假设一个像素点的移动向量，例如(1, 1)，代表向右下方移动了一个像素单位
u = np.array([1, 1])

# 为了简化，我们只计算一个点的梯度和移动，选择L中的一个点
point_grad_L = grad_L[1][1, 1], grad_L[0][1, 1]

# 计算点积，得到光强变化速度
change_rate = -np.dot(point_grad_L, u)

print("光强变化速度: ", change_rate)

