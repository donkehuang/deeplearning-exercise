#!/usr/bin/env python
# coding=utf-8
# 数字识别分类回归任务

# 下载数据集
import torch
import gzip
import pickle
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)


# 解压数据集

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid),
     _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot
import numpy as np

# 直接运行的话，python会崩掉
# pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
x_train[0].reshape((28, 28))
print(x_train.shape)


# 将数据集中的数据转换为tensor格式，便于继续计算
# 利用map函数的序列映射，快速进行格式转换
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

# 如果模型中有待学习的参数，如卷积层和全连接层，使用nn.module模块比较方便
# 反之，使用nn.function模块更为简单

# 导入Function模块
import torch.nn.functional as F

# # 交叉熵计算损失
# loss_func = F.cross_entropy
# def model(xb):
#     return xb.mm(weights)+bias

# bs = 64
# xb = x_train[0:bs]  # a mini-batch from x
# yb = y_train[0:bs]
# weights = torch.randn([784, 10], dtype = torch.float,  requires_grad = True) 
# bs = 64
# bias = torch.zeros(10, requires_grad=True)

# # model(xb) 为预测追
# # yb为真实值
# print(loss_func(model(xb), yb))

# Model模块可以简化代码
# - 必须继承nn.Module且在其构造函数中需调用nn.Module的构造函数
# - 无需写反向传播函数，nn.Module能够利用autograd自动实现反向传播
# - Module中的可学习参数可以通过named_parameters()或者parameters()返回迭代器

from torch import nn

# 继承才能调用构造函数
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()

        # 只是定义哪些需要用到的层
        # 自己定义隐藏层
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        # 输出层
        self.out  = nn.Linear(256, 10)

    # 只要定义前向传播就行，pytorch自己定义反向传播
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

net = Mnist_NN()
print(net)


### 使用TensorDataset和DataLoader来简化

# 通过下述函数，能够实现一个batch一个batch的取数据
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

# - 一般在训练模型时加上model.train()，这样会正常使用Batch Normalization和 Dropout
# - 测试的时候一般选择model.eval()，这样就不会使用Batch Normalization和 Dropout
# 训练的时候，使用比如normalization会让过拟合的风险降低

import numpy as np

def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:'+str(step), '验证集损失：'+str(val_loss))

# 定义优化器
from torch import optim
def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)

# 计算损失值
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

# 获取数据
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

# 获取模型
model, opt = get_model()

# 训练过程
fit(25, model, loss_func, opt, train_dl, valid_dl)
