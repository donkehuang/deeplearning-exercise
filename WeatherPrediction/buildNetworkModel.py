'''
Author: enfuri51 huangzhichao@xzrobot.com
Date: 2022-11-10 22:37:56
LastEditors: enfuri51 huangzhichao@xzrobot.com
LastEditTime: 2022-11-10 23:12:23
FilePath: \DeepLearning exervise\WeatherPrediction\buildNetworkModel.py
brief: 

Copyright (c) 2022 by enfuri51 huangzhichao@xzrobot.com, All Rights Reserved. 
'''
import torch
import torch.optim as optim
import data_preprocess
from sklearn import preprocessing

# 导入数据处理模块处理后的数据
features = data_preprocess.features
labels = data_preprocess.labels


# 将数据标准化处理
input_features = preprocessing.StandardScaler().fit_transform(data_preprocess.features)

# 将输出和输出的数据类型由numpy.ndarray 转换为 tensor格式
# 创建归一化的变量x，它的取值是0.02,0.04,...,1
# 创建归一化的预测变量y，它的取值范围是0～1
x = torch.tensor(features, dtype=float)
y = torch.tensor(labels, dtype=float)


# 初始化所有神经网络的权重（weights）和阈值（biases）
# 输入数据 X 的矩阵维度为【348 * 14】
# 当隐藏的神经元个数设计为128个时，权重参数矩阵W的数据维度应该为【14*128】
# 每个特征的偏执项 b 的矩阵维度则为【1*128】
# rand ->标准正太分布作为初始值
# requires_grad ->需要计算梯度，便于回归运算
weights = torch.randn((14, 128), dtype = torch.double, requires_grad = True) #14*128的输入到隐含层的权重矩阵
biases = torch.randn(128, dtype = torch.double, requires_grad = True) #尺度为128的隐含层节点偏置向量

# 预测任务的输出应该为1个值，因此需要再添加一层转换，将上述的矩阵维度转为1
# 因此，有【348*128】*W【128*1】+b【1】 = y
weights2 = torch.randn((128, 1), dtype = torch.double, requires_grad = True) #128*1的隐含到输出层权重矩阵
biases2 = torch.randn(1, dtype = torch.double, requires_grad = True)

#设置学习率
learning_rate = 0.001 
losses = []

for i in range(100000):
    # 前向传播
    # 1 从输入层到隐含层的计算
    hidden = x.mm(weights) + biases
    # 2 将relu激活函数作用在隐含层的每一个神经元上
    hidden = torch.relu(hidden)
    # 3 隐含层输出到输出层，计算得到最终预测
    predictions = hidden.mm(weights2) + biases2

    # 计算误差
    # 通过与标签数据y比较，计算均方误差
    loss = torch.mean((predictions - y) ** 2) 
    losses.append(loss.data.numpy())
    
    # 每隔10000个周期打印一下损失函数数值
    if i % 1000 == 0:
        print('loss:', loss)
        
    # 对损失函数进行梯度反传
    # 反向传播计算
    loss.backward()
    
    # 利用上一步计算中得到的weights，biases等梯度信息
    # 更新weights或biases中的data数值
    # “-” 表示反方向
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)
    
    # 清空所有变量的梯度值。
    # 因为pytorch中backward一次梯度信息会自动累加到各个变量上，因此需要清空
    # 否则下一次迭代会累加，造成很大的偏差
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

