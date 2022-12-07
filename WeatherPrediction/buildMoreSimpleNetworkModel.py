'''
Author: enfuri51 huangzhichao@xzrobot.com
Date: 2022-11-12 21:58:52
LastEditors: enfuri51 huangzhichao@xzrobot.com
LastEditTime: 2022-11-13 22:10:12
FilePath: \DeepLearning exervise\WeatherPrediction\buildMoreSimpleNetworkModel.py
brief: 

Copyright (c) 2022 by enfuri51 huangzhichao@xzrobot.com, All Rights Reserved. 
'''
import numpy as np
import torch
import torch.optim as optim
import data_preprocess
from sklearn import preprocessing
# 处理时间数据
import datetime
# 读取csv文件的库
import pandas as pd  

import matplotlib.pyplot as plt

#from torch.autograd import Variable

# 解决plot绘图会使得python崩溃的问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 读取data_process中的数据
years = data_preprocess.years
months = data_preprocess.months
days = data_preprocess.days
feature_list = data_preprocess.feature_list
features = data_preprocess.features
labels = data_preprocess.labels


# 将数据标准化处理
input_features = preprocessing.StandardScaler().fit_transform(data_preprocess.features)

# size为标准化的数据的列向量个数
# shape[0] 行向量个数
# shape[1] 列向量个数
input_size = input_features.shape[1]

# 设计的隐藏层的神经元个数
hidden_size = 128

# 设计的输出值的维度
# 因为结果为预测的温度，所以维度为1
output_size = 1

# 不是直接计算所有数据，而是分为多个batch
# batch_size代表依次输入数据的维度为16
batch_size = 16

# 调用torch的nn模块，直接构建了两层的神经网络
# Linear为全连接层
# sigmoid为激活函数层 
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)

# cost表示前向传播的损失值，损失函数的类型为平均值
cost = torch.nn.MSELoss(reduction='mean')

# 表示优化器的学习的优化求解
# Adam表示动态调整学习率
optimizer = torch.optim.Adam(my_nn.parameters(), lr = 0.001)

# 训练网络
losses = []
for i in range(1000):
    batch_loss = []
    # MINI-Batch方法来进行训练
    for start in range(0, len(input_features), batch_size):

        # 控制batch的end迭代器
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype = torch.float, requires_grad = True)
        yy = torch.tensor(labels[start:end], dtype = torch.float, requires_grad = True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)

        # 每次梯度回归计算后，需要重置梯度值
        # 不然梯度累积，会造成结果错误
        optimizer.zero_grad()

        # 反向传播
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    
    # 打印损失
    if i % 100==0:
        losses.append(np.mean(batch_loss))
        # print(i, np.mean(batch_loss))

# 预测训练结果
x = torch.tensor(input_features, dtype = torch.float)
predict = my_nn(x).data.numpy()

# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

# 同理，再创建一个来存日期和其对应的模型预测值
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predict.reshape(-1)}) 

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()

# 图名
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values')

# 输出的预测曲线比较平滑，说明没有过拟合
# 模型具有更多场景的应用能力
