# 数字识别分类回归任务

# 下载数据集
import torch
import numpy as np
from matplotlib import pyplot
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


# 查看数据集的数据格式
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
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
