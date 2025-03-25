import numpy as np 
import math
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils import data
from matplotlib import pyplot as plt
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.visualization.animator_tool import Animator
from src.utils.accumulator import Accumulator
from src.utils.load_mnist import load_fashion_mnist

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 用于生成数据集
def synthetic_data(w, b, num_examples):
    """
        使用 y = xw + b + ε 来模拟收集到的数据集, 由于收集到的数据一般都会有噪声, 所以使用ε(0, 0.01)这个服从正太分布的作为噪声
        使用 synthetic_data 生成的数据来进行训练和测试, 以此来得到真实模型(w、b)
    """
    X = torch.normal(0, 1, (num_examples, len(w)))      # 生成形状为(num_eamples, len(w))的服从0-1分布的元素构成的张量
    y = torch.matmul(X, w) + b                          # 可能发生广播机制
    y += torch.normal(0, 0.01, y.shape)                 # 加上噪声    
    return X, y.reshape(-1, 1)         


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def data_iter(n_train, n_test, num_inputs, batch_size, is_train=False):
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = synthetic_data(true_w, true_b, n_train)
    train_iter = load_array(train_data, batch_size)
    test_data = synthetic_data(true_w, true_b, n_test)
    test_iter = load_array(test_data, batch_size, is_train=is_train)


def init_net(num_inputs):
    return [torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True), torch.zeros(1, requires_grad=True)]

# 定义线性回归模型
def linear_regression(X, w, b):
    return torch.matmul(X, w) + b                       # b会执行广播机制

# 定义损失函数: 使用均方误差
def loss_linear(y_hat, y):                              # y_hat为预测值(即每次模型输入训练数据后, 模型的输出值), y为标签值
    return 0.5 * (y_hat - y.reshape(y_hat.shape)) ** 2


def train(num_epochs, batch_size, lr, net, w, b, loss, evlauate_loss, train_iter, optimer, lambd, penalty):
    pass



if __name__ == "__main__":
    pass