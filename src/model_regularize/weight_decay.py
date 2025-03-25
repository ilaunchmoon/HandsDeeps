import torch
import torch.nn as nn 
from torch.utils import data
import torch.optim as optim 
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



def data_iter(n_train, n_test, num_inputs, batch_size, is_train=False):
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = synthetic_data(true_w, true_b, n_train)
    train_iter = load_array(train_data, batch_size)
    test_data = synthetic_data(true_w, true_b, n_test)
    test_iter = load_array(test_data, batch_size, is_train=is_train)
    return train_iter, test_iter


def evaluate_loss(net, data_iter, loss_func):
    metric = Accumulator(2)                              # 损失的总和、样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss_func(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(num_inputs, num_epochs, lr, wd, train_iter, test_iter):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    trainer = optim.SGD([
        {"params":net[0].weight, 'weight_decay':wd},
        {"params":net[0].bias}], lr=lr
    )
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                    (evaluate_loss(net, test_iter, loss))))
    print("w的L2范数: ", net[0].weight.norm().item())

if __name__ == "__main__":
    n_train, n_test, num_inputs, batch_size, num_epochs, lr, wd = 20, 100, 200, 5, 100, 0.01, 3
    train_iter, test_iter = data_iter(n_train, n_test, num_inputs, batch_size)
    train(num_inputs, num_epochs, lr, wd, train_iter, test_iter)

