import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.linear_regression.man_impl_linear_regression import synthetic_data

# 定义生成训练特征与标签
def generate_data(num_eamples=1000):
    true_w, true_b = torch.tensor([2, -3.4]), 4.2
    return synthetic_data(true_w, true_b, num_eamples)


# 定义读取数据集方法
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 定义模型并初始化模型的参数
def define_net():
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    return net

# 定义损失函数
def define_loss():
    return nn.MSELoss()


# 定义优化算法
def define_optim(net_params, lr=0.03):
    return optim.SGD(net_params, lr=lr)


# 定义训练函数
def train(num_epoch, net, loss, features, labels, data_iter, optimer):
    """   
        num_epoch:      训练次数
        net:            模型
        loss:           损失函数
        features:       训练集中的特征
        labels:         训练集中的标签
        data_iter:      训练数据迭代器
        optimer:        优化算法: 一般是随机梯度下降
    """
    for epoch in range(num_epoch):
        for X, y in data_iter:
            l = loss(net(X), y)
            optimer.zero_grad()
            l.backward()
            optimer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

if __name__ == "__main__":
    features, labels = generate_data()
    num_epoch, lr, batch_size = 3, 0.03, 10
    data_iter = load_array((features, labels), batch_size)
    net = define_net()
    loss = define_loss()
    lr = 0.03
    optimer = define_optim(net.parameters(), lr)
    train(num_epoch, net, loss, features, labels, data_iter, optimer)



