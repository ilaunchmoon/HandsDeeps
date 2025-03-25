import torch
import torch.nn as nn 
from torch import optim 
from matplotlib import pyplot as plt

from src.utils.load_mnist import load_fashion_mnist
from src.utils.accumulator import Accumulator
from src.visualization.animator_tool import Animator

# softmax的简洁实现


# 定义模型
def net():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


# 初始化模型参数
def init_net(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 定义损失函数
def get_loss():
    return nn.CrossEntropyLoss(reduction='none')


# 定义优化器
def optimer(net, lr):
    return optim.SGD(net.parameters(), lr=lr)



def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 定义训练周期函数
def train_epoch(net, train_iter, loss_func, optimer):
    if isinstance(net, nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss_func(y_hat, y)
        if isinstance(optimer, optim.Optimizer):
            optimer.zero_grad()
            l.mean().backward()
            optimer.step()
        else:
            l.sum().backward()
            optimer(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


# 定义训练函数
def train(net, train_iter, test_iter, loss_func, num_epochs, optimer):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=["train_loss", "train_acc", "test_acc"])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss_func, optimer)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc, ))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and  test_acc > 0.7, test_acc


if __name__ == "__main__":
    num_epochs, batch_size, num_inputs, num_outputs, lr = 10, 256, 784, 10, 0.1     # 初始化超参数
    num_works = 4                                                                   # 初始化线程数
    train_iter, test_iter = load_fashion_mnist(batch_size, num_works)
    net_softmax = net()
    net_softmax.apply(init_net)
    opt = optimer(net_softmax, lr)
    loss_func = get_loss()  # 初始化损失函数实例
    train(net_softmax, train_iter, test_iter, loss_func, num_epochs, opt)
    plt.show()

