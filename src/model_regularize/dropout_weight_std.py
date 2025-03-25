import torch
import torch.nn as nn 
import torch.optim as optim
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.visualization.animator_tool import Animator
from src.utils.accumulator import Accumulator
from src.utils.load_mnist import load_fashion_mnist

def dropout(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return (mask * X) / (1.0 - dropout)

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2, is_training=True):
        super().__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = dropout1 
        self.dropout2 = dropout2

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(-1, self.num_inputs)))
        if self.train:
            H1 = dropout(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout(H2, self.dropout2)
        return self.lin3(H2)
    


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






if __name__ == "__main__":
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    num_epochs, lr, batch_size, num_works= 10, 0.5, 256, 4
    train_iter, test_iter = load_fashion_mnist(batch_size, num_works)
    dropout1, dropout2 = 0.2, 0.5
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2)
    trainer = optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    train(net,train_iter, test_iter, loss, num_epochs, trainer)

