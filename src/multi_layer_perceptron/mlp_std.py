import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.load_mnist import load_fashion_mnist, get_fashion_mnist_labels, show_images
from src.visualization.animator_tool import Animator
from src.utils.accumulator import Accumulator

# 初始化模型
def init_net(num_inputs, num_outputs, num_hiddens):
    w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    return [w1, b1, w2, b2]

# 定义relu激活函数
def relu_func(x):
    return torch.max(x, torch.zeros_like(x))

# 定义模型
def net(x, num_inputs, w1, b1, w2, b2):
    x = x.reshape((-1, num_inputs))
    h = relu_func(x @ w1 + b1)
    return h @ w2 + b2

# 定义损失函数
def loss_fun():
    return nn.CrossEntropyLoss()

# 定义优化器
def optimer(params, lr):
    return torch.optim.SGD(params, lr=lr)

# 计算准确率
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 评估模型
def evaluate_accuracy(net, num_inputs, w1, b1, w2, b2, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X, num_inputs, w1, b1, w2, b2)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]

# 定义单轮训练
def train_epoch(net, num_inputs, params, train_iter, loss, opt):
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X, num_inputs, *params)
        l = loss(y_hat, y)
        opt.zero_grad()
        l.mean().backward()
        opt.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

# 定义训练函数
def train(net, num_inputs, params, lr, train_iter, test_iter, loss, num_epochs, opt):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 1], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, num_inputs, params, lr, train_iter, loss, opt)
        test_acc = evaluate_accuracy(net, num_inputs, *params, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 1.0, train_loss
    assert train_acc > 0.7, train_acc
    assert test_acc > 0.7, test_acc


# 预测
def predict(net, num_inputs, params, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X, num_inputs, *params).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

# 主函数
if __name__ == "__main__":
    num_epochs, batch_size, num_inputs, num_outputs, num_hiddens, lr = 10, 256, 784, 10, 256, 0.5
    num_workers = 4
    params = init_net(num_inputs, num_outputs, num_hiddens)
    train_iter, test_iter = load_fashion_mnist(batch_size, num_workers)
    loss = loss_fun()
    opt = optimer(params, lr)
    train(net, num_inputs, params, lr, train_iter, test_iter, loss, num_epochs, opt)
    plt.show()
