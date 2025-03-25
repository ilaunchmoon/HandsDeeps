import torch 
import torch.nn as nn 
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.utils.load_mnist import load_fashion_mnist, get_fashion_mnist_labels, show_images
from src.visualization.animator_tool import Animator
from src.utils.accumulator  import Accumulator
from matplotlib import pyplot as plt


# 定义softmax操作
def softmax_operator(x):
    x = torch.exp(x)    
    normal_num = x.sum(1, keepdim=True)         # 对张量x的第1个维度(即列)所在的维度进行求和, 作为规范数
    return x /  normal_num


# 定义返回训练数据
def data_iter(batch_size, num_works, resize=None):
    return load_fashion_mnist(batch_size, num_works, resize)

def init_net():
    init_w, init_b = torch.normal(0, 0.01, size=(784, 10), requires_grad=True), torch.zeros(10, requires_grad=True)
    return init_w, init_b

# 定义模型
def softmax_net(x, init_w, init_b):
    return softmax_operator(torch.matmul(x.reshape((-1, init_w.shape[0])), init_w) + init_b)
    
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:     # 检查y_hat是否为二维张量, 第1个维度的长度是否大于1
        y_hat = y_hat.argmax(axis=1)                    # 对 y_hat 沿着第1个维度（即类别维度）取最大值的索引, 此时y_hat就变成了一维张量
    cmp = y_hat.type(y.dtype) == y                      # cmp是一个布尔张量, 用于表示预测值和标签值是否一致
    return float(cmp.type(y.dtype).sum())               # 由于pytorch中布尔张量不能直接求和, 所以需要将cmp转成float类型之后再使用sum()求和, 凡是True的都会转成1, False转成0, 此时sum()之后就是预测正确的数量

def evaluate_accuracy(net, init_w, init_b, data_iter):
    if isinstance(net, nn.Module):
        net.eval()                                      # 开启模型
    metric = Accumulator(2)                                # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X, init_w, init_b), y), y.numel())  # accuracy(net(X), y)是返回当前X批次中预测正确的数量, y.numel()是统计当前批次中参加预测的总数(即当前批次的样本总数), y.numel()是统计y张量的所有元素的个数
    return metric[0] / metric[1]                        # metric[0]: 是预测正确的总数, metric[1]: 是样本总数


# 定义损失函数
def loss_func(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])      # [range(len(y_hat))]生成从[0, len(y_hat) - 1]的整数用作于y_hat二维张量的行索引, y是标签用作为y_hat二维张量的列索引

# 定义优化器
def optimer(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 定义批次的训练函数
def train_epoch(net, init_w, init_b, lr, batch_size, train_iter, loss, optimer):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X, init_w, init_b)
        l = loss(y_hat, y)
        if isinstance(optimer, torch.optim.Optimizer):
            optimer.zero_grad()
            l.mean().backward()
            optimer.step()
        else:
            l.sum().backward()
            optimer([init_w, init_b], lr, batch_size)
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


# 定义训练函数
def train(net, init_w, init_b, lr, batch_size, train_iter, test_iter, loss, num_epochs, optimer):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, init_w, init_b, lr, batch_size, train_iter, loss, optimer)
        test_acc = evaluate_accuracy(net, init_w, init_b, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc, ))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# 定义预测函数
def predict(net, init_w, init_b, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X, init_w, init_b).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == "__main__":
    num_epochs, batch_size, num_inputs, num_outputs, lr = 10, 256, 784, 10, 0.1     # 初始化超参数
    num_works = 4                                                                   # 初始化线程数
    init_w, init_b = init_net()
    train_iter, test_iter = load_fashion_mnist(batch_size, num_works)
    train(softmax_net, init_w, init_b, lr, batch_size, train_iter, test_iter, loss_func, num_epochs, optimer)
    plt.show()
    pred = predict(softmax_net, init_w, init_b, test_iter)
    print(pred)
    


    
