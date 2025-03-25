import torch 
import random
from matplotlib import pyplot as plt
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.visualization.plot import set_figsize       
# 使用在根目录下运行指令: python -m src.linearRegression.ManImplinearRegression
# 如果你在当前模块下(python -u "/Users/icur/VScode/DeepL/src/linearRegression/ManImplinearRegression.py")
# from ..visualization.DrawTool import set_figsize  会报ImportError: attempted relative import with no known parent package导入错误


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

#  用于批量读取数据集(即特征和标签)
def data_iter(batch_size, features, labels):
    num_eamples = len(features)                         # len(张量)就是返回这个张量第0个维度的大小(即第0个维度的元素个数)
    indices = list(range(num_eamples))                  # 或num_eamples以内的所有元素作为索引
    random.shuffle(indices)                             # 打乱
    for i in range(0, num_eamples, batch_size):         # 随机去batch_size个样本
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_eamples)])       # 由于features和labels都是tensor类型, 所以batch_indices也要为tensor类型
        yield features[batch_indices], labels[batch_indices]

# 定义初始化模型参数的函数
def init_net():
    """
        初始化模型参数是指得初始化所有需要学习的参数
        由于这里是线性回归模型, 所以该模型的学习参数就是w、b
        也就是说损失对可学习参数进行求梯度, 然后向着让损失函数减少的方向来更新可学习参数
        则对多有可学习w、b的requires_grad必须设置为True
    """
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b

# 定义线性回归模型
def linear_regression(X, w, b):
    return torch.matmul(X, w) + b                       # b会执行广播机制

# 定义损失函数: 使用均方误差
def loss_linear(y_hat, y):                              # y_hat为预测值(即每次模型输入训练数据后, 模型的输出值), y为标签值
    return 0.5 * (y_hat - y.reshape(y_hat.shape)) ** 2

# 定义优化算法
def sgd(params, lr, batch_size):                        # 小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size      # 梯度更新
            param.grad.zero_()                         # 梯度清零



# 定义训练函数
def train(batch_size, num_epoch, lr, net, loss, features, labels, init_w, init_b, optimer):
    """
        batch_size:     批次大小    
        num_epoch:      训练次数
        lr:             学习率
        net:            模型
        loss:           损失函数
        features:       训练集中的特征
        labels:         训练集中的标签
        init_w:         初始化的模型参数w
        init_b:         初始化的模型参数b
        optimer:        优化算法: 一般是随机梯度下降
    """
    for epoch in range(num_epoch):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, init_w, init_b), y)
            l.sum().backward()
            optimer([init_w, init_b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, init_w, init_b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
            

if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])                    # 真实模型参数w, 待使用数据来训练得出
    true_b = 4.2                                        # 真实模型参数b, 待使用数据来训练得出
    features, labels = synthetic_data(true_w, true_b, 1000)      # 生成特征和标签
    set_figsize()
    plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
    plt.show()

    w, b = init_net()                                   # 初始化模型参数
    batch_size, lr, num_epochs = 10, 0.03, 3            # 设置超参数
    train(batch_size, num_epochs, lr, linear_regression, loss_linear, features, labels, w, b, sgd)      # 训练
    plt.show()

        
    

