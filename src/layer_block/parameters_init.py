import torch 
import torch.nn as nn 

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)     # 使用指定均值和方差对权重矩阵进行随机初始化
        nn.init.zeros_(m.bias)                          # 使用0矩阵对偏置进行初始化


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)                  # 使用全1对权重矩阵进行初始化, 即使用常量进行初始化
        nn.init.zeros_(m.bias)                          # 使用0矩阵对偏置进行初始化

def init_constant_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)                 # 使用全42对权重矩阵进行初始化, 即使用常量进行初始化
        nn.init.zeros_(m.bias)    

def init_xavier(m):
    if type(m) == nn.Linear:                        
        nn.init.xavier_uniform_(m.weight)               # 使用xavier方法对权重矩阵初始化为符合均匀分布的随机值

# 自定义的初始化方法, 将其初始化为开发者自定义的分布
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5
        



if __name__ == "__main__":
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        nn.Linear(8, 1))
    
    net.apply(init_normal)
    print(net[0].weight.data)
    print(net[0].weight.data[0])
    print(net[0].bias.data)
    print(net[0].bias.data[0])

    net.apply(init_constant)
    print(net[0].weight.data)
    print(net[0].weight.data[0])
    print(net[0].bias.data)
    print(net[0].bias.data[0])

    # 以下演示同一个网络中的不同层进行不同的初始化
    net[0].apply(init_constant_42)
    print(net[0].weight.data)
    print(net[0].weight.data[0])
    print(net[0].bias.data)
    print(net[0].bias.data[0])

    net[2].apply(init_xavier)
    print(net[2].weight.data)
    print(net[2].weight.data[0])
    print(net[2].bias.data)
    print(net[2].bias.data[0])

    