import torch 
import torch.nn as nn 


if __name__ == "__main__":
    shared = nn.Linear(8, 8)        # 共享参数层
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        shared, nn.ReLU(),
                        shared, nn.ReLU(), 
                        nn.Linear(8, 1))
    x = torch.rand(2, 4)
    print(net(x))
    print(net[2].weight.data[0] == net[4].weight.data[0])       # 验证参数是否相同: 因为shared就是同一参数层, 所以值一定是相同的
    net[2].weight.data[0,0] = 100
    print(net[2].weight.data[0] == net[4].weight.data[0])       # 验证参数是否相同: 因为shared是同一个实例对象, 则也是相同的

    # 共享参数: 一般是有个相同的实例对象, 即任意一个发生改变, 其他n-1个都会发生改变, 并且在反向传播中梯度值会发生累积
    