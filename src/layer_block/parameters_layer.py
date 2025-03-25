import torch 
import torch.nn as nn 
from torch.nn import functional as F

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))
    
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

    
if __name__ == "__main__":
    linear = MyLinear(5, 3)
    print(linear.weight)

    # 使用自定义层进行前向传播
    linear(torch.rand(2, 5))
    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))
    