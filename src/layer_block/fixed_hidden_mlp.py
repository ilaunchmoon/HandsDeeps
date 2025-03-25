import torch 
import torch.nn as nn 
from torch.nn import functional as F


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
    

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU())
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        return self.linear(self.net(X))


if __name__ == "__main__":
    net = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    X = torch.rand(10, 20)
    print(net(X))   

