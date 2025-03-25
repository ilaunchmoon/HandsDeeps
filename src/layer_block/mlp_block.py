import torch
import torch.nn as nn 
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    


if __name__ == "__main__":
    X = torch.rand(10, 20)
    net = MLP()
    print(net(X))
