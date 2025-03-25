import torch 
import torch.nn as nn 
from torch.nn import functional as F


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return X - X.mean()     # X - X.mean()时, 会发生广播机制, 即X.mean()会成为一个扩展为一个与X形状一致的张量
    

if __name__ == "__main__":
    layer = CenteredLayer()
    print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

