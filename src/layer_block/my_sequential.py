import torch
import torch.nn as nn 

class MySequent(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X 
    

if __name__ == "__main__":
    net = MySequent(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 10), nn.ReLU())
    X = torch.rand(10, 10)
    print(net(X))