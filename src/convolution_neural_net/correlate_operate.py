import torch
import torch.nn as nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y 

class Conv2d(nn.Module):
    def __init__(self, kernel_size):
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(kernel_size))
    
    def forward(self, X):
        return corr2d(X, self.weight) + self.bias



if __name__ == "__main__":
    X = torch.rand(3, 3)
    K = torch.rand(2, 2)
    print(corr2d(X, K))



