import torch 
import torch.nn as nn 

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net


if __name__ == "__main__":
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8,1))
    X = torch.rand(2, 4)
    print(net(X))
    print('\n')
    print(net[0].state_dict())
    print('\n')
    print(net[1].state_dict())
    print('\n')
    print(net[2].state_dict())
    print('\n')
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)
    print("get all parameters")
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print("\n")
    print(*[(name, param.shape) for name, param in net.named_parameters()])
    print("block layer parameters")
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    print(rgnet(X))
    print(rgnet)
    print(rgnet[0][1][2].bias.data)
    
