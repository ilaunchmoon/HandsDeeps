import torch
import torch.nn as nn 
from torch.nn import functional as F 


if __name__ == "__main__":
    # 存放张量
    x = torch.arange(4)
    torch.save(x, 'x_tensor')
    x1 = torch.load('x_tensor')         # 加载张量

    # 存放多个张量
    y = torch.ones(2, 3)
    torch.save([x, y], 'x_y_tensor')
    x2, y1 = torch.load('x_y_tensor')   # 加载张量

    # 以字典的形式保存和加载张量
    mydict = {'x': x, 'y': y}
    torch.save(mydict, 'mydict_x_y')
    mydict1 = torch.load('mydict_x_y')

    
