import torch 
import torch.nn as nn 
from torch.nn import functional as F 


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))
    

if __name__ == "__main__":
    net = MLP()
    X = torch.rand(10, 20)
    net_impl = net(X)

    torch.save(net.state_dict(), './result_data/net_impl')      # 保存模型: 注意没有保存模型的实例对象./result_data/net_impl, 而是保存的net.state_dict(), 这里./result_data/net_impl只是一个本地存放路径
    net_cone = MLP()                                            # 注意加载保存的模型前, 要新创建一个MLP()的实例
    # net_cone.load_state_dict(torch.load('net_impl'))          # 使用新创建的模型对象来加载保存的模型, 若不是设置weights_only=True, 则会报警告
    net_cone.load_state_dict(torch.load('./result_data/net_impl', weights_only=True))            # 使用新创建的模型对象来加载保存的模型
    net_cone.eval()
    Y = net_cone(X)
    print(Y == net_impl)                                        # 验证Y和net_impl是否完全一致: 如果一致会输出一个全为True的张量

    




    