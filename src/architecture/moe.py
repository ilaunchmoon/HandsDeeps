import torch
import torch.nn as nn 
from torch.nn import functional as F



class BasicExpert(nn.Module):
    def __init__(self, in_dim:int, out_dim:int)->None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x)->torch.Tensor:
        return self.fc(x)
    

class BasicMoE(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, num_experts:int)->None:
        super().__init__()
        self.gate = nn.Linear(in_dim, out_dim)
        self.experts = nn.ModuleList([BasicExpert(in_dim, out_dim) for _ in range(num_experts)])
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        expert_weight = self.gate(x)
        expert_out_list = [
            expert(x) for expert in self.experts
        ]

        expert_output = torch.concat(
            expert_out_list,
            dim=1
        )

        expert_weight = F.softmax(expert_weight, dim=1)     # 门口网络, 用于输出一个概率分布, 然后按照概率的前k个概率值去获取前k个专家的输出, 用于后面concat这个k和专家的输出
        output = expert_weight @ expert_output
        return output.squeeze(1)






class Expert(nn.Module):
    def __init__(self, n_embed:int, dropout_rate:float=0.1)->None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.net(x)

class TopkGating(nn.Module):
    def __init__(self, n_embed:int, n_experts:int, top_k:int, dropout_rate:float=0.1)->None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gates = nn.Linear(n_embed, n_experts)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        logits = self.gates(x)
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)            # 获取前k个最大值的值和索引
        zeros = torch.full_like(logits, float("-inf"))                           # 生成一个与logits相同形状的全为负无穷的张量
        zeros.scatter(dim=-1, index=top_k_indices, src=top_k_logits)             # 将top_k_logits的值填充到zeros中
        gates = F.softmax(zeros, dim=-1)                                         # 对zeros进行softmax操作
        return self.dropout(gates), top_k_indices                                # 返回dropout后的gates和top_k_indices
    

