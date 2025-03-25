import torch 
import torch.nn as nn 
import math 


"""
Group Query Attention的原理

在MultiHead Attention中, Q、K、V都是(batch_size, seq_len, hidden_dim)分成head_num个, 即(batch_size, seq_len, head_num, head_dim)
现在GQA中会将KV分成kv_head_num个小组来共享Q, 也就是说假如原理MHA是有head_hum个注意力头来进行注意力计算, 现在又会将KV以更大些如kv_head_num个为一组来共享Q进行注意力计算
即:
    Q: (batch_size, head_num, seq_len, head_dim)
    K: (batch_size, )
"""
class GroupQueryAtt1(nn.Module):
    def __init__(self, hidden_dim, head_num, num_group, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim    
        self.head_num = head_num                        # 注意力头数
        self.head_dim = hidden_dim // head_num          # 每个注意力头的维度
        self.num_group = self.head_num // num_group     # 组数
        self.head_num_pre_group = self.head_num // self.num_group   # 每组的注意力头数

        
        self.q_weight = nn.Linear(hidden_dim, hidden_dim)
        self.k_weight = nn.Linear(hidden_dim, self.head_dim * num_group)
        self.v_weight = nn.Linear(hidden_dim, self.head_dim * num_group)
        self.out_weight = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # (batch_size, seq_len, hidden_dim) --> (batch_size, seq_len, group_num, head_num_pre_group, head_dim) --->
        # (batch_size, group_num, head_num_pre_group, seq_len, head_dim)
        query = self.q_weight(x).view(batch_size, seq_len, self.num_group, self.head_num_pre_group).permute(0, 2, 3, 1, 4)

        # (batch_size, seq_len, hidden_dim) --> (batch_size, seq_len, group_num, head_dim) --> (batch_size, group_num, 1, seq_len, head_dim)
        key = self.k_weight(x).view(batch_size, seq_len, self.num_group, self.head_dim).transpose(1, 2).unsqueeze(2)
        value = self.v_weight(x).view(batch_size, seq_len, self.num_group, self.head_dim).transpose(1, 2).unsqueeze(2)

        attention = torch.matmul(query, key) / math.sqrt(self.head_dim)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf")) 
        
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, value)
        x = x.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, seq_len, -1)
        return self.out_weight(x)
    



if __name__ == "__main__":
    pass