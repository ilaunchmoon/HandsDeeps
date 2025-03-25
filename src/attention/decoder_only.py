import torch 
import torch.nn as nn 
import math 

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.query_weight = nn.Linear(hidden_dim, hidden_dim)
        self.key_weight = nn.Linear(hidden_dim, hidden_dim)
        self.value_weight = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.ffn_up_dim = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn_down_dim = nn.Linear(hidden_dim * 4, hidden_dim)
        
    
    
    def ffn(self, x):
        up = self.ffn_up_dim(x)
        up = self.gelu(up)
        down = self.dropout(self.ffn_down_dim(up))
        return self.layer_norm(x + down)                # 残差
    
    def multi_head_attention(self, q, k, v, masked=None):
        # q,k,v: (batch_size, head_num, seq_len, head_dim)
        # (batch_size, head_num, seq_len, seq_len)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 注意q和k进行矩阵运算其非批次维度要满足矩阵乘法

        # 注意力机制掩码和填充掩码(如果有填充的话), 但是注意力一定有掩码操作
        if masked is not None:
            # 说明没有填充掩码, 只有注意力机制掩饰码
            masked = masked.tril()      # 就使用原本用于填充掩码的来作为注意力机制的掩码矩阵, 用一个下三角矩阵, 现在是因为没有填充掩码, 所以可以不同单独声明一个新的掩码矩阵来存放
            attention = attention.masked_fill(
                masked==0,
                float("-inf")
            )
        else:
            # 说明有填充掩码, 所以此时既要考虑填充掩码, 也要考虑注意力掩码
            masked = torch.ones_like(attention).tril()
            attention = attention.masked_fill(
                masked==0,
                float("-inf")
            )
         # (batch_size, head_num, seq_len, seq_len)
        attention = torch.softmax(attention, dim=-1)        # 注意力得分函数进行softmax操作
        attention = self.dropout(attention)                 # 注意力得分进行dropout后再与v乘

        # (batch_size, head_num, seq_len, head_dim)
        out_tmp = torch.matmul(attention, v)        
        out_tmp = out_tmp.transpose(1, 2).contiguous()      # (batch_size, head_num, seq_len, head_dim) ---> (batch_size, seq_len, head_num, head_dim)
        batch_size, seq_len, _, _ = out_tmp.size()
        out_tmp = out_tmp.view(batch_size, seq_len, -1)     #  (batch_size, seq_len, head_num, head_dim) --->  (batch_size, seq_len, hidden_dim)
        return self.output(out_tmp)


    def mha_layer(self, x, masked=None):
        batch_size, seq_len, _ = x.size()
        # (batch_size, seq_len, hidden_dim) --> (batch_size, seq_len, head_num, head_dim) ---> (batch_size, head_num, seq_len, head_dim)
        query = self.query_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        key = self.key_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        value = self.value_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        out = self.multi_head_attention(query, key, value, masked)   # 多头注意力机制
        return self.layer_norm(x + out)                 # 残差

    
    def forward(self, x, masked=None):
        x = self.mha_layer(x, masked)
        x = self.ffn(x)
        return x 




class Decoder(nn.Module):
    def __init__(self, seq_len, hidden_dim, head_num, dropout_rate=0.1, layer_num=5):
        super().__init__()
        self.multi_att_layer = nn.ModuleList(
            DecoderLayer(hidden_dim, head_num, dropout_rate) for _ in range(layer_num)
        )
        self.embedding = nn.Embedding(seq_len, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, masked=None):
        # (batch_size, seq_len) ----> (batch_size, seq_len, hidden_dim)
        x = self.embedding(x)  
        for layer in self.multi_att_layer:  # 逐层传递 x
            x = layer(x, masked)
        x = self.out_linear(x)
        return torch.softmax(x, dim=-1)


if __name__ == "__main__":
    x = torch.randint(0, 10, (3, 4))   # x: batch_size, seq_len = 3, 4, 但是nn.Embeding()中的seq_len必须等于 = 3 * 4, 否则会报错
    net = Decoder(10, 8, 2)            # seq_len, hidden_dim, head_num
    masked = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 0]
        ]
    ).unsqueeze(1).unsqueeze(2)

    print(net(x, masked))


