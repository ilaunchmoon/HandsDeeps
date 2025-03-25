import torch
import math 
import torch.nn as nn 
from torchinfo import summary


class PositionFNN(nn.Module):
    def __init__(self, hidden_dim:int, max_seq_len:int=512)->None:
        super().__init__()
        pe = torch.zeros(max_seq_len, hidden_dim)       # 生成一个全0的位置编码矩阵

        # 为输入的token序列中的每个token生成位置序列, 并且由[max_seq_len]扩展为[max_seq_len, 1]
        # 因为输入token序列是(batch_size, seq_len, hidden_dim), 先要扩展为(seq_len, hidden_size), 为最后一步从(max_seq_len, hidden_dim) ---> (batch_size, max_seq_len, hidden_dim)做准备
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)     
        # print(position.shape)

        # 用于计算sin和cos中的分母部分
        div = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float) *         # 用于生产(0, hidden_dim - 1)中隔位等差序列, 作为第0, 2, 4, .. hidden_dim维, 即i
            -(torch.tensor(10000.0, dtype=torch.float) / hidden_dim)
        )
        # print(div.shape)


        pe[:,0::2] = torch.sin(position * div)      # pos位置上的这个token上的偶数维度用sin
        pe[:,1::2] = torch.cos(position * div)      # pos位置上的这个token上的奇数维度用cos

        self.register_buffer('pe', pe.unsqueeze(0)) # 由于pe是不可学习的参数, 可以直接注册到缓冲区, 可以提高速度, 由[max_seq_len, hidden_dim] 扩充一个批次维度(即第0维) --> (1, max_seq_len, hidden_dim)


    def forward(self, x):
        # (batch_size, seq_len, hidden_dim)
        # self.pe --> (1, max_seq_len, hidden_dim): self.pe[:, x.size(1)]代表第0个维度全取, 第1个维度取x的第1维度(即x.size(1)), 最后一个维度会自动匹配上, 这里其实就是广播
        return self.pe[:, :x.size(1)]  # 动态适配序列长度

        

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim:int, head_num:int, dropout_rate:float=0.1)-> None: 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num

        self.q_weight = nn.Linear(hidden_dim, hidden_dim)
        self.k_weight = nn.Linear(hidden_dim, hidden_dim)
        self.v_weight = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            self.dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x:torch.tensor, masked:torch.tensor=None):
        residual = x 
        batch_size, seq_len, _ = x.size()
        
        # (batch_size, seq_len, hidden_dim) -->
        # (batch_size, seq_len, head_num, head_dimm) -->
        # (batch_size, head_num, seq_len, head_dim)
        query = self.q_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        key = self.k_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        value = self.v_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)


        # (batch_size, head_num, seq_len, seq_len)
        attention_tmp = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if masked is not None:
            attention_tmp = attention_tmp.masked_fill(
                masked==0,
                float("-inf")
            )

        # (batch_size, head_num, seq_len, seq_len)
        attention_score = torch.softmax(attention_tmp, dim=-1)
        attention_score = self.dropout(attention_score)
        attention = torch.matmul(attention_score, value)            # 注意力评分与value矩阵相乘

        # 合并多头结果
        x = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 残差连接 + 层归一化
        x = self.layer_norm(x + residual)  # 使用合并后的x
        
        # FFN部分
        residual = x
        x = self.ffn(x)
        x = self.layer_norm(x + residual)
        
        return x
    

class Encoder(nn.Module):
    def __init__(self, hidden_dim:int, head_num:int, dropout_rate:float=0.1, layer_num:int=8, max_seq_len:int = 512, vocab_size:int=151643)->None:
        super().__init__()
        self.pos_embedding = PositionFNN(hidden_dim, max_seq_len)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_block = nn.Sequential(
            *[EncoderLayer(hidden_dim, head_num, dropout_rate) for _ in range(layer_num)]
        )
        self.ffn = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x, masked=None):
        x_emb = self.embedding(x)  # 先完成嵌入
        x = x_emb + self.pos_embedding(x_emb)  # 再添加位置编码
        for layer in self.encoder_block:
            x = layer(x, masked)
        x = self.ffn(x)
        return x 
    


if __name__ == "__main__":
    hidden_dim, head_num = 512, 8
    vocab_size = 151643
    batch_size, seq_len = 100, 45
    
    # 生成符合要求的输入张量
    x = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]
    encoder = Encoder(hidden_dim, head_num, layer_num=32)

    # 打印模型结构（包含参数统计）
    summary(encoder, input_data=x)  # 显式传入输入张量和masked=None
    
    # 验证前向传播
    output = encoder(x)
    print("Output shape:", output.shape)  # 预期输出 [2, 32, 151643] [batch_size, seq_len, vocab_size]





        
        



        


