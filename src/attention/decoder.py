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
    


class MultiAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.q_weight = nn.Linear(hidden_dim, hidden_dim)
        self.k_weight = nn.Linear(hidden_dim, hidden_dim)
        self.v_weight = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x, masked=None):
        batch_size, seq_len, _ = x.size()
        query = self.q_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        key = self.k_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        value = self.v_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        atten_tmp = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if masked is not None:
            atten_tmp = atten_tmp.masked_fill(
                masked==0, 
                float("-inf")
            )
        
        attention_score = torch.softmax(atten_tmp, dim=-1)
        attention_score = self.dropout(attention_score)
        x = torch.matmul(attention_score, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return x 
    



class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiAttention(hidden_dim, head_num, dropout_rate)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    
    def forward(self, x, masked=None):
        residual = x 
        x = self.attention(x)
        x = self.layer_norm(x + residual)
        residual = x 
        x = self.dense(x)
        return self.layer_norm(x + residual)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate=0.2, layer_num=8, max_seq_len = 512, vocab_size=151643):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = PositionFNN(hidden_dim, max_seq_len)
        self.decoder_layer = nn.ModuleList([DecoderLayer(hidden_dim, head_num, dropout_rate) for _ in range(layer_num)])
        self.ffn = nn.Linear(hidden_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
    
    def forward(self, x, marsked=None):
        x_emb = self.embedding(x)
        x = x_emb + self.pos_embedding(x_emb)
        for layer in self.decoder_layer:
            x = layer(x, marsked)
        x = self.ffn(x)
        return x 
    


if __name__ == "__main__":
    hidden_dim, head_num = 512, 8
    vocab_size = 151643
    batch_size, seq_len = 100, 45
    
    # 生成符合要求的输入张量
    x = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]
    decoder = Decoder(hidden_dim, head_num, layer_num=32)
    # 验证前向传播
    output = decoder(x)
    print("Output shape:", output.shape)  # 预期输出 [2, 32, 151643] [batch_size, seq_len, vocab_size]

    # 打印模型结构（包含参数统计）
    summary(decoder, input_data=x)  # 显式传入输入张量和masked=None
    
    


