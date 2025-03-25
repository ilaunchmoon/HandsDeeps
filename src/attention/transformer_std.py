import torch
import math
import torch.nn as nn 

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, ffn_dim, out_dim, dropout_rate=0.5):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, out_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.ffn(x)
    

class PositionEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, hidden_dim)
        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32).unsqueeze(0) * (-(math.log(10000.0) / hidden_dim)))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)
        self.register_buffer("pe", self.pe)

    def forward(self, x):
        return x + self.pe[:, x.size(1)]
    


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate=0.5):
        super().__init__()
        self.hidden_dim - hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.q_weight = nn.Linear(hidden_dim, hidden_dim)
        self.k_weight = nn.Linear(hidden_dim, hidden_dim)
        self.v_weight = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key=None, value=None, masked=None):
        batch_size, seq_len, _ = query.size()
        if key is  None:
            key = query
        if value is None:
            value = query
        
        query = self.q_weight(query).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        key = self.k_weight(key).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        value = self.v_weight(value).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        att = torch.matmul(query, key.transpose(-2, -1))
        if masked is None:
            att  = att.masked_fill(
                masked == 0, 
                float("-inf")
            )    
        att_score = torch.softmax(att, dim=-1)
        att_score = self.dropout(att_score)
        attention = torch.matmul(att_score, value)

        x = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return x 


        
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate):
        pass
    
    def forward(self, x, src_masked=None):
        pass



class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate):
        pass
    
    def forward(self, x, encoder_output, src_masked=None, tgt_masked=None):
        pass

