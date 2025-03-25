import torch 
import torch.nn as nn 
import math 

class SelfAttention1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim) 
        self.dropout = nn.Dropout()

    
    def forward(self, x, maksed_att=None):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_values = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.hidden_dim)
    
        if maksed_att is not None:
            attention_scores = attention_scores.masked_fill(
                attention_scores == 0, 
                float("-inf")
            )
        attention_scores = torch.softmax(attention_values, dim=-1)          # 注意进行softmax()前完成padding的屏蔽
        attention_scores = self.dropout(attention_scores)
        attention = attention_scores @ v 
        return attention
    

class SelfAttention2(nn.Module):
    def _init__(self):
        super().__init__()
    
    def forward(self, x):
        pass


class SelfAttention3(nn.Module):
    def _init__(self):
        super().__init__()
    
    def forward(self, x):
        pass



class SelfAttention4(nn.Module):
    def _init__(self):
        super().__init__()
    
    def forward(self, x):
        pass
    
        
        

        
        
