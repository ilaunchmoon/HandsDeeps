import torch
import torch.nn as nn 
import math


class MultiSelfAtten(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = self.hidden_dim // self.head_num

        # Q、K、V: (batch_size, seq_len, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.values = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    
    def forward(self, x, masked_attention=None):
        query = self.query(x)
        key = self.key(x)
        values = self.values(x)

        # (batch_size, seq_len, hidden_dim) --> (batch_size, seq_len, head_num, head_dim)
        # --> (batch_size, head_num, seq_len, head_dim)
        batch_size, seq_len, _ = x.size()
        query_state = query.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        key_state = key.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        values_state = values.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        # (batch_size, head_num, seq_len, seq_len)
        attention_score_weight = torch.matmul(
            query_state, key_state.transpose(-1, 2 )
        ) / math.sqrt(self.head_dim)

        if masked_attention is not None:
            attention_score_weight = attention_score_weight.masked_fill(
                masked_attention == 0,
                float("-inf")
            )
        
        attention_score_weight = torch.softmax(attention_score_weight, dim=-1)
        attention_score_weight = self.dropout(attention_score_weight)
        
        output = torch.matmul(
            attention_score_weight, values_state
        )

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)
        output = self.output(output)
        return output


if __name__ == "__main__":
    x = torch.rand(4, 5, 10)
