import torch
import math
import torch.nn as nn 
from torch.nn import functional as F 
from dataclasses import dataclass
from torch.utils.data import Dataset

@dataclass
class GPTConfig:
    block_size: int = 512                   # seq_size: 文本的最大长度
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768                       # hidden_dim, hidden_size
    dropout: float = 0.1
    head_size: int = n_embd // batch_size   
    vocab_size: int = 50257
    hidden_dim: int = n_embd
    


class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)

        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float('inf')
        )
        weight = F.softmax(weight / math.sqrt(self.head_size), dim=-1) 
        weight = self.dropout(weight)
        out = weight @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config:GPTConfig):
        self.heads = nn.ModuleList(
            [SingleHeadAttention(config)]
            for _ in range(config.n_head)
        )
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, config:GPTConfig):
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, x):
        x = x + self.att(x)
        x = x + self.ffn(x)
        return x 

class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        # (embedding, position, norm, mlp, block)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_embd)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight     # tie weight, 权重共享
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)    
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        batch, seq_len = idx.size()         # (batch_size, seq_len, n_embd)
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embeding_table(
            torch.arange(seq_len, device=idx.device)        # 确保位置编码和输入的idx是在同一个设备上
        )
        x = token_emb + pos_emb             # (batch_size, seq_len, n_embd)
        x = self.blocks(x)            
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len, vocab_size)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        pass



class MyDataSet(Dataset):
    def __init__(self, path, block_size=512):
        super().__init__()
        import tiktoken
        self.enc = tiktoken.get_encoding('gpt2')
        self.block_size = block_size                    # 序列最大长度
        self.encoded_data = []

        self.eos_token = self.enc.encode(               # 特殊符号分割训练文本
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        # 读取原始文本, 由于此处的原始文本是json格式存放, 则使用json包每次加载读取1000行, 并存放在raw_data列表中
        self.max_lines = 1000
        import json
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i > self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())["text"]
                    raw_data.append(text)
                except Exception as e:
                    continue
        
        # 将原始文本进行编码, 存放在full_encoded列表中
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])        # 注意将每行原始文本加上分割符合的编码抖添加到full_encoded
                
        # 将原始文本的编码变成最大编码长度512, 即需要截断
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i+self.block_size+1]
            if len(chunk) < self.block_size + 1:                        # 如果不能整除512, 则选择padding填满, 其实也可以选择直接丢弃
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
    
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, index):
        chunk = self.encoded_data[index]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def encode(self, text):
        return self.enc.decode(text)

    def decode(self, ids):
        return self.enc.decode(ids)
    

if __name__ == "__main__":
    pass
    
