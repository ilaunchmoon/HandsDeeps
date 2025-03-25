import torch
import torch.nn as nn 
from dataclasses import dataclass

@dataclass
class Config:
    hidden_dim: int
    head_num: int 
    head_dim: int = hidden_dim // head_num

    q_lora_rank: int            # 压缩Q的维度
    kv_lora_rank: int           # 压缩K和V的维度
    q_head_dim: int             # Q的头维度
    qk_nope_dim: int            # Q和K不带位置编码的维度
    kv_head_dim: int            # K和V的头维度
    qk_rope_dim: int            # Q和K的旋转位置的维度, Q和K的进行位置编码的维度是一致的
    v_head_dim: int             # V的头维度

    dropout: float = 0.1


class MultiLatentAttention(nn.Module):
    def __init__(self, config:Config)->None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.head_num = config.head_num
        self.head_dim = config.head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.q_head_dim = config.q_head_dim
        self.kv_head_dim = config.kv_head_dim
        self.qk_rope_dim = config.qk_rope_dim
        self.qk_nope_dim = config.qk_nope_dim
        self.v_head_dim = config.v_head_dim
        self.dropout = nn.Dropout(config.dropout)

        # q的压缩和升维
        self.q_compression = nn.Linear(self.hidden_dim, self.q_lora_rank)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.q_head_dim * self.head_num)
        
        # kv压缩和升维
        self.kv_compression = nn.Linear(self.hidden_dim, self.kv_lora_rank + self.qk_rope_dim)
        # self.q_head_dim - self.qk_rope_dim是 Q和K不带位置编码的维度, 两个维度是一致的
        # 由于K和V升维需要分裂为不带位置编码的k和v, 所以这里需要self.v_head_dim来预留V的维度
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.head_num * (self.q_head_dim - self.qk_rope_dim + self.v_head_dim))   

        # 输出维度, 其实这里的v_head_dim == q_head_dim == k_head_dim
        self.out_proj = nn.Linear(self.v_head_dim, self.hidden_dim)     


    def forward(self, hidden_size:torch.Tensor, masked:torch.Tensor=None)->torch.Tensor:
        batch_size, seq_len, _ = hidden_size.size()

        # (batch_size, seq_len, hidden_dim) --> (batch_size, seq_len, q_lora_rank)
        # q 压缩
        q = self.q_compression(hidden_size)
        # q 升维
        q = self.q_up_proj(q).view(batch_size, seq_len, self.head_num, self.q_head_dim).transpose(1, 2)
        # (batch_size, seq_len, hidden_dim) --> (batch_size, seq_len, kv_lora_rank + qk_rope_dim)
        # q分裂为不带位置编码的q和带位置编码的rope
        q_nope, q_rope = torch.split(q, [self.q_head_dim - self.qk_rope_dim, self.qk_rope_dim], dim=-1)

        # (batch_size, seq_len, hidden_dim) --> (batch_size, seq_len, kv_lora_rank + qk_rope_dim)
        # kv压缩: kv分裂为不带位置编码的kv和带位置编码的rope
        kv = self.kv_compression(hidden_size)
        # kv 分裂
        kv_nope, kv_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_dim], dim=-1)
        # kv 升维, 这里的v_head_dim是为了预留V的维度, 因为升维度要分裂为不带位置编码的k和v
        # self.q_head_dim - self.qk_rope_dim = self.qk_nope_dim
        kv = self.kv_up_proj(kv_nope).view(batch_size, seq_len, self.head_num, self.qk_nope_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_dim, self.v_head_dim], dim=-1)

        







    

    
        
