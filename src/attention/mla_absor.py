import torch
import torch.nn as nn 
from typing import Optional, Tuple

"""
带矩阵吸收版的MLA

"""

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,       # 是否在缩放时添加单位偏移（即 1 + weight），默认为 True
    )->None:    
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x)->torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x)->torch.Tensor:
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output


"""
    DeepSeek的RMSNorm: 没有添加1缩放, 而且可学习参数也是初始化为全1
"""
class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RopeEmbedding(nn.Module):
    def  __init__(self, hidden_dim:int, max_position_embeddings:int=2048, base:int=10000, device:torch.device=None)->None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_dim, 2).float().to(device) / hidden_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cached(max_position_embeddings, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        self.max_seq_len_cached:Optional[int] = None 

    def _set_cos_sin_cached(self, seq_len:int, device:torch.device, dtype:torch.dtype)->None:          
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq.to(device))    # 或 sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq) 其实就是求外积
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cache", emb.sin().to(dtype), persistent=False)

    def forward(self, x:torch.Tensor, seq_len:int)->Tuple[torch.Tensor, torch.Tensor]:
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cached(seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cache[:seq_len].to(dtype=x.dtype),
            self.sin_cache[:seq_len].to(dtype=x.dtype)
        )


class MLA(nn.Module):
    def __init__(self)->None:
        pass

    def forward(self)->torch.Tensor:
        pass



if __name__ == "__main__":
    dim = 512 
    x = torch.rand(2, 100, dim)
    rope_embedding = RopeEmbedding(dim)
    print(rope_embedding(x, 100))

