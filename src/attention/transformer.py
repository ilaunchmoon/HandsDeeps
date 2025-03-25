import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# 位置编码（Positional Encoding）
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, hidden_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 前馈神经网络（Feed Forward Network）
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# 多头注意力机制（Multi-Head Attention）
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        
        self.q_weight = nn.Linear(hidden_dim, hidden_dim)
        self.k_weight = nn.Linear(hidden_dim, hidden_dim)
        self.v_weight = nn.Linear(hidden_dim, hidden_dim)
        self.out_weight = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        query = self.q_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        key = self.k_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        value = self.v_weight(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_weight(x)

# Transformer 编码器层（Encoder Layer）
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_dim, head_num, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attention(x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

# Transformer 编码器（Encoder）
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, head_num, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, head_num, ffn_dim, dropout_rate)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Transformer 解码器层（Decoder Layer）
class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_dim, head_num, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attention = MultiHeadAttention(hidden_dim, head_num, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout_rate)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attention(x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

# Transformer 解码器（Decoder）
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, head_num, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, head_num, ffn_dim, dropout_rate)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

# Transformer 模型（完整架构）
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, head_num, ffn_dim, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.encoder = TransformerEncoder(num_layers, hidden_dim, head_num, ffn_dim, dropout_rate)
        self.decoder = TransformerDecoder(num_layers, hidden_dim, head_num, ffn_dim, dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_encoding(self.embedding(src))
        tgt = self.pos_encoding(self.embedding(tgt))
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)      # 交叉注意力机制
        return self.fc_out(decoder_output)



# Generate src_mask (Padding Mask)
def create_padding_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

# Generate tgt_mask (Causal Mask + Padding Mask)
def create_tgt_mask(tgt, pad_idx):
    seq_len = tgt.size(1)
    padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(tgt.device)  # (seq_len, seq_len)
    return padding_mask & causal_mask  # Combine both masks



if __name__ == "__main__":
    # Hyperparameters
    input_dim = 100  # Vocabulary size
    output_dim = 100  # Vocabulary size
    hidden_dim = 512
    num_layers = 6
    head_num = 8
    ffn_dim = 2048
    dropout_rate = 0.1
    seq_len = 10  # Length of input sequence
    batch_size = 32
    pad_idx = 0  # Padding index in vocab

    # Sample input and target sequences (random data for demonstration)
    src = torch.randint(1, input_dim, (batch_size, seq_len))  # Source sequence (avoid pad index 0)
    tgt = torch.randint(1, output_dim, (batch_size, seq_len))  # Target sequence

    # Randomly introduce padding (set some tokens to pad_idx)
    src[:, -2:] = pad_idx  # Last two tokens in each src sequence are padding
    tgt[:, -3:] = pad_idx  # Last three tokens in each tgt sequence are padding

    # Transformer model
    model = Transformer(input_dim, output_dim, hidden_dim, num_layers, head_num, ffn_dim, dropout_rate)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # Ignore padding when computing loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    src_mask = create_padding_mask(src, pad_idx)  # (batch_size, 1, 1, seq_len)
    tgt_mask = create_tgt_mask(tgt, pad_idx)  # (batch_size, 1, seq_len, seq_len)

    # Forward pass
    optimizer.zero_grad()
    output = model(src, tgt, src_mask, tgt_mask)

    # Assume we're predicting the next token, so we use the target for calculating loss
    output_dim = output.size(-1)  # Should be equal to output_dim (vocab size)
    loss = criterion(output.view(-1, output_dim), tgt.view(-1))  # Flatten to (batch_size * seq_len, vocab_size)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")
    