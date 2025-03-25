import sys
import os
import re
import collections
import torch
import random
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.utils.download_tool import DATA_HUB, DATA_URL, download


# 下载time_machine文件
# 将将所有非单词都使用空格替换, 将所有单词都转成小写
def get_time_machine_data():
    DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt', 
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# 词元化
# 即一个单词是一个token, 这是最简单的分词方法
def tokenize(lines, token='word'):                      
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print("error: unkown token type: " + token)

# 统计每个单词出现的频率
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):     # token是1D或2D列表
        tokens = [token for line in tokens for token in line]   # 将词元展成一个列表
    return collections.Counter(tokens)


# 利用每个单词出现的频率来构建词汇表
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        
        if reserved_tokens is None:
            reserved_tokens = []
        
        counter = count_corpus(tokens)      # 按照出现频率排序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens     # 未知词元的索引为0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self): # 未知词元的索引为0
        return 0
    

    @property 
    def token_freqs(self):
        return self._token_freqs


def get_vocab_token_idx(max_tokens=-1):                 # 获取词汇表和词汇表中所有词元的索引
    lines = get_time_machine_data()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    token_idx = [vocab[token] for line in tokens for token in line]     # 词汇表中的词元索引
    if max_tokens > 0:
        token_idx = token_idx[:max_tokens]
    return token_idx, vocab


# 随机采样
# 利用词元在词汇表中的索引来随机转为tensor
def seq_data_iter_random(token_idx, batch_size, num_steps):
    token_idx = token_idx[random.randint(0, num_steps - 1):]
    num_subseqs = (len(token_idx) - 1) // num_steps             # 减去1, 因为有标签
    init_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(init_indices)

    def data(pos):
        return token_idx[pos: pos + num_steps]
    
    num_batchs = num_subseqs // batch_size
    for i in range(0, batch_size * num_batchs, batch_size):
        init_indices_per_batch = init_indices[i:i + batch_size]
        X = [data(j) for j in init_indices_per_batch]
        Y = [data(j + 1) for j in init_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)              # 转成tensor


# 顺序采样
# 利用词元在词汇表中的索引来随机转为tensor
def seq_data_iter_sequential(token_idx, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(token_idx) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(token_idx[offset:offset + num_tokens])
    Ys = torch.tensor(token_idx[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batchs = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batchs, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        yield X, Y 


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.token_idx, self.vocab = get_vocab_token_idx(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.token_idx, self.batch_size, self.num_steps)

# 加载词汇表中每个词元转为tensor之后的数据
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab



if __name__ == "__main__":
    lines = get_time_machine_data()
    print(lines[0])
    print(lines[10])
