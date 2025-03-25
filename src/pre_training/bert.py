import torch
import random
import torch.nn as nn


def get_tokens_and_segments(token_a, token_b=None):
    """
        将输入进来的句子拼接为 <cls> 句子a <sep> 句子b <sep>, 如果句子b存在的话
    """
    tokens = ['<cls>']  + token_a + ['<sep>']
    segments = [0] * (len(token_a) + 2)
    if token_b is not None:
        tokens += token_b +['<sep>']
        segments += [1] * (len(token_b) + 1)
    return tokens, segments



def get_next_sentence(sentence, next_sentence, paragraphs):
    """
        当前函数的功能是, 输入同一个段落中紧邻的两句话和所有段落的集合
        如果当前生成了概率小于0.5, 就不将next_sentence替换为随机句子, 此时is_next=True, 也就是说sentence没有被替换为不是紧邻它的下一句
        如果当前生成了概率大于0.5, 就将next_sentence替换为随机句子, 此时is_next=Flase, 也就是说sentence被替换为不是紧邻它的下一句
        sentence: 第一个句子
        next_sentence: sentence的下一个句子, 也就是说next_sentence和sentence同一个段落的紧邻下一句
        paragraphs: 所有段落的集合, 例如

        [
            [  # 段落A
                ["我", "在", "读书"], 
                ["突然", "听到", "敲门声"], 
                ["然后", "打开", "了门"]
            ],
            [  # 段落B
                ["今天", "天气", "晴朗"], 
                ["适合", "去", "散步"]
            ],
            [  # 段落C
                ["深度", "学习", "很", "有趣"]
            ]
        ]

    """
    if random.random() < 0.5:       
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next



def get_nsp_data_from_paragrahs(paragraph, paragraphs, vocab, max_len):
    """
        用于获取下一句预测任务的数据
        paragraph: 一个段落
        paragraphs: 所有段落的集合
        vocab: 词汇表
        max_len: 获取句子最大长度
        注意: 一个段落中的句子是紧邻的, 也就是说一个段落中的句子是连续的
    """
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) -  1):
        token_a, token_b, is_next = get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        if len(token_a) + len(token_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(token_a, token_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph



def replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    """
        用于替换mask掉的词元
        tokens: 词元
        candidate_pred_positions: 候选预测位置
        num_mlm_preds: 预测的词元个数
        vocab: 词汇表
    """
    # 为了让模型更好的学习, 我们不会替换句子的第一个词元和最后一个词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        if random.random() < 0.8:
            masked_token = "<mask>"
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def get_mlm_data_from_tokens(tokens, vocab):
    """
        用于获取mask语言模型的数据
        tokens: 词元
        vocab: 词汇表
    """
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token not in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pre_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pre_positions, vocab[mlm_pred_labels]


def pad_bert_inputs(examples, max_len, vocab):
    """
        用于填充bert输入, 也就是说将输入的数据填充到相同的长度max_len
        examples: 一个batch的数据
        max_len: 最大长度
        vocab: 词汇表
        返回值: 返回填充后的数据, 包括所有的词元, 所有的段落, 所有的有效长度, 所有的预测位置, 所有的mask权重, 所有的mask标签
    """
    max_num_tokens = round(max_len * 0.15)  # 最大mask的词元个数
    all_token_ids, all_segments, valid_lens = [], [], []    # 词元, 段落, 有效长度
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []    # 预测位置, mask权重, mask标签
    nsp_labels = []   # 下一句预测标签
    for (token_ids, pred_positions, mlm_pred_labels, segments, is_next) in examples:  # 一个batch的数据
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long)) # 填充词元, 转为tensor
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long)) # 填充段落, 转为tensor
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))    # 有效长度, 转为tensor
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_tokens - len(pred_positions)), dtype=torch.long)) # 填充预测位置, 转为tensor
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_labels) + [0.0] * (max_num_tokens - len(mlm_pred_labels)), dtype=torch.float32))   # 填充mask权重, 转为tensor
        all_mlm_labels.append(torch.tensor(mlm_pred_labels + [0] * (max_num_tokens - len(mlm_pred_labels)), dtype=torch.long))  # 填充mask标签, 转为tensor
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))  #  下一句预测标签, 转为tensor
    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)






class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, ffn_dim, output_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, output_dim),
            nn.Dropout()
        )
    
    def forward(self, x):
        return self.ffn(x)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, masked=None):
        batch_size, seq_len, _ = query.size()
        


