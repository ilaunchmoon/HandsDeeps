import sys
import os
import torch.nn as nn 
import torch 
import math
from torch.nn import functional as F
import torch.optim.sgd
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.utils.load_time_machine import load_data_time_machine
from src.utils.time_tool import Timer
from src.utils.accumulator import Accumulator
from src.visualization.animator_tool import Animator


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    
    # Ht = φ(XtWxh + Ht-1Whh + bh)
    # hidden layer Params: Wxh、Whh、bh
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    
    # Ot = HtWhq + bq
    # output layer params: Whq、bq
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params 


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    Wxh, Whh, bh, Whq, bq = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, Wxh) + torch.mm(H, Whh) + bh)    # Ht = φ(XtWxh + Ht-1Whh + bh)
        Y = torch.mm(H, Whq) + bq       # Ot = HtWhq + bq
        outputs.append(Y)   
    return torch.cat(outputs, dim=0), (H, )

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state            # 初始化               
        self.forward_fn = forward_fn            # 前向传播
    
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
    

def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    output = [vocab[prefix[0]]]
    get_inputs = lambda: torch.tensor([output[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_inputs(), state)
        output.append(vocab[y])
    
    for _ in range(num_preds):
        y, state = net(get_inputs(), state)
        output.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in output])
        
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def sgd(params, lr, batch_size):
    # 小批量随机梯度下降
    with torch.no_grad():  # 禁用梯度计算
        for param in params:
            param -= lr * param.grad / batch_size  # 参数更新
            param.grad.zero_()  # 梯度清零

def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state, time = None, Timer()
    metric = Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(1 * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / time.stop()


def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size : sgd(net.params, lr, batch_size)
    predict1 = lambda prefix:predict(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict1('time  traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} token/sec on {str(device)}')
    print(predict1('timer traveller'))
    print(predict1('traveller'))


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens, num_epochs, lr= 512, 100, 0.3
    X = torch.arange(10).reshape((2, 5))
    device = "gpu" if torch.xpu.is_available() else "cpu"
    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], device)
    print(predict('time traveller', 4, net, vocab, device))
    train(net, train_iter, vocab, lr, num_epochs, device)









