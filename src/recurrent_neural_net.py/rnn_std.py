import sys
import os
import torch.nn as nn 
import torch 
import math
from torch.nn import functional as F
from man_impl_rnn import train

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.utils.load_time_machine import load_data_time_machine
from src.utils.time_tool import Timer
from src.utils.accumulator import Accumulator
from src.visualization.animator_tool import Animator



if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens, num_epochs, lr= 512, 100, 0.3
    print(len(vocab))



