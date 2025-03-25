import time
import numpy as np


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    


if __name__ == "__main__":
    import torch
    a = torch.ones([1000000])
    b = torch.ones([1000000])
    c = torch.zeros(1000000)
    timer = Timer()
    for i in range(1000000):
         c[i] = a[i] + b[i]
    print(f'{timer.stop(): .5f} sec')
