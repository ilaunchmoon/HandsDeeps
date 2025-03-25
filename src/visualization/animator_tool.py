import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.visualization.plot import use_svg_display, set_axes
from matplotlib import pyplot as plt
from IPython import display

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]  # 转换为列表以便后续统一处理

        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y = [], []  # 初始化为空列表，而不是 None
        self.fmts = fmts
        self.is_interactive = plt.isinteractive()  # 确保调用方法时使用括号

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        # 如果 X 和 Y 尚未初始化为与数据长度一致的列表，则初始化
        if len(self.X) == 0:
            self.X = [[] for _ in range(n)]
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()  # 清除当前的绘图
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)  # 绘制曲线
        self.config_axes()  # 设置坐标轴
        # display.display(self.fig)
        # display.clear_output(wait=True)  # 动态更新
        plt.pause(0.01)  # 动态刷新

    def show(self):
        if self.is_interactive:
            plt.show()
        else:
            plt.close(self.fig)