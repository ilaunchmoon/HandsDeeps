import torch
import torch.nn as nn
from typing import Tuple
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.convolution_neural_net.le_net import train
from src.utils.load_mnist import load_fashion_mnist



class Inception(nn.Module):
    """ GoogLeNet 的 Inception 模块 """
    def __init__(self, in_channels: int, c1: int, c2: Tuple[int, int], 
                 c3: Tuple[int, int], c4: int) -> None:
        """
        :param in_channels: 输入通道数
        :param c1: 1x1 卷积通道数
        :param c2: (1x1, 3x3) 卷积通道数
        :param c3: (1x1, 5x5) 卷积通道数
        :param c4: 3x3 最大池化 + 1x1 卷积通道数
        """
        super().__init__()

        # 线路1：1x1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU()
        )

        # 线路2：1x1 卷积 + 3x3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 线路3：1x1 卷积 + 5x5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),  # 确保 5x5 的 padding 设为 2 以保持尺寸
            nn.ReLU()
        )

        # 线路4：3x3 最大池化 + 1x1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: 输入特征图
        :return: 拼接后的输出
        """
        return torch.cat((self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)), dim=1)


class GoogLeNet(nn.Module):
    """ GoogLeNet 主体模型 """

    def __init__(self, num_classes: int = 10) -> None:
        """
        :param num_classes: 输出类别数
        """
        super().__init__()

        # 第一部分：初始特征提取
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第二部分：卷积块
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第三部分：Inception 块
        self.stage3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第四部分：Inception 深化
        self.stage4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第五部分：最后几层 Inception + 平均池化
        self.stage5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化
            nn.Flatten()
        )

        # 全连接分类层
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: 输入图像
        :return: 分类输出
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.fc(x)



if __name__ == "__main__":
    # x = torch.rand(size=(1, 1, 96, 96))
    # model = GoogLeNet(num_classes=10)
    # for name, layer in model.named_children():
    #     x = layer(x)
    #     print(f"{name}: {layer.__class__.__name__}, Output shape: {x.shape}")
    net = GoogLeNet(num_classes=10)
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = load_fashion_mnist(batch_size, 4, resize=96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(net, train_iter, test_iter, num_epochs, lr, device)