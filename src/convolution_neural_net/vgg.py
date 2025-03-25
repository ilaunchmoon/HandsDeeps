import torch
import torch.nn as nn
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.convolution_neural_net.le_net import train
from src.utils.load_mnist import load_fashion_mnist

class VGG(nn.Module):
    def __init__(self, conv_arch, num_classes=10, in_channels=3, dropout_rate=0.5, use_batchnorm=True):
        """
            VGG 网络结构，支持自定义卷积层架构
            :param conv_arch: list, [(num_convs, out_channels), ...] 指定 VGG 结构
            :param num_classes: int, 分类类别数量
            :param in_channels: int, 输入图像通道数（默认为 3, 即 RGB 图像）
            :param dropout_rate: float, dropout 比例
            :param use_batchnorm: bool, 是否使用批量归一化(BatchNorm2d)
        """
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.features = self._make_layers(conv_arch, in_channels)
        
        # 计算全连接层的输入维度（自动推导）
        self.flatten_size = self._get_flatten_size(in_channels, conv_arch)

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )

    def _make_layers(self, conv_arch, in_channels):
        """
        构建 VGG 的特征提取部分（卷积层）
        """
        layers = []
        for num_convs, out_channels in conv_arch:
            layers.append(self._vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _vgg_block(self, num_convs, in_channels, out_channels):
        """
        VGG 的单个卷积块，包括多个 `Conv2D + ReLU`，最后 `MaxPool2D`
        """
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))  # 添加 BatchNorm
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 池化层
        return nn.Sequential(*layers)

    def _get_flatten_size(self, in_channels, conv_arch):
        """
            计算全连接层的输入大小（自动推导）
            VGG 结构通常假设输入为 224x224, 经过 5 次池化后变为 7x7
        """
        size = 224 // (2 ** len(conv_arch))  # 计算池化后的特征图大小
        return conv_arch[-1][1] * size * size

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # x = torch.randn(size=(1, 1, 224, 224))    
    # # 指定 VGG-11 结构
    vgg11_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
    # # 传入 conv_arch
    # net = VGG(conv_arch=vgg11_arch, in_channels=1)  # in_channels=1 因为输入是 1 通道
    # # 遍历网络
    # for blk in net.features:
    #     x = blk(x)
    #     print(blk.__class__.__name__, 'output shape: \t', x.shape)

    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in vgg11_arch]
    net = VGG(conv_arch=vgg11_arch, in_channels=1)  # in_channels=1 因为输入是 1 通道
    lr, num_epochs, batch_size = 0.05, 10, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iter, test_iter = load_fashion_mnist(batch_size, 4, resize=224)
    train(net, train_iter, test_iter, num_epochs, lr, device)
