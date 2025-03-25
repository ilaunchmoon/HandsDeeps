import torch
import torch.nn as nn
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.convolution_neural_net.le_net import train
from src.utils.load_mnist import load_fashion_mnist


class NiN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, dropout_rate=0.5):
        """
            NiN (Network in Network) 实现，适用于图像分类任务
            :param num_classes: 分类类别数
            :param in_channels: 输入图像的通道数
            :param dropout_rate: Dropout 比例
        """
        super(NiN, self).__init__()

        self.features = nn.Sequential(
            self._nin_block(in_channels, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),

            self._nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),

            self._nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),

            nn.Dropout(dropout_rate),

            # 输出通道设为 num_classes，作为分类预测
            self._nin_block(384, num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )

        self.classifier = nn.Sequential(
            nn.Flatten()
        )

    def _nin_block(self, in_channels, out_channels, kernel_size, strides, padding):
        """
            NiN Block:包含多个 1x1 卷积层进行特征映射。
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(size=(1, 1, 224, 224))  # 假设输入 224x224 灰度图
    model = NiN(num_classes=10, in_channels=1)
    # print("Input shape: ", x.shape)
    # for name, module in model.named_children():
    #     print(f"Processing {name}...")
    #     for layer in module:
    #         x = layer(x)
    #         print(layer.__class__.__name__, "Output shape: ", x.shape)

    lr, num_epochs, batch_size = 0.1, 10, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iter, test_iter = load_fashion_mnist(batch_size, 4, resize=224)
    train(model, train_iter, test_iter, num_epochs, lr, device)
