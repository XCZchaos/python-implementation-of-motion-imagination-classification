# 导入工具包
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(22, 1),  # filter size
                # group8意味着八组滤波器
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((8, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                # 十六组滤波器
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear((16 * 31), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        # print("block1", x.shape)
        x = self.block_2(x)
        # print("block2", x.shape)
        x = self.block_3(x)
        # print("block3", x.shape)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        # return F.softmax(x, dim=1), x  # return x for visualization
        return x
# if __name__ == '__main__':
#     input = torch.randn(32,1,22,1125)
#
#     model = EEGNet(4)
#
#     out = model(input)
#
#     print(model)
