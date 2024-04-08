import torch

from EEGNet_version_01 import *


num = torch.rand((32, 1, 22, 1000))

net = EEGNet(4)

out = net(num)

print(net)

