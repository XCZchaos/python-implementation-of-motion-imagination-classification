import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
 
 
 # ## 帮助函数
def show_plot(iteration,accuracy,loss):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('interation')
    plt.ylabel('number(0-1)')
    line1, = plt.plot(iteration,loss,color='r',linewidth=1.5,linestyle='-',label='loss')
    line2, = plt.plot(iteration,accuracy,color='b',linewidth=1.5,linestyle='-',label='accuracy')
    plt.legend(handles=[line1, line2], labels=['loss','accuracy'], loc='best')
    plt.show()


# ## 用于配置的帮助类
class Config():
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    # batch_size也会影响模型的精度
    train_batch_size = 48 # 64
    test_batch_size = 48
    train_number_epochs = 100 # 100
    test_number_epochs = 20

    

class EEGNetDataset(Dataset):
    # Dataset模块提供了一些接口可供实现 属于是抽象基类
    def __init__(self,file_path,target_path,transform=None,target_transform=None):
        self.file_path = file_path
        self.target_path = target_path
        # 读取文件 EEGdata与label
        self.data = self.parse_data_file(file_path)
        self.target = self.parse_target_file(target_path)
        
        self.transform = transform
        self.target_transform = target_transform
        
    def parse_data_file(self,file_path):
        
        data = torch.load(file_path)
        data = data.to('cpu')
        return np.array(data, dtype=np.float32)
    
    def parse_target_file(self,target_path):
        
        target = torch.load(target_path)
        target = target.to('cpu')
        return np.array(target, dtype=np.float32)
    # dataset的抽象方法 需要自己实现，下同
    # 返回data的长度 size为样本量总和 22*20*50 即channels*sample -> channels * h * w
    def __len__(self):
        
        return len(self.data)
    # dataset的抽象方法
    # 加载数据特征的index进行截取    index参数是由getitem自动生成的
    def __getitem__(self,index):
        # 只要创建了对象就会迭代 迭代48次也就是一个batch_size
        # data 已变成 287*22*1000的数据 
        # 选择第一个维度index个样本 每个样本的shape为(22,20,50) 每一个index即为一个trail

        item = self.data[index, :]
        
        # label
        target = self.target[index]
        # 目前不会执行
        if self.transform:
            item = self.transform(item)
        if self.target_transform:
            target = self.target_transform(target)
        
        return item,target
    

 
# 深度卷积 
class DepthwiseConv(nn.Module):
    # inp oup 为in_channels out_channels  
    def __init__(self, inp, oup):
        # 调用父类的构造函数
        super(DepthwiseConv, self).__init__()
        self.depth_conv = nn.Sequential(
            # dw 将两个inp参数为输入的Tensor通道数和out_put的featuremap数
            # 分组卷积使得参数变少 每个filter不会连接所有的参数
            nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw  将上面得到的feature map继续进行卷积
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup)
        )
    
    def forward(self, x):
        
        return self.depth_conv(x)
 
 
 
# 深度卷积  和上面类可以得到相同的结果
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
 
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        
        return x
 
 
 
 

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 500 
        self.conv1 = nn.Conv2d(22,48,(3,3),padding=0)
        self.batchnorm1 = nn.BatchNorm2d(48,False)
        self.Depth_conv = DepthwiseConv(inp=48,oup=22)
        self.pooling1 = nn.AvgPool2d(4,4)
        
        self.Separable_conv = depthwise_separable_conv(ch_in=22, ch_out=48)
        self.batchnorm2 = nn.BatchNorm2d(48,False)
        self.pooling2 = nn.AvgPool2d(2,2)
              
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,4)
    
    def forward(self, item):
        
        x = F.relu(self.conv1(item))
        x = self.batchnorm1(x)
        x = F.relu(self.Depth_conv(x))
        x = self.pooling1(x)
        x = F.relu(self.Separable_conv(x))
        x = self.batchnorm2(x)
        x = self.pooling2(x)
        #flatten
        x = x.contiguous().view(x.size()[0],-1) 
        #view函数：-1为计算后的自动填充值=batch_size，或x = x.contiguous().view(batch_size,x.size()[0])
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.25)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,0.5)
        x = F.softmax(self.fc3(x),dim=1)
        
        return x
 
 
 
 