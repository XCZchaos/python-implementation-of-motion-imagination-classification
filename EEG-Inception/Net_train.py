import numpy as np
import torch
from torch.utils.data import DataLoader
from EEGNet_pytorch_version import *
from torch import optim
from EEG_Inception import *

from sklearn.model_selection import cross_val_score, StratifiedKFold




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
# 加载EEGNetdata

EEGNetdata = EEGNetDataset(file_path='C:\\Users\\24242\\PycharmProjects\\paper_rebuild\\EEG-dataprocessing\\EEG-Conformerprocessing\\combine_data_and_label\\A01_combine\\train_data_A01.pt', target_path='C:\\Users\\24242\\PycharmProjects\\paper_rebuild\\EEG-dataprocessing\\EEG-Conformerprocessing\\combine_data_and_label\\A01_combine\\train_label_A01.pt', transform=False, target_transform=False)

train_dataloader = DataLoader(EEGNetdata, batch_size=48, shuffle=False)

# 构建EEGNet
print(device)

net = EEGInception(input_time=1000, fs=250, ncha=22, filters_per_branch=22, n_classes=4).to(device)



# 损失函数
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)
counter = []
# 画图要用
loss_history = []

iteration_number = 0
train_correct = 0
total = 0

classNum = 4
# 画图要用
acc_history = []
# 开启训练模式
net.train()

for epoch in range(0, 250):
    for i,data in enumerate(train_dataloader, 0):
        item, target = data
        item, target = item.to(device), target.to(device)
        item = item.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)

        optimizer.zero_grad()

        output = net(item)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = torch.max(output.data, 1)[1]
        train_correct += (pred == target).sum().item()
        total += target.size(0)
        train_acc = train_correct / total
        train_acc = np.array(train_acc)
        if i % 50 == 0:
            print('Epoch number {}\n acc {}\n loss {}'.format(epoch, train_acc, loss))

        iteration_number += 1
        counter.append(iteration_number)
        acc_history.append(train_acc.item())
        loss_history.append(loss.item())


show_plot(counter, acc_history, loss_history)
torch.save(net, 'EEGNet_paper.pth')


