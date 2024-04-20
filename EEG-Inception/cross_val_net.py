from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from EEGNet_pytorch_version import *
from EEG_Inception import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load('C:\\Users\\24242\\PycharmProjects\\paper_rebuild\\EEG-dataprocessing\\EEG-Conformerprocessing\\combine_data_and_label\\A03_combine\\A03_combine_data.pt')
label = torch.load('C:\\Users\\24242\\PycharmProjects\\paper_rebuild\\EEG-dataprocessing\\EEG-Conformerprocessing\\combine_data_and_label\\A03_combine\\A03_combine_label.pt')
data = data.detach().to('cpu')
label = label.detach().to('cpu')
cv = StratifiedKFold(n_splits=8)

acc = []
count = 0
for train_index, test_index in cv.split(data, label):
    count += 1
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    train_dataset = TensorDataset(X_train, y_train)
    batch_size = 48
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = EEGInception(input_time=1000, fs=250, ncha=22, filters_per_branch=22, n_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()


    print(f'-------------The {count} number run the model------------------')

    for epoch in range(500):

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device).type(torch.cuda.FloatTensor)
            batch_labels = batch_labels.to(device).type(torch.cuda.LongTensor)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print("Epoch: {}/{}.. ".format(epoch + 1, 500))
            print("Loss: {:.12f}".format(loss.item()))


    model.eval()
    all_pre = []
    all_label = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device).type(torch.cuda.FloatTensor)
            batch_labels = batch_labels.to(device).type(torch.cuda.LongTensor)
            output = model(batch_data)
            pre = torch.max(output.data, 1)[1]
            pre = pre.cpu().numpy()
            all_pre.extend(pre)
            all_label.extend(batch_labels.cpu().numpy())

    acc_num = accuracy_score(y_true=np.array(all_label), y_pred=np.array(all_pre))
    acc.append(acc_num)
    print(f'-------------Finish {count} acc recognize-------------------')
    print(f'-------------Test Accuracy: {acc_num}-----------------------')


print('-------------Finish process-------------')
print(f'cross_val_scores is{np.mean(acc)}')



