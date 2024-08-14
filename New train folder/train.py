import torch
import torch.nn as nn
import numpy as np
from EEGNet import EEGNet
import scipy
from torch.autograd import Variable
from sklearn.metrics import cohen_kappa_score
from metrics import plot_confusion_matrix, plot_metrics
import random

gpus = [0]

# 输入的shape
# (288, 1, 22, 1000)   (trial, cov_number, channel, timepiont)  (batch_size, channel, width, height)
# (batch_size, RGB, height, width)
class Trans():
    def __init__(self, nsub):
        super(Trans, self).__init__()
        self.batch_size = 50
        self.n_epochs = 1000
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.9
        self.nSub = nsub
        self.start_epoch = 0
        self.root = 'C:\\Users\\24242\\Desktop\\AI_Reference\\data_bag\\BCICIV_2a_gdf\\'  # the path of data
        self.pretrain = False
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.model = EEGNet(4).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        self.centers = {}
        self.criterion = nn.CrossEntropyLoss()

    def get_source_data(self):
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']
        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)  # (288, 1, 22, 1000)
        self.train_label = np.transpose(self.train_label)
        self.allData = self.train_data
        self.allLabel = self.train_label[0]
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']
        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)
        self.testData = self.test_data
        self.testLabel = self.test_label[0]
        # 归一化
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std
        return self.allData, self.allLabel, self.testData, self.testLabel

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def calculate_kappa(self, y_true, y_pred):
        kappa = cohen_kappa_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        return kappa

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]
            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]
        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label - 1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def train(self):
        img, label, test_data, test_label = self.get_source_data()
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        train_losses = []
        train_accuracies = []

        for e in range(self.n_epochs):
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                outputs = self.model(img)
                print(label.shape)
                loss = self.criterion(outputs, label)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            

            if (e + 1) % 1 == 0:
                self.model.eval()
                outputs_test= self.model(test_data)
                loss_test = self.criterion(outputs_test, test_label)
                y_pred = torch.max(outputs_test, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                print('Epoch:', e,
                      '  Train loss:', loss.detach().cpu().numpy(),
                      '  Test loss:', loss_test.detach().cpu().numpy(),
                      '  Train accuracy:', train_acc,
                      '  Test accuracy is:', acc)
                train_losses.append(loss.detach().cpu().numpy())
                train_accuracies.append(train_acc)
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

                  

                # if e == self.n_epochs - 1:
                    
                #     plt_tsne(outputs_test, test_label, per=30, nsub=self.nSub)
                #     plt_tsne(test_label, per=30, nsub=self.nSub)
               
        # you can save the model state_dict in your path
        # torch.save(self.model.module.state_dict(), '/root/autodl-tmp/model_picture/TFCformer_model_Subject_%d.pth' % (self.nSub))
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        kappa = self.calculate_kappa(Y_true, Y_pred)
        print('The kappa score is:', kappa)
        plot_metrics(train_losses, train_accuracies, self.nSub)
        return bestAcc, averAcc, Y_true, Y_pred, kappa
    
    
    
def main():
    best = 0
    aver = 0
    for i in range(9):
        seed_n = np.random.randint(500)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print('Subject %d' % (i+1))
        trans = Trans(i + 1)
        bestAcc, averAcc, Y_true, Y_pred, kappa = trans.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        plot_confusion_matrix(Y_true, Y_pred, i+1)
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))
    plot_confusion_matrix(yt, yp, 666)
    
    
if __name__ == '__main__':
    
    main()