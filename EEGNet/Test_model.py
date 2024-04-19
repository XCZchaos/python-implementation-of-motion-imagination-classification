import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary


def accuracy(output, target):
    pred = torch.argmax(output, dim=1)
    pred = pred.float()
    correct = torch.sum(pred == target)
    return 100 * correct / len(target)
 


def plot_loss(epoch_number, loss):
    plt.plot(epoch_number, loss, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss during test')
    plt.savefig("loss.jpg")
    plt.show()
    


def plot_accuracy(epoch_number, accuracy):
    plt.plot(epoch_number, accuracy, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy during test')
    plt.savefig("accuracy.jpg")
    plt.show()
    


def plot_recall(epoch_number, recall):
    plt.plot(epoch_number, recall, color='purple', label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('Recall during test')
    plt.savefig("recall.jpg")
    plt.show()
 


def plot_precision(epoch_number,  precision):
    plt.plot(epoch_number, precision, color='black', label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('Precision during test')
    plt.savefig("precision.jpg")
    plt.show()
 


def plot_f1(epoch_number,  f1):
    plt.plot(epoch_number, f1, color='yellow', label='f1')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('f1 during test')
    plt.savefig("f1.jpg")
    plt.show()
    


def calc_recall_precision(output, target):
    pred = torch.argmax(output, dim=1)
    pred = pred.float()
    tp = ((pred == target) & (target == 1)).sum().item()  # 正确预测为“相同”的样本数
    tn = ((pred == target) & (target == 0)).sum().item()  # 正确预测为“不相同”的样本数
    fp = ((pred != target) & (target == 0)).sum().item()  # 错误预测为“相同”的样本数
    fn = ((pred != target) & (target == 1)).sum().item()  # 错误预测为“不相同”的样本数
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0  # 计算召回率
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # 计算精确度
    return recall, precision



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
    def __init__(self,file_path,transform=None):
        self.file_path = file_path
        
        # 读取文件 EEGdata与label
        self.data = self.parse_data_file(file_path)
        
        
        self.transform = transform
        
        
    def parse_data_file(self,file_path):
        
        data = torch.load(file_path)
        return np.array(data,dtype=np.float32)
    
    
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
        item = self.data[index,:]
        
        
        # 目前不会执行
        if self.transform:
            item = self.transform(item)
        
        
        return item

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model_EEG_net = torch.load('EEGNet_paper.pth', map_location=torch.device(device))
    data_test = torch.load('A01T_new.pt')
    print(data_test.shape)
    labels = torch.load('A01T_new_label.pt').to(device)

    
    
    model_EEG_net.eval()
    test_label = []
    
    with torch.no_grad():
        # EEGnetdata = EEGNetDataset(file_path ='C:\\Users\\24242\\DataspellProjects\\EEG_Project\\EEGNet\\A01T.pt',transform=False)
        # test_dataloader = DataLoader(EEGnetdata,shuffle=False,num_workers=0,batch_size=Config.train_batch_size,drop_last=True)

        
            
        
        output = model_EEG_net(data_test.to(device))  #输出
        # output = torch.max(output, 1)[1]

        # labels = labels[:-1]
        # print(label_test)
        # print(labels)
        
        # result = accuracy(output,labels)
        # result1 = accuracy(output=output, target=labels.to(device))
        #
        # print(result1)
        # summary(model_EEG_net,input_size=(1, 22, 1000), batch_size=20)
        # print(result1)
        # 模型打印
        summary(model_EEG_net, batch_size=48, input_size=(1, 22, 1000))

        # 0.89 0.85 0.71 0.86 0.87
        print(accuracy(output, labels))
        

           
