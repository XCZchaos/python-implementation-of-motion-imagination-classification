import mne
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import scipy.io
import torchvision.transforms as transforms

"""进行预处理前请检查数据是否存在过多的噪声，或者数据是否稳定"""

# 进行数据预处理 只适合T结尾的data数据，其他的需要修改函数 filename需以.npy结尾
def transform_save_data(filename,save_filename):
    raw = mne.io.read_raw_gdf(filename)
    print(raw.info['ch_names'])
    events,event_id = mne.events_from_annotations(raw)
    raw.info['bads'] += ['EOG-left','EOG-central','EOG-right']
    # 运动想象时间2-6秒
    tmin,tmax = 2,6 
    event_id = {'769':7,'770':8,'771':9,'772':10}
    # 需要重新加载raw对象进行滤波处理
    raw.load_data()
    raw.filter(7.0,35.0,fir_design='firwin')
    picks = mne.pick_types(raw.info,meg=False,eeg=True,stim=False,exclude='bads')
    epochs = mne.Epochs(raw=raw,events=events,event_id=event_id,tmin=tmin,tmax=tmax,preload=True,baseline=None,picks=picks)
    epoch_data = epochs.get_data()
    # 将最后一位数据进行去除
    epoch_data = epoch_data[:,:,:-1]


    epoch_data = epoch_data.reshape(epoch_data.shape[0], 1, 22, 1000)
    np.save(save_filename,epoch_data)


# 进行归一化处理
def data_processing(BCI_IV_2a_data,label_filename):
    Scaler = StandardScaler()
    X_train = BCI_IV_2a_data.reshape(BCI_IV_2a_data.shape[0], 22000)
    X_train_Scaler = Scaler.fit_transform(X_train)
    # 进行reshape第二个维度为channels W H
    acc_train = X_train_Scaler.reshape(BCI_IV_2a_data.shape[0], 1, 22, 1000)
    data_label = scipy.io.loadmat(label_filename)
    Label = data_label['classlabel']
    # 查看由多少类
    n_classes = len(np.unique(Label))
    print(n_classes)
    encoder = OneHotEncoder(handle_unknown='ignore')
    y = np.array(Label)
    y_oh = encoder.fit_transform(y).toarray()
    y_oh = y_oh[:-1]

    return acc_train,y_oh


# 进行转换成Tensor格式的数据  保存的文件的格式应该以pt为后缀
def data_transform_tensor(acc_train,y_oh,save_datafilename,save_labelfilename):
    transf = transforms.ToTensor()
    d = transf(y_oh)
    # 去除另外四个维度的标签，标签就是最大值
    label = torch.argmax(d,dim=2).long()

    

    h = torch.squeeze(label)
    data = torch.tensor(acc_train,dtype=torch.float32)
    labels = torch.tensor(h,dtype=torch.long)

    torch.save(data,save_datafilename)
    torch.save(labels,save_labelfilename)



# 将数据进行联合
def combine_data(data_list,label_list,data_filename,label_filename):
    """_summary_
    将增强的EEG_data数据进行拼接 并保存为pt后缀文件
    Parameters
    ----------
    data_list : _type_  tensor
        EEGdata list
    label_list : _type_ tensor
        label list
    data_filename : _type_, optional
        _description_, by default None 
    label_filename : _type_, optional
        _description_, by default None 
    """
    data_combine = torch.cat(data_list,axis=0)
    label_combine = torch.cat(label_list,axis=0)
    torch.save(data_combine,data_filename)
    torch.save(label_combine,label_filename)




# 进行时域上EEG数据增强  通过分割，重构 打乱数据

def interaug(timg, label,batch_size):
    """_summary_
    函数还在完善中 根据需求进行更改     来自Conformer论文
    Parameters
    ----------
    timg : _type_ ndarray
        original EEGdata
    label : _type_ ndarray
        original label
    batch_size : _type_
        想要随机划分新的EEGdata的trial

    Returns
    -------
    EEG_data:shape为[batch_size,channels,sample]  type is tensor
    label:shape为[batch_size]   type is tensor
    
    """
    aug_data = []
    aug_label = []
    for cls4aug in range(4):
        # 条件判断 找出对应的label和data
        
        cls_idx = np.where(label == cls4aug)
        
        tmp_data = timg[cls_idx]
        
        tmp_label = label[cls_idx]
        # 分epoch
        tmp_aug_data = np.zeros((int(batch_size / 4), 22, 1000))
        for ri in range(int(batch_size / 4)):
        # 随机取8个时间片段，每个时间片段为1000即sample_rate 一个label对应的是1000个sample
            for rj in range(8):
                
                rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                # 进行数据的打乱重构 
                tmp_aug_data[ri, :, rj * 1000:(rj + 1) * 1000] = tmp_data[rand_idx[rj], :, rj * 1000:(rj + 1) * 1000]


        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:int(batch_size / 4)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    aug_data = torch.from_numpy(aug_data)
    aug_data = aug_data.float()
    aug_data = aug_data.reshape(aug_data.shape[0], 1, 22,1000)
    
    aug_label = torch.from_numpy(aug_label)
    aug_label = aug_label.long()
    
    return aug_data, aug_label


if __name__ == '__main__':
    # filename = 'C:\\Users\\24242\\Desktop\\AI_Reference\\data_bag\\BCICIV_2a_gdf\\A06T.gdf'
    # save_filename = 'A06T.npy'
    # transform_save_data(filename,save_filename)
    # data = np.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\\EEGNet\\A06T.npy')
    # label_filename = 'C:\\Users\\24242\\Desktop\\AI_Reference\\data_bag\\BCICIV_2a_gdf\\A06T.mat'
    # acc_train,y_oh = data_processing(data,label_filename)
    # save_datafilename = 'A06T.pt'
    # save_labelfilename = 'A06T_target.pt'
    # data_transform_tensor(acc_train,y_oh,save_datafilename,save_labelfilename)
    # 进行数据的拼接 使深度学习模型训练样本不会太少
     
    # data1 = np.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\\EEGNet\\A01T.npy')
    # data2 = torch.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\\EEGNet\\A02T.pt')
    # data3 = torch.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\\EEGNet\\A03T.pt')
    # data5 = torch.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\\EEGNet\\A05T.pt')
    # label1 = torch.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\EEGNet\\A01T_target.pt')
    # label1 = label1[:-1]
    # label2 = torch.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\EEGNet\\A02T_target.pt')
    # label2 = label2[:-1]
    # label3 = torch.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\EEGNet\\A03T_target.pt')
    # label3 = label3[:-1]
    # label1 = label1.numpy()
    # label5 = torch.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\EEGNet\\A05T_target.pt')
    
    # data_list = [data1,data2,data3,data5]
    # label_list = [label1,label2,label3,label5]
    # data_filename = 'combine_data_01.pt'
    # label_filename = 'combine_label_01.pt'
    # label1 = label1[:-1]
    #
    # aug_data, aug_label = interaug(timg=data1,label=label1,batch_size=287)
    #
    # data_1 = torch.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\EEGNet\\A01T.pt')
    # label_1 = torch.load('C:\\Users\\24242\\DataspellProjects\\EEG_Project\EEGNet\\A01T_target.pt')
    # print(label_1.shape)
    # print(data_1.shape)
    # print(aug_data.shape)
    # print(aug_label.shape)
    # data_list = [data_1,aug_data]
    # label_list = [label_1,aug_label]
    # combine_data(data_list=data_list,label_list=label_list,data_filename=data_filename,label_filename=label_filename)
    print('------------------new FileProcess------------------')
    filename = 'C:\\Users\\24242\\Desktop\\AI_Reference\\data_bag\\BCICIV_2a_gdf\\A01T.gdf'
    save_filename = 'A01T_new.npy'
    # transform_save_data(filename, save_filename)
    data = np.load(save_filename)

    label_name = 'C:\\Users\\24242\\Desktop\\AI_Reference\\data_bag\\BCICIV_2a_gdf\\A01T.mat'
    acc_train, y_oh = data_processing(data,label_name)

    save_dataFilename = 'A01T_new.pt'
    save_labelFilename = 'A01T_new_label.pt'
    data_transform_tensor(acc_train, y_oh, save_dataFilename, save_labelFilename)
    data_final = torch.load('A01T_new.pt')
    label_final = torch.load('A01T_new_label.pt')


