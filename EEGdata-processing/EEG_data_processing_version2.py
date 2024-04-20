import mne
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import scipy.io
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from scipy import signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""进行预处理前请检查数据是否存在过多的噪声，或者数据是否稳定"""

# 进行数据预处理 只适合T结尾的data数据，其他的需要修改函数 filename需以.npy结尾
def transform_save_data(filename,save_filename=None):
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
    if save_filename is not None:
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
def data_transform_tensor(acc_train,y_oh,save_datafilename=None,save_labelfilename=None):
    transf = transforms.ToTensor()
    d = transf(y_oh)
    # 去除另外四个维度的标签，标签就是最大值
    label = torch.argmax(d,dim=2).long()

    

    h = torch.squeeze(label)
    data = torch.tensor(acc_train,dtype=torch.float32)
    labels = torch.tensor(h,dtype=torch.long)
    if save_datafilename is not None:
        torch.save(data,save_datafilename)
    if save_labelfilename is not None:
        torch.save(labels,save_labelfilename)

    return data, labels



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
    data_combine = torch.cat(data_list, axis=0)
    label_combine = torch.cat(label_list, axis=0)
    torch.save(data_combine, data_filename)
    torch.save(label_combine, label_filename)

    return data_combine, label_combine


# 数据滤波 利用巴特沃斯滤波器
def buttferfiter(data):
    Fs = 250
    b, a = signal.butter(6, [8, 30], 'bandpass', fs=Fs)
    data = signal.filtfilt(b, a, data, axis=1)
    return data


# 进行时域上EEG数据增强  通过分割，重构 打乱数据
def interaug(timg, label, batch_size):
    """timg是data label是标签"""
    """
    tmp_aug_data 用于保存生成的增强样本数据，其形状为 (batch_size / 4, 1, 22, 1000)，
    即每个增强样本包含8个时间片段，每个时间片段的形状为 (1, 22, 125)
    
    
    rand_idx 是随机选择的8个时间片段的索引，用于从原始数据中获取时间片段。
    aug_data 和 aug_label 分别保存所有类别的增强样本和对应的标签。
    aug_shuffle 对增强样本和标签进行随机打乱。

    """
    aug_data = []
    aug_label = []
    for cls4aug in range(4):
        # 条件判断 找出对应的label和data
        cls_idx = np.where(label == cls4aug)  # label == cls4aug + 1
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        # 分epoch
        tmp_aug_data = np.zeros((int(batch_size / 4), 1, 22, 1000))
        for ri in range(int(batch_size / 4)):
            # 随机取8个时间片段
            for rj in range(8):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                # 进行数据的打乱重构
                tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:int(batch_size / 4)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    aug_data = torch.from_numpy(aug_data).cuda()
    aug_data = aug_data.float()
    aug_label = torch.from_numpy(aug_label).cuda()  # aug_label - 1
    aug_label = aug_label.long()
    return aug_data, aug_label



# 切分部分数据进行test
def split_EEGdata(data, label):
    data = data.view(data.shape[0], 1, 22, 1000)
    data = data[:100]
    label = label[:100]
    torch.save(data, 'EEG_data_split.pt')
    torch.save(label, 'EEG_label_split.pt')


#  没有进行滤波处理原数据获取
def transform_save_data_version2(filename,save_filename=None):
    """

    :param filename: data filename
    :param save_filename: save data filename
    :return: save a np style file
    """
    raw = mne.io.read_raw_gdf(filename)
    print(raw.info['ch_names'])
    events,event_id = mne.events_from_annotations(raw)
    raw.info['bads'] += ['EOG-left','EOG-central','EOG-right']
    # 运动想象时间2-6秒
    tmin,tmax = 2,6
    event_id = {'769':7,'770':8,'771':9,'772':10}
    # 需要重新加载raw对象进行滤波处理
    raw.load_data()
    picks = mne.pick_types(raw.info,meg=False,eeg=True,stim=False,exclude='bads')
    epochs = mne.Epochs(raw=raw,events=events,event_id=event_id,tmin=tmin,tmax=tmax,preload=True,baseline=None,picks=picks)
    epoch_data = epochs.get_data()
    # 将最后一位数据进行去除
    epoch_data = epoch_data[:,:,:-1]


    epoch_data = epoch_data.reshape(epoch_data.shape[0], 1, 22, 1000)
    if save_filename is not None:
        np.save(save_filename,epoch_data)

    return epoch_data



if __name__ == '__main__':
    count = input('please input your subject:')
    filename = 'C:\\Users\\24242\\Desktop\\AI_Reference\\data_bag\\BCICIV_2a_gdf\\A0' + count + 'T.gdf'
    BCI_data = transform_save_data_version2(filename)
    label_filename = 'C:\\Users\\24242\\Desktop\\AI_Reference\\data_bag\\BCICIV_2a_gdf\\A0' + count + 'T.mat'
    acc_train, y_oh = data_processing(BCI_data, label_filename)
    data, label = data_transform_tensor(acc_train, y_oh)
    data1, label1 = interaug(data, label, 48)
    data2, label2 = interaug(data, label, 48)
    data3, label3 = interaug(data, label, 48)
    data4, label4 = interaug(data, label, 48)
    data5, label5 = interaug(data, label, 48)
    data6, label6 = interaug(data, label, 48)
    data7, label7 = interaug(data, label, 48)
    data8, label8 = interaug(data, label, 48)
    data9, label9 = interaug(data, label, 48)
    data10, label10 = interaug(data, label, 48)
    data11, label11 = interaug(data, label, 48)
    data12, label12 = interaug(data, label, 48)
    data13, label13 = interaug(data, label, 48)
    data14, label14 = interaug(data, label, 48)
    data15, label15 = interaug(data, label, 48)
    data16, label16 = interaug(data, label, 48)

    data_list = [data.to('cuda'), data1.to('cuda'), data2.to('cuda'), data3.to('cuda'), data4.to('cuda'), data5.to('cuda'), data6.to('cuda'), data7.to('cuda'), data8.to('cuda'), data9.to('cuda'), data10.to('cuda'), data11.to('cuda'), data12.to('cuda'), data13.to('cuda'), data14.to('cuda'), data15.to('cuda'), data16.to('cuda')]
    label_list = [label.to('cuda'),label1.to('cuda'), label2.to('cuda'), label3.to('cuda'), label4.to('cuda'), label5.to('cuda'), label6.to('cuda'), label7.to('cuda'), label8.to('cuda'), label9.to('cuda'), label10.to('cuda'), label11.to('cuda'), label12.to('cuda'), label13.to('cuda'), label14.to('cuda'), label15.to('cuda'), label16.to('cuda')]
    data_filename = 'combine_data_and_label/A0'+ count + '_combine/A0'+ count + '_combine_data.pt'
    label_filename = 'combine_data_and_label/A0'+ count + '_combine/A0'+ count + '_combine_label.pt'

    data_combine, label_combine = combine_data(data_list, label_list, data_filename, label_filename)
    print(data_combine.shape)
    print(label_combine.shape)
    data_combine = data_combine.detach().cpu().numpy()
    label_combine = label_combine.detach().cpu().numpy()
    train_data, test_data, train_label, test_label = train_test_split(data_combine, label_combine, test_size=0.2, train_size=0.8, shuffle=True)
    train_data = torch.from_numpy(train_data).float()
    test_data = torch.from_numpy(test_data).float()
    train_label = torch.from_numpy(train_label).long()
    test_label = torch.from_numpy(test_label).long()
    torch.save(train_data, 'combine_data_and_label/A0' + count + '_combine/train_data_A0' + count + '.pt')
    torch.save(test_data, 'combine_data_and_label/A0' + count + '_combine/test_data_A0' + count + '.pt')
    torch.save(train_label, 'combine_data_and_label/A0' + count + '_combine/train_label_A0' + count + '.pt')
    torch.save(test_label, 'combine_data_and_label/A0' + count + '_combine/test_label_A0' + count + '.pt')
    print(train_data.shape)
    print(test_data.shape)
    print(train_label.shape)
    print(test_label.shape)





