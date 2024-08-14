import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from scipy import stats
from sklearn import manifold
from einops import reduce
from scipy.linalg import eigh

def plot_confusion_matrix(y_true, y_pred, sub, title = "Confusion matrix - 2a",
                          cmap=plt.cm.Blues, save_flg=True):

    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    classes = [str(i) for i in range(4)]
    labels = range(4)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    # print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=30)

    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    # save your path,you can choose your path and change it
    if save_flg:
        plt.savefig("confusion_matrix" + str(sub) + ".png")
        
        
        
def plt_tsne(data, label, per, nsub):

    data = data.cpu().detach().numpy()
    data = reduce(data, 'b n e -> b e', reduction='mean')
    label = label.cpu().detach().numpy()

    tsne = manifold.TSNE(n_components=2, perplexity=per, init='pca', random_state=166, learning_rate=200, n_iter=1000)
    X_tsne = tsne.fit_transform(data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 8))

    color_list = ['blue', 'red', 'green', 'orange']

    unique_labels = np.unique(label)
    num_classes = len(unique_labels)

    label_to_color = {unique_labels[i]: color_list[i % len(color_list)] for i in range(num_classes)}

    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=label_to_color[label[i]], s=50, alpha=0.8)  # 增加点的大小和透明度

    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE visualization')
    
    plt.savefig('EEGNet_%d.png' % (nsub), dpi=600)
    
    

def plot_metrics(train_losses, train_accuracies, nSub):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax2 = ax1.twinx()
    ax1.plot(epochs, train_losses, 'g-', label='Training loss')
    ax2.plot(epochs, train_accuracies, 'r-', label='Training accuracy')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Accuracy', color='r')


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()


    combined_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.5, 0.5))

    ax1.add_artist(combined_legend)  # 添加合并的图例

    plt.title('Training Loss and Accuracy')
    plt.savefig('training_metrics_subject_%d.png' % (nSub))
    plt.show()
    
