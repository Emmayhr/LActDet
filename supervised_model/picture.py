# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   picture.py
# @Time     :   2022/01/25

'''draw pictures'''

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.rc('font', family = 'Times New Roman')
    plt.rcParams['figure.figsize'] = (8,7)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=75, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.show()




from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def ROC(ground_truth, predicted, classes):
    fpr, tpr, thersholds = roc_curve(ground_truth, predicted, pos_label=2)
 
    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))
    
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == "__main__":
    x = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

    '''miss-recall'''
    # y1 = [1,1,1,1,1,1,1,0.98,0.98,0.98]
    # y2 = [1,1,1,1,1,1,1,1,0.99,0.99]
    # y3 = [1,1,1,1,1,1,1,1,1,0.99]
    # y4 = [1,1,1,1,1,0.99,0.99,0.99,0.97,0.96]
    # y5 = [1,1,1,0.99,0.98,0.98,0.9,0.89,0.89,0.89]
    # y6= [1,1,1,1,1,1,1,1,1,1]
    # y7 = [1,1,1,1,1,1,1,1,0.99,0.99]
    # y8 = [1,1,1,1,1,0.99,0.98,0.95,0.86,0.83]
    # y9 = [1,1,1,0.95,0.95,0.95,0.95,0.90,0.90,0.90]
    # y10 = [1,1,1,1,1,1,1,0.99,1,0.98]
    # y11 = [1,1,1,1,1,0.99,0.99,0.98,0.96,0.96]
    # y12 = [1,1,1,1,1,1,1,1,0.97,0.96]

    '''miss-f1'''
    y1 = [1,1,1,1,1,1,1,0.99,0.99,0.99]
    y2 = [1,1,1,1,1,1,1,1,0.995,0.995]
    y3 = [1,1,1,1,1,1,1,1,1,0.995]
    y4 = [1,1,1,1,1,0.995,0.995,0.995,0.985,0.98]
    y5 = [1,1,1,0.995,0.99,0.99,0.947,0.942,0.942,0.942]
    y6 = [1,1,1,1,1,1,1,1,1,1]
    y7 = [1,1,1,1,1,1,1,1,0.995,0.995]
    y8 = [1,1,1,1,1,0.995,0.99,0.974,0.925,0.907]
    y9 = [1,1,1,0.974,0.974,0.974,0.974,0.947,0.947,0.947]
    y10 = [1,1,1,1,1,1,1,0.995,1,0.99]
    y11 = [1,1,1,1,1,0.995,0.995,0.99,0.98,0.98]
    y12 = [1,1,1,1,1,1,1,1,0.985,0.98]
    

    plt.rc('font', family = 'Times New Roman')
    plt.rcParams['figure.figsize'] = (8,6)

    plt.plot(x, y1,  'o-', color = '#cb997e', lw = 3, label = 'LLDOS1.0', ms=10)
    plt.plot(x, y2,  'v-', color = '#e76f51', lw = 3, label = 'LLDOS2.0.2', ms=10)
    plt.plot(x, y3,  's-', color = '#deaaff', lw = 3, label = 'Infiltration', ms=10)
    plt.plot(x, y4,  'D-', color = '#283845', lw = 3, label = 'HTTP DoS', ms=10)
    plt.plot(x, y5,  'H-', color = '#7251b5', lw = 3, label = 'Brute force SSH', ms=10)
    plt.plot(x, y6,  'h-', color = '#b5179e', lw = 3, label = 'Heartbleed', ms=10)
    plt.plot(x, y7,  '2-', color = '#469d89', lw = 3, label = 'Wannacry', ms=10)
    plt.plot(x, y8,  'p-', color = '#e56b6f', lw = 3, label = 'Andariel-2019', ms=10)
    plt.plot(x, y9,  '^-', color = '#0096c7', lw = 3, label = 'APT-29', ms=10)
    plt.plot(x, y10, '<-', color = '#f48c06', lw = 3, label = 'AZORult Neutrino', ms=10)
    plt.plot(x, y11, '*-', color = '#538d22', lw = 3, label = 'IcedID', ms=10)
    plt.plot(x, y12, '>-', color = '#c9184a', lw = 3, label = 'Raccoon', ms=10)
    
    plt.xlabel("Missing rate", fontsize=20)
    plt.ylabel("F1-score", fontsize=20)
    plt.xticks(x, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()

    # '''绘制混淆矩阵'''
    # cnf_matrix = np.array([[416, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 420, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 420, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 3, 412, 3, 0, 0, 0, 2, 0, 0, 0],
    #                        [0, 0, 0, 1, 382, 5, 9, 0, 0, 0,23, 0],
    #                        [0, 0, 0, 1, 3, 400, 15, 0, 0, 1, 0, 0],
    #                        [0, 0, 0, 3, 0, 4, 133, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 2, 8, 410, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0,10, 0, 6, 324, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 11, 0, 0, 0, 309, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 420, 0],
    #                        [4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 415]])

    # attack_type = ['LLDOS1.0', 'LLDOS2.0.2', 'Infiltration', 'HTTP DoS', 'Brute force SSH', 'Andariel-2019', 'APT-29', 'AZORult Neutrino', 
    #                'IcedID', 'Raccoon', 'Heartbleed', 'Wannacry']

    # plot_confusion_matrix(cnf_matrix, classes=attack_type, normalize=True, title='Normalized confusion matrix')