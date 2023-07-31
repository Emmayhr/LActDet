# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   svm.py
# @Time     :   2022/03/10

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.metrics import classification_report, confusion_matrix

from dataset import MyDataset

# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# from torchvision.datasets import MNIST

import sys
sys.path.append("/home/kk/attack_activity_detect/json2mysql")
from log import ApiLog

log_name = "/home/kk/attack_activity_detect/log/doc_svm22-04-21.log"
log = ApiLog(log_name)
log.logger.debug("------- Attack Activity Detection -------")


def load_data():
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])

    data_loaders = {}
    data_sizes = {}

    # train_loader = torch.load("/home/kk/attack_activity_detect/dataset/train_dataset.t")
    # val_loader = torch.load("/home/kk/attack_activity_detect/dataset/test_dataset.t")

    miss_train = torch.load("/home/kk/attack_activity_detect/dataset/miss_train.t")
    miss_val = torch.load("/home/kk/attack_activity_detect/dataset/miss_val.t")
    false_train = torch.load("/home/kk/attack_activity_detect/dataset/false_train.t")
    false_val = torch.load("/home/kk/attack_activity_detect/dataset/false_val.t")

    Train = ConcatDataset([miss_train,false_train])
    Test = ConcatDataset([miss_val,false_val])

    train_loader = DataLoader(Train, batch_size = 8,shuffle = True)
    val_loader = DataLoader(Test, batch_size = 8)

    data_loaders['train'] = train_loader
    data_sizes['train'] = len(train_loader)

    data_loaders['val'] = val_loader
    data_sizes['val'] = len(val_loader)

    # for name in ['train', 'val']:
    #     data_set = MNIST('./data', download=True, transform=transform)
    #     # 测试
    #     # img, target = data_set.__getitem__(0)
    #     # print(img.shape)
    #     # print(target)
    #     # exit(0)

    #     data_loader = DataLoader(data_set, shuffle=True, batch_size=128, num_workers=8)
    #     data_loaders[name] = data_loader
    #     data_sizes[name] = len(data_set)
    return data_loaders, data_sizes


def hinge_loss(outputs, labels):
    """
    折页损失计算
    :param outputs: 大小为(N, num_classes)
    :param labels: 大小为(N)
    :return: 损失值
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # # 正则化强度
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)

    return loss


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=30, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                total_predict = torch.zeros(len(data_loaders[phase].dataset))
                total_gt = torch.zeros(len(data_loaders[phase].dataset))
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i,(inputs, labels) in enumerate(data_loaders[phase]):
                # print(inputs.shape)
                # print(labels.shape)
                inputs = inputs.reshape(-1, 14*512)
                inputs = F.normalize(inputs)
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        total_gt[i*8:(i+1)*8] = labels
                        total_predict[i*8: (i+1)*8] = preds
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

                report = classification_report(total_gt, total_predict, output_dict = True)
                confusion = confusion_matrix(total_gt, total_predict)
                log.logger.debug('Best Accuracy: {:f}'.format(best_acc))
                log.logger.debug('Report: \n{}'.format(report))
                log.logger.debug('Confusion matrix: \n{}'.format(confusion))


                torch.save(model, "/home/kk/attack_activity_detect/checkout/svm.t")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_loaders, data_sizes = load_data()
    # print(data_loaders)
    # print(data_sizes)

    model = nn.Linear(14*512, 12).to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = hinge_loss
    optimizer = optim.Adam(model.parameters(), lr = 0.000001) 
    # optimizer = optim.SGD(model.parameters(), lr=0.0000001, momentum=0.9)
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(data_loaders, model, criterion, optimizer, lr_schduler, num_epochs=100, device=device)
