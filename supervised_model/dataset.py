# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   dataset.py
# @Time     :   2022/01/04

'''自定义数据集格式'''

import numpy as np
from torch.utils.data import Dataset

'''
class MyDataset(Dataset):
    def __init__(self, activity, label, ratio):
        self.data = activity
        self.label = label
        self.ratio = ratio

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.ratio[idx]

    def __len__(self):
        return len(self.label)
'''
class MyDataset(Dataset):
    def __init__(self, activity, label):
        self.data = activity
        self.label = label

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


# prepare dataset
import os
import torch
from torch.utils.data import random_split, ConcatDataset

def splitDataset(np_dir):

    train_datasets = []
    test_datasets = []

    for dirname in os.listdir(np_dir):
        np_path = np_dir + dirname  # "/home/kk/attack_activity_detect/data/np_array/ddos1"

        sample_list = []
        label_list = []

        for npdir in os.listdir(np_path):
            print(dirname + "/" + npdir)

            # if npdir == str(1):
            #     percent_path = np_path + "/" + npdir    # "/home/kk/attack_activity_detect/data/np_array/ddos1/0.05"
            #     files = os.listdir(percent_path)   # read folder
            #     filename = files[0]
            #     label = int(filename.split("_")[0]) - 1
            #     file_path = percent_path + "/" + filename
            #     activity_vector = np.load(file_path)
            #     sample_list.append(activity_vector)
            #     label_list.append(label)
            # else:
            percent_path = np_path + "/" + npdir    # "/home/kk/attack_activity_detect/data/np_array/ddos1/0.05"
            files = os.listdir(percent_path)   # read folder
            file_num = len(files)
            if file_num <= 100:
                for i in range(100):
                # for filename in files:
                    filename = files[i % file_num]
                    label = int(filename.split("_")[0]) - 1
                    file_path = percent_path + "/" + filename
                    activity_vector = np.load(file_path)
                    sample_list.append(activity_vector)
                    label_list.append(label)
            else:
                for i in range(100):
                    filename = files[i]
                    label = int(filename.split("_")[0]) - 1
                    file_path = percent_path + "/" + filename
                    activity_vector = np.load(file_path)
                    sample_list.append(activity_vector)
                    label_list.append(label)

        dataset = MyDataset(sample_list, label_list)
        num_rows = len(label_list)
        print("count of dataset is : ", num_rows)
        test_split_num = int (num_rows * 0.2)   # choose 80% as train data
        train_split_num = num_rows - test_split_num
        train_set, test_set = random_split(dataset, [train_split_num, test_split_num])

        train_sample_list = []
        train_sample_list.append(train_set)

        if len(train_set) < 800:
            n = 800 / (len(train_set))
            train_sample_list = train_sample_list * int(n)
        
        train_datasets.append(ConcatDataset(train_sample_list))
        test_datasets.append(test_set)


    Train = ConcatDataset(train_datasets)
    Test = ConcatDataset(test_datasets)

    # train_loader = DataLoader(Train, batch_size = BATCH_SIZE,shuffle = True)
    # test_loader = DataLoader(Test, batch_size = BATCH_SIZE)
    torch.save(Train, "dataset/miss_word_train.t")  # save ground truth
    torch.save(Test, "dataset/miss_word_val.t")  # save ground truth

if __name__ == '__main__':
    splitDataset("test/miss_sample/")
