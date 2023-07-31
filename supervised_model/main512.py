# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   main.py
# @Time     :   2022/01/04

'''attack activity detection'''

import time
import torch
from torch.utils.data import DataLoader, ConcatDataset

import modelFunc as mfunc
from dataset import MyDataset

# ----------------- LSTM -------------------
# input: (seq_len, batch, input_size)
#   seq_len: 14 (attack phase)
#   input_size: 1000 (attack phase vector)


# train and test dataset
print("-- construct train and test dataset......")
miss_train = torch.load("dataset/miss_train.t")
miss_val = torch.load("dataset/miss_val.t")
false_train = torch.load("dataset/false_train.t")
false_val = torch.load("dataset/false_val.t")
#print("miss_train.shape is : ", miss_train.cumsum())
#print("miss_val.shape is : ", miss_val.shape)

Train = ConcatDataset([miss_train,false_train])
Test = ConcatDataset([miss_val,false_val])

train_loader = DataLoader(Train, batch_size = 32,shuffle = True)
test_loader = DataLoader(Test, batch_size =32)
print(len(train_loader))
start = time.time()
# LSTM model
print("-- train the model ......")
mfunc.trainLstm(train_loader, test_loader)
end = time.time()
print((end-start)/5)
