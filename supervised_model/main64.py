# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   main.py
# @Time     :   2022/01/04

'''attack activity detection'''


import torch
from torch.utils.data import DataLoader, ConcatDataset

import modelFunc2 as mfunc
from dataset import MyDataset

import time
# ----------------- LSTM -------------------
# input: (seq_len, batch, input_size)
#   seq_len: 14 (attack phase)
#   input_size: 1000 (attack phase vector)

miss_train = torch.load("dataset/train_miss_phase.t") #LWS
miss_val = torch.load("dataset/test_miss_phase.t")
false_train = torch.load("dataset/train_false_phase.t")
false_val = torch.load("dataset/test_false_phase.t")

# train and test dataset
print("-- construct train and test dataset......")
#miss_train = torch.load("dataset/train.t")
#miss_val = torch.load("dataset/test.t")
# miss_train = torch.load("dataset/train_miss_msg64_more.t") #LActDet
# miss_val = torch.load("dataset/test_miss_msg64_more.t")
# false_train = torch.load("dataset/train_false_msg64_more.t")
# false_val = torch.load("dataset/test_false_msg64_more.t")

Train = ConcatDataset([miss_train, false_train])
Test = ConcatDataset([miss_val, false_val])

train_loader = DataLoader(Train, batch_size = 32,shuffle = True)
test_loader = DataLoader(Test, batch_size = 32)
print(len(train_loader))

# LSTM model
print("-- train the model ......")
start = time.time()
mfunc.trainLstm(train_loader, test_loader)
end = time.time()
delay = (end - start)/40
print("computing delay is : {}".format(delay))
#mfunc.testLstm(train_loader)
