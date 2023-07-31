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


# train and test dataset
print("-- construct train and test dataset......")
miss_train = torch.load("dataset/miss_event_train.t")
miss_val = torch.load("dataset/miss_event_test.t")
false_train = torch.load("dataset/false_event_train.t")
false_val = torch.load("dataset/false_event_test.t")

Train = ConcatDataset([miss_train, false_train])
Test = ConcatDataset([miss_val, false_val])

train_loader = DataLoader(Train, batch_size = 32,shuffle = True)
test_loader = DataLoader(Test, batch_size = 32, shuffle = True)

print("-- train the model ......")
start = time.time()
mfunc.trainLstm(train_loader, test_loader)
end = time.time()
delay = (end - start)/40
print("computing delay is : {}".format(delay))

