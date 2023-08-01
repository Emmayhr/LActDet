# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   main.py
# @Time     :   2022/01/04

'''attack activity detection'''


import torch
from torch.utils.data import DataLoader, ConcatDataset

import modelFunc2 as mfunc
from dataset import MyDataset
from dataset_e_LActDet import MyDataset_e_LActDet
from log import ApiLog
import time
import argparse
 

parser = argparse.ArgumentParser(description='Demo of LActDet')
parser.add_argument('--method', type=str, default="LActDet")
args = parser.parse_args()
method = args.method


if method == "e-LActDet":
    miss_train = torch.load("dataset/e-LActDet/miss_train.t") #LWS
    miss_val = torch.load("dataset/e-LActDet/miss_test.t")
    false_train = torch.load("dataset/e-LActDet/false_train.t")
    false_val = torch.load("dataset/e-LActDet/false_test.t")
    log_name = "log/e-LActDet.log"
    print("-- train the e-LActDet model ......")

if method == "LWS":
    miss_train = torch.load("dataset/LWS/miss_train.t") #LWS
    miss_val = torch.load("dataset/LWS/miss_test.t")
    false_train = torch.load("dataset/LWS/false_train.t")
    false_val = torch.load("dataset/LWS/false_test.t")
    log_name = "log/LWS.log"
    print("-- train the LWS model ......")

if method == "LActDet":
    miss_train = torch.load("dataset/LActDet/miss_train.t") #LWS
    miss_val = torch.load("dataset/LActDet/miss_test.t")
    false_train = torch.load("dataset/LActDet/false_train.t")
    false_val = torch.load("dataset/LActDet/false_test.t")
    log_name = "log/LActDet.log"
    print("-- train the LActDet model ......")

log = ApiLog(log_name)
log.logger.debug("------- Attack Activity train -------")

# train and test dataset
print("-- construct train and test dataset......")

Train = ConcatDataset([miss_train, false_train])
Test = ConcatDataset([miss_val, false_val])

train_loader = DataLoader(Train, batch_size = 32,shuffle = True)
test_loader = DataLoader(Test, batch_size = 32)

# LSTM model
mfunc.trainLstm(train_loader, test_loader, log)

