# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   modelFunc.py
# @Time     :   2022/01/25

'''train and test model'''

import os
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset as ds
import lstm as mlstm
import lossFunc as lossf
import copy
import sys
sys.path.append("json2mysql")
from log import ApiLog
classes = ['ddos1', 'ddos2', 'infiltration', 'http_dos', 'brute_force', 'Andariel', 'APT29', 'AZORult', 'IcedID', 'Raccon', 'Heartbleed', 'Wannacry']

# hyper-parameter
HIDDEN_SIZE = 128  # 隐层size通常为输入size两倍
BATCH_SIZE = 32 # (8, 16, 32)
LAYERS_NUM = 2
EPOCHS_NUM = 40  # 记录每一轮的输出，取最好的结果保存
INPUT_FEATURES_NUM = 64
OUTPUT_FEATURES_NUM = 12

# Define execution device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print("The model will be running on", device, "device\n") 

 
# Training Function
def trainLstm(train_loader, test_loader, log):
    
    # LSTM model
    lstm_model = mlstm.LstmRNN(input_size = INPUT_FEATURES_NUM, hidden_size = HIDDEN_SIZE, output_size = OUTPUT_FEATURES_NUM, 
                                num_layers = LAYERS_NUM)
    lstm_model.to(device)    # Convert model parameters and buffers to CPU or Cuda
    loss_function = nn.CrossEntropyLoss()
    # loss_function = lossf.LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr = 0.001)   # lr 为默认。loss收敛的快减小学习率，收敛的慢增加学习率

    best_accuracy = 0.0
    for epoch in range(EPOCHS_NUM):

        running_tmp_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        val_total = 0

        for i, (data, label, ratio) in enumerate(train_loader):
            p = data.permute(1,0,2)
            p = F.normalize(p)

            p = p.to(device)

            output = lstm_model(p)
            label = label.type(torch.LongTensor)
            label = label.to(device)

            loss = loss_function(output, label)

            optimizer.zero_grad()   # Set gradient = 0 after each batch
            loss.backward()
            optimizer.step()    # update weight
            
            running_tmp_loss += loss.item()

            if i % 100 == 99:
                log.logger.debug('Epoch [{}/{}], Train Loss: {:.5f}'.format(epoch+1, EPOCHS_NUM, running_tmp_loss / 10000))
                running_tmp_loss = 0.0
            else:
                pass

        # Validation Loop
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                #! predict
                total_predict = torch.zeros(len(test_loader.dataset))
                #! gt
                total_gt = torch.zeros(len(test_loader.dataset))
                for i, (x_val, y_val, ratio) in enumerate(test_loader):
                    x_data = x_val.permute(1,0,2)
                    x_data = F.normalize(x_data)
                    x_data = x_data.to(device)

                    # x_data = x_data.type(torch.FloatTensor)
                    y_data = y_val.type(torch.LongTensor)
                    y_data = y_data.to(device)

                    #! save ground truth
                    total_gt[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = y_data
                    val_outputs = lstm_model(x_data)
                    val_loss = loss_function(val_outputs, y_data)

                    # The label with the highest value will be our prediction 
                    _, val_predicted = torch.max(val_outputs, 1)
                    #! save predict
                    total_predict[i*BATCH_SIZE: (i+1)*BATCH_SIZE] = val_predicted
                    running_val_loss += val_loss.item()
                    val_total += y_data.size(0)
                    running_accuracy += (val_predicted == y_data).sum().item()

            # Calculate validation loss value 
            # val_loss_value = running_val_loss/len(test_loader)
            # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
            val_accuracy = (100 * running_accuracy / val_total)

            report = classification_report(total_gt, total_predict, target_names = classes, output_dict = True)
            # report = classification_report(total_gt, total_predict, output_dict = True)
            confusion = confusion_matrix(total_gt, total_predict)
            
            # Save the model if the accuracy is the best 
            if val_accuracy > best_accuracy:
                log.logger.debug('report: \n{}'.format(report))
                log.logger.debug('Confusion matrix: \n{}'.format(confusion))
                log.logger.debug('Accuracy: {:f}'.format(val_accuracy))

                torch.save(lstm_model, "checkout/doc2vec_64_lstmModel.t")    # save the model
                torch.save(total_predict, "checkout/val_predict_new.t")  # save ground truth
                torch.save(total_gt, "checkout/val_gt_new.t")

                best_accuracy = val_accuracy
            else:
                pass

        else:
            pass


# Function to test the model 
def testLstm(test_loader, log):
    lstm_model = torch.load("checkout/doc2vec_64_lstmModel.t")
    lstm_model.to(device)

    total = 0
    running_accuracy = 0
    details = {}
    with torch.no_grad():
        #! predict
        total_predict = torch.zeros(len(test_loader.dataset))
        #! gt
        total_gt = torch.zeros(len(test_loader.dataset))
        
        for i, (x_val, y_val, ratio) in enumerate(test_loader):
            x_data = x_val.permute(1,0,2)
            x_data = F.normalize(x_data)
            x_data = x_data.to(device)

            # x_data = x_data.type(torch.FloatTensor)
            y_data = y_val.type(torch.LongTensor)
            y_data = y_data.to(device)
            
                       #! save ground truth
            total_gt[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = y_data
            test_outputs = lstm_model(x_data)
            # val_loss = loss_function(test_outputs, y_data)

            # The label with the highest value will be our prediction 
            _, test_predicted = torch.max(test_outputs, 1)

            for index in range(len(ratio)):
                r = ratio[index]
                label = int(y_data.cpu()[index])
                pred = test_predicted.cpu()[index]
                if label not in details:
                    details[label] = {}
                if r not in details[label]:
                    details[label][r] = {"label":[], "pred":[]}
                details[label][r]["label"].append(copy.deepcopy(label))
                details[label][r]["pred"].append(copy.deepcopy(pred))
            #print("details" , details)
            #! save predict
            total_predict[i*BATCH_SIZE: (i+1)*BATCH_SIZE] = test_predicted
            # running_val_loss += val_loss.item()
            total += y_data.size(0)
            running_accuracy += (test_predicted == y_data).sum().item()

    # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.  
    test_accuracy = (100 * running_accuracy / total)

    report = classification_report(total_gt, total_predict, output_dict = True)
    print('report: \n{}'.format(report))
    # log.logger.debug('Confusion matrix: {}'.format(confusion))
    print('Accuracy: {:f}\n'.format(test_accuracy))

    # report = classification_report(total_gt, total_predict, output_dict = True)
    # confusion = confusion_matrix(total_gt, total_predict)
    for label, rl in details.items():
        log.logger.debug("label is : {}\n".format(label))
        for ratio, result in rl.items():
            log.logger.debug("\tratio is : {}\n".format(ratio))
            labels = result["label"]
            #print(labels)
            preds = result["pred"]
            report = classification_report(labels, preds, output_dict = True)
            log.logger.debug('\t\t{}'.format(report))



if __name__ == '__main__':
    # classes = ['ddos1', 'ddos2', 'infiltration', 'httpdos', 'brute_force', 'Andariel', 'APT-29', 'AZORult', 
    #            'IcedID', 'Raccoon', 'Heartbleed', 'Wannacry']

    log_name = "log/doc2vec_lstm_64.log"
    log = ApiLog(log_name)
    log.logger.debug("------- Attack Activity test -------")
    
    np_dir = "/data/kk/code/wcmc_attack_activity_detect/test/miss_doc2vec/Wannacry/"
        
    for percent in tqdm(os.listdir(np_dir)):
        if percent == str(1):
            pass

        else:
            percent_path = np_dir + percent  # "/home/kk/attack_activity_detect/test/miss_doc2vec/ddos1/0.2/"

            sample_list = []
            label_list = []

            files = os.listdir(percent_path)   # read folder
            file_num = len(files)

            if (file_num > 100):
                for filename in files[-100:]:
                    label = int(filename.split("_")[0]) - 1
                    file_path = percent_path + "/" + filename
                    activity_vector = np.load(file_path)
                    sample_list.append(activity_vector)
                    label_list.append(label)
            else:
                for filename in files:
                    label = int(filename.split("_")[0]) - 1
                    file_path = percent_path + "/" + filename
                    activity_vector = np.load(file_path)
                    sample_list.append(activity_vector)
                    label_list.append(label)
            
            log.logger.debug('Test: Wannacry {}'.format(percent))
            test_dataset = ds.MyDataset(sample_list, label_list)
            testLoader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
            testLstm(testLoader, log)
