import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import random_split, ConcatDataset

class MyDataset_e_LActDet(Dataset):
    def __init__(self, event_dir):
        #event_dir = "/home/ubuntu/wcmc_attack_activity_detect_code/test/miss_event_msg64"
        i = 0
        for act_dir in os.listdir(event_dir):
            print(act_dir)
            act_path = os.path.join( event_dir, act_dir)
            event_vector_path = os.path.join(act_path , "event_vector_sequence.npy")
            label_vector_path = os.path.join(act_path , "label_vector_sequence.npy")

            if i == 0:
                self.events = np.load(event_vector_path)
                self.labels = np.load(label_vector_path)
                self.events = self.events.astype('float32')
                i = 1
            else:
                self.events = np.vstack((self.events, np.load(event_vector_path)))
                self.labels = np.hstack((self.labels, np.load(label_vector_path)))
                self.events = self.events.astype('float32')
            print(self.events.shape)
            print(self.labels.shape)

    def __getitem__(self, idx):
        return self.events[idx], self.labels[idx], 0

    def __len__(self):
        return len(self.labels)

def restore(dataset, train_path, test_path):
    train_datasets = []
    test_datasets = []
    num_rows = dataset.labels.shape[0]
    test_split_num = int (num_rows * 0.2)   # choose 80% as train data
    train_split_num = num_rows - test_split_num
    train_set, test_set = random_split(dataset, [train_split_num, test_split_num])
    train_datasets.append(train_set)
    test_datasets.append(test_set)
    print("train_datasets len : ", len(train_datasets[0]))
    print("test_datasets len : ", len(test_datasets[0]))
    Train = ConcatDataset(train_datasets)
    Test = ConcatDataset(test_datasets)
    torch.save(Train, train_path)
    torch.save(Test, test_path)


if __name__== "__main__" :
    
    miss_train_datasets = []
    miss_test_datasets = []
    false_train_datasets = []
    false_test_datasets = []

    miss_event_dir = "/home/ubuntu/wcmc_attack_activity_detect_code/test/miss_event_msg64"
    false_event_dir = "/home/ubuntu/wcmc_attack_activity_detect_code/test/miss_event_msg64"

    miss_dataset = MyDataset_e_LActDet(miss_event_dir)
    restore(miss_dataset, "dataset/e-LActDet/miss_train.t", "dataset/e-LActDet/miss_test.t")
    false_dataset = MyDataset_e_LActDet(false_event_dir)
    restore(false_dataset, "dataset/e-LActDet/false_train.t", "dataset/e-LActDet/false_test.t")
