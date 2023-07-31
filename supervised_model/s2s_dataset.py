import numpy as np
import os
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        miss_s2s_dir = "test/miss_s2s/"
        i = 0
        for act_dir in os.listdir(miss_s2s_dir):
            print(act_dir)
            act_path = os.path.join( miss_s2s_dir, act_dir)
            phase_vector_path = os.path.join(act_path , "phase_vector_sequence.npy")
            event_vector_path = os.path.join(act_path , "event_vector_sequence.npy")
            label_vector_path = os.path.join(act_path , "label_vector_sequence.npy")
        
            if i == 0:
                self.phases = np.load(phase_vector_path)   # .npy
                self.events = np.load(event_vector_path)
                self.labels = np.load(label_vector_path)
                i = 1
            else:
                self.phases = np.vstack((self.phases, np.load(phase_vector_path) ))  # .npy
                self.events = np.vstack((self.events, np.load(event_vector_path) ))
                self.labels = np.hstack((self.labels, np.load(label_vector_path) ))
            print(self.phases.shape)
            print(self.events.shape)
            print(self.labels.shape)

    def __getitem__(self, idx):
        return self.events[idx], self.phases[idx]

    def __len__(self):
        return len(self.label)

if __name__== "__main__" :
    dataset = MyDataset()
    torch.save(Train, "/data/kk/code/wcmc_attack_activity_detect/dataset/miss_word_train.t")  # save ground truth
    torch.save(Test, "/data/kk/code/wcmc_attack_activity_detect/dataset/miss_word_val.t")  # save ground truth


