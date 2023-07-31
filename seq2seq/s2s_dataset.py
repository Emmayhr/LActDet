import numpy as np
import os
from torch.utils.data import Dataset

class S2sDataset(Dataset):
    def __init__(self, usage, path):
        miss_s2s_dir = path #"/home/ubuntu/wcmc_attack_activity_detect_code/test/false_s2s_msg64_more"
        i = 0
        for act_dir in os.listdir(miss_s2s_dir):
            print(act_dir)
            act_path = os.path.join( miss_s2s_dir, act_dir)
            phase_vector_path = os.path.join(act_path , "phase_vector_sequence.npy")
            event_vector_path = os.path.join(act_path , "event_vector_sequence.npy")
            label_vector_path = os.path.join(act_path , "label_vector_sequence.npy")
            ratio_vector_path = os.path.join(act_path, "ratio_vector_sequence.npy")
            line = int(0.2 * np.load(label_vector_path).shape[0])

            if i == 0:
                if usage == 'train':
                    self.phases = np.load(phase_vector_path)[:line]   # .npy
                    self.events = np.load(event_vector_path)[:line]
                    self.labels = np.load(label_vector_path)[:line]
                    self.ratios = np.load(ratio_vector_path)[:line]
                if usage == 'test':
                    self.phases = np.load(phase_vector_path)[line:]   # .npy
                    self.events = np.load(event_vector_path)[line:]
                    self.labels = np.load(label_vector_path)[line:]
                    self.ratios = np.load(ratio_vector_path)[line:]
                self.phases = self.phases.astype('float32')
                self.events = self.events.astype('float32')
                i = 1
            else:
                if usage == 'train':
                    print(self.phases.shape)
                    print(np.load(phase_vector_path).shape)
                    self.phases = np.vstack((self.phases, np.load(phase_vector_path)[:line] ))  # .npy
                    self.events = np.vstack((self.events, np.load(event_vector_path)[:line] ))
                    self.labels = np.hstack((self.labels, np.load(label_vector_path)[:line] ))
                    self.ratios = np.hstack((self.ratios, np.load(ratio_vector_path)[:line] ))

                if usage == 'test':
                    self.phases = np.vstack((self.phases, np.load(phase_vector_path)[line:] ))  # .npy
                    self.events = np.vstack((self.events, np.load(event_vector_path)[line:] ))
                    self.labels = np.hstack((self.labels, np.load(label_vector_path)[line:] ))
                    self.ratios = np.hstack((self.ratios, np.load(ratio_vector_path)[line:] ))

                self.events = self.events.astype('float32')
                self.phases = self.phases.astype('float32')
            print(self.phases.shape)
            print(self.events.shape)
            print(self.labels.shape)

    def __getitem__(self, idx):
        return self.events[idx], self.phases[idx], self.labels[idx], self.ratios[idx]

    def __len__(self):
        return len(self.labels)

if __name__== "__main__" :
    dataset = S2sDataset()
