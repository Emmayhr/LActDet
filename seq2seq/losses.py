import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.margin = 1.
        self.eps = 1e-9

    def forward(self, output1, output2, size_average=True):
        #np.nan_to_num(output1, nan=-1)
        #np.nan_to_num(output2, nan=-1)

        #print("prediction : ", output1)
        #print("output : ", output2)
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        return distances.mean() if size_average else distances.sum()



