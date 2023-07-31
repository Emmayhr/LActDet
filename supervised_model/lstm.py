# -*- coding: utf-8 -*-
# @Author   :   kk
# @File     :   model.py
# @Time     :   2021/12/27


'''lstm模型训练与测试'''

from torch import nn
import torch.nn.functional as F

# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters:
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of ouf LSTM to stack
    """
    def __init__(self, input_size, embedding_size=512, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.linear = nn.Linear(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn
        self.drop = nn.Dropout(p=0.92)
        self.classifier  = nn.Linear(hidden_size, output_size) # 全连接层

    def forward(self, _x):
        """
            x: input batch data, size: [sequence len, batch size, feature size]
            for word2vec, size(x) is [14, batch size, 2413]
            for doc2vec, size(x) is [14, batch size, 512]
        """
        # embedded: [sequence len, batch size, embedding size]
        embedded = F.relu(self.linear(_x))
        # out,_ = self.lstm(_x)
        out, _ = self.lstm(embedded)
        out = self.drop(out)
        out = out[-1, :, :]
        out = self.classifier(out)
        return out
