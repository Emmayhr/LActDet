import torch
import torch.nn as nn
import torch.optim as optim
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        outputs, (hidden, cell) = self.rnn(src)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        # input = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(input, (hidden, cell))

        prediction = self.fc_out(output)
        # print("prediction", prediction.shape)
        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim


        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        output, hidden, cell=self.decoder(trg, hidden, cell)
        # first input to the decoder is the <sos> tokens
        #input = trg[:,0, :]
        return output
