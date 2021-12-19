import torch
from torch import nn, ones


class ValinaLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int, num_of_classes: int):
        super(ValinaLSTM, self).__init__()
        self.layer = layers
        self.hidden_units = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers)
        self.linear = nn.Linear(hidden_size, num_of_classes)

    def forward(self, hidden, X):
        X = X.transpose(0, 1)
        outputs, hidden = self.lstm(X, hidden)
        outputs = outputs[-1]
        out = self.linear(outputs)
        return out

    def init_hidden(self, batch_size, device):
        hidden = (torch.ones(self.layer, batch_size, self.hidden_units, device=device),
                  torch.ones(self.layer, batch_size, self.hidden_units, device=device))
        return hidden
