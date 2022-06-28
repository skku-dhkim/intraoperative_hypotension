import torch
import torch.nn.functional as F
from torch import nn


class ValinaLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_of_classes: int, **kwargs):
        super(ValinaLSTM, self).__init__()
        self.layers = kwargs['layers']
        self.hidden_units = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers)
        self.linear = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_of_classes)

    def forward(self, X, hidden):
        X = X.transpose(0, 1)
        outputs, hidden = self.lstm(X, hidden)
        outputs = outputs[-1]
        out = self.linear(outputs)
        out = F.relu(out)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size, device):
        hidden = (torch.ones(self.layer, batch_size, self.hidden_units, device=device),
                  torch.ones(self.layer, batch_size, self.hidden_units, device=device))
        return hidden
