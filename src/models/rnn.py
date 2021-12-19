from torch import nn


class ValinaLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int):
        super(ValinaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers)
        self.linear = nn.Linear(128, 1)

    def forward(self, hidden, X):
        X = X.transpose(0, 1)
        outputs, hidden = self.lstm(X, hidden)
        outputs = outputs[-1]
        out = self.linear(outputs)
        return out
