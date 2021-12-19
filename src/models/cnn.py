import torch
import torch.nn.functional as F
from torch import nn, ones


class OneDimCNN(nn.Module):
    def __init__(self, input_size: int, num_of_classes: int):
        super(OneDimCNN, self).__init__()
        init_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, init_channel, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, init_channel*2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, init_channel*4, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(128),
            nn.Linear(128, num_of_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = nn.Flatten()(out)
        out = self.fc(out)
        return out
