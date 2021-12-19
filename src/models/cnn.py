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
            nn.Conv1d(init_channel, init_channel*2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(init_channel*2, init_channel*4, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(init_channel*4, init_channel*4, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, num_of_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = nn.Flatten()(out)
        out = self.fc(out)
        return out


class MultiChannelCNN(nn.Module):
    def __init__(self, input_size: int, num_of_classes: int):
        super(MultiChannelCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 14, kernel_size=3, stride=2),
            nn.BatchNorm1d(14),
            nn.ReLU(),
            nn.Conv1d(14, 14, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(input_size, 14, kernel_size=5, stride=2),
            nn.BatchNorm1d(14),
            nn.ReLU(),
            nn.Conv1d(14, 14, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(input_size, 14, kernel_size=7, stride=2),
            nn.BatchNorm1d(14),
            nn.ReLU(),
            nn.Conv1d(14, 14, kernel_size=7, stride=2, padding=3),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(42, 62, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(62, 62, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(620, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_of_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        out = torch.concat([out1, out2, out3], dim=1)
        out = self.conv4(out)
        out = nn.Flatten()(out)
        out = self.fc(out)
        return out
