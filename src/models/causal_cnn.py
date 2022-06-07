import torch
import torch.nn.functional as F
import copy
import math
from torch import nn, ones
from src.models.attentions import MultiHeadAttention

##test later

class CausalOneDimCNN(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, num_of_classes: int):
        super(CausalOneDimCNN, self).__init__()
        init_channel = 32
        self.net = nn.Sequential(
            ResUnit(in_channels=8, size=3, dilation=2),
            ResUnit(in_channels=8, size=3, dilation=2),
            ResUnit(in_channels=8, size=3, dilation=2),
            ResUnit(in_channels=8, size=3, dilation=2),
            nn.BatchNorm1d(8),

        )

        self.fc = nn.Sequential(
            nn.Linear(24000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_of_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.net(x)
        out = nn.Flatten()(out)
        out = self.fc(out)
        return out



class ResUnit(nn.Module):
    def __init__(self, in_channels, size=3, dilation=1, causal=True, in_ln=True):
        super(ResUnit, self).__init__()
        self.size = size
        self.dilation = dilation
        self.causal = causal
        self.in_ln = in_ln
        if self.in_ln:
            self.ln1 = nn.InstanceNorm1d(in_channels, affine=True)
            self.ln1.weight.data.fill_(1.0)
        self.conv_in = nn.Conv1d(in_channels, in_channels//2, 1)
        self.ln2 = nn.InstanceNorm1d(in_channels//2, affine=True)
        self.ln2.weight.data.fill_(1.0)
        self.conv_dilated = nn.Conv1d(in_channels//2, in_channels//2, size, dilation=self.dilation,
                                      padding=((dilation*(size-1)) if causal else (dilation*(size-1)//2)))
        self.ln3 = nn.InstanceNorm1d(in_channels//2, affine=True)
        self.ln3.weight.data.fill_(1.0)
        self.conv_out = nn.Conv1d(in_channels//2, in_channels, 1)

    def forward(self, inp):
        x = inp
        if self.in_ln:
            x = self.ln1(x)
        x = nn.functional.leaky_relu(x)
        x = nn.functional.leaky_relu(self.ln2(self.conv_in(x)))
        x = self.conv_dilated(x)
        if self.causal and self.size>1:
            x = x[:,:,:-self.dilation*(self.size-1)]
        x = nn.functional.leaky_relu(self.ln3(x))
        x = self.conv_out(x)
        return x+inp
