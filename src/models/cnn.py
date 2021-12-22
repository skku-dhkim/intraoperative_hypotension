import torch
import torch.nn.functional as F
import copy
import math
from torch import nn, ones
from src.models.attentions import MultiHeadAttention

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


class Attention(nn.Module):
    def __init__(self, embedding_size: int, output_dim: int):
        super(Attention, self).__init__()
        self.Q = nn.Linear(in_features=embedding_size, out_features=output_dim)
        self.K = nn.Linear(in_features=embedding_size, out_features=output_dim)
        self.V = nn.Linear(in_features=embedding_size, out_features=output_dim)

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        d_k = k.size(-1)

        attention_score = torch.matmul(q, k.transpose(-2, -1))
        attention_score = attention_score / math.sqrt(d_k)
        attention_prob = F.softmax(attention_score, dim=-1)
        out = torch.matmul(attention_prob, v)
        return out


class Encoder(nn.Module):
    def __init__(self, input_size: int, embedding_dim: int, sequences: int, device):
        super(Encoder, self).__init__()
        self.layers = []
        self.linear = nn.Sequential(
            nn.Linear(in_features=sequences, out_features=embedding_dim),
            nn.ReLU()
        )
        for _ in range(input_size):
            self.layers.append(copy.deepcopy(self.linear).to(device))

    def forward(self, x):
        x = x.transpose(1, 2)
        output_list = []
        for idx in range(x.shape[1]):
            _x = x[:, idx, :]
            _x = _x.reshape(-1, 1, _x.shape[-1])
            output_list.append(self.layers[idx](_x))
        out = torch.concat(output_list, dim=1)
        return out


class AttentionCNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 embedding_dim: int,
                 attention_dim: int,
                 sequences: int,
                 num_of_classes: int,
                 device):
        super(AttentionCNN, self).__init__()
        self.encoder = Encoder(input_size=input_size,
                               embedding_dim=embedding_dim,
                               sequences=sequences,
                               device=device).to(device)
        self.attention = Attention(embedding_size=embedding_dim, output_dim=attention_dim).to(device)
        self.linear = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=1, kernel_size=1),
            nn.ReLU()
        )
        self.FC = nn.Linear(in_features=attention_dim, out_features=num_of_classes)

    def forward(self, x):
        out = self.encoder(x)
        out = self.attention(out)
        out = self.linear(out)
        out = nn.Flatten()(out)
        out = self.FC(out)
        return out


class MultiHeadAttentionCNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 embedding_dim: int,
                 attention_dim: int,
                 num_heads: int,
                 sequences: int,
                 num_of_classes: int,
                 device):
        super(MultiHeadAttentionCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder = Encoder(input_size=input_size,
                               embedding_dim=embedding_dim,
                               sequences=sequences,
                               device=device).to(device)
        self.multi_head_attn = MultiHeadAttention(d_model=embedding_dim, num_heads=num_heads)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=embedding_dim)
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim/2)),
            # nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc = nn.Linear(int(embedding_dim/2), num_of_classes)

    def forward(self, x, hidden):
        out = self.encoder(x)
        context, attn = self.multi_head_attn(key=out,
                                             query=out,
                                             value=out)
        context = context.transpose(0, 1)
        out, _ = self.rnn(context, hidden)
        out = out[-1]
        out = self.linear(out)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size, device):
        hidden = torch.ones(1, batch_size, self.embedding_dim, device=device)
        return hidden
