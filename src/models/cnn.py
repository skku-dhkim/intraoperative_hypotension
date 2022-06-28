import torch
import torch.nn.functional as F
import math
from torch import nn
from src.models.attentions import MultiHeadAttention


class OneDimCNN(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, num_of_classes: int):
        super(OneDimCNN, self).__init__()
        init_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(init_channel, 2*hidden_dim, kernel_size=3),
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
            nn.BatchNorm1d(init_channel*4),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(23680, 512),
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


class Embedding(nn.Module):
    def __init__(self, input_size: int, embedding_dim: int, sequences: int, nonlinear=False, sequence_first=False):
        super(Embedding, self).__init__()
        self.layers = []
        self.sequence_first = sequence_first

        def create_linear(in_c, out_c):
            if nonlinear:
                linear = nn.Sequential(
                    nn.Linear(in_features=in_c, out_features=out_c, bias=False),
                    nn.ReLU()
                )
            else:
                linear = nn.Sequential(
                    nn.Linear(in_features=in_c, out_features=out_c, bias=False)
                )
            return linear
        self.layers = nn.ModuleList([create_linear(sequences, embedding_dim) for _ in range(input_size)])

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()             # (batch, features, sequences)
        output_list = []
        for idx in range(x.shape[1]):
            _x = x[:, idx, :]
            _x = _x.reshape(-1, 1, _x.shape[-1])        # (batches ,1, sequences)
            output_list.append(self.layers[idx](_x))    # (batches, idx, 1, embedding_dim)
        out = torch.concat(output_list, dim=1)          # (batches, num_of_features, embedding_dim)
        if self.sequence_first:
            out = out.transpose(1, 2).contiguous()
        return out


class AttentionCNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 embedding_dim: int,
                 attention_dim: int,
                 sequences: int,
                 num_of_classes: int):
        super(AttentionCNN, self).__init__()
        self.encoder = Encoder(input_size=input_size,
                               embedding_dim=embedding_dim,
                               sequences=sequences)
        self.attention = Attention(embedding_size=embedding_dim, output_dim=attention_dim)
        self.linear = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=1, kernel_size=(1, 1)),
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
                 num_of_classes: int):
        super(MultiHeadAttentionCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder = Embedding(input_size=input_size,
                                 embedding_dim=embedding_dim,
                                 sequences=sequences)
        self.multi_head_attn = MultiHeadAttention(d_model=embedding_dim, num_heads=num_heads)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=embedding_dim)
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim/2)),
            # nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc = nn.Linear(int(embedding_dim/2), num_of_classes)

    def forward(self, x, hidden):
        out = self.encoder(x)#(batch_size,features,len)
        context, attn = self.multi_head_attn(key=out,
                                             query=out,
                                             value=out)
        context = context.transpose(0, 1) #(features,batch_size,len) -> (batch_size,features,len)
        out, _ = self.rnn(context, hidden)
        out = out[-1]
        out = self.linear(out)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size, device):
        hidden = torch.ones(1, batch_size, self.embedding_dim, device=device)
        return hidden


# TODO: causal_cnn --> cnn.py
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