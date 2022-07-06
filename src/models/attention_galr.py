from src.models.cnn import Embedding, gEmbedding
from src.models.galr import GALR, Segment1d
from torch import nn
from torch.nn import functional as F
import torch


class AttentiveGALR(nn.Module):
    def __init__(self,
                 input_size: int,
                 embedding_dim: int,
                 sequences: int,
                 num_heads: int,
                 chunk_size: int,
                 hop_size: int,
                 hidden_channels: int,
                 low_dimension: bool,
                 num_layers: int,
                 num_classes: int,
                 non_linear: bool,
                 **kwargs):
        super(AttentiveGALR, self).__init__()

        # Multi-head attention layer
        self.multihead_attention = nn.Sequential(
            Embedding(input_size=input_size, embedding_dim=embedding_dim, sequences=sequences, nonlinear=non_linear),
            torch.nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True),
                num_layers=num_layers)
        )

        # GALR layer
        self.galr = nn.Sequential(
            gEmbedding(input_size=input_size, embedding_dim=embedding_dim, sequences=sequences, nonlinear=non_linear),
            Segment1d(chunk_size=chunk_size, hop_size=hop_size),
            GALR(num_features=input_size,
                 hidden_channels=hidden_channels,
                 low_dimension=low_dimension,
                 num_heads=num_heads,
                 **kwargs)
        )
        # self.mthd_embedding = Embedding(input_size=input_size, embedding_dim=embedding_dim, sequences=sequences)
        # self.galr_embedding = Embedding(input_size=input_size, embedding_dim=embedding_dim, sequences=sequences)

        self.conv2d = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=(2, 1), dilation=(1, 1), bias=False)

        # TODO: Implement suitable FC layer in the future.
        self.fc = nn.Sequential(
            nn.Linear(sequences * input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_x):
        batch, sequences, features = input_x.size()

        mthd_out = self.multihead_attention(input_x)
        galr_out = self.galr(input_x)
        galr_out = galr_out.view(batch, features, -1)

        out = torch.stack((mthd_out, galr_out), dim=1)
        out = out.view(-1, sequences, 2, features)
        out = self.conv2d(out)

        out = out.view(-1, sequences * features)
        out = self.fc(out)
        return out


class MultiheadAttentionGALR(nn.Module):
    def __init__(self,
                 input_size: int,
                 embedding_dim: int,
                 sequences: int,
                 num_heads: int,
                 chunk_size: int,
                 hop_size: int,
                 hidden_channels: int,
                 low_dimension: bool,
                 num_classes: int,
                 linear: bool,
                 **kwargs):
        super(MultiheadAttentionGALR, self).__init__()

        self.embedding = Embedding(input_size=input_size, embedding_dim=embedding_dim, sequences=sequences, linear=linear)

        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        # GALR layer
        self.galr = nn.Sequential(
            Segment1d(chunk_size=chunk_size, hop_size=hop_size),
            GALR(num_features=input_size,
                 hidden_channels=hidden_channels,
                 low_dimension=low_dimension,
                 num_heads=num_heads,
                 **kwargs)
        )

        self.conv1d = nn.Conv1d(8, 1, kernel_size=1, stride=1)

        # TODO: Implement suitable FC layer in the future.
        self.fc = nn.Sequential(
            nn.Linear(sequences, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_x):
        batch, sequences, features = input_x.size()

        embedded_out = self.embedding(input_x)
        mthd_out, attention_weights = self.multihead_attention(embedded_out, embedded_out, embedded_out)

        galr_out = self.galr(embedded_out)
        galr_out = galr_out.view(batch, features, -1)
        out = torch.bmm(attention_weights, galr_out)

        out = F.relu(self.conv1d(out))
        out = out.view(-1, sequences)
        out = self.fc(out)
        return out
