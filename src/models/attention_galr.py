from src.models.cnn import Embedding
from src.models.galr import GALR, Segment1d
from torch import nn
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
                 **kwargs):
        super(AttentiveGALR, self).__init__()

        # Multi-head attention layer
        self.multihead_attention = nn.Sequential(
            Embedding(input_size=input_size, embedding_dim=embedding_dim, sequences=sequences),
            torch.nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True),
                num_layers=num_layers)
        )

        # GALR layer
        self.galr = nn.Sequential(
            Embedding(input_size=input_size, embedding_dim=embedding_dim, sequences=sequences),
            Segment1d(chunk_size=chunk_size, hop_size=hop_size),
            GALR(num_features=input_size,
                 hidden_channels=hidden_channels,
                 low_dimension=low_dimension,
                 num_heads=num_heads,
                 **kwargs)
        )
        self.mthd_embedding = Embedding(input_size=input_size, embedding_dim=embedding_dim, sequences=sequences)
        self.galr_embedding = Embedding(input_size=input_size, embedding_dim=embedding_dim, sequences=sequences)

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
