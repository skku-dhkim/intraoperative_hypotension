from .. import *
from .cnn import OneDimCNN, CausalOneDimCNN, MultiChannelCNN, AttentionCNN, MultiHeadAttentionCNN
from .rnn import ValinaLSTM
from .galr import GALRBlock
from .attention_galr import AttentiveGALR
# import torch
# from torch import nn


def call_models(model_name: str,
                features: Optional[int] = None,
                sequences: Optional[int] = None,
                # batch_size: Optional[int] = None,
                num_of_classes: Optional[int] = 2,
                **kwargs):

    # TODO: Need to be re-arranged in the future.
    # NOTE: Model Setting
    if model_name.lower() == "lstm":
        hidden = True
        model = ValinaLSTM(features,
                           hidden_dim=kwargs['hidden_dim'],
                           layers=kwargs['layers'],
                           num_of_classes=num_of_classes)
    # elif model_name.lower() == 'cnn':
    #     hidden = False
    #     model = OneDimCNN(data_shape[-1], hidden_dim=hidden_dim, num_of_classes=num_of_classes)
    # elif model_name.lower() == 'causal-cnn':
    #     # TODO: Check Future
    #     hidden = False
    #     model = CausalOneDimCNN(data_shape[-1], hidden_dim=hidden_dim, num_of_classes=num_of_classes)
    # elif model_name.lower() == 'multi-channel-cnn':
    #     hidden = False
    #     model = MultiChannelCNN(input_size=data_shape[-1], num_of_classes=num_of_classes)
    # elif model_name.lower() == 'attention_cnn':
    #     hidden = False
    #     model = AttentionCNN(input_size=data_shape[-1],
    #                          embedding_dim=hidden_dim,
    #                          attention_dim=attention_dim,
    #                          sequences=data_shape[1],
    #                          num_of_classes=2)
    elif model_name.lower() == 'galr':
        hidden = False
        model = GALRBlock(num_features=8,
                          hidden_channels=64,
                          num_heads=8,
                          norm=True,
                          dropout=1e-1,
                          num_of_classes=2,
                          block_num=1)

    # elif model_name.lower() == 'multi_head_attn':
    #     hidden = True
    #     model = MultiHeadAttentionCNN(
    #         input_size=data_shape[-1],
    #         embedding_dim=hidden_dim,
    #         attention_dim=attention_dim,
    #         num_heads=num_of_heads,
    #         sequences=data_shape[1],
    #         num_of_classes=2)
    elif model_name.lower() == 'attention_galr':
        model = AttentiveGALR(input_size=features,
                              # embedding_dim=kwargs['embedding_dim'],
                              sequences=sequences,
                              num_heads=kwargs['num_of_heads'],
                              # chunk_size=kwargs['chunk_size'],
                              # hop_size=kwargs['hop_size'],
                              # hidden_channels=kwargs['hidden_channels'],
                              # low_dimension=kwargs['low_dimension'],
                              # num_layers=kwargs['num_layers'],
                              num_classes=num_of_classes,
                              **kwargs)
    else:
        raise NotImplementedError()
    return model
