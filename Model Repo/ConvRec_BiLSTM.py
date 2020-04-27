from argparse import Namespace
from collections import Counter
import json
import os
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



class ConvRec_BiLSTM(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels,
                 hidden_dim, num_classes, dropout_p,
                 pretrained_embeddings=None, padding_idx=0):
        """
        Args:
            embedding_size (int): size of the embedding vectors
            num_embeddings (int): number of embedding vectors
            filter_width (int): width of the convolutional kernels
            num_channels (int): number of convolutional kernels per layer
            hidden_dim (int): the size of the hidden dimension
            num_classes (int): the number of classes in classification
            dropout_p (float): a dropout parameter
            pretrained_embeddings (numpy.array): previously trained word embeddings
                default is None. If provided,
            padding_idx (int): an index representing a null position
        """
        super(ConvRec_BiLSTM, self).__init__()

        if pretrained_embeddings is None:

            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)

        self._extractor1 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            #Permute(),
            #nn.Dropout()
            )
            #Flatten()

        self._extractor2 = nn.LSTM(73,num_channels, batch_first=True, bidirectional=True)


        self._classifier = nn.Linear(in_features=2 * num_channels, out_features=num_classes)

        self.apply(self._init_weights)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
        """

        # embed and permute so features are channels
        x_embedded = self.emb(x_in).permute(0, 2, 1)
        feature = self._extractor1(x_embedded)
        outputs, hc = self._extractor2(feature)
        feature = torch.cat([*hc[0]], dim=1)

        prediction_vector = self._classifier(feature)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)


class Flatten(nn.Module):
    """Flatten class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Permute(nn.Module):
    """Permute class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)
