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
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    def __init__(self, text_df,args):
        self.text_df = text_df
        self._max_seq_length = args.max_text_length
        train_df, self.test_df = train_test_split(text_df)
        self.train_df, self.val_df = train_test_split(train_df)

        self.train_size = len(self.train_df)
        self.validation_size = len(self.val_df)
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')


    def get_splits(self):
        return self.train_df, self.val_df, self.text_df

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X, y)
