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


class TCDataset(Dataset):
    def __init__(self, text_df,args):
        self.text_df = text_df
        self._max_seq_length = args.max_text_length
        train_df, self_test_df = train_test_split(text_df)
        self.train_df, self.val_df = train_test_split(train_df)

        self.train_size = len(self.train_df)
        self.validation_size = len(self.val_df)
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        class_counts = text_df.category.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.category_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        text_vector = \
            self._vectorizer.vectorize(row.text, self._max_seq_length)
        category_index = \
            self._vectorizer.category_vocab.lookup_token(row.category)
        return {'x_data': text_vector,
                'y_target': category_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


class Vectorizer(object):
    def __init__(self, tex_df,token_type ='w', cutoff=25):
        self.text_vocab, self.category_vocab, self_token_type = \
                self.create_vocab(tex_df, token_type, cutoff)

    def vectorize(self, text, vector_length=-1):
        indices = [self.text_vocab.begin_seq_index]
        if self.token_type == 'w':
            indices.extend(self.text_vocab.lookup_token(token)
                           for token in text.split(" "))
        else:
            indices.extend(self.text_vocab.lookup_token(token)
                           for token in text)
        indices.append(self.text_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.text_vocab.mask_index

        return out_vector

    def create_vocab(self,df, token_type ='w', cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            df (pandas.DataFrame): the target dataset
            cutoff (int): frequency threshold for including in Vocabulary
        Returns:
            an instance of the TextVectorizer
        """
        category_vocab = Vocabulary()
        for category in sorted(set(df.category)):
            category_vocab.add_token(category)

        def tokenized_word_counter(df):
            word_counts = Counter()
            for text in df.text:
                for word in text.split(" "):
                    if word not in string.punctuation:
                        word_counts[word] += 1
            return word_counts

        text_vocab = SequenceVocabulary()
        if token_type == 'w':
            word_counts = tokenized_word_counter(df)
            for word, word_count in word_counts.items():
                if word_count >= cutoff:
                    text_vocab.add_token(word)
        else:
            for text in df.text:
                for letter in text:
                    text_vocab.add_token(letter)

        return (text_vocab, category_vocab,token_type)


    def tokenized_word_counter(self,df):
        word_counts = Counter()
        for text in df.text:
            for word in text.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1
        return word_counts


class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary

        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Retrieve the index associated with the token

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        """
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


