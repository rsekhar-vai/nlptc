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
from utils.classes import *

def setup_environment(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)

    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    # Set seed for reproducibility
    set_seed_everywhere(args.seed, args.cuda)

    return


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings

    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r", encoding="utf-8") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings


def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


def predict_category(text, classifier, vectorizer, max_length):
    """Predict a Text category for a new text

    Args:
        text (str): a raw text string
        classifier (TextClassifier): an instance of the trained classifier
        vectorizer (TextVectorizer): the corresponding vectorizer
        max_length (int): the max sequence length
            Note: CNNs are sensitive to the input data tensor size.
                  This ensures to keep it the same size as the training data
    """
    text = preprocess_text(text)
    vectorized_text = \
        torch.tensor(vectorizer.vectorize(text, vector_length=max_length))
    result = classifier(vectorized_text.unsqueeze(0), apply_softmax=True)
    probability_values, indices = result.max(dim=1)
    predicted_category = vectorizer.category_vocab.lookup_index(indices.item())

    return {'category': predicted_category,
            'probability': probability_values.item()}

def get_samples():
    samples = {}
    for cat in dataset.val_df.category.unique():
        samples[cat] = dataset.val_df.text[dataset.val_df.category==cat].tolist()[:5]
    return samples


def build_model(args,dataset,classifier,loss_func,optimizer,scheduler):

    train_state = make_train_state(args)

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index
            print("--------------------- @epoch ",epoch_index,"---------------------")

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on

            dataset.set_split('train')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = classifier(batch_dict['x_data'])

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
#                 print("@batch ",batch_index,"-->",running_acc)

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)
            print('  training loss/accuracy {:.5f} / {:.2f}'.format(running_loss, running_acc))

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('val')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.
            running_acc = 0.
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # compute the output
                y_pred = classifier(batch_dict['x_data'])

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)
            print('validation loss/accuracy {:.5f} / {:.2f}'.format(running_loss, running_acc))

            train_state = update_train_state(args=args, model=classifier,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break

    except KeyboardInterrupt:
        print("Exiting loop")

    # compute the loss & accuracy on the test set using the best available model

    classifier.load_state_dict(torch.load(train_state['model_filename']))

    classifier = classifier.to(args.device)
    dataset.class_weights = dataset.class_weights.to(args.device)
    loss_func = nn.CrossEntropyLoss(dataset.class_weights)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(batch_dict['x_data'])

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

    print("Test loss: {:.3f}".format(running_loss))
    print("Test Accuracy: {:.2f}".format(running_acc))

    return train_state

def load_vectorize_data(args):

    if args.reload_from_files:
        # training from a checkpoint
        dataset = TextDataset.load_dataset_and_load_vectorizer(args.text_csv,
                                                               args.vectorizer_file)
    else:
        # create dataset and vectorizer
        dataset = TextDataset.load_dataset_and_make_vectorizer(args)
        dataset.save_vectorizer(args.vectorizer_file)
    vectorizer = dataset.get_vectorizer()

    return dataset, vectorizer


def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    return text.strip()


