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
from .utilclasses import *

def setup_environment(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.expand_filepaths_to_save_dir:
        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)

    print("Expanded filepaths: ")
    print("\t{}".format(args.model_state_file))

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    # Set seed for reproducibility
    set_seed_everywhere(args.seed, args.cuda)

    return

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

def predict_category(text,Field_TEXT,Field_LABEL,classifier):
    preprocessed_sample = [Field_TEXT.preprocess(sample)]
    processed_sample = Field_TEXT.process(preprocessed_sample).to(args.device)
    y_pred = classifier(processed_sample)
    y_pred_np = y_pred.to(torch.device("cpu")).detach().numpy()

    return Field_LABEL.vocab.itos[np.argmax(y_pred_np)]

def build_model(args,dataset,classifier,Batches_train,Batches_val,Batches_test,loss_func,optimizer,scheduler):

    train_state = make_train_state(args)

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index
            print("--------------------- @epoch ",epoch_index,"---------------------")
            dataset.set_split('train')
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()
            batches = BatchGenerator(Batches_train, 'text', 'category')

            for batch_index, batch_dict in enumerate(batches):
              optimizer.zero_grad()
              y_pred = classifier(batch_dict[0])
              loss = loss_func(y_pred, batch_dict[1])
              loss_t = loss.item()
              running_loss += (loss_t - running_loss) / (batch_index + 1)
              loss.backward()
              optimizer.step()
              acc_t = compute_accuracy(y_pred, batch_dict[1])
              running_acc += (acc_t - running_acc) / (batch_index + 1)

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)
            print('  training loss/accuracy {:.5f} / {:.2f}'.format(running_loss, running_acc))

            dataset.set_split('val')

            batches = BatchGenerator(Batches_val, 'text', 'category')
            running_loss = 0.
            running_acc = 0.
            classifier.eval()

            for batch_index, batch_dict in enumerate(batches):
              optimizer.zero_grad()
              y_pred = classifier(batch_dict[0])
              loss = loss_func(y_pred, batch_dict[1])
              loss_t = loss.item()
              running_loss += (loss_t - running_loss) / (batch_index + 1)
              loss.backward()
              optimizer.step()
              acc_t = compute_accuracy(y_pred, batch_dict[1])
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
    loss_func = nn.CrossEntropyLoss()
    dataset.set_split('test')
    batches = BatchGenerator(Batches_test, 'text', 'category')
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    for batch_index, batch_dict in enumerate(batches):
      optimizer.zero_grad()
      y_pred = classifier(batch_dict[0])
      loss = loss_func(y_pred, batch_dict[1])
      loss_t = loss.item()
      running_loss += (loss_t - running_loss) / (batch_index + 1)
      loss.backward()
      optimizer.step()
      acc_t = compute_accuracy(y_pred, batch_dict[1])
      running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

    print("Test loss: {:.3f}".format(running_loss))
    print("Test Accuracy: {:.2f}".format(running_acc))

    return train_state

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    return text.strip()


