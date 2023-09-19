# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import os
import logging
import time
import datetime
from tqdm import tqdm
import argparse

import os
import logging
import time
import pandas as pd

from models.Simstock import model
from utils.helper import make_noise
from utils.prepro import dataset_for_modeling
from exp.training import train, test


# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def log(str): logger.info(str)


parser = argparse.ArgumentParser(description="SimStock")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = ['train_nasdaq', 'train_sse', 'train_szse', 'train_tse']
test_datasets = ['test_nasdaq', 'tset_sse', 'test_szse', 'test_tse', "ARKK", "LIT", "BOTZ", "SKYY"]

# dataset param
parser.add_argument("--train_dataset", default="train_nasdaq", type=str, help="one of: {}".format(", ".join(sorted(datasets))))
parser.add_argument("--test_dataset", default="test_nasdaq", type=str, help="one of: {}".format(", ".join(sorted(test_datasets))))
parser.add_argument("--batch_size", default=512, type=int,      help="the number of epoches for each task.")
parser.add_argument("--data_size", default=25, type=int,      help="the number of input features.")

# model param
parser.add_argument("--noise_dim", default=25, type=float,     help="the dimension of the LSTM input noise.")
parser.add_argument("--latent_dim", default=25, type=float,     help="the latent dimension of RNN variables.")
parser.add_argument("--hidden_dim", default=128, type=float,     help="the latent dimension of RNN variables.")
parser.add_argument("--noise_type", choices=["Gaussian", "Uniform"], default="Gaussian", help="The noise type to feed into the generator.")
parser.add_argument("--num_rnn_layer", default=1, type=float,   help="the number of RNN hierarchical layers.")
parser.add_argument("--sector_size", default=138, type=int,help="the number of sector size. WARNING : total + 1")
parser.add_argument("--sector_emb", default=256, type=int,help="the number of sector embedding size")
parser.add_argument("--lambda_values", default=0.7, type=float,help="the number of sector argument")

# training param
parser.add_argument("--learning_rate", default=1e-3, type=float,help="the unified learning rate for each single task.")
parser.add_argument("--epoches", default=5, type=int, help="the number of epoches for each task.")
parser.add_argument("--save_name", default="test", type=str,help="model save weight")

args = parser.parse_args([])


def main(arsgs):
    train_out = dataset_for_modeling(args, train_type = False)
    test_out = dataset_for_modeling(args, train_type = True)
    models =  model(args, device).to(device)
    optimizer = torch.optim.Adam(models.parameters(), lr=args.learning_rate)

    starting_time = time.time()


    Es, hiddens = [None], [None]
    for task_id, dataloader in enumerate(train_out[:-1]):
        E, hidden, rnn_unit = train(dataloader, optimizer, models, args, log, device, Es[-1], hiddens[-1], task_id)
        Es.append(E)
        hiddens.append(hidden)    
    ending_time = time.time()

    print("Training time:", ending_time - starting_time)

    # Testing
    representation_l = test(train_out[-1], model, args, log, device, Es[-1], hiddens[-1], is_repre = True)
    test(train_out[-1], model, args, log, device, Es[-1], hiddens[-1], is_repre = False)

    # Testing 2
    representation_ll = test(test_out[-1], model, args, log, device, Es[-1], hiddens[-1], is_repre = True)
    
    return representation_l, representation_ll


if __name__ == "__main__":
    print("Start Training and get embeddings")
    r1, r2 = main(args)
