
import torch
from pathlib import Path
import numpy as np
import json
import pandas as pd
from BaseExperiment import BaseExperiment, test_model
from Dataset import HeadCTScan
from utils import read_files
from sklearn.model_selection import train_test_split
import os
import argparse

# Argparsers
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='/mnt/dados/dataset_jpr_train/dataset_36slices', help='Path to the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--num_workers', type=int, default=8, help='Number of Workers in Dataloader')
parser.add_argument('--debug', action='store_false' ,help='Activate Debug Mode')
parser.add_argument('--exp_name', type=str, default='exp01', help='Experiment Name')
args = parser.parse_args()

batch_size = args.batch_size
num_workers = args.num_workers
Debug = args.debug
root_dir = Path(args.root_dir)

# root_dir = Path('/media/SSD2/IDOR/spr-head-ct-age-prediction-challenge/dataset_jpr_train/dataset_36slices')
# root_dir = Path('/mnt/dados/dataset_jpr_train/dataset_36slices')
# /mnt/dados/dataset_jpr_train/dataset_36slices

print(root_dir)

transform = None

# TODO: Load Dataset here
# data_files = read_files(root_dir, Debug)
filenames = os.listdir(root_dir)
train_files, val_files = train_test_split(filenames, test_size=0.2, random_state=42)

train_set = HeadCTScan(root_dir, train_files, transform=transform, Debug=Debug)
val_set = HeadCTScan(root_dir, val_files, transform=transform, Debug=Debug)
print(len(train_set), len(val_set))

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
# Exp17 training with categories 2 in input
custom_training_config = {
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 100,
    'lr': 1e-4,
    'ex_name': args.exp_name, # custom_exp
    'patience': 10,
    'delta': 0.0001,
    'in_shape': (36, 512, 512)
}

exp = BaseExperiment(dataloader_train, dataloader_val, custom_training_config)

exp.train()

test_set = HeadCTScan(root_dir=root_dir, Debug=Debug, mode='val', val_data=test_data,\
    means=[train_set.mean], stds=[train_set.std])

dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, pin_memory=True)

# test_model(dataloader_test, custom_training_config, custom_model_config)
