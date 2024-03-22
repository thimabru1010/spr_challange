
import torch
from pathlib import Path
import numpy as np
import json
import pandas as pd
from BaseExperiment import BaseExperiment, test_model
from Dataset import HeadCTScan, HeadCTScan_TestSubmission
from utils import read_files
from sklearn.model_selection import train_test_split
import os
import argparse

# Argparsers
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='/mnt/dados/dataset_jpr_train/dataset_36slices', help='Path to the dataset')
parser.add_argument('--test_dir', type=str, default='/mnt/dados/dataset_jpr_test/segmented_dataset_36slices', help='Path to the test dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--num_workers', type=int, default=8, help='Number of Workers in Dataloader')
parser.add_argument('--debug', action='store_false',help='Activate Debug Mode')
parser.add_argument('--exp_name', type=str, default='custom_exp09', help='Experiment Name')
parser.add_argument('--deactivate_test', action='store_true', help='Deactivate Test')
parser.add_argument('--deactivate_train', action='store_true', help='Deactivate Train')
parser.add_argument('--aux_clssf', action='store_true', help='Activates auxiliary classification head for the model')
parser.add_argument('--clssf_weights', action='store_true', help='Enable classification weights on classification loss')
args = parser.parse_args()

batch_size = args.batch_size
num_workers = args.num_workers
Debug = args.debug
root_dir = Path(args.root_dir)
test_dir = Path(args.test_dir)

clssf_weights = None
if args.clssf_weights:
    clssf_weights = [1.46, 4.62, 10.24]

# root_dir = Path('/media/SSD2/IDOR/spr-head-ct-age-prediction-challenge/dataset_jpr_train/dataset_36slices')
# root_dir = Path('/mnt/dados/dataset_jpr_train/dataset_36slices')
# /mnt/dados/dataset_jpr_train/dataset_36slices

print(root_dir)

transform = None

training_config = {
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 100,
    'lr': 1e-4,
    'ex_name': args.exp_name, 
    'patience': 10,
    'delta': 0.0001,
    'in_shape': (36, 512, 512),
    'classification_head': args.aux_clssf,
    'clssf_weights': clssf_weights,
}

if not args.deactivate_train:
    # data_files = read_files(root_dir, Debug)
    filenames = os.listdir(root_dir)
    train_files, val_files = train_test_split(filenames, test_size=0.2, random_state=42)

    train_set = HeadCTScan(root_dir, train_files, transform=transform, Debug=Debug, aux_clssf=args.aux_clssf)
    val_set = HeadCTScan(root_dir, val_files, transform=transform, Debug=Debug, aux_clssf=args.aux_clssf)
    print(len(train_set), len(val_set))

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    exp = BaseExperiment(dataloader_train, dataloader_val, training_config)

    exp.train()

if not args.deactivate_test:
    test_files = os.listdir(test_dir)

    test_set = HeadCTScan_TestSubmission(root_dir=test_dir, data_files=test_files, Debug=Debug)

    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_model(dataloader_test, training_config)
