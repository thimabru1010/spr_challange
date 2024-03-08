
import torch
from pathlib import Path
import numpy as np
import json
import pandas as pd
from BaseExperiment import BaseExperiment, test_model

batch_size = 16
num_workers = 8
Debug = False
pixel_size = '1K'

patch_size = 64
overlap = 0.1
window_size = 3
min_def = 0.02

root_dir = Path(f'/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/IBAMA_INPE/{pixel_size}')
print(root_dir)

transform = None

# TODO: Load Dataset here

print(len(train_set), len(val_set))

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
# Exp17 training with categories 2 in input
custom_training_config = {
    'pre_seq_length': 2,
    'aft_seq_length': 1,
    'total_length': 3,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 100,
    'lr': 1e-4,
    # 'metrics': ['mse', 'mae', 'acc', 'Recall', 'Precision', 'f1_score', 'CM'],
    'metrics': ['mse', 'mae'],

    'ex_name': 'custom_exp09', # custom_exp
    'dataname': 'custom',
    'patience': 10,
    'delta': 0.0001,
    'amazon_mask': True,
    'pixel_size': pixel_size,
    'patch_size': patch_size,
    'overlap': overlap
}

exp = BaseExperiment(dataloader_train, dataloader_val, custom_model_config, custom_training_config)

exp.train()

test_data, mask_test_data = train_set.get_test_set()
test_set = IbamaDETER1km_Dataset(root_dir=root_dir, Debug=Debug, mode='val', val_data=test_data,\
    mask_val_data=mask_test_data, means=[train_set.mean], stds=[train_set.std])

dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, pin_memory=True)

# test_model(dataloader_test, custom_training_config, custom_model_config)
