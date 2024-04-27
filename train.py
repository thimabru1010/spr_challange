
import torch
from pathlib import Path
import numpy as np
import json
import pandas as pd
from BaseExperiment import BaseExperiment, test_model
from Dataset import HeadCTScan, HeadCTScan_TestSubmission, HeadCTScan_Val
from utils import read_files
from sklearn.model_selection import train_test_split
import os
import argparse
from torchvision import transforms

# Argparsers
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='/mnt/dados/dataset_jpr_train/segmented_dataset_9slices', help='Path to the dataset')
parser.add_argument('--root_dir2', type=str, default='/mnt/dados/dataset_jpr_train/segmented_dataset_9slices_single', help='Path to the dataset')
parser.add_argument('--label_path', type=str, default='/mnt/dados/train_test_groups.csv', help='Path to label csv')
parser.add_argument('--label_path2', type=str, default='/mnt/dados/train_test_groupsX9.csv', help='Path to label csv')
parser.add_argument('--test_dir', type=str, default='/mnt/dados/dataset_jpr_test/segmented_dataset_36slices', help='Path to the test dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of Workers in Dataloader')
parser.add_argument('--debug', action='store_true',help='Activate Debug Mode')
parser.add_argument('--exp_name', type=str, default='custom_exp22', help='Experiment Name')
parser.add_argument('--deactivate_test', action='store_true', help='Deactivate Test')
parser.add_argument('--deactivate_train', action='store_true', help='Deactivate Train')
parser.add_argument('--aux_clssf', action='store_true', help='Activates auxiliary classification head for the model')
parser.add_argument('--clssf_weights', action='store_true', help='Enable classification weights on classification loss')
parser.add_argument('--input_channels', type=int, default=1, help='Channels of the input image')
parser.add_argument('--n_slices', type=int, default=9, help='Number of slices used to cut the CT scan')
parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone model for the experiment')
parser.add_argument('--norm_min', type=int, default=-15, help='Normalization min bound')
parser.add_argument('--norm_max', type=int, default=1024, help='Normalization max bound')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('-optm', '--optmizer', type=str, default='adam', help='Optmizer')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--sched_step_size', type=int, default=1, help='SGD momentum')
parser.add_argument('--sched_decay_factor', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--use_data_aug', action='store_true', help='Enables data augmentation')
args = parser.parse_args()

batch_size = args.batch_size
num_workers = args.num_workers
Debug = args.debug
root_dir = args.root_dir
root_dir2 = args.root_dir2
test_dir = args.test_dir

clssf_weights = None
if args.clssf_weights:
    clssf_weights = [1.46, 4.62, 10.24]

# root_dir = Path('/media/SSD2/IDOR/spr-head-ct-age-prediction-challenge/dataset_jpr_train/dataset_36slices')
# root_dir = Path('/mnt/dados/dataset_jpr_train/dataset_36slices')
# /mnt/dados/dataset_jpr_train/dataset_36slices

print(root_dir)

if args.use_data_aug:
    prob = 0.5
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256))
            # transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=prob),
            # transforms.RandomApply([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()], p=prob)
        ])

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256))
        ])
    img_size = 256
else:
    transform = None
    val_transform = None
    img_size = 512


print(args.backbone)
training_config = {
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 150,
    'lr': args.learning_rate,
    'ex_name': args.exp_name, 
    'patience': 10,
    'delta': 0.0001,
    'in_shape': (args.input_channels, img_size, img_size),
    'classification_head': args.aux_clssf,
    'clssf_weights': clssf_weights,
    'backbone': args.backbone,
    'n_slices': args.n_slices,
    'norm_min': args.norm_min,
    'norm_max': args.norm_max,
    'optimizer': args.optmizer,
    'momentum': args.momentum,
    'sched_step_size': args.sched_step_size,
    'sched_decay_factor': args.sched_decay_factor
}

if not args.deactivate_train:
    # data_files = read_files(root_dir, Debug)
    
    # patients = ['000648', '002394', '002469']
    # estratificado
    df = pd.read_csv(args.label_path, converters={'StudyID': str})
    df['StudyID_pure'] = df['StudyID'].apply(lambda x: x.split('_')[0])

    grouped_channels = df.groupby('StudyID_pure').size().reset_index(name='n_channels')
    print(df.shape)
    
    df = pd.merge(df, grouped_channels, on='StudyID_pure')
    df_tmp = df[['Group', 'StudyID_pure', 'n_channels']]
    df_tmp = df_tmp.drop_duplicates()
    print(df_tmp.head(10))
    1/0
    list_groups = df_tmp['Group'].tolist()
    # list_groups = groups['Age'].tolist()
    filenames = df_tmp['StudyID_pure'].tolist()
    n_channels = df_tmp['n_channels'].tolist()
    
    # print(df_tmp.shape)
    # _train_files = filenames[:2]
    # val_files = filenames[-1:]
    # print(_train_files, val_files)
    _train_files, val_files = train_test_split(filenames, test_size=0.2, random_state=42, stratify=list_groups)
    
    print(f'Patients Count Train: {len(_train_files)} - Val: {len(val_files)}')
    
    if args.n_slices != args.input_channels:
        train_files = []
        n_slices_lst = list(range(args.n_slices))
        if args.n_slices == -1:
            n_slices_lst = n_channels
        for i, filename in enumerate(_train_files):
            for j in n_slices_lst:
                train_files.append(filename + '_' + str(j))
    else:
        train_files = _train_files
    
            
    train_set = HeadCTScan(root_dir, args.label_path, train_files, transform=transform, Debug=Debug,\
        aux_clssf=args.aux_clssf, training_config=training_config)
    val_set = HeadCTScan_Val(root_dir2, args.label_path, val_files, transform=val_transform, Debug=Debug,\
        aux_clssf=args.aux_clssf, training_config=training_config)
    print(len(train_set), len(val_set))

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    if training_config['n_slices'] != training_config['in_shape'][0]:
        batch_size_val = 1
    else:
        batch_size_val = batch_size
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=num_workers)

    exp = BaseExperiment(dataloader_train, dataloader_val, training_config)

    exp.train()

if not args.deactivate_test:
    test_files = os.listdir(test_dir)

    test_set = HeadCTScan_TestSubmission(root_dir=test_dir, data_files=test_files, Debug=Debug, transform=val_transform)

    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_val, shuffle=False, pin_memory=True)

    test_model(dataloader_test, training_config)
