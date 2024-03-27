import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import nibabel as nib

class HeadCTScan(Dataset):
    def __init__(self, root_dir: Path, data_files: list, normalize: bool=True, transform: torchvision.transforms=None,
                Debug: bool=False, aux_clssf: bool=False):
        super(HeadCTScan, self).__init__()
        
        self.root_dir = root_dir
        self.data_files = data_files
        if Debug:
            self.data_files = self.data_files[:20]
        
        self.labels = pd.read_csv('/mnt/dados/train_test_groups.csv',  converters={'StudyID': str})      
        
        # self.labels = pd.read_csv('/mnt/dados/train_test.csv')
        self.labels['StudyID'] = self.labels['StudyID'].apply(lambda x: x.lstrip('0'))
        # self.groups = self.labels.set_index('StudyID').to_dict()['Group']
        self.labels = self.labels.set_index('StudyID').to_dict()['Age']
        # print(self.labels)
        # # print(self.labels.head(10))
        # 1/0
        self.normalize = normalize
        self.transform = transform
        
        self.aux_clssf = aux_clssf
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = self.data_files[index]
        
        # print(file_name)
        data = nib.load(self.root_dir + '/' + file_name).get_fdata()
        # data = nib.load(self.root_dir + '/' + file_name + '.nii.gz').get_fdata()
                
        b = data.reshape((data.shape[0], data.shape[1], 1))
        # print(b.shape)
        data = b.transpose(2, 0, 1)
        data = data[:, :512, :512]
        
        labels = self.labels[file_name.split('/')[-1].split('_')[0].lstrip('0')]
        #labels = self.labels[file_name.split('.')[0].lstrip('0')]
        if self.aux_clssf:
            # Idade min = 18; Idade Max = 89
            labels = (labels - 18) / (89 - 18)
        
        # groups = self.groups[file_name.split('.')[0].lstrip('0')]        
        
        if self.normalize:
            # data = data - data.mean() / data.std()
            
            dcm_min, dcm_max = -1024, 1024
            data = 2*((data - dcm_min) / (dcm_max - dcm_min)) - 1    # min-max normalization (-1,1)
            # data = ((data - dcm_min) / (dcm_max - dcm_min))   # min-max normalization (0,1)
            # data = data - data.mean() / data.std()
            
        # data = np.expand_dims(data, axis=1)
            
        if self.transform:
            data = self.transform(data)
        else:
            data = torch.tensor(data, dtype=torch.float32)
            labels = torch.tensor(labels)
            # groups = torch.tensor(groups)
        # print(data.shape, labels.shape)
        # print(data.dtype, labels.dtype)
        return data, labels.unsqueeze(0), int(file_name.split('.')[0]) #, groups


class HeadCTScan_TestSubmission(Dataset):
    def __init__(self, root_dir: Path, data_files: list, normalize: bool=True, transform: torchvision.transforms=None,
                Debug: bool=False):
        super(HeadCTScan_TestSubmission, self).__init__()
        
        self.root_dir = root_dir
        self.data_files = data_files
        if Debug:
            self.data_files = self.data_files[:20]
        
        # self.labels = pd.read_csv('/media/SSD2/IDOR/spr-head-ct-age-prediction-challenge/train.csv')
        # self.labels = pd.read_csv('/mnt/dados/train.csv')
        # self.labels['StudyID'] = self.labels['StudyID'].apply(lambda x: x.lstrip('0'))
        # self.labels = self.labels.set_index('StudyID').to_dict()['Age']

        self.normalize = normalize
        self.transform = transform
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = self.data_files[index]
        
        data = nib.load(self.root_dir + '/' + file_name).get_fdata()
        data.reshape((data.shape[0], data.shape[1], 1))
        data = data.transpose(2, 0, 1)
        data = data[:, :512, :512]

        # labels = self.labels[file_name.split('/')[-1].split('.')[0].lstrip('0')]
        
        if self.normalize:
            # data = data - data.mean() / data.std()
            
            dcm_min, dcm_max = -1024, 1024
            data = 2*((data - dcm_min) / (dcm_max - dcm_min)) - 1    # min-max normalization (-1,1)
            # data = ((data - dcm_min) / (dcm_max - dcm_min))   # min-max normalization (0,1)
            # data = data - data.mean() / data.std()
            
        if self.transform:
            data = self.transform(data)
        else:
            data = torch.tensor(data, dtype=torch.float32)
            # labels = torch.tensor(labels)
        # print(data.shape, labels.shape)
        # print(data.dtype, labels.dtype)
        # print(int(file_name.split('.')[0]))
        return data, int(file_name.split('.')[0])