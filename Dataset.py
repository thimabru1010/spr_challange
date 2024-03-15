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
                Debug: bool=False, means=None, stds=None):
        super(HeadCTScan, self).__init__()
        
        self.root_dir = root_dir
        self.data_files = data_files
        if Debug:
            self.data_files = self.data_files[:20]
        
        # self.labels = pd.read_csv('/media/SSD2/IDOR/spr-head-ct-age-prediction-challenge/train.csv')
        self.labels = pd.read_csv('/mnt/dados/train.csv')
        self.labels['StudyID'] = self.labels['StudyID'].apply(lambda x: x.lstrip('0'))
        self.labels = self.labels.set_index('StudyID').to_dict()['Age']
        # print(self.labels)
        # # print(self.labels.head(10))
        # 1/0
        self.normalize = normalize
        self.transform = transform
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = self.data_files[index]
        
        data = nib.load(self.root_dir / file_name).get_fdata()
        data = data.transpose(2, 0, 1)
        data = data[:, :512, :512]
        # print(data.shape)
        # print(file_name.split('/')[-1].split('.')[0].lstrip('0'))
        # print(self.labels)
        # 1/0
        labels = self.labels[file_name.split('/')[-1].split('.')[0].lstrip('0')]
        
        if self.normalize:
            data = data - data.mean() / data.std()
            
        # data = np.expand_dims(data, axis=1)
            
        if self.transform:
            data = self.transform(data)
        else:
            data = torch.tensor(data, dtype=torch.float32)
            labels = torch.tensor(labels)
        # print(data.shape, labels.shape)
        # print(data.dtype, labels.dtype)
        return data, labels.unsqueeze(0)