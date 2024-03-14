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
    def __init__(self, data_files: Path, normalize: bool=True, transform: torchvision.transforms=None,
                Debug: bool=False, means=None, stds=None):
        super(HeadCTScan, self).__init__()
        
        self.data_files = data_files
        if Debug:
            self.data_files = self.data_files[:20]
        
        self.labels = pd.read_csv('/media/SSD2/IDOR/spr-head-ct-age-prediction-challenge/train.csv').to_dict()
        
        self.normalize = normalize
        self.transform = transform
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_name = self.data_files[index]
        
        data = nib.load(file_name).get_fdata()
        print(data.shape)
        labels = self.labels[file_name.split('/')[-1]]
        
        if self.normalize:
            data = data - data.mean / data.std
            
        data = np.expand_dims(data, axis=1)
            
        if self.transform:
            # For albumentations the image needs to be in shape (H, W, C)
            transformed = self.transform(image=data)
        else:
            data = torch.tensor(data)
            labels = torch.tensor(labels)
         
        return data.float(), labels.unsqueeze(1).float()