"""
-----------------------------------------------------------------------------------
# Author: Youshaa Murhij
# DoC: 2021.01.5
# email: yosha.morheg@gmail.com
-----------------------------------------------------------------------------------
# Description: Cutsom Dataset loading script for Semantic Head
"""

import os
import torch
from torch._C import device
from torch.utils.data import Dataset
import numpy as np
from os import listdir
from os.path import isfile, join

class FeaturesDataset(Dataset):
    """segmentation features dataset."""

    def __init__(self, feat_dir, label_dir, device):
        
        self.device = device
        self.feat_dir = feat_dir
        self.label_dir = label_dir
        self.features_files = [f for f in listdir(self.feat_dir) if isfile(join(self.feat_dir, f))]

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feat_name = os.path.join(self.feat_dir,'{:07}'.format(idx)+'.bin.pt')
        feature = torch.load(feat_name, map_location=self.device)

        label_name = os.path.join(self.label_dir,'{:07}'.format(idx)+'.txt')
        label = np.loadtxt(label_name, dtype=int, delimiter=',')
        #label = label.astype('int')

        sample = {'feature': feature, 'label': label, 'index': idx}

        return sample