import os
import torch
from torch.utils.data import Dataset
import numpy as np
from os import listdir
from os.path import isfile, join

class FeaturesDataset(Dataset):
    """segmentation features dataset."""

    def __init__(self, feat_dir, label_dir):
        
        self.feat_dir = feat_dir
        self.label_dir = label_dir

    def __len__(self):
        self.files = [f for f in listdir(self.feat_dir) if isfile(join(self.feat_dir, f))]
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feat_name = os.path.join(self.feat_dir,'{:06}'.format(idx)+'.bin.pt')
        feature = torch.load(feat_name)

        label_name = os.path.join(self.label_dir,'{:06}'.format(idx)+'.txt')
        label = np.loadtxt(label_name, dtype=int, delimiter=',')
        #label = label.astype('int')

        sample = {'feature': feature, 'label': label}

        return sample