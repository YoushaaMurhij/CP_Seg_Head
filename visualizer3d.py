# import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d

# # construct a dataset by specifying dataset_path
# dataset = ml3d.datasets.SemanticKITTI(dataset_path='/home/cds-josh/data/')

# # get the 'all' split that combines training, validation and test set
# all_split = dataset.get_split('validation')

# # print the attributes of the first datum
# print(all_split.get_attr(0))

# # print the shape of the first point cloud
# print(all_split.get_data(0)['point'].shape)

# # show the first 100 frames using the visualizer
# vis = ml3d.vis.Visualizer()
# vis.visualize_dataset(dataset, 'validation', indices=range(5))

#!/usr/bin/env python
import open3d.ml.torch as ml3d
from open3d.ml.vis import Visualizer, LabelLUT
from open3d.ml.utils import get_module

import math
import numpy as np
import os
import random
import sys
from os.path import exists, join, isfile, dirname, abspath, split

# ------ for custom data -------
kitti_labels = {
    0: 'unlabeled',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign'
}

def get_custom_data(pc_names, path):

    pc_data = []
    for i, name in enumerate(pc_names):
        pc_path = join(path, 'points', name + '.bin')
        label_path = join(path, 'labels', name + '.label')
        point = np.load(pc_path)[:, 0:3]
        label = np.squeeze(np.load(label_path))

        data = {
            'point': point,
            'feat': None,
            'label': label,
        }
        pc_data.append(data)

    return pc_data

def main():

    path = "/home/cds-josh/data/"
    v = Visualizer()
    lut = LabelLUT()
    for val in sorted(kitti_labels.keys()):
        lut.add_label(kitti_labels[val], val)
    v.set_lut("labels", lut)
    path = os.path.dirname(os.path.realpath(__file__)) + "/demo_data"

    pcs = get_custom_data(pc_names, path)
    v.visualize_dataset(pcs, "PCs")

if __name__ == "__main__":
    main()