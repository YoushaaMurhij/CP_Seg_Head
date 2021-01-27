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
from label_mapping import label_name_mapping

import numpy as np
from os import listdir
from os.path import exists, join, isfile, dirname, abspath, split

def get_custom_data(bin_path, lbl_path):

    pc_data = []
    bin_files = [f for f in listdir(bin_path) if isfile(join(bin_path, f))]
    lbl_files = [f for f in listdir(lbl_path) if isfile(join(lbl_path, f))]
    bin_files.sort()
    lbl_files.sort()

    for i, (bin,lbl) in enumerate(zip(bin_files[:100], lbl_files[:100])):
        print(f'{bin} & {lbl} mapped!')
        num_features = 4
        cloud = np.fromfile(bin_path + "/" + bin, dtype=np.float32, count=-1).reshape([-1, num_features])
        label = np.loadtxt(lbl_path + "/" + lbl, dtype=np.int32)
        label = np.squeeze(label)
        data = {
            'name': bin,
            'points': cloud,
            'feat': None,
            'label': label,
        }
        pc_data.append(data)
    return pc_data

def main():
    bin_path = "/home/cds-josh/data/bins"
    label_path = "/home/cds-josh/data/gen_labels"
    v = Visualizer()
    lut = LabelLUT()
    for val in sorted(label_name_mapping.keys()):
        lut.add_label(label_name_mapping[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)
    point_clouds = get_custom_data(bin_path, label_path)
    v.visualize(point_clouds)

if __name__ == "__main__":
    main()


# improve memory usage here ()