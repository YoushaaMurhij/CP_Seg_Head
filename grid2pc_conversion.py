import sys
import logging
import numpy as np
from os import listdir
from os.path import isfile, join

from numpy.core.fromnumeric import shape
from label_mapping import *

#pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),

def load_cloud_from_bin_file(pc_f, lb_f):
        logging.info('loading cloud from: {} and grids from : {}'.format(pc_f, lb_f))
        num_features = 4
        cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
        grid = np.loadtxt(lb_f, dtype=int, delimiter=',')
        return cloud, grid

def main():
    grid_size = 256
    pos_offset = 51.2 
    pc_width = 51.2 * 2

    if len(sys.argv) < 3:
        logging.error('Enter a lidar bin folder[1] path and grid folder[2] path!')
    else:
        bin_path=sys.argv[1]
        grd_path=sys.argv[2]
        bin_files = [f for f in listdir(bin_path) if isfile(join(bin_path, f))]
        grd_files = [f for f in listdir(grd_path) if isfile(join(grd_path, f))]
        assert(len(bin_files) == len(grd_files)),"Number of Points and grid files should be the same!"
        bin_files.sort()
        grd_files.sort()
    for i, (bin, grd) in enumerate(zip(bin_files, grd_files)):

        cloud, grid = load_cloud_from_bin_file(bin_path + "/" + bin, grd_path + "/" + grd)
        label = []
        for j, pt in enumerate(cloud):
            if (pt[0] >= pos_offset) or (pt[1] >= pos_offset) or (pt[0] < -1 * pos_offset) or (pt[1] < -1 * pos_offset) :
                label.append(0)
                continue
            elif (int((pt[0] + pos_offset) * grid_size / pc_width) > 255) or (int((pt[1] + pos_offset) * grid_size / pc_width) > 255):
                label.append(0)
                continue
            label.append(id2class[grid[int((pt[0] + pos_offset) * grid_size / pc_width), int((pt[1] + pos_offset) * grid_size / pc_width)]])
        assert(len(cloud) == len(label)),"Points and labels lists should be the same lenght!"
        np.savetxt('./data/gen_labels/'+'{:0>8}'.format(int(bin[:6]))+'.label', label,  fmt='%d')
        print(f'{i}: {str(int(bin[:6]))} - label was saved!')
        

if __name__=="__main__":
    main()
