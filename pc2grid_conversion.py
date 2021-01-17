import sys
import logging
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from label_mapping import *

#pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),

def load_cloud_from_bin_file(pc_f, lb_f):
        logging.info('loading cloud from: {} and labels from : {}'.format(pc_f, lb_f))
        num_features = 4
        cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
        label = np.fromfile(lb_f, dtype=np.uint32)
        label = label.reshape((-1))
        cloud = np.hstack((cloud, np.zeros([cloud.shape[0], 1])))
        return cloud, label

def main():
    grid_size = 256
    pos_offset = 51.2 
    pc_width = 51.2 * 2
    num_classes = 33 
    save_png = False

    if len(sys.argv) < 3:
        logging.error('Enter a lidar bin folder[1] path and label folder[2] path!')
    else:
        bin_path=sys.argv[1]
        lbl_path=sys.argv[2]
        bin_files = [f for f in listdir(bin_path) if isfile(join(bin_path, f))]
        lbl_files = [f for f in listdir(lbl_path) if isfile(join(lbl_path, f))]
        assert(len(bin_files) == len(lbl_files)),"Number of Points and labels files should be the same!"
        bin_files.sort()
        lbl_files.sort()
    for i, (bin, lbl) in enumerate(zip(bin_files, lbl_files)):
        seg_grid = np.zeros([grid_size, grid_size, num_classes])
        seg_grid.astype(int)
        cloud, label = load_cloud_from_bin_file(bin_path+"/"+bin, lbl_path+"/"+lbl)
        # print(f'cloud shape os :{np.shape(cloud)}')
        assert(len(cloud) == len(label)),"Points and labels lists should be the same lenght!"
        for j, (pt, lb) in enumerate(zip(cloud, label)):
            if (lb > 259) or (pt[0] > pos_offset) or (pt[1] > pos_offset) or (pt[0] < -1 * pos_offset) or (pt[1] < -1 * pos_offset) :
                continue
            seg_grid[int((pt[0] + pos_offset) * grid_size / pc_width - 1), int((pt[1] + pos_offset) * grid_size / pc_width - 1), class2id[lb]] += 1  
        fin_grid = np.argmax(seg_grid, axis=2)
        np.savetxt('00'+str(int(lbl[:6])+5632)+'.txt', fin_grid,  fmt='%d' , delimiter=',')

        if save_png:
            plt.figure()
            plt.imshow(fin_grid, interpolation='nearest')
            plt.savefig(lbl[:6]+'.png')
            plt.clf()
        

if __name__=="__main__":
    main()
