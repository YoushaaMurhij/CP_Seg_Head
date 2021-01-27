import os
import sys
import logging
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

def visual2d(grid, index, save_dir, epoch=None):
    '''Visualizing a sample for the every batch as a picture'''
    plt.figure() #TODO add colors
    plt.imshow(grid, interpolation='bilinear')
    np.savetxt( save_dir + '/{:0>7}'.format(index) + '.txt', grid,  fmt='%d', delimiter=',')
    if epoch is not None:
        plt.savefig(save_dir + '/Epoch_' + epoch + '_{:0>7}'.format(index) + '.png')
    else:
        plt.savefig(save_dir + '/_{:0>7}'.format(index) + '.png')
    plt.clf()
    plt.close()

def visual(grid, label, save_dir):
    '''Visualizing a sample for the every batch as a picture'''
    plt.figure() #TODO add colors
    plt.imshow(grid, interpolation='bilinear')
    plt.savefig(save_dir + label + '.png')
    plt.clf()
    plt.close()


def main():
    # if len(sys.argv) < 2:
    #     logging.error('Enter targets folder path!')
    # else:
    #     lbl_path=sys.argv[1]
    lbl_path = '/home/josh94mur/data/targets'
    save_dir = '/home/josh94mur/data/figures/'
    lbl_files = [f for f in listdir(lbl_path) if isfile(join(lbl_path, f))]
    lbl_files.sort()
    for label in lbl_files:
        label_name = os.path.join(lbl_path,label)
        grid = np.loadtxt(label_name, dtype=int, delimiter=',')
        visual(grid, label, save_dir)
        #print('Image '+label+'.png saved!')

if __name__ == "__main__":
    main()