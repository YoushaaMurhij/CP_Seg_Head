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
    if epoch != 'None':
        plt.savefig(save_dir + '/Epoch_' + epoch + '_{:0>7}'.format(index) + '.png')
    else:
        plt.savefig(save_dir + '/{:0>7}'.format(index) + '.png')
    plt.clf()
    plt.close()

def visual(grid, label, save_dir):
    '''Visualizing a sample for the every batch as a picture'''
    plt.figure() #TODO add colors
    plt.imshow(grid, interpolation='bilinear')
    plt.savefig(save_dir + label[:7] + '.png')
    plt.clf()
    plt.close()


def main():
    # if len(sys.argv) < 2:
    #     logging.error('Enter targets folder path!')
    # else:
    #     lbl_path=sys.argv[1]
    targets_path = '/home/josh94mur/data/targets'
    save_dir = '/home/josh94mur/data/'
    targets_files = [f for f in listdir(targets_path) if isfile(join(targets_path, f))]
    targets_files.sort()
    for target in targets_files:
        target_name = os.path.join(targets_path,target)
        grid = np.loadtxt(target_name, dtype=int, delimiter=',')
        visual(grid, target, save_dir)

if __name__ == "__main__":
    main()