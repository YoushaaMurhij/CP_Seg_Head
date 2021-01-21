import os
import sys
import logging
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

def visual2d(grid, index, writer=None, save_dir=None):
    '''Visualizing a sample for the every batch as a picture'''
    plt.figure() #TODO add colors
    plt.imshow(grid, interpolation='bilinear')
    plt.savefig('logs/' + save_dir + '/figs/' + str(index) + '.png')
    plt.clf()
    plt.close()

def main():
    if len(sys.argv) < 2:
        logging.error('Enter labels folder path!')
    else:
        lbl_path=sys.argv[1]

    lbl_files = [f for f in listdir(lbl_path) if isfile(join(lbl_path, f))]
    lbl_files.sort()
    for index, label in enumerate(lbl_files):
        label_name = os.path.join(lbl_path,label)
        grid = np.loadtxt(label_name, dtype=int, delimiter=',')
        visual2d(grid, index)
        print('Image '+str(index)+'.png saved!')

if __name__ == "__main__":
    main()