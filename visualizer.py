import os
import sys
import logging
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import tensorflow as tf

def visual2d(grid, index):
    '''Visualizing a sample for the every batch as a picture'''
    plt.figure() #TODO add colors
    plt.imshow(grid, interpolation='bilinear')
    plt.savefig('data/figs/'+str(index)+'.png')
    print('Image '+str(index)+'.png saved!')
    plt.clf()
    plt.close()
    img = np.reshape(grid, (-1, 256, 256, 3))
    #rm -rf logs
    logdir = "logs/eval_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        tf.summary.image("Eval data", img, step=0)
   
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

if __name__ == "__main__":
    main()