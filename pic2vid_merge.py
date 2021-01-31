import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

two_frames = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width_1 = 640
frame_width_2 = 640
frame_height = 480
fps = 10
idx_max = 2000 
target_path = 'seg_test.avi'

path_1 = "./1"
list_1 = sorted([f for f in listdir(path_1) if isfile(join(path_1, f))])

if two_frames:
    path_2 = "./2"
    list_2 = sorted([f for f in listdir(path_2) if isfile(join(path_2, f))])
    out = cv2.VideoWriter(target_path, fourcc, fps, (frame_width_1 + frame_width_2, frame_height), True)
else:
    out = cv2.VideoWriter(target_path, fourcc, fps, (frame_width_1, frame_height), True)    

for idx, img_1 in enumerate(list_1):
    if idx < idx_max:
        print(img_1)
        frame_1 = cv2.imread(path_1 + '/' + img_1)
        frame_1 = cv2.resize(frame_1, (frame_width_1, frame_height))
        if two_frames:
            print(list_2[idx])
            frame_2 = cv2.imread(path_2 + '/' + list_2[idx])
            frame_2 = cv2.resize(frame_2, (frame_width_2, frame_height))
            frame_3 = np.concatenate((frame_1, frame_2), axis=1)
            # write the flipped frame
            out.write(frame_3)
        else:
            out.write(frame_1)
    else:
        break
print('Done!.')
#cv2.destroyAllWindows()