import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

is_two = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
img_width_1 = 640
img_width_2 = 640
img_height = 480

fps = 10
idx_max = 2000 #First idx_max images will be recorded

target_path = 'seg_test.avi'

path_1 = "./1"
if is_two:
     path_2 = "./2"

if is_two:
    out = cv2.VideoWriter(target_path, fourcc, fps, (img_width_1 + img_width_2, img_height), True)
else:
    out = cv2.VideoWriter(target_path, fourcc, fps, (img_width_1, img_height), True)

list_1 = sorted([f for f in listdir(path_1) if isfile(join(path_1, f))])
if is_two:
    list_2 = sorted([f for f in listdir(path_2) if isfile(join(path_2, f))])

for idx, img in enumerate(list_1):
    if idx < idx_max:
        print(img)
        frame_1 = cv2.imread(path_1 +'/' + img)
        frame_1 = cv2.resize(frame_1, (img_width_1, img_height))
        if is_two:
            print(list_2[idx])
            frame_2 = cv2.imread(path_2 + '/' +list_2[idx])
            frame_2 = cv2.resize(frame_2, (img_width_2, img_height))
            frame_3 = np.concatenate((frame_1, frame_2), axis=1)
            # write the flipped frame
            out.write(frame_3)
        else:
            out.write(frame_1)
    else:
        break
print('End.')
#cv2.destroyAllWindows()