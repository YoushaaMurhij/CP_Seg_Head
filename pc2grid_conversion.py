import numpy as np

def load_cloud_from_deecamp_file(pc_f, lb_f):
        logging.info('loading cloud from: {} and labels from : {}'.format(pc_f, lb_f))
        num_features = 4
        cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
        # last dimension should be the timestamp.
        label = np.fromfile(lb_f, dtype=np.uint32)
        label = label.reshape((-1))
        cloud = np.hstack((cloud, np.zeros([cloud.shape[0], 1])))
        return cloud, label

def main():

    grid_size = 256
    pc_length = 51.2 * 2
    pc_width = 51.2 * 2
    num_classes = 19 
    
    grid = np.empty([grid_size, grid_size, num_classes])

    if len(sys.argv) < 3:
        logging.error('Enter a lidar bin folder[1] path and label folder[2] path!')
    else:
        bin_path=sys.argv[1]
        lbl_path=sys.argv[2]
        bin_files = [f for f in listdir(bin_path) if isfile(join(bin_path, f))]
        lbl_files = [f for f in listdir(lbl_path) if isfile(join(lbl_path, f))]
        bin_files.sort()
        lbl_files.sort()
    for bin, lbl in enumerate(bin_files, lbl_files):
        print(lbl)
        cloud, label = load_cloud_from_deecamp_file(mypath+"/"+bin, mypath+"/"+lbl)



if __name__="__main__":
    main()
