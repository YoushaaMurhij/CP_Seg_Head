import sys
import time

import numpy as np
import open3d as o3d # need to be imported befor torch
import torch
#from loguru import logger as logging
import logging
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.models import build_detector
from det3d.torchie import Config
from tools.demo_utils import visual

from os import listdir
from os.path import isfile, join

class CenterPointDetector(object):
    def __init__(self, config_file, model_file, calib_data=None):
        self.config_file = config_file
        self.model_file = model_file
        self.calib_data = calib_data
        self.points = None
        self.inputs = None
        self._init_model()

    def _init_model(self):
        cfg = Config.fromfile(self.config_file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_file)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.voxel_generator = VoxelGenerator(
            voxel_size=cfg.voxel_generator.voxel_size,
            point_cloud_range=cfg.voxel_generator.range,
            max_num_points=cfg.voxel_generator.max_points_in_voxel,
            max_voxels=cfg.voxel_generator.max_voxel_num,
        )

    @staticmethod
    def load_cloud_from_nuscenes_file(pc_f):
        logging.info('loading cloud from: {}'.format(pc_f))
        num_features = 5
        cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
        # last dimension should be the timestamp.
        cloud[:, 4] = 0
        return cloud

    @staticmethod
    def load_cloud_from_deecamp_file(pc_f):
        logging.info('loading cloud from: {}'.format(pc_f))
        num_features = 4
        cloud = np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, num_features])
        # last dimension should be the timestamp.
        cloud = np.hstack((cloud, np.zeros([cloud.shape[0], 1])))
        return cloud

    def load_cloud_from_my_file(seld, pc_f):
        logging.info('loading cloud from: {}'.format(pc_f))
        ins = open(pc_f, "r" )
        num_features = 4
        data = []
        for line in ins:
            number_strings = line.split() # Split the line on runs of whitespace
            numbers = [float(n) for n in number_strings] # Convert to integers
            data.append(numbers) 
        cloud = np.array(data,dtype=np.float32).reshape([-1, num_features])
        cloud = np.hstack((cloud, np.zeros([cloud.shape[0], 1])))
        return cloud

    def predict_on_local_file(self, cloud_file, i, f):

        # load sample from file
        #self.points = self.load_cloud_from_nuscenes_file(cloud_file)
        self.points = self.load_cloud_from_deecamp_file(cloud_file)
        #self.points = self.load_cloud_from_my_file(cloud_file)

        # prepare input
        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)

        self.inputs = dict(
            voxels=voxels,
            num_points=num_points,
            num_voxels=num_voxels,
            coordinates=coords,
            shape=[grid_size]
        )

        # predict
        torch.cuda.synchronize()
        tic = time.time()
        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)  #[0]

        #print(outputs.shape)
        torch.save(outputs, f+'.pt')
        # features = outputs.detach().cpu().numpy()
        # np.savetxt(f, features, newline=" ")

        torch.cuda.synchronize()
        logging.info("Prediction time: {:.3f} s".format(time.time() - tic))

        # for k, v in outputs.items():
        #     if k not in [
        #         "metadata",
        #     ]:
        #         outputs[k] = v.to('cpu')

        # visualization
        #print(outputs)
        #visual(i, np.transpose(self.points), outputs, conf_th=0.5, show_plot=False, show_3D=False)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error('Input a lidar bin folder pls.')
    else:
        mypath=sys.argv[1]
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        onlyfiles.sort()
        config_file = 'configs/centerpoint/nusc_centerpoint_pp_02voxel_circle_nms_demo.py'
        model_file = 'work_dirs/centerpoint_pp_512_circle_nms_tracking/last1.pth'
        #config_file = 'configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_circle_nms.py'
        #model_file = 'work_dirs/centerpoint_voxel_1440_dcn_flip_circle_nms/latest.pth'
        detector = CenterPointDetector(config_file, model_file)
        for i, f in enumerate(onlyfiles):
            print(f)
            detector.predict_on_local_file(mypath+"/"+f, i, f)


#/datasets/KITTI_Odometry/dataset/sequences/00/velodyne