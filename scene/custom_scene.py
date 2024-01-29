#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from loguru import logger

from ..utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from .cameras import Camera
from .dataset_readers import readColmapSceneInfo


class Args:
    data_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resolution = -1


class Scene:
    def __init__(self, source_path: Path, output_path: Path, shuffle=True, resolution_scales=[1.0]):
        """B :param path: Path to colmap scene main folder."""
        # XXX #
        args = Args()
        # XXX #

        self.source_path = source_path
        self.output_path = output_path
        self.loaded_iter = None

        self.train_cameras: Dict[float, List[Camera]] = {}
        self.test_cameras: Dict[float, List[Camera]] = {}

        assert (source_path / 'sparse').exists(), 'Could not find sparse folder in source path!'

        logger.info('Loading Scene Info')
        scene_info = readColmapSceneInfo(path=source_path, images='images', eval=False)
        shutil.copyfile(scene_info.ply_path, output_path / 'input.ply')

        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(output_path / 'cameras.json', 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization['radius']

        for resolution_scale in resolution_scales:
            logger.info('Loading Training Cameras')
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            logger.info('Loading Test Cameras')
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        self.point_cloud = scene_info.point_cloud

    # def save(self, iteration):
    #     point_cloud_path = os.path.join(self.model_path, 'point_cloud/iteration_{}'.format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, 'point_cloud.ply'))

    def getTrainCameras(self, scale=1.0) -> List[Camera]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> List[Camera]:
        return self.test_cameras[scale]
