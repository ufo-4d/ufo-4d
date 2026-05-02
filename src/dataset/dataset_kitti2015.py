"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import pickle
import mediapy as media
import skimage.io as io
import cv2

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_geometric_augmentation

from .types import Stage
from ..misc.geometry import get_3d_motion_vector, pixel2point
from ..misc.cam_utils import normalize_intrinsics, camera_to_pose_encoding


width_to_date = {
    1242: '2011_09_26',
    1224: '2011_09_28',
    1238: '2011_09_29',
    1226: '2011_09_30',
    1241: '2011_10_03',
}


def read_png_disp(disp_file):
    disp_np = io.imread(disp_file).astype(np.uint16) / 256.0
    disp_np = np.expand_dims(disp_np, axis=2)
    mask_disp = (disp_np > 0).astype(np.float64)
    return disp_np, mask_disp


def read_png_flow(flow_file):
    flow = cv2.imread(flow_file, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)[:,:,::-1].astype(np.float64)
    flow, valid = flow[:, :, :2], flow[:, :, 2:]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def disp2depth_kitti(disp, k, use_numpy=True):
    depth = k * 0.54 / (disp + 1e-8)
    if use_numpy:
        return np.clip(depth, 1e-3, 80.)
    return torch.clamp(depth, 1e-3, 80.)


def get_date_from_width(width):
    return width_to_date[width]


def read_image_as_byte(filename):
    return io.imread(filename)


def read_raw_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def read_calib_into_dict(path_dir):

    calibration_file_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
    intrinsic_dict_l = {}
    intrinsic_dict_r = {}

    for ii, date in enumerate(calibration_file_list):
        file_name = "kitti_helpers/calib_cam_to_cam_" + date + '.txt'
        file_name_full = os.path.join(path_dir, file_name)
        file_data = read_raw_calib_file(file_name_full)
        P_rect_02 = np.reshape(file_data['P_rect_02'], (3, 4))
        P_rect_03 = np.reshape(file_data['P_rect_03'], (3, 4))
        intrinsic_dict_l[date] = P_rect_02[:3, :3]
        intrinsic_dict_r[date] = P_rect_03[:3, :3]

    return intrinsic_dict_l, intrinsic_dict_r


def convert_depthflow_to_sceneflow(depth0, depth0_mask, depth1, depth1_mask, flow, flow_mask, intrinsics, use_numpy=True):

    inputs = [depth0, depth0_mask, depth1, depth1_mask, flow, flow_mask]
    for item in inputs:
        assert len(item.shape) == 4

    if use_numpy:
        point_traj_3d = np.concatenate([flow, depth1], axis=1)
    else:
        point_traj_3d = torch.cat([flow, depth1], dim=1)
    sceneflow = get_3d_motion_vector(depth0, intrinsics, point_traj_3d, use_numpy=use_numpy)
    return sceneflow, depth0_mask * flow_mask * depth1_mask


@dataclass
class DatasetKITTI2015Cfg(DatasetCfgCommon):

    name: str
    test_root: str
    test_aug_type: str


@dataclass
class DatasetKITTI2015CfgWrapper:
    kitti2015: DatasetKITTI2015Cfg


@dataclass
class DatasetKITTI2015SceneKey:
    root: str
    key: str
    num_images: int | None


class DatasetKITTI2015(IterableDataset):
    cfg: DatasetKITTI2015Cfg
    stage: Stage

    to_tensor: tf.ToTensor
    near: float = 1.
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetKITTI2015Cfg,
        stage: Stage,
        global_rank: int,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.depth_valid_min = 0.01
        self.depth_valid_max = 80.
        self.global_rank = global_rank

        images_root = os.path.join(cfg.test_root, "training", "image_2")
        flow_root = os.path.join(cfg.test_root, "training", "flow_occ")
        disp0_root = os.path.join(cfg.test_root, "training", "disp_occ_0")
        disp1_root = os.path.join(cfg.test_root, "training", "disp_occ_1")

        self.file_list = []

        for ii in range(200):
            file_idx = f"{ii:06d}"
            im_l1 = os.path.join(images_root, file_idx + "_10.png")
            im_l2 = os.path.join(images_root, file_idx + "_11.png")
            flow_occ = os.path.join(flow_root, file_idx + "_10.png")
            disparity0_occ = os.path.join(disp0_root, file_idx + "_10.png")
            disparity1_occ = os.path.join(disp1_root, file_idx + "_10.png")
            item_list = [im_l1, im_l2, flow_occ, disparity0_occ, disparity1_occ]

            for _, item in enumerate(item_list):
                if not os.path.isfile(item):
                    raise ValueError("File not exist: %s", item)

            self.file_list.append(item_list)

        ## loading calibration matrix
        path_dir = os.path.dirname(os.path.realpath(__file__))
        intrinsic_dict_l, _ = read_calib_into_dict(path_dir)

        self.intrinsic_dict_l = intrinsic_dict_l

        if self.stage == "val":
            self.file_list = self.file_list[::5]

        print(f" List of KITTI scenes for {stage}: {len(self.file_list)}")

        # Augmentation parameters.
        assert self.cfg.test_aug_type in ['resize', 'resize_test']


    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]


    def load_test_data(self, image_list):

        assert self.stage in ["val", "test"]

        image_fn = image_list[:2]
        flow_fn = image_list[2]
        disp_fn = image_list[3:]

        # Load data
        image0, image1 = [read_image_as_byte(img) for img in image_fn]
        flow, flow_mask = read_png_flow(flow_fn)
        disp0, disp0_mask = read_png_disp(disp_fn[0])
        disp1, disp1_mask = read_png_disp(disp_fn[1])

        # example filename
        basename = os.path.basename(image_fn[0])[:6]
        intrinsics = self.intrinsic_dict_l[get_date_from_width(image0.shape[1])]

        # input size
        h_orig, w_orig, _ = image0.shape
        input_im_size = np.array([h_orig, w_orig])

        # To torch
        def convert_to_torch(np_input):
            return self.to_tensor(np_input).float()[None, ...]

        image0 = convert_to_torch(image0)
        image1 = convert_to_torch(image1)
        flow = convert_to_torch(flow)
        flow_mask = convert_to_torch(flow_mask)
        disp0 = convert_to_torch(disp0)
        disp0_mask = convert_to_torch(disp0_mask)
        disp1 = convert_to_torch(disp1)
        disp1_mask = convert_to_torch(disp1_mask)
        intrinsics = convert_to_torch(intrinsics)[0]

        # Convert data format
        depth0 = disp2depth_kitti(disp0, intrinsics[0,0,0], use_numpy=False)
        depth1 = disp2depth_kitti(disp1, intrinsics[0,0,0], use_numpy=False)
        depth0_mask, depth1_mask = disp0_mask, disp1_mask

        gt_motion, valid_mask_motion = convert_depthflow_to_sceneflow(depth0, depth0_mask, depth1, depth1_mask, flow, flow_mask, intrinsics, use_numpy=False)
        gt_depth = depth0
        gt_pointmap = pixel2point(intrinsics, depth0, use_numpy=False)
        valid_mask_pointmap = depth0_mask

        # Data preprocess.
        context_images = torch.cat([image0, image1], dim=0)
        target_images = torch.cat([image0, image1], dim=0)
        imh, imw = image0.shape[-2:]

        # Camera parameters.
        assert len(intrinsics.shape) == 3
        intrinsics = intrinsics.repeat(2, 1, 1)
        # placeholders.
        pose1 = torch.eye(4).to(torch.float32)
        extrinsics = torch.stack((pose1, pose1), dim=0)  # (2, 4, 4)

        # normalize K
        intrinsics = normalize_intrinsics(intrinsics, imh, imw)

        # Convert extrinsics to quaternion
        extrinsics_quat = camera_to_pose_encoding(extrinsics, pose_encoding_type="absT_quaR")

        def repeat_2nd_view(tensor):
            return tensor.repeat(2, 1, 1, 1)

        gt_depth = repeat_2nd_view(gt_depth)
        gt_motion = repeat_2nd_view(gt_motion)
        gt_pointmap = repeat_2nd_view(gt_pointmap)
        valid_mask_pointmap = repeat_2nd_view(valid_mask_pointmap)
        valid_mask_motion = repeat_2nd_view(valid_mask_motion)

        example = {
            "context": {
                "extrinsics": extrinsics.float(),
                "extrinsics_quat": extrinsics_quat.float(),
                "intrinsics": intrinsics.float(),
                "image": context_images.float(),
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "target": {
                "extrinsics": extrinsics.float(),
                "extrinsics_quat": extrinsics_quat.float(),
                "intrinsics": intrinsics.float(),
                "image": target_images.float(),
                "depth": gt_depth.float(),
                "motion3d": gt_motion.float(),
                "pointmap": gt_pointmap.float(),
                "valid_mask_depth": valid_mask_pointmap,
                "valid_mask_motion": valid_mask_motion,
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "scene": basename,
            "dataset_alias": "kt",
        }

        return example

    def __iter__(self):

        assert self.stage in ["val", "test"]

        for file_list in self.file_list:
            example = self.load_test_data(file_list)
            # If an empty dictionary was return, skip this example.
            if example == {}:
                continue
            example = apply_geometric_augmentation(
                example,
                target_shape=tuple(self.cfg.test_image_shape),
                augmentation_type=self.cfg.test_aug_type)

            yield example

    def convert_images(
        self,
        images,
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
