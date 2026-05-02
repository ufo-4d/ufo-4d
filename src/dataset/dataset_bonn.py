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
import mediapy

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
from ..misc.geometry import pixel2point, transform_points
from ..misc.cam_utils import normalize_intrinsics, unnormalize_intrinsics, camera_to_pose_encoding

import glob


def get_pose(pose_string):
    initial_pose = np.matrix(pose_string.split(' ', 1)[1])
    # Transformation matrices
    T_ros = np.matrix('-1 0 0 0;\
                        0 0 1 0;\
                        0 1 0 0;\
                        0 0 0 1')

    T_m = np.matrix('1.0157    0.1828   -0.2389    0.0113;\
                    0.0009   -0.8431   -0.6413   -0.0098;\
                    -0.3009    0.6147   -0.8085    0.0111;\
                            0         0         0    1.0000')

    # Extract translation vector
    t = initial_pose[0, 0:3].transpose()

    # Extract quaternion values
    qx, qy, qz, qw = initial_pose[0, 3], initial_pose[0, 4], initial_pose[0, 5], initial_pose[0, 6]

    # Compute the rotation matrix from the quaternion
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
    ])

    # Form T_0 matrix
    T_0 = np.block([[R, t], [np.zeros((1, 3)), np.ones((1, 1))]])

    # Compute T_g
    T_g = T_ros @ T_0 @ T_ros @ T_m
    return np.array(T_g)


@dataclass
class DatasetBonnCfg(DatasetCfgCommon):
    name: str
    test_root: str
    test_aug_type: str


@dataclass
class DatasetBonnCfgWrapper:
    bonn: DatasetBonnCfg


class DatasetBonn(IterableDataset):
    cfg: DatasetBonnCfg
    stage: Stage

    to_tensor: tf.ToTensor
    near: float = 0.01
    far: float = 200.0

    def __init__(
        self,
        cfg: DatasetBonnCfg,
        stage: Stage,
        global_rank: int,
    ) -> None:
        super().__init__()

        # del view_sample
        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.depth_valid_min = 0.01
        self.depth_valid_max = 150.
        self.global_rank = global_rank

        # filter out folders
        raw_dir = cfg.test_root
        seq_list = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]

        # Collect paths of data
        img_patterns = [f"{raw_dir}/rgbd_bonn_dataset/rgbd_bonn_{seq}/rgb_110/*.png" for seq in seq_list]
        img_lists = []
        for pattern in img_patterns:
            img_lists.append(sorted(glob.glob(pattern)))

        depth_patterns = [f"{raw_dir}/rgbd_bonn_dataset/rgbd_bonn_{seq}/depth_110/*.png" for seq in seq_list]
        depth_lists = []
        for pattern in depth_patterns:
            depth_lists.append(sorted(glob.glob(pattern)))

        camera_pose_files = [f"{raw_dir}/rgbd_bonn_dataset/rgbd_bonn_{seq}/groundtruth_110.txt" for seq in seq_list]
        camera_pose_lists = []
        for fn in camera_pose_files:
            with open(fn, 'r') as f:
                camera_poses_str = f.readlines()
                camera_poses = []
                for camera_poses_str_i in camera_poses_str:
                    camera_poses.append(get_pose(camera_poses_str_i))
            camera_pose_lists.append(camera_poses)

        ## Sampling
        n_stride = 1
        img_lists = [img_lists[n][::n_stride] for n in range(len(img_lists))]
        depth_lists = [depth_lists[n][::n_stride] for n in range(len(img_lists))]
        camera_pose_lists = [camera_pose_lists[n][::n_stride] for n in range(len(img_lists))]

        ## Get sample.
        intrinsics = [
            [542.822841, 0., 315.593520],
            [0., 542.576870, 237.756098],
            [0., 0., 1.]
        ]
        data_dict = {
            "img": img_lists,
            "depth": depth_lists,
            "extrinsics": camera_pose_lists,
            "intrinsics": intrinsics,
        }

        self.data_dict = data_dict
        self.num_scenes = len(img_lists)
        self.num_images = [len(l) for l in img_lists]
        print(f" List of Bonn scenes for {stage}: {sum(self.num_images)}")

        # Augmentation parameters.
        aug_list = ['random_scale_crop_resize_aspect_ratio', 'resize', 'no_augmentation', 'resize_test']
        assert self.cfg.test_aug_type in aug_list


    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]


    def load_data(self, scene_idx: int, frame_idx:int) -> dict:

        image_src = self.data_dict["img"][scene_idx][frame_idx]
        image_tgt = self.data_dict["img"][scene_idx][frame_idx+1]
        depth_src = self.data_dict["depth"][scene_idx][frame_idx]
        depth_tgt = self.data_dict["depth"][scene_idx][frame_idx+1]
        intrinsics_src = self.data_dict["intrinsics"]
        intrinsics_tgt = self.data_dict["intrinsics"]
        extrinsics_src = self.data_dict["extrinsics"][scene_idx][frame_idx]
        extrinsics_tgt = self.data_dict["extrinsics"][scene_idx][frame_idx+1]

        # Load data
        image_src = mediapy.read_image(image_src)
        image_tgt = mediapy.read_image(image_tgt)

        def load_depth_bonn(filename):
            # Loads depth from png and returns a numpy array.
            depth_png = mediapy.read_image(filename)
            # make sure to have 16bit depth map, not 8bit.
            assert np.max(depth_png) > 255
            depth = depth_png.astype(np.float64) / 5000.0
            depth[depth_png == 0] = -1.0
            return depth
        depth_src = load_depth_bonn(depth_src)
        depth_tgt = load_depth_bonn(depth_tgt)

        def get_depth_mask_bonn(depth):
            return (0.01 <= depth) * (depth <= 200.)
        depth_mask_src = get_depth_mask_bonn(depth_src)
        depth_mask_tgt = get_depth_mask_bonn(depth_tgt)

        # Merge inputs.
        images = self.convert_images([image_src, image_tgt]).float()
        depths = self.convert_images([depth_src, depth_tgt]).float()
        depth_masks = self.convert_images([depth_mask_src, depth_mask_tgt]).float()
        intrinsics = torch.tensor(np.stack([intrinsics_src, intrinsics_tgt])).float()

        # 3D pointmap from depth, intrinsics, and extrinsics.
        points = pixel2point(intrinsics, depths, use_numpy=False)

        extrinsics_src = torch.tensor(extrinsics_src).float()
        extrinsics_tgt = torch.tensor(extrinsics_tgt).float()
        tform_src = torch.eye(4)
        tform_tgt = torch.inverse(extrinsics_src) @ extrinsics_tgt
        extrinsics = torch.stack([tform_src, tform_tgt])

        def transform_cam_to_canonical(points, intrinsics, extrinsics):
            # transform
            tgt2ref = extrinsics[1]
            pt1_cam, pt2_cam = points[:1], points[1:]
            pt1_cano = pt1_cam
            pt2_cano = transform_points(tgt2ref, pt2_cam)
            return torch.cat([pt1_cano, pt2_cano], dim=0)

        # Transfrom points to the canonical coordinate.
        points = transform_cam_to_canonical(points, intrinsics, extrinsics)
        sceneflow = torch.zeros_like(points)  # Placeholder
        imh, imw = images.shape[2:]

        # Depth is also at the canonical coordinate.
        depths = points[:, -1:]
        depths = torch.clamp(depths, max=self.depth_valid_max)

        # Convert extrinsics to quaternion
        extrinsics_quat = camera_to_pose_encoding(extrinsics, pose_encoding_type="absT_quaR")

        # Normalize intrinsics
        intrinsics = normalize_intrinsics(intrinsics, imh, imw)

        example = {
            "context": {
                "extrinsics": extrinsics,
                "extrinsics_quat": extrinsics_quat,
                "intrinsics": intrinsics,
                "image": images,
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "target": {
                "extrinsics": extrinsics,
                "extrinsics_quat": extrinsics_quat,
                "intrinsics": intrinsics,
                "image": images,
                "depth": depths,
                "motion3d": sceneflow,
                "pointmap": points,
                "valid_mask_depth": depth_masks,
                "valid_mask_motion": depth_masks,
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "scene": f"{scene_idx}_{frame_idx}",
            "dataset_alias": "bo",
        }

        return example


    def __iter__(self):

        for scene_idx in range(self.num_scenes):
            for frame_idx in range(self.num_images[scene_idx]-1):
                example = self.load_data(scene_idx, frame_idx)
                # If an empty dictionary was return, skip this example.
                if example == {}:
                    continue
                yield apply_geometric_augmentation(
                    example,
                    target_shape=tuple(self.cfg.test_image_shape),
                    augmentation_type=self.cfg.test_aug_type)

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
