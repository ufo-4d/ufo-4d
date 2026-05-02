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
import cv2

import torch
import torchvision.transforms as tf
from einops import repeat, rearrange
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
import pandas as pd

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_geometric_augmentation
from torchvision.transforms import ColorJitter

from .types import Stage
from ..misc.geometry import pixel2point, transform_points
from ..misc.cam_utils import normalize_intrinsics, camera_to_pose_encoding


def read_intrinsics_txt(file_path):
    intrinsics_list = []
    df = pd.read_csv(file_path, sep='\\s+', header=0)
    camera_0_df = df[df['cameraID'] == 0]

    for index, row in camera_0_df.iterrows():
        fx = row['K[0,0]']
        fy = row['K[1,1]']
        cx = row['K[0,2]']
        cy = row['K[1,2]']

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        intrinsics_list.append(K)
    return intrinsics_list


def read_extrinsics_txt(file_path):
    col_names = [
        'frame', 'cameraID',
        'r11', 'r12', 'r13', 't1',
        'r21', 'r22', 'r23', 't2',
        'r31', 'r32', 'r33', 't3',
        'd1', 'd2', 'd3', 'd4'
    ]

    df = pd.read_csv(file_path, sep='\\s+', header=0, names=col_names)
    extrinsics_list = []
    camera_0_df = df[df['cameraID'] == 0].copy()

    # Define the columns that make up the 3x4 matrix
    matrix_cols = col_names[2:]

    # Iterate and create the list of matrices
    for index, row in camera_0_df.iterrows():
        # Select the 12 matrix values, convert to a NumPy array, and reshape
        extrinsic_matrix = row[matrix_cols].to_numpy(dtype=float).reshape(4, 4)
        extrinsics_list.append(extrinsic_matrix)
 
    return extrinsics_list


@dataclass
class DatasetVKITTI2Cfg(DatasetCfgCommon):

    name: str
    train_root: str
    train_aug_type: str
    test_aug_type: str

    # Data augmentation.
    photometric_aug: bool
    temporal_flip: bool
    horizontal_flip: bool


@dataclass
class DatasetVKITTI2CfgWrapper:
    vkitti2: DatasetVKITTI2Cfg


class DatasetVKITTI2(IterableDataset):
    cfg: DatasetVKITTI2Cfg
    stage: Stage

    to_tensor: tf.ToTensor
    near: float = 0.01
    far: float = 200.0

    def __init__(
        self,
        cfg: DatasetVKITTI2Cfg,
        stage: Stage,
        global_rank: int,
    ) -> None:
        super().__init__()

        # del view_sample
        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.photo_aug = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1/3.14)
        self.depth_valid_min = 0.01
        self.depth_valid_max = 150.
        self.supervise_faraway_pixels = False
        self.global_rank = global_rank

        val_scene = ['Scene18']
        skip_cam = ['rain', 'fog']

        # filter out folders
        raw_dir = cfg.train_root
        list_scene = sorted(os.listdir(raw_dir))

        if stage == 'train':
            list_scene = [f for f in list_scene if f not in val_scene]
        elif stage in ['val', 'test']:
            list_scene = val_scene

        # Collect paths of data
        data_dict = {}

        for scene_idx in list_scene:
            path_scene = os.path.join(raw_dir, scene_idx)
            list_cams = [f for f in os.listdir(path_scene) if f not in skip_cam]
            for cam in list_cams:
                key = f'{scene_idx}/{cam}'

                # Load intrinsics
                intrinsic_name = os.path.join(raw_dir, key, "intrinsic.txt")
                list_intrinsics = read_intrinsics_txt(intrinsic_name)
                n_frames = len(list_intrinsics)

                # Load extrinsics
                extrinsic_name = os.path.join(raw_dir, key, "extrinsic.txt")
                list_extrinsics = read_extrinsics_txt(extrinsic_name)
                assert len(list_extrinsics) == n_frames

                # RGB paths
                rgb_path = os.path.join(raw_dir, key, "frames", "rgb", "Camera_0")
                list_rgbs = [os.path.join(rgb_path, f'rgb_{n:05d}.jpg') for n in range(n_frames)]

                # Depth paths
                depth_path = os.path.join(raw_dir, key, "frames", "depth", "Camera_0")
                list_depths = [os.path.join(depth_path, f'depth_{n:05d}.png') for n in range(n_frames)]

                # Scene flow paths
                sf_fw_path = os.path.join(raw_dir, key, "frames", "forwardSceneFlow", "Camera_0")
                list_sf_fw = [os.path.join(sf_fw_path, f'sceneFlow_{n:05d}.png') for n in range(n_frames-1)]
                sf_bw_path = os.path.join(raw_dir, key, "frames", "backwardSceneFlow", "Camera_0")
                list_sf_bw = [os.path.join(sf_bw_path, f'backwardSceneFlow_{n:05d}.png') for n in range(1, n_frames)]

                data_dict[key] = {
                    'n_frames': n_frames,
                    'intrinsics': list_intrinsics,
                    'extrinsics': list_extrinsics,
                    'rgb': list_rgbs,
                    'depth': list_depths,
                    'sceneflow_forward': list_sf_fw,
                    'sceneflow_backward': list_sf_bw,
                }

        self.data_dict = data_dict
        self.scene_keys = list(data_dict.keys())
        self.scene_keys = self.shuffle(self.scene_keys)
        print(f" List of VKITTI2 dataset for {stage}: {len(self.scene_keys)}")

        # Augmentation parameters.
        aug_list = ['random_scale_crop_resize_aspect_ratio', 'resize', 'no_augmentation', 'resize_test']
        assert self.cfg.train_aug_type in aug_list
        assert self.cfg.test_aug_type in aug_list

        # Coordinate definitions.
        self.coordinate_type = 'canonical_coordinate'


    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]


    def load_data(self, scene_key: str, source_index: int, target_index: int) -> dict:

        data_key_dict = self.data_dict[scene_key]

        # Given indices, read image, depth, mask, trajectory, camera matrix
        image_src = data_key_dict['rgb'][source_index]
        image_tgt = data_key_dict['rgb'][target_index]
        depth_src = data_key_dict['depth'][source_index]
        depth_tgt = data_key_dict['depth'][target_index]
        sceneflow_src = data_key_dict['sceneflow_forward'][source_index]
        sceneflow_tgt = data_key_dict['sceneflow_backward'][source_index]

        # Load data
        image_src = media.read_image(image_src)
        image_tgt = media.read_image(image_tgt)

        def load_depth_vkitti2(fn):
            return cv2.imread(fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.
        depth_src = load_depth_vkitti2(depth_src)
        depth_tgt = load_depth_vkitti2(depth_tgt)
        depth_src = np.clip(depth_src, self.depth_valid_min, self.depth_valid_max)
        depth_tgt = np.clip(depth_tgt, self.depth_valid_min, self.depth_valid_max)

        def get_depth_mask_vkitti2(depth):
            if self.supervise_faraway_pixels:
                return (self.depth_valid_min <= depth) * (depth <= self.depth_valid_max)
            else:
                return (self.depth_valid_min < depth) * (depth < self.depth_valid_max)

        depth_mask_src = get_depth_mask_vkitti2(depth_src)
        depth_mask_tgt = get_depth_mask_vkitti2(depth_tgt)

        def load_sceneflow_vkitti2(fn):
            sf_bgr = cv2.imread(fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            h, w, c = sf_bgr.shape
            assert sf_bgr.dtype == np.uint16 and c == 3
            sf_rgb = cv2.cvtColor(sf_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            return ((sf_rgb * 2.0 / 65535.0) - 1.0) * 10.0
        sceneflow_src = load_sceneflow_vkitti2(sceneflow_src)
        sceneflow_tgt = load_sceneflow_vkitti2(sceneflow_tgt)

        # Camera parameters
        intrinsics_src = data_key_dict['intrinsics'][source_index]
        intrinsics_tgt = data_key_dict['intrinsics'][target_index]
        extrinsics_src = data_key_dict['extrinsics'][source_index]
        extrinsics_tgt = data_key_dict['extrinsics'][target_index]

        # Merge inputs.
        images = self.convert_images([image_src, image_tgt]).float()
        depths = self.convert_images([depth_src, depth_tgt]).float()
        depth_masks = self.convert_images([depth_mask_src, depth_mask_tgt]).float()
        intrinsics = torch.tensor(np.stack([intrinsics_src, intrinsics_tgt])).float()
        sceneflow = self.convert_images([sceneflow_src, sceneflow_tgt]).float()

        # Photometric augmentation.
        if self.stage == "train" and self.cfg.photometric_aug and (random.random() > 0.5):
            images = self.photo_aug(images)
        imh, imw = images.shape[2:]

        # 3D pointmap from depth, intrinsics, and extrinsics.
        points = pixel2point(intrinsics, depths, use_numpy=False)
        extrinsics_src = torch.tensor(extrinsics_src).float()
        extrinsics_tgt = torch.tensor(extrinsics_tgt).float()
        tform_src = torch.eye(4)
        tform_tgt = extrinsics_src @ torch.inverse(extrinsics_tgt)
        extrinsics = torch.stack([tform_src, tform_tgt])

        def transform_cam_to_canonical(points, intrinsics, pose, sceneflow):

            # transform
            tgt2ref = pose[1]
            ref2tgt = torch.inverse(pose[1])

            pt1_cam, pt2_cam = points[:1], points[1:]
            pt1_cano = pt1_cam  # pose[0] is identity
            pt2_cano = transform_points(tgt2ref, pt2_cam)
            
            # forward
            cam_motion_pt1_cano = transform_points(ref2tgt, pt1_cam) - pt1_cam
            sceneflow_fw_cano = sceneflow[:1] - cam_motion_pt1_cano

            # backward
            pt2_cam_tform_cam = transform_points(tgt2ref, pt2_cam)
            pt2_sf_tform_cam = sceneflow[1:] + pt2_cam
            pt2_sf_tform_cano = transform_points(tgt2ref, pt2_sf_tform_cam)
            pt2_cam_tform_cano = transform_points(tgt2ref, pt2_cam_tform_cam)
            sceneflow_bw_cano = pt2_sf_tform_cano - pt2_cam_tform_cano

            # Transform 2nd view's point and motion to the 1st view.
            points = torch.cat([pt1_cano, pt2_cano], dim=0)
            sceneflow = torch.cat([sceneflow_fw_cano, sceneflow_bw_cano], dim=0)

            return points, sceneflow

        # Compensate motion by cam
        points, sceneflow = transform_cam_to_canonical(points, intrinsics, extrinsics, sceneflow)
        # Depth is also at the canonical coordinate.
        depths = points[:, -1:]
        depths = torch.clamp(depths, max=self.depth_valid_max)

        # Horizontal flip augmentation:
        if self.stage == "train" and self.cfg.horizontal_flip and random.random() > 0.5:
            images = torch.flip(images, dims=[-1])
            depths = torch.flip(depths, dims=[-1])
            points = torch.flip(points, dims=[-1])
            sceneflow = torch.flip(sceneflow, dims=[-1])
            depth_masks = torch.flip(depth_masks, dims=[-1])
            depth_masks = torch.flip(depth_masks, dims=[-1])
            # Adjust point and scene flow x coordinate.
            points[:, 0, :, :] *= -1
            sceneflow[:, 0, :, :] *= -1
            # Camera intrinsics.
            intrinsics[:, 0, 2] = (imw - 1) - intrinsics[:, 0, 2]  # cx = w - 1 - cx
            # Camera extrinsics.
            extrinsics[1, 0, 1:4] *= -1
            extrinsics[1, 1:3, 0] *= -1

        # Normalize intrinsics
        intrinsics = normalize_intrinsics(intrinsics, imh, imw)

        # Convert extrinsics to quaternion
        extrinsics_quat = camera_to_pose_encoding(extrinsics, pose_encoding_type="absT_quaR")

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
            "scene": scene_key,
            "dataset_alias": "vk",
        }

        return example


    def __iter__(self):

        def get_random_indices(n_frames):
            source_index = random.randint(0, n_frames-2)
            target_index = source_index + 1
            return source_index, target_index

        def get_list_test_indices(n_frames):
            list_indices = [i for i in range(n_frames)]
            list_pairs = []
            for source_index, target_index in zip(list_indices[:-1], list_indices[1:]):
                list_pairs.append([source_index, target_index])
            list_pairs = list_pairs[::30]
            return list_pairs

        for scene_key in self.scene_keys:

            data_key_dict = self.data_dict[scene_key]

            # Train stage.
            if self.stage in ["train"]:
                source_index, target_index = get_random_indices(data_key_dict['n_frames'])
                example = self.load_data(scene_key, source_index, target_index)
                # If an empty dictionary was return, skip this example.
                if example == {}:
                    continue
                yield example

            elif self.stage in ["val", "test"]:
                list_pairs = get_list_test_indices(data_key_dict['n_frames'])
                for source_index, target_index in list_pairs:                    
                    example = self.load_data(scene_key, source_index, target_index)
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
