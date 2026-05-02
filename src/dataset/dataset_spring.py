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
import torch.nn.functional as F

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_geometric_augmentation
from torchvision.transforms import ColorJitter

from .types import Stage
from ..misc.geometry import get_3d_motion_vector, pixel2point, transform_points
from ..misc.cam_utils import normalize_intrinsics, camera_to_pose_encoding

import h5py

BASELINE_WIDTH = 0.065


def read_intrinsics_txt(filepath):
    def get_intrinsics(values):
        fx, fy, cx, cy = values
        return np.array([
            [fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.],
        ])

    intrinsics_list = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            values = [float(num) for num in line.split()]
            intrinsics_list.append(get_intrinsics(values))
    return intrinsics_list


def read_extrinsics_txt(filepath):
    def get_extrinsics(values):
        values_4x4 = [values[4*i:4*(i+1)] for i in range(4)]
        return np.array([
            values_4x4[0],
            values_4x4[1],
            values_4x4[2],
            values_4x4[3],
        ])

    extrinsics_list = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            values = [float(num) for num in line.split()]
            extrinsics_list.append(get_extrinsics(values))
    return extrinsics_list


def readFlo5Flow(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
        return f["flow"][()]


def readDsp5Disp(filename):
    with h5py.File(filename, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
        return f["disparity"][()]


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


def infilling_invalid(tensor):

    assert len(tensor.shape) == 4, f"{tensor.shape} is not 4 dimensional."
    b, c, _, _ = tensor.shape

    # Prepare data.
    nan_mask = is_invalid(tensor)
    tensor_no_nan = torch.nan_to_num(tensor, nan=0.0)
    valid_mask = (~nan_mask).float()
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32).repeat(c, 1, 1, 1)

    # Perform convolutions.
    sum_conv = F.conv2d(tensor_no_nan, kernel, groups=c, padding='same')
    count_conv = F.conv2d(valid_mask, kernel, groups=c, padding='same')
    neighbor_avg = sum_conv / (count_conv + 1e-8)

    # Overwrite the input.
    tensor[nan_mask] = neighbor_avg[nan_mask]
    return tensor


def is_invalid(tensor):
    return ~torch.isfinite(tensor)


@dataclass
class DatasetSpringCfg(DatasetCfgCommon):

    name: str
    train_root: str
    train_aug_type: str
    test_aug_type: str

    # Data augmentation.
    temporal_flip: bool
    horizontal_flip: bool
    photometric_aug: bool


@dataclass
class DatasetSpringCfgWrapper:
    spring: DatasetSpringCfg


class DatasetSpring(IterableDataset):
    cfg: DatasetSpringCfg
    stage: Stage

    to_tensor: tf.ToTensor
    near: float = 0.01
    far: float = 80.0

    def __init__(
        self,
        cfg: DatasetSpringCfg,
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
        self.depth_valid_max = 80.
        self.supervise_faraway_pixels = False
        self.global_rank = global_rank

        # filter out folders
        raw_dir = cfg.train_root
        list_scene = sorted(os.listdir(raw_dir))

        val_scene = ['0005', '0009', '0013', '0017', '0023', '0037', '0044']

        if stage == 'train':
            list_scene = [f for f in list_scene if f not in val_scene]
        elif stage in ['val', 'test']:
            list_scene = val_scene
            
        # Collect paths of data
        data_dict = {}

        for scene_idx in list_scene:
    
            path_scene = os.path.join(raw_dir, scene_idx)

            # Load intrinsics
            intrinsic_name = os.path.join(path_scene, "cam_data", "intrinsics.txt")
            list_intrinsics = read_intrinsics_txt(intrinsic_name)
            n_frames = len(list_intrinsics)

            # Load extrinsics
            extrinsic_name = os.path.join(path_scene, "cam_data", "extrinsics.txt")
            list_extrinsics = read_extrinsics_txt(extrinsic_name)
            assert len(list_extrinsics) == n_frames

            # RGB paths
            rgb_path = os.path.join(path_scene, "frame_left")
            list_rgbs = [os.path.join(rgb_path, fn) for fn in sorted(os.listdir(rgb_path))]
            
            # Disp
            disp1_left_path = os.path.join(path_scene, "disp1_left_resize")
            list_disp1_left = [os.path.join(disp1_left_path, fn) for fn in sorted(os.listdir(disp1_left_path))]

            # Disp change
            disp2_FW_left_path = os.path.join(path_scene, "disp2_FW_left_resize")
            list_disp2_FW_left = [os.path.join(disp2_FW_left_path, fn) for fn in sorted(os.listdir(disp2_FW_left_path))]
            disp2_BW_left_path = os.path.join(path_scene, "disp2_BW_left_resize")
            list_disp2_BW_left = [os.path.join(disp2_BW_left_path, fn) for fn in sorted(os.listdir(disp2_BW_left_path))]

            # Flow
            flow_FW_left_path = os.path.join(path_scene, "flow_FW_left_resize")
            list_flow_FW_left = [os.path.join(flow_FW_left_path, fn) for fn in sorted(os.listdir(flow_FW_left_path))]
            flow_BW_left_path = os.path.join(path_scene, "flow_BW_left_resize")
            list_flow_BW_left = [os.path.join(flow_BW_left_path, fn) for fn in sorted(os.listdir(flow_BW_left_path))]

            assert len(list_rgbs) == len(list_disp1_left), f"{len(list_rgbs)} != {len(list_disp1_left)}."
            assert len(list_rgbs) == len(list_disp2_FW_left)+1, f"{len(list_rgbs)} != {len(list_disp2_FW_left)}."
            assert len(list_rgbs) == len(list_disp2_BW_left)+1, f"{len(list_rgbs)} != {len(list_disp2_BW_left)}."
            assert len(list_rgbs) == len(list_flow_FW_left)+1, f"{len(list_rgbs)} != {len(list_flow_FW_left)}."
            assert len(list_rgbs) == len(list_flow_BW_left)+1, f"{len(list_rgbs)} != {len(list_flow_BW_left)}."

            data_dict[scene_idx] = {
                'n_frames': len(list_rgbs),
                'intrinsics': list_intrinsics,
                'extrinsics': list_extrinsics,
                'rgb': list_rgbs,
                'disp': list_disp1_left,
                'disp_change_fw': list_disp2_FW_left,
                'disp_change_bw': list_disp2_BW_left,
                'flow_fw': list_flow_FW_left,
                'flow_bw': list_flow_BW_left,
            }

        self.data_dict = data_dict
        self.scene_keys = list(data_dict.keys())
        self.scene_keys = self.shuffle(self.scene_keys)

        # Augmentation parameters.
        aug_list = ['random_scale_crop_resize_aspect_ratio', 'resize', 'no_augmentation', 'resize_test']
        assert self.cfg.train_aug_type in aug_list
        assert self.cfg.test_aug_type in aug_list

        # Coordinate definitions.
        self.coordinate_type = 'canonical_coordinate'

        print(f" List of Spring scenes for {stage}: {len(self.scene_keys)}")

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]


    def load_data(self, scene_key: str, source_index: int, target_index: int) -> dict:

        data_key_dict = self.data_dict[scene_key]

        # Given indices, read image, depth, mask, trajectory, camera matrix
        image_src = data_key_dict['rgb'][source_index]
        image_tgt = data_key_dict['rgb'][target_index]
        disp_src = data_key_dict['disp'][source_index]
        disp_tgt = data_key_dict['disp'][target_index]
        disp_change_fw = data_key_dict['disp_change_fw'][source_index]
        disp_change_bw = data_key_dict['disp_change_bw'][source_index]
        flow_fw = data_key_dict['flow_fw'][source_index]
        flow_bw = data_key_dict['flow_bw'][source_index]

        # Camera parameters
        intrinsics_src = data_key_dict['intrinsics'][source_index]
        intrinsics_tgt = data_key_dict['intrinsics'][target_index]
        extrinsics_src = data_key_dict['extrinsics'][source_index]
        extrinsics_tgt = data_key_dict['extrinsics'][target_index]

        if self.stage == "train" and self.cfg.temporal_flip:
            image_src, image_tgt = image_tgt, image_src
            disp_src, disp_tgt = disp_tgt, disp_src
            disp_change_fw, disp_change_bw = disp_change_bw, disp_change_fw
            flow_fw, flow_bw = flow_bw, flow_fw
            intrinsics_src, intrinsics_tgt = intrinsics_tgt, intrinsics_src
            extrinsics_src, extrinsics_tgt = extrinsics_tgt, extrinsics_src

        # Load data
        image_src = media.read_image(image_src)
        image_tgt = media.read_image(image_tgt)

        # Load disparity
        disp_src = readDsp5Disp(disp_src)
        disp_tgt = readDsp5Disp(disp_tgt)
        disp_change_fw = readDsp5Disp(disp_change_fw)
        disp_change_bw = readDsp5Disp(disp_change_bw)

        # Load flow
        flow_fw = readFlo5Flow(flow_fw)
        flow_bw = readFlo5Flow(flow_bw)

        # Merge inputs.
        images = self.convert_images([image_src, image_tgt]).float()
        if self.stage == "train" and self.cfg.photometric_aug and (random.random() > 0.5):
            images = self.photo_aug(images)
        intrinsics = torch.tensor(np.stack([intrinsics_src, intrinsics_tgt])).float()
        extrinsics_src = self.to_tensor(extrinsics_src).float()[0]
        extrinsics_tgt = self.to_tensor(extrinsics_tgt).float()[0]

        flow_fw = self.to_tensor(flow_fw)[None, ...].float()
        flow_bw = self.to_tensor(flow_bw)[None, ...].float()
        disp_src = self.to_tensor(disp_src)[None, ...].float()
        disp_tgt = self.to_tensor(disp_tgt)[None, ...].float()
        disp_change_fw = self.to_tensor(disp_change_fw)[None, ...].float()
        disp_change_bw = self.to_tensor(disp_change_bw)[None, ...].float()

        # Prepare depth
        def disp2depth(focal_length, disp):
            depth = focal_length * BASELINE_WIDTH / (disp + 1e-6)
            depth[torch.isinf(depth)] = 10000
            return depth

        depth_src = disp2depth(intrinsics[0,0,0], disp_src)
        depth_tgt = disp2depth(intrinsics[1,0,0], disp_tgt)
        depth_src_fw = disp2depth(intrinsics[0,0,0], disp_change_fw)
        depth_tgt_bw = disp2depth(intrinsics[1,0,0], disp_change_bw)

        data_list = [depth_src, depth_tgt, depth_src_fw, depth_tgt_bw, flow_fw, flow_bw]
        for d_idx in range(len(data_list)):
            if is_invalid(data_list[d_idx]).any():
                data_list[d_idx] = infilling_invalid(data_list[d_idx])
        depth_src, depth_tgt, depth_src_fw, depth_tgt_bw, flow_fw, flow_bw = data_list

        def get_depth_mask_spring(depth):
            return (0.01 <= depth) * (depth <= 80.)
        depth_mask_src = get_depth_mask_spring(depth_src)
        depth_mask_tgt = get_depth_mask_spring(depth_tgt)
        depth_mask_src_fw = get_depth_mask_spring(depth_src_fw)
        depth_mask_tgt_bw = get_depth_mask_spring(depth_tgt_bw)

        def get_flow_mask_spring(flow):
            mask = is_invalid(flow_fw)
            mask = mask[:, :1] + mask[:, 1:]
            return ~mask
        flow_mask_fw = get_flow_mask_spring(flow_fw)
        flow_mask_bw = get_flow_mask_spring(flow_bw)

        # Prepare scene flow and mask
        sceneflow_fw, sceneflow_mask_fw = convert_depthflow_to_sceneflow(
            depth_src, depth_mask_src,
            depth_src_fw, depth_mask_src_fw,
            flow_fw, flow_mask_fw,
            intrinsics[:1], use_numpy=False)

        sceneflow_bw, sceneflow_mask_bw = convert_depthflow_to_sceneflow(
            depth_tgt, depth_mask_tgt,
            depth_tgt_bw, depth_mask_tgt_bw,
            flow_bw, flow_mask_bw,
            intrinsics[1:], use_numpy=False)

        # Merge output
        depths = torch.cat([depth_src, depth_tgt], dim=0)
        points = pixel2point(intrinsics, depths, use_numpy=False)
        depth_masks = torch.cat([depth_mask_src, depth_mask_tgt], axis=0).float()
        motion3d = torch.cat([sceneflow_fw, sceneflow_bw], dim=0)
        motion3d_mask = torch.cat([sceneflow_mask_fw, sceneflow_mask_bw], axis=0).float()
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
        points, motion3d = transform_cam_to_canonical(points, intrinsics, extrinsics, motion3d)
        imh, imw = images.shape[2:]

        # Depth is also at the canonical coordinate.
        depths = points[:, -1:]
        depths = torch.clamp(depths, max=self.depth_valid_max)

        # Horizontal flip augmentation:
        if self.stage == "train" and self.cfg.horizontal_flip and random.random() > 0.5:
            images = torch.flip(images, dims=[-1])
            depths = torch.flip(depths, dims=[-1])
            points = torch.flip(points, dims=[-1])
            motion3d = torch.flip(motion3d, dims=[-1])
            depth_masks = torch.flip(depth_masks, dims=[-1])
            motion3d_mask = torch.flip(motion3d_mask, dims=[-1])
            # Adjust point and scene flow x coordinate.
            points[:, 0, :, :] *= -1
            motion3d[:, 0, :, :] *= -1
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
                "motion3d": motion3d,
                "pointmap": points,
                "valid_mask_depth": depth_masks,
                "valid_mask_motion": motion3d_mask,
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "scene": scene_key,
            "dataset_alias": "sp",
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
