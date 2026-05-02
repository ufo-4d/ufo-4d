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

import torch
import torchvision.transforms as tf
from einops import repeat, rearrange
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_geometric_augmentation
from torchvision.transforms import ColorJitter

from .types import Stage
from ..misc.geometry import pixel2point, transform_points
from ..misc.cam_utils import normalize_intrinsics, camera_to_pose_encoding


skip_sequence = [
    'ani2_s', 'ani_s', 'animal1_s', 'animal2_s', 'animal3_s', 'animal4_s',
    'animal6_s', 'animal_s', 'animal_smoke', 'cab_e1_3rd', 'cab_e1_ego2',
    'cab_e_ego2', 'cab_h_bench_ego2', 'cnb_dlab_0225_ego1', 'dancingroom_3rd',
    'scene_d78_0318_3rd', 'scene_d78_0318_ego1', 'scene_d78_0318_ego2',
    'character', 'character0', 'character0_', 'character0_f', 'character0_f2',
    'character1', 'character1_f', 'character2', 'character2_', 'character2_f',
    'character3', 'character3_f', 'character4', 'character4_', 'character4_f',
    'character5', 'character5_', 'character5_f', 'character6', 'character6_f',
    'gso_in_big', 'gso_out_big']

heavy_fog_sequence = [
    'cab_e1_ego2', 'cab_e1_3rd', 'cab_e_ego2', 'cnb_dlab_0225_ego1',
    'dancingroom_3rd', 'scene_d78_0318_ego1', 'scene_d78_0318_ego2']

invalid_info_sequence = ['gso_in_big', 'gso_out_big']

test_clean_sequence = ['cab_e_3rd', 'egobody_egocentric', 'seminar_g110_0315_3rd', 'seminar_g110_0315_ego1']

@dataclass
class DatasetPointOdysseyCfg(DatasetCfgCommon):

    name: str
    train_root: str
    test_root: str
    train_aug_type: str
    test_aug_type: str
    
    # Data augmentation.
    temporal_flip: bool
    horizontal_flip: bool
    photometric_aug: bool
    min_distance_to_target: float
    max_distance_to_target: float


@dataclass
class DatasetPointOdysseyCfgWrapper:
    pointodyssey: DatasetPointOdysseyCfg


class DatasetPointOdyssey(IterableDataset):
    cfg: DatasetPointOdysseyCfg
    stage: Stage

    to_tensor: tf.ToTensor
    near: float = 0.01
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetPointOdysseyCfg,
        stage: Stage,
        global_rank: int,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.photo_aug = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1/3.14)
        self.depth_valid_min = 0.01
        self.depth_valid_max = 200.
        self.global_rank = global_rank

        if stage == 'train':
            dataset_root = cfg.train_root
            list_sequence = sorted(os.listdir(dataset_root))
            list_sequence = [f for f in list_sequence if not f[-3:] in ['mp4', '.py']]

            # Maximal skip
            skip_sequence = []
            for seq in list_sequence:
                if seq.startswith('ani') or seq.startswith('char') or seq.startswith('r') or (seq in heavy_fog_sequence) or (seq in invalid_info_sequence):
                    skip_sequence.append(seq)
            list_sequence = [f for f in list_sequence if f not in skip_sequence]

        elif stage in ['val', 'test']:
            dataset_root = cfg.test_root
            list_sequence = test_clean_sequence

        print(f" List of PointOdyssey scenes for {stage}: {len(list_sequence)}")

        # Collect paths of data
        data_dict = {}

        for i, key in enumerate(list_sequence):
            data_dict[key] = {}
            sequence_dir = os.path.join(dataset_root, key)

            # Read info
            info_name = os.path.join(sequence_dir, "info.npz")
            assert os.path.exists(info_name)
            with np.load(info_name) as info:
                n_frames, n_points = info['trajs_3d'][:2]
            data_dict[key]['n_frames'] = n_frames
            data_dict[key]['n_points'] = n_points
            
            # RGB lists 
            rgb_dir = os.path.join(sequence_dir, "rgbs")
            rgb_list = [os.path.join(rgb_dir, f'rgb_{n:05d}.jpg') for n in range(n_frames)]
            data_dict[key]['rgb_list'] = rgb_list

            # Depth list
            depth_dir = os.path.join(sequence_dir, "depths")
            depth_list = [os.path.join(depth_dir, f'depth_{n:05d}.png') for n in range(n_frames)]
            data_dict[key]['depth_list'] = depth_list

            # Mask list
            mask_dir = os.path.join(sequence_dir, "masks")
            mask_list = [os.path.join(mask_dir, f'mask_{n:05d}.png') for n in range(n_frames)]
            data_dict[key]['mask_list'] = mask_list

            # Anno list
            anno_dir = os.path.join(sequence_dir, "annos")
            anno_list = [os.path.join(anno_dir, f'anno_{n:05d}.npz') for n in range(n_frames)]
            data_dict[key]['anno_list'] = anno_list

        self.data_dict = data_dict
        self.scene_keys = list(data_dict.keys())
        self.scene_keys = self.shuffle(self.scene_keys)

        # Augmentation parameters.
        aug_list = ['random_scale_crop_resize_aspect_ratio', 'resize', 'crop_resize', 'no_augmentation', 'resize_test']
        assert self.cfg.train_aug_type in aug_list
        assert self.cfg.test_aug_type in aug_list


    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]


    def load_data(self, scene_key: str, source_index: int, target_index: int) -> dict:

        data_key_dict = self.data_dict[scene_key]

        # Given indices, read image, depth, mask, trajectory, camera matrix
        image_src = data_key_dict['rgb_list'][source_index]
        image_tgt = data_key_dict['rgb_list'][target_index]
        depth_src = data_key_dict['depth_list'][source_index]
        depth_tgt = data_key_dict['depth_list'][target_index]
        mask_src = data_key_dict['mask_list'][source_index]
        mask_tgt = data_key_dict['mask_list'][target_index]
        anno_src = data_key_dict['anno_list'][source_index]
        anno_tgt = data_key_dict['anno_list'][target_index]

        # Load data
        image_src = media.read_image(image_src)
        image_tgt = media.read_image(image_tgt)
        depth_src = media.read_image(depth_src)[:, :, None]
        depth_tgt = media.read_image(depth_tgt)[:, :, None]

        # Depth rescale. 1000 is the max depth in the dataset
        depth_src = depth_src.astype(np.float32) / 65535.0 * 1000.0
        depth_tgt = depth_tgt.astype(np.float32) / 65535.0 * 1000.0

        depth_mask_src = (self.depth_valid_min < depth_src) * (depth_src < self.depth_valid_max)
        depth_mask_tgt = (self.depth_valid_min < depth_tgt) * (depth_tgt < self.depth_valid_max)

        # Load annotation.
        with np.load(anno_src) as anno:
            intrinsics_src = anno['intrinsics']
            extrinsics_src = anno['extrinsics']
            traj3d_src = anno['trajs_3d']
            valid_vis = anno['valids'] * anno['visibs']

        with np.load(anno_tgt) as anno:
            intrinsics_tgt = anno['intrinsics']
            extrinsics_tgt = anno['extrinsics']
            traj3d_tgt = anno['trajs_3d']
            valid_vis_tgt = anno['valids'] * anno['visibs']

        # If valid trajectory is two few
        if (valid_vis * valid_vis_tgt).sum() < 3000:
            return {}

        # Merge inputs.
        images = self.convert_images([image_src, image_tgt])
        depths = self.convert_images([depth_src, depth_tgt])
        depth_masks = self.convert_images([depth_mask_src, depth_mask_tgt]).float()
        intrinsics = torch.tensor(np.stack([intrinsics_src, intrinsics_tgt]))

        # Photometric augmentation.
        if self.stage == "train" and self.cfg.photometric_aug and (random.random() > 0.5):
            images = self.photo_aug(images)
        imh, imw = images.shape[2:]

        # 3D pointmap from depth, intrinsics, and extrinsics.
        points = pixel2point(intrinsics, depths, use_numpy=False)
        extrinsics_src = torch.tensor(extrinsics_src)
        extrinsics_tgt = torch.tensor(extrinsics_tgt)
        pose_src = torch.eye(4)
        pose_tgt = extrinsics_src @ torch.inverse(extrinsics_tgt)
        extrinsics = torch.stack([pose_src, pose_tgt])

        # Transform 2nd view's point to the 1st view.
        points[1:] = transform_points(pose_tgt, points[1:])
        # Depth is also at the canonical coordinate
        depths = points[:, -1:]

        # 3D trajectory annotation.
        traj3d_src = traj3d_src[valid_vis] # n, 3
        traj3d_tgt = traj3d_tgt[valid_vis] # n, 3
        traj3d = torch.tensor(np.stack([traj3d_src, traj3d_tgt])).permute(0, 2, 1) # v, 3, n

        # Transform 3D trajectory into 1st frame's coordinate.
        traj3d_at_view0 = transform_points(extrinsics_src, traj3d)
        traj3d_at_view1 = transform_points(extrinsics_tgt, traj3d)

        # Convert it to into scene flow map (projection into 2D image.)
        def convert_to_sceneflow_map(
            traj3d_at_view0,
            traj3d_at_view1,
            intrinsics,
            image_size):

            # Compute scene flow
            sceneflow_fw = traj3d_at_view0[1] - traj3d_at_view0[0]
            sceneflow_bw = traj3d_at_view0[0] - traj3d_at_view0[1]
            sceneflow = torch.stack([sceneflow_fw, sceneflow_bw], dim=0)
            proj_pts_view0 = torch.matmul(intrinsics[:1].float(), traj3d_at_view0[:1])
            proj_pts_view1 = torch.matmul(intrinsics[1:].float(), traj3d_at_view1[1:])
            proj_pts = torch.cat([proj_pts_view0, proj_pts_view1], dim=0)
            pixels_mat = proj_pts.div(proj_pts[:, 2:3, :] + 1e-8)[:, :2, :]
            x_coords = pixels_mat[:, 0, :].long()
            y_coords = pixels_mat[:, 1, :].long()

            height, width = image_size
            image = torch.zeros((2, 3, height, width), dtype=sceneflow.dtype, device=sceneflow.device)
            mask = torch.zeros((2, 3, height, width), dtype=sceneflow.dtype, device=sceneflow.device)

            if x_coords.max() >= width or y_coords.max() >= height or \
                x_coords.min() < 0 or y_coords.min() < 0:
                x_coords = torch.clamp(x_coords, 0, width - 1)
                y_coords = torch.clamp(y_coords, 0, height - 1)
            
            for t in range(2):
                image[t, :, y_coords[t], x_coords[t]] = sceneflow[t]
                mask[t, :, y_coords[t], x_coords[t]] = 1.
            return image, mask[:, :1]

        motion3d, motion_mask = convert_to_sceneflow_map(
            traj3d_at_view0,
            traj3d_at_view1,
            intrinsics,
            images.shape[2:])

        # Horizontal flip augmentation:
        if self.stage == "train" and self.cfg.horizontal_flip and random.random() > 0.5:
            images = torch.flip(images, dims=[-1])
            depths = torch.flip(depths, dims=[-1])
            points = torch.flip(points, dims=[-1])
            motion3d = torch.flip(motion3d, dims=[-1])
            depth_masks = torch.flip(depth_masks, dims=[-1])
            motion_mask = torch.flip(motion_mask, dims=[-1])
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
                "valid_mask_motion": motion_mask,
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "scene": scene_key,
            "dataset_alias": "po",
        }

        return example


    def __iter__(self):

        # Get random index given the number of frames in the sequence.
        def get_random_indices(n_frames):
            sequence_length = random.randint(
                self.cfg.min_distance_to_target,
                min(n_frames-1,
                    self.cfg.max_distance_to_target))
            source_index = random.randint(
                0,
                (n_frames-1)-sequence_length)
            target_index = source_index + sequence_length
            
            if self.stage == "train" and self.cfg.temporal_flip and random.random() > 0.5:
                source_index, target_index = target_index, source_index

            return source_index, target_index

        def get_list_test_indices(n_frames, frame_dist=15):
            list_indices = [i for i in range(n_frames)]
            list_indices = list_indices[::frame_dist]
            list_pairs = []
            for source_index, target_index in zip(list_indices[:-1], list_indices[1:]):
                list_pairs.append([source_index, target_index])
            list_pairs = list_pairs[::8]
            return list_pairs

        # Iterate through scene keys.
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

            # Val and test shares the same index.
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
