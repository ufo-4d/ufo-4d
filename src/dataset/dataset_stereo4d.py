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
from ..misc.cam_utils import normalize_intrinsics, camera_to_pose_encoding
from .stereo4d_helpers.utils_stereo4d import Track3d, load_rgbd_cam_from_pkl

# jaxcam doesn't work on GPU
# import jax
# jax.config.update('jax_default_device', jax.devices('cpu')[0])

import webdataset as wds
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time


STEREO4D_DATA_SPLIT_KEY = ['5N07zZzi8Vw', 'C8FDo1lzAg8', 'JHHAtpUsEJQ', 'OTzRun5q5Cc', 'UfAWb9qKRlQ', '_t672qRgCQg', 'hQ6xDzDijrI', 'mxvMt_A2jHU', 'tmRDRCgdi9w', 'zzW5NZGERmU']


@dataclass
class DatasetStereo4dCfg(DatasetCfgCommon):

    name: str
    train_root: str
    test_root: str
    train_aug_type: str
    test_aug_type: str

    # Data augmentation.
    temporal_flip: bool
    horizontal_flip: bool
    photometric_aug: bool


@dataclass
class DatasetStereo4dCfgWrapper:
    stereo4d: DatasetStereo4dCfg


@dataclass
class DatasetStereo4dSceneKey:
    root: str
    key: str
    num_images: int | None


class DatasetStereo4d(IterableDataset):
    cfg: DatasetStereo4dCfg
    stage: Stage

    to_tensor: tf.ToTensor
    near: float = 0.01
    far: float = 100.0

    video_suffix: str = 'left_rectified.mp4'
    camera_suffix: str = 'left_camera_c2w.pkl'
    depth_suffix: str = 'left_depth.pkl'
    track_suffix: str = 'optimized_tracks_filtered_v2.pkl'

    def __init__(
        self,
        cfg: DatasetStereo4dCfg,
        stage: Stage,
        global_rank: int,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.photo_aug = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1/3.14)
        self.depth_valid_min = 0.01
        self.depth_valid_max = 80.
        self.data_filtering_threshold = 0.01
        self.global_rank = global_rank

        # Collect test scenes.
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        txt_path = f'{curr_dir}/stereo4d_helpers/dataset_stereo4d_test_key.txt'

        def read_test_text(fn):
            list_test_key = []
            with open(fn, "r") as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    list_test_key.append(line.strip())
            return list_test_key

        list_test_scenes = sorted(read_test_text(txt_path))
        split_prefix = '-clip'

        def get_prefix(input_list, prefix):
            return [f.split(prefix)[0] for f in input_list]
        test_prefix = set(get_prefix(list_test_scenes, split_prefix))

        def convert_scene_key(main_root, list_scenes, list_num_images):
            return [DatasetStereo4dSceneKey(
                root=main_root,
                key=list_scenes[i],
                num_images=None if list_num_images==[] else list_num_images[i]) for i in range(len(list_scenes))]
        list_test_scene_keys = convert_scene_key(
            main_root=cfg.test_root,
            list_scenes=list_test_scenes,
            list_num_images=[])

        # Collect train scenes.
        list_train_scene_keys = []

        if stage == "train":
            print(f' Loading {stage} data.')

            num_tar_files = 13036
            multi_shard = True
            if not multi_shard:
                chunk_str_idx = 0
                chunk_end_idx = num_tar_files
            else:    
                chunk_size = num_tar_files // torch.cuda.device_count()
                chunk_str_idx = self.global_rank * chunk_size
                chunk_end_idx = chunk_str_idx + chunk_size

            chunk_str_idx = f'{chunk_str_idx:06d}'
            chunk_end_idx = f'{chunk_end_idx:06d}'

            tar_string = cfg.train_root + "/{" + chunk_str_idx + ".." + chunk_end_idx + "}_00.tar"
            
            print(f'  Reading {tar_string}...')
            self.wds_dataset = wds.WebDataset(
                tar_string,
                shardshuffle=False,
                )
            print(f'  wds_dataset is ready.')

        # Split train and test        
        if stage == "train":
            pass
        elif stage == "val":
            self.scene_keys = list_test_scene_keys[::2]
        elif stage == "test":
            self.scene_keys = list_test_scene_keys
        else:
            raise ValueError

        if stage == "train":
            print(f" List of Stereo4d scenes for {stage}.")
        else:
            print(f" List of Stereo4d scenes for {stage}: {len(self.scene_keys)}")

        # Augmentation parameters.
        aug_list = ['random_scale_crop_resize_aspect_ratio', 'resize', 'no_augmentation', 'resize_test']
        assert self.cfg.train_aug_type in aug_list
        assert self.cfg.test_aug_type in aug_list


    def get_thread_pool_for_worker(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0 # Main process gets ID 0

        if worker_id not in self.worker_thread_pools:
            # Each DataLoader worker (or the main process if num_workers=0)
            # gets its own single-thread executor.
            self.worker_thread_pools[worker_id] = ThreadPoolExecutor(max_workers=1)
            # print(f"[PID {os.getpid()}, Worker {worker_id}] Created ThreadPoolExecutor.")
        return self.worker_thread_pools[worker_id]


    def get_wids_item(self, index):

        pool = self.get_thread_pool_for_worker()
        future = pool.submit(lambda: self.wids_dataset[index])

        try:
            # Wait for the result with a timeout
            return future.result(timeout=self.wids_timeout_seconds)
        except FuturesTimeoutError:
            future.cancel() # Best effort to cancel
            pid = os.getpid()
            worker_info = torch.utils.data.get_worker_info()
            worker_id_str = str(worker_info.id) if worker_info else "main"
            print(f" [PID {pid}, worker {worker_id_str}] timeout, index {index}.")
            raise
        except Exception as e:
            # Handle other exceptions that might occur within wids_dataset[i]
            pid = os.getpid()
            worker_info = torch.utils.data.get_worker_info()
            worker_id_str = str(worker_info.id) if worker_info else "main"
            print(f" [PID {pid}, worker {worker_id_str}] error, index {index}.")
            raise # Re-raise the original exception


    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]


    def process_train_data(self, data_src: dict, data_tgt: dict, scene_key_id: str) -> dict:

        # Camera-World extrinsics.
        data_src['extrinsics_w2c'] = torch.tensor(data_src['extrinsics_w2c']).float()
        data_tgt['extrinsics_w2c'] = torch.tensor(data_tgt['extrinsics_w2c']).float()
        data_src['extrinsics_c2w'] = torch.tensor(data_src['extrinsics_c2w']).float()
        data_tgt['extrinsics_c2w'] = torch.tensor(data_tgt['extrinsics_c2w']).float()

        # Camera Intrinsics
        intrinsics = np.stack([data_src['intrinsics'], data_tgt['intrinsics']], axis=0)  # (2, 3, 3)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

        # Camera extrinsics, also used for rendering.
        pose1 = torch.eye(4).to(torch.float32)
        # Inverse here because in the decoder: view_matrix = extrinsics.inverse()
        pose2 = torch.inverse(data_tgt['extrinsics_w2c'] @ data_src['extrinsics_c2w'])
        extrinsics = torch.stack((pose1, pose2), dim=0)  # (2, 4, 4)

        # To tensor, normalized between [0., 1.]
        context_images = [data_src['image'], data_tgt['image']]
        context_images = self.convert_images(context_images)

        # # Photometric augmentation.
        if self.stage == "train" and self.cfg.photometric_aug and (random.random() > 0.5):
            context_images = self.photo_aug(context_images)
        target_images = context_images.clone()  # (v, c, h, w)
        imh, imw = target_images.shape[2:]

        # Get valid indices and pixel coordinates for image coordinate projection.
        valid_indices = data_src['valid_mask'] & data_tgt['valid_mask']
        pixel_coord_src = torch.tensor(data_src['points_at_frame'][valid_indices]).int()
        pixel_coord_tgt = torch.tensor(data_tgt['points_at_frame'][valid_indices]).int()

        # Get pointmaps at each camera's coordinate
        points_traj_src = torch.tensor(data_src['points_at_cam'][valid_indices])[None].permute(0, 2, 1).float()
        points_traj_tgt = torch.tensor(data_tgt['points_at_cam'][valid_indices])[None].permute(0, 2, 1).float()

        # Transform points_tgt to canonical coordinate (1st frame's coordinate)
        pose_t2s = data_src['extrinsics_w2c'] @ data_tgt['extrinsics_c2w']
        def tform_pts(pose, pts):
            assert pose.shape == (4, 4), f'{pose.shape=}'
            assert len(pts.shape) == 3
            b, c, n = pts.shape
            assert c == 3
            pose = pose[None, ...].repeat(b, 1, 1)
            ones = torch.ones_like(pts)[:, :1, :]
            pts = torch.cat([pts, ones], dim=1)
            return (pose @ pts)[:, :3]
        points_traj_tgt = tform_pts(pose_t2s, points_traj_tgt)

        # Get pointmaps and depths defined at the canonical coordinate
        # pointmaps = torch.cat([points_traj_src, points_traj_tgt], dim=0)  # 2, 3, n
        pointmaps = torch.zeros((2, 3, imh, imw)).float()
        pointmaps[
            0,
            :,
            pixel_coord_src[:, 1],
            pixel_coord_src[:, 0],
        ] = points_traj_src[0]
        pointmaps[
            1,
            :,
            pixel_coord_tgt[:, 1],
            pixel_coord_tgt[:, 0],
        ] = points_traj_tgt[0]
        depths = pointmaps[:, -1:]

        # Scene flow and depth_optimized share the same validity mask.
        valid_mask = ((depths > 0.) * (depths < 80.)).float()  # 2, 1, h, w
        
        # Pad zero values for invalid pixels.
        pointmaps = torch.where(
            valid_mask.bool().expand(-1, 3, -1, -1),
            pointmaps,
            torch.zeros_like(pointmaps))

        # Get scene flow defined at canonical coordinate
        sceneflow = torch.zeros((2, 3, imh, imw)).float()
        sceneflow_src = (points_traj_tgt - points_traj_src).float()
        sceneflow_tgt = (points_traj_src - points_traj_tgt).float()
        sceneflow[
            0,
            :,
            pixel_coord_src[:, 1],
            pixel_coord_src[:, 0],
        ] = sceneflow_src[0]
        sceneflow[
            1,
            :,
            pixel_coord_tgt[:, 1],
            pixel_coord_tgt[:, 0],
        ] = sceneflow_tgt[0]

        # Filter out very small motion.
        sceneflow_mask = (sceneflow.norm(2, dim=1, keepdim=True) >= 0.02).repeat(1, 3, 1, 1)
        sceneflow = torch.where(sceneflow_mask, sceneflow, torch.zeros_like(sceneflow))

        # Skip the sample if the number of valid trajectory is too few.
        data_filtering_threshold = 0.02
        if (
            float(valid_mask[0].sum() / (imh * imw)) < data_filtering_threshold or
            float(valid_mask[1].sum() / (imh * imw)) < data_filtering_threshold
        ):
            return {}

        # # Horizontal flip augmentation:
        if self.cfg.horizontal_flip and random.random() > 0.5:
            context_images = torch.flip(context_images, dims=[-1])
            target_images = torch.flip(target_images, dims=[-1])
            depths = torch.flip(depths, dims=[-1])
            pointmaps = torch.flip(pointmaps, dims=[-1])
            sceneflow = torch.flip(sceneflow, dims=[-1])
            valid_mask = torch.flip(valid_mask, dims=[-1])
            # Adjust point and scene flow x coordinate.
            pointmaps[:, 0, :, :] *= -1
            sceneflow[:, 0, :, :] *= -1
            # Camera intrinsics.
            intrinsics[:, 0, 2] = (imw - 1) - intrinsics[:, 0, 2]  # cx = w - 1 - cx
            # Camera extrinsics.
            extrinsics[1, 0, 1:4] *= -1
            extrinsics[1, 1:3, 0] *= -1

        # normalize K
        intrinsics = normalize_intrinsics(intrinsics, imh, imw)

        # Convert extrinsics to quaternion
        extrinsics_quat = camera_to_pose_encoding(extrinsics, pose_encoding_type="absT_quaR")

        example = {
            "context": {
                "extrinsics": extrinsics,
                "extrinsics_quat": extrinsics_quat,
                "intrinsics": intrinsics,
                "image": context_images,
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "target": {
                "extrinsics": extrinsics,
                "extrinsics_quat": extrinsics_quat,
                "intrinsics": intrinsics,
                "image": target_images,
                "depth": depths,
                "motion3d": sceneflow,
                "pointmap": pointmaps,
                "valid_mask_depth": valid_mask,
                "valid_mask_motion": valid_mask,
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "scene": scene_key_id,
            "dataset_alias": "s4",
        }

        return example


    def load_test_data(self, scene_key):

        assert self.stage in ["val", "test"]

        # Load data.
        with open(os.path.join(scene_key.root, scene_key.key, scene_key.key + '.pkl'), 'rb') as f:
            input_dict = pickle.load(f)

        # Loading images.
        imh0, imw0, _ = input_dict['reference']['rgb'].shape
        if imh0 != 512 or imw0 != 512:
            print("Warning: Resizing input image")
            input_dict['reference']['rgb'] = media.resize_image(input_dict['reference']['rgb'], (512, 512))
        imh1, imw1, _ = input_dict['target']['rgb'].shape
        if imh1 != 512 or imw1 != 512:
            print("Warning: Resizing input image")
            input_dict['target']['rgb'] = media.resize_image(input_dict['target']['rgb'], (512, 512))

        # Load images.
        image_a = input_dict['reference']['rgb']
        image_b = input_dict['target']['rgb']
        context_images = [image_a, image_b]
        # To tensor, normalized between [0., 1.]
        context_images = self.convert_images(context_images)

        target_images = context_images.clone()  # (v, c, h, w)
        imh, imw = target_images.shape[2:]

        # Camera Intrinsics
        def compose_intrinsics(camera_dict):
            focal = camera_dict.focal_length
            principal_point = camera_dict.principal_point
            intrinsics = [
                [focal, 0., principal_point[0]],
                [0., focal, principal_point[1]],
                [0., 0., 1]
            ]
            return torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0)
        
        intrinsics = torch.cat(
            [compose_intrinsics(input_dict['reference']['camera']),
            compose_intrinsics(input_dict['target']['camera'])], dim=0)

        # Camera Extrinsic is initialized as identity
        def compose_extrinsic(camera_dict):
            trans = np.array(camera_dict.position).reshape(3, 1)
            mat = np.eye(4)
            mat[:3, :3] = camera_dict.orientation
            mat[:3, 3] = camera_dict.translation
            return mat
        pose_ref = compose_extrinsic(input_dict['reference']['camera'])
        pose_tgt = compose_extrinsic(input_dict['target']['camera'])
        pose_ref = torch.from_numpy(pose_ref).float()
        pose_tgt = torch.from_numpy(pose_tgt).float()

        pose1 = torch.eye(4).to(torch.float32)
        pose2 = pose_ref @ torch.inverse(pose_tgt)
        extrinsics = torch.stack((pose1, pose2), dim=0)  # (2, 4, 4)

        # 3D trajectory.
        num_valid_gt_tracks = input_dict['reference']['motionmap'][0][..., 3].sum()
        valid_pixel_ratio_motion = num_valid_gt_tracks / (imh * imw)

        gt_motion_and_mask = np.stack([
            input_dict['reference']['motionmap'][0],
            input_dict['target']['motionmap'][0]],
            axis=0)  # 2, h, w, 4
        gt_motion = torch.from_numpy(gt_motion_and_mask[..., :3]).float().permute(0, 3, 1, 2)  # 2, 3, h, w
        valid_mask_motion = gt_motion_and_mask[..., 3] > 0.5  # Last channel is valid mask
        valid_mask_motion = torch.from_numpy(valid_mask_motion).float().unsqueeze(1)  # 2, 1, h, w

        # Depth.
        if input_dict['reference']['pointmap'][..., 3].sum() == 0:
            print(f'{name} no valid pixels in depth gt')

        num_valid_gt_depth = input_dict['reference']['pointmap'][..., 3].sum()
        valid_pixel_ratio_depth = num_valid_gt_depth / (imh * imw)

        source_pointmap = input_dict['reference']['pointmap']  # h, w, 4
        target_pointmap = input_dict['target']['pointmap']  # h, w, 4
        gt_pointmap_and_mask = np.stack([
            source_pointmap,
            target_pointmap],
            axis=0)  # 2, h, w, 4
        gt_pointmap = torch.from_numpy(gt_pointmap_and_mask[..., :3]).float().permute(0, 3, 1, 2)  # 2, 3, h, w
        gt_depth = gt_pointmap[:, 2:3, :, :]
        valid_mask_pointmap = gt_pointmap_and_mask[..., 3] > 0.5  # Last channel is valid mask
        valid_mask_pointmap = torch.from_numpy(valid_mask_pointmap).float().unsqueeze(1)  # 2, 1, h, w
        valid_mask_pointmap = valid_mask_pointmap * (0. < gt_depth) * (gt_depth < self.depth_valid_max)

        # normalize K
        intrinsics = normalize_intrinsics(intrinsics, imh, imw)

        # Convert extrinsics to quaternion
        extrinsics_quat = camera_to_pose_encoding(extrinsics, pose_encoding_type="absT_quaR")

        example = {
            "context": {
                "extrinsics": extrinsics,
                "extrinsics_quat": extrinsics_quat,
                "intrinsics": intrinsics,
                "image": context_images,
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "target": {
                "extrinsics": extrinsics,
                "extrinsics_quat": extrinsics_quat,
                "intrinsics": intrinsics,
                "image": target_images,
                "depth": gt_depth,
                "motion3d": gt_motion,
                "pointmap": gt_pointmap,
                "valid_mask_depth": valid_mask_pointmap,
                "valid_mask_motion": valid_mask_motion,
                "near": self.get_bound("near", 2),
                "far": self.get_bound("far", 2),
            },
            "scene": scene_key.key,
            "dataset_alias": "s4",
        }

        return example


    def load_train_data_wds(self, data_bytes) -> dict:
        try:
            data_dict = pickle.loads(data_bytes)
        except pickle.UnpicklingError as e:
            print(f"WDS dataloader error {e}")
            return None
        except Exception as e:
            print(f"WDS dataloader exception {e}")
            return None
        return data_dict


    def process_wds_sample(self, input_dict) -> dict:

        if not input_dict:
            return None

        data_dict = input_dict.get("pkl")
        key = input_dict.get("__key__")

        # Loading train data,
        if self.stage == "train" and self.cfg.temporal_flip and random.random() > 0.5:
            data_dict['source'], data_dict['target'] = data_dict['target'], data_dict['source']

        example = self.process_train_data(
            data_dict['source'],
            data_dict['target'],
            key)
        
        if example == {}:
            return None

        return example

    def __iter__(self):

        assert self.stage in ["train", "val", "test"]

        if self.stage in ["train"]:
            dataset_wds_decoded = self.wds_dataset \
                .decode(
                    wds.handle_extension("pkl", self.load_train_data_wds) # Decode .pkl files
                ) \
                .map(self.process_wds_sample) \
                .select(lambda sample: sample is not None)
            yield from dataset_wds_decoded

        elif self.stage in ["val", "test"]:
            for scene_key in self.scene_keys:
                example = self.load_test_data(scene_key)
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
