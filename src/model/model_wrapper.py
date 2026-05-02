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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any, Union

import cv2
import copy
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import random
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor, nn, optim
import torchvision.utils as vutils
import os
import mediapy

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_psnr, compute_epe, compute_depth_eval
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker, Timer
from ..misc.cam_utils import pose_encoding_to_camera
from ..misc.image_io import prep_image
from ..misc.LocalLogger import LocalLogger
from ..misc.step_tracker import StepTracker
from ..misc.utils import vis_depth_map, vis_3dmotion_map, vis_normalized_map
from ..misc.geometry import estimate_pointmap_scale
from ..misc.estimate_intrinsics import estimate_intrinsics
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat, vcat
from .decoder.decoder import Decoder
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .types import Gaussians, DynamicGaussians


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    use_point_for_eval: bool
    median_scale_eval: bool


@dataclass
class TrainCfg:
    print_log_every_n_steps: int
    save_image_every_n_steps: int
    detach_cross_grad_for_rendering: bool
    detach_mg_grad_for_rendering: bool
    train_width: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[Union[WandbLogger, TensorBoardLogger]]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.detach_cross_grad_for_rendering = self.train_cfg.detach_cross_grad_for_rendering
        self.detach_mg_grad_for_rendering = self.train_cfg.detach_mg_grad_for_rendering
    
        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.time_logger = Timer()

        self.n_views = 2

        # This is for metric logging.
        metric_keys = ['psnr', 'focal_abs']
        single_view_metric_keys = []
        two_view_metric_keys = ['point_epe', 'motion_g_epe', 'motion_r_epe', 'point_r_epe', 'depth_abs_rel']

        # Each item will be metrics of each dataset, and key will be the dataloader index.
        self.list_dataloader_key = ['s4', 'po', 'vk', 'kt', 'bo', 'sp']
        self.metrics_accumulator = {}
        # Each item will be the counter of a dataloader index
        self.metrics_count = {}
        self.metrics_keys = (
            [f'{k}' for k in metric_keys]
            + [f'{k}/{view}' for k in single_view_metric_keys for view in [0, 'all'] ]
            + [f'{k}/{view}' for k in two_view_metric_keys for view in [0, 1, 'all'] ]
        )
        for n in self.list_dataloader_key:
            self.init_metrics(n)

        self.key_to_skip_print = (
            ['lpips', 'ssim']
            + [f'{k}/{view}' for k in single_view_metric_keys for view in ['all'] ]
            + [f'{metric}/{view}' for view in [0, 1] for metric in ['point_epe', 'motion_g_epe'] ]
        )

    def init_metrics(self, n):
        self.metrics_count[n] = 0
        self.metrics_accumulator[n] = {}
        for key in self.metrics_keys:
            self.metrics_accumulator[n][key] = 0.


    def postprocess(
        self,
        batch,
        encoder_dict):

        pred_dict = dict()

        b, _, _, h, w = batch["context"]["image"].shape
        v_render = batch["target"]["image"].shape[1]

        pred_dict['gaussians'] = encoder_dict['gaussians']
        
        # Get scales
        pred_dict["pointmap_scale"], batch["target"]["pointmap_scale"] = self.get_scale(batch, pred_dict['gaussians'])

        # We assume both views has the same intrinsics. Infer intrinsics from pointmap.
        pred_intrinsics = estimate_intrinsics(
            rearrange(
                pred_dict['gaussians'].means,
                "b (v h w) xyz -> b v h w xyz",
                v=self.n_views, h=h, w=w),
            focal_mode='weiszfeld')  # b v c d
        pred_intrinsics = pred_intrinsics[:, :1, :, :].repeat(1, v_render, 1, 1)

        # Overwrite GT intrinsics if estimated intrinsics is singular.
        for b_idx in range(b):
            for v_idx in range(pred_intrinsics.shape[1]):
                det = abs(torch.linalg.det(pred_intrinsics[b_idx, v_idx]))
                if det < 1e-6:
                    pred_intrinsics[b_idx, v_idx] = batch["target"]["intrinsics"][b_idx, v_idx]
        pred_dict['intrinsics'] = pred_intrinsics

        # Prepare pose for rendering.
        if encoder_dict['pose'] is not None:  # If pose is estimated.
            # Convert quaternion to pose matrix
            pose_est_view2 = pose_encoding_to_camera(
                encoder_dict['pose'], pose_encoding_type="absT_quaR")
            pose_est_view1 = torch.eye(4).to(device=pose_est_view2.device)
            pose_est_view1 = pose_est_view1.unsqueeze(0).expand(pose_est_view2.shape[0], -1, -1)
            post_list = [pose_est_view1, pose_est_view2]

            pose_est = torch.stack(post_list, dim=1)
            pred_dict["extrinsics"] = pose_est  # for rendering
            pred_dict["extrinsics_quat"] = encoder_dict['pose']  # for loss

        batch["pred"] = pred_dict

        return batch 


    def interpolate_output(self, input_tensor, output_h, output_w):
        
        if input_tensor.ndim == 5:
            b, v, c, h, w = input_tensor.shape
            input_tensor = input_tensor.reshape(b*v, c, h, w)
            input_resized = F.interpolate(
                input_tensor, size=(output_h, output_w), mode='bilinear', align_corners=False)
            return input_resized.reshape(b, v, c, output_h, output_w)
        else:
            raise NotImplementedError


    def get_scale(self, batch, gaussians):

        assert "pointmap" in batch["target"]

        # Infer scale from pointmap for scale-invariant loss.
        b, _, _, h, w = batch["context"]["image"].shape
        pred_points = rearrange(gaussians.means, "b (v h w) c -> b v c h w", v=self.n_views, h=h, w=w)
        target_points = batch["target"]["pointmap"]
        pred_points = self.interpolate_output(
            pred_points,
            output_h=target_points.shape[3],
            output_w=target_points.shape[4])

        if 'valid_mask_depth' in batch["target"]:
            valid_mask = batch["target"]["valid_mask_depth"][:, :, 0]  # b v h w
        else:
            valid_mask = torch.ones_like(pred_points)[:, :, 0]  # b v h w
        
        # Estimate scale. (the first two are context GT.)
        pred_pointmap_scale, target_pointmap_scale = estimate_pointmap_scale(
            pred_points[:, :2],
            target_points[:, :2], 
            valid_mask[:, :2])

        return pred_pointmap_scale, target_pointmap_scale


    def prepare_rendering(
        self,
        gaussians: DynamicGaussians,
        render_intrinsics: Float[Tensor, "batch view 3 3"],
        render_extrinsics: Float[Tensor, "batch view 4 4"],
        intp_type: str,
        is_train: bool = False,
        ):

        # Prepare a set of gaussians at {t=0, v=0} and {t=1, v=1}.
        if intp_type == 'standard':
            v_model = v_render = 2
            # All estimations are at the canonical coordinate.
            means = rearrange(gaussians.means, "b (v g) xyz -> b v g xyz", v=v_model)
            motions = rearrange(gaussians.motions, "b (v g) xyz -> b v g xyz", v=v_model)
            means_v0, means_v1 = means.split([1,1], dim=1)  # b 1 g xyz
            motions_v0, motions_v1 = motions.split([1,1], dim=1)  # b 1 g xyz

            if self.detach_cross_grad_for_rendering:
                means_t0 = torch.cat([means_v0, means_v1.detach() + motions_v1], dim=2)  # b 1 gg xyz
                means_t1 = torch.cat([means_v0.detach() + motions_v0, means_v1], dim=2)  # b 1 gg xyz
            else:
                means_t0 = torch.cat([means_v0, means_v1 + motions_v1], dim=2)  # b 1 gg xyz
                means_t1 = torch.cat([means_v0 + motions_v0, means_v1], dim=2)  # b 1 gg xyz
            
            motion_t0 = torch.cat([motions_v0, -motions_v1], dim=2)  # b 1 gg xyz
            motion_t1 = torch.cat([-motions_v0, motions_v1], dim=2)  # b 1 gg xyz
            means_merged = torch.cat([means_t0, means_t1], dim=1)  # b v gg xyz
            motion_merged = torch.cat([motion_t0, motion_t1], dim=1)  # b v gg xyz

            if self.detach_mg_grad_for_rendering:
                means_merged = means_merged.detach()
                motion_merged = motion_merged.detach()

            covariances = repeat(gaussians.covariances, "b gg i j -> b v gg i j", v=v_render)
            harmonics = repeat(gaussians.harmonics, "b gg c d_sh -> b v gg c d_sh", v=v_render)
            opacities = gaussians.opacities
            opacities = repeat(opacities, "b gg -> b v gg", v=v_render)

            render_gaussian = DynamicGaussians(
                rearrange(means_merged, "b v gg xyz -> b (v gg) xyz"),
                rearrange(covariances, "b v gg i j -> b (v gg) i j"),
                rearrange(harmonics, "b v gg c d_sh -> b (v gg) c d_sh"),
                rearrange(opacities, "b v gg -> b (v gg)"),
                rearrange(motion_merged, "b v gg xyz -> b (v gg) xyz"),
            )
        else:
            raise ValueError(f'{intp_type} not implemented.')

        return render_gaussian, render_intrinsics, render_extrinsics


    def training_step(self, batch, batch_idx):

        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        visualization_dump = {}
        encoder_dict = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump)

        # Postprocess.
        batch = self.postprocess(batch, encoder_dict)

        # Rendering camera model
        render_intrinsics = batch["target"]["intrinsics"]
        render_extrinsics = batch["pred"]["extrinsics"]

        # Preparing gaussians
        (render_gaussian, render_intrinsics, render_extrinsics) = self.prepare_rendering(
            batch["pred"]["gaussians"],
            render_intrinsics,
            render_extrinsics,
            intp_type='standard',
            is_train=True)

        # Render using the decoder
        output = self.decoder.forward(
            gaussians=render_gaussian,
            extrinsics=render_extrinsics,
            intrinsics=render_intrinsics,
            near=batch["target"]["near"],
            far=batch["target"]["far"],
            image_shape=(h, w),
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_train = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_train", psnr_train.mean())

        # Compute and log loss.
        total_loss = 0
        loss_log = {}
        for loss_fn in self.losses:
            loss_weighted, loss_val = loss_fn.forward(output, batch, batch["pred"]["gaussians"], self.global_step)
            total_loss = total_loss + loss_weighted

            # Logging.
            self.log(f"loss/{loss_fn.name}", loss_val)
            loss_log[loss_fn.name] = loss_val
   
        if torch.isnan(total_loss):
            print('Loss is NaN.')
            exit()
            total_loss = torch.zeros_like(total_loss)

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            elapsed_time_per_step = self.time_logger.time(
                num_calls=self.train_cfg.print_log_every_n_steps)
            loss_string = " ".join([f"{key}={value:.3f}" for key, value in loss_log.items()])
            print(
                f"train step {self.global_step}; "
                f"{elapsed_time_per_step:.2f}s/step; "
                f"loss={total_loss:.6f} "
                f"{loss_string}"
            )
            
        # Save train sample
        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.save_image_every_n_steps == 0
        ):  
            self.visualize_output(batch, visualization_dump, output, mode='train', batch_idx=0)  # 0 is placeholder

        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    @rank_zero_only
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.
        visualization_dump = {}
        with self.benchmarker.time("encoder"):
            encoder_dict = self.encoder(
                batch["context"],
                self.global_step,
                visualization_dump=visualization_dump
            )

        # Postprocess.
        batch = self.postprocess(batch, encoder_dict)

        # Rendering camera model
        render_intrinsics = batch["pred"]["intrinsics"]
        render_extrinsics = batch["pred"]["extrinsics"]

        # Preparing gaussians
        (render_gaussian, render_intrinsics, render_extrinsics) = self.prepare_rendering(
            batch["pred"]["gaussians"],
            render_intrinsics,
            render_extrinsics,
            intp_type='standard')

        _, _, _, h, w = batch["context"]["image"].shape
        # with self.benchmarker.time("decoder", num_calls=v):
        output = self.decoder.forward(
            gaussians=render_gaussian,
            extrinsics=render_extrinsics,
            intrinsics=render_intrinsics,
            near=batch["target"]["near"],
            far=batch["target"]["far"],
            image_shape=(h, w),
        )

        # Get metrics.
        dataset_key = batch["dataset_alias"][0]
        eval_count = self.get_metrics(
            batch=batch,
            gaussians=batch["pred"]["gaussians"],
            output=output,
            metrics_dict=self.metrics_accumulator[dataset_key],
            dataset_key=dataset_key,
            mode='test')
        self.metrics_count[dataset_key] += eval_count

        # # Slow if it's turned on
        # if batch_idx <= 32 and batch_idx % 4 == 0:
        #     self.visualize_output(
        #         batch,
        #         visualization_dump,
        #         output,
        #         mode='test',
        #         batch_idx=batch_idx)


    def get_valid_mask(self, batch, key, view):
        if key in batch["target"]:
            mask = batch["target"][key][:, :view]  # b v c h w
            assert len(mask.shape) == 5, f'{mash.shape}'
            assert mask.shape[2] == 1, f'{mask.shape=}'
        else:
            mask = None
        return mask


    def get_valid_mask_depth_eval(self, batch, mask_key, target_depth, view):

        mask = self.get_valid_mask(batch, mask_key, view)
        if mask == None:
            mask = torch.ones_like(target_depth)
        assert mask.shape == target_depth.shape
        return (mask * (target_depth > 0.) * (target_depth < 80.)).bool()


    def align_scale_of_prediction_to_target(
        self, batch, target, pred, output_list, use_first_frame_for_median_scale=False, align_type='median'):
        
        for out in output_list:
            assert len(out.shape) == 5, f'{target.shape=}'

        with torch.no_grad():

            # Get scale. Target and pred are pointmaps.
            assert len(target.shape) == 5, f'{target.shape=}'
            assert target.shape[2] == 3, f'{target.shape=}'
            assert len(pred.shape) == 5, f'{pred.shape=}'
            assert pred.shape[2] == 3, f'{pred.shape=}'

            pred_depth = pred[:, :, 2:3]  # b v 1 h w
            target_depth = target[:, :, 2:3]  # b v 1 h w

            mask = self.get_valid_mask_depth_eval(
                batch=batch,
                mask_key='valid_mask_depth',
                target_depth=target_depth,
                view=target_depth.shape[1]
            ).bool()  # b v 1 h w

            scales = []
            shift = []

            if align_type == 'median':
                for i in range(pred.shape[0]):
                    if use_first_frame_for_median_scale or (self.n_views==1):
                        scales.append(
                            torch.median(target_depth[i,:1][mask[i,:1]] / pred_depth[i,:1][mask[i,:1]]).detach()
                            )
                    else:
                        scales.append(
                            torch.median(target_depth[i][mask[i]] / pred_depth[i][mask[i]]).detach()
                            )
                scales = torch.stack(scales, dim=0).to(device=pred.device)
                scales = scales.reshape(pred.shape[0], 1, 1, 1, 1)
                shift = torch.zeros_like(scales)

            elif align_type == 'scale_shift_invariant':
                for b in range(target_depth.shape[0]):
                    gt_b = target_depth[b]
                    pred_b = pred_depth[b]
                    mask_b = mask[b]

                    gt_valid = gt_b[mask_b]
                    pred_valid = pred_b[mask_b]

                    A = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)
                    solution = torch.linalg.lstsq(A, gt_valid)[0]
                    s, t = solution.squeeze().tolist() # Convert to Python floats
                    scales.append(s)
                    shift.append(t)
                scales = torch.tensor(scales).to(device=pred.device)
                shift = torch.tensor(shift).to(device=pred.device)
                scales = scales.reshape(pred.shape[0], 1, 1, 1, 1)
                shift = shift.reshape(pred.shape[0], 1, 1, 1, 1)

        # Update with the median scale.
        return [out * scales + shift for out in output_list]


    def transform_canonical_to_cam(
        self,
        extrinsics,
        point_gaussian,
        point_rendered,
        motion_gaussian,
        motion_rendered):

        # Convert canonical coordinate to camera coordinate
        def tform_point2(pose, pts):
            b, v, c, h, w = pts.shape
            assert v == 1, f'{v=}'
            assert pose.shape == (b, 4, 4), f'{pose.shape=}'
            pts = pts.reshape(b, c, h*w)
            ones = torch.ones_like(pts)[:, :1, :]
            pts = torch.cat([pts, ones], dim=1)
            return (pose @ pts)[:, :3].reshape(b, 1, c, h, w)

        tgt2ref = extrinsics[:,1]
        ref2tgt = torch.inverse(extrinsics[:,1])

        # Convert motion
        tform_pt_g_1 = tform_point2(ref2tgt, point_gaussian[:, :1])
        tform_pt_r_1 = tform_point2(ref2tgt, point_rendered[:, :1])
        motion_by_cam_pt_g = tform_pt_g_1 - point_gaussian[:, :1]
        motion_by_cam_pt_r = tform_pt_r_1 - point_rendered[:, :1]

        motion_gaussian[:, :1] = motion_gaussian[:, :1] + motion_by_cam_pt_g
        motion_rendered[:, :1] = motion_rendered[:, :1] + motion_by_cam_pt_r

        # For two view estimation, transform 2nd view's point to its camera coordinate.
        if self.n_views != 1:
            point_gaussian[:, 1:] = tform_point2(ref2tgt, point_gaussian[:, 1:])
            point_rendered[:, 1:] = tform_point2(ref2tgt, point_rendered[:, 1:])

        return point_gaussian, point_rendered, motion_gaussian, motion_rendered


    def get_metrics(self, batch, gaussians, output, metrics_dict, dataset_key, mode):
        '''
        Accumulate the sum of metrics in the batch.
        Need to divide them when averaging.
        '''

        assert mode in ['val', 'test'], f'{mode=}'

        # Resize output if necessary
        b, _, c, context_h, context_w = batch["context"]["image"].shape
        _, _, _, target_h, target_w = batch["target"]["image"].shape

        # FOr KITTI, evaluate only the first frame.
        if dataset_key in ['kt']:
            eval_view_indices = [0]
        else:
            eval_view_indices = list(range(self.n_views))

        if (context_h != target_h) or (context_w != target_w):
            resize_output = True
        else:
            resize_output = False

        # Skip if GT is not valid:
        mask = self.get_valid_mask(batch, key='valid_mask_depth', view=2)
        mask_check = mask[:, eval_view_indices] 
        mask_check = mask_check.sum(axis=(2, 3, 4)).reshape(b * len(eval_view_indices), -1)
        if (mask_check == 0).any():
            print(f'skipping eval {mask_check}')
            return

        # Prepare all in (b, v, c, h, w)
        point_gaussian = rearrange(
            gaussians.means, 
            "b (v h w) xyz -> b v xyz h w", v=self.n_views, h=context_h, w=context_w)  # gaussian's center
        motion_gaussian = rearrange(
            gaussians.motions, 
            "b (v h w) xyz -> b v xyz h w", v=self.n_views, h=context_h, w=context_w)  # gaussian's center

        if resize_output:
            point_gaussian = self.interpolate_output(point_gaussian, target_h, target_w)
            motion_gaussian = self.interpolate_output(motion_gaussian, target_h, target_w)
        
        point_rendered = point_gaussian if self.test_cfg.use_point_for_eval else output.point_rendered
        motion_rendered = motion_gaussian if self.test_cfg.use_point_for_eval else output.motion_rendered
        image_rendered = output.color

        if resize_output:
            point_rendered = self.interpolate_output(point_rendered, target_h, target_w)
            motion_rendered = self.interpolate_output(motion_rendered, target_h, target_w)
            image_rendered = self.interpolate_output(image_rendered, target_h, target_w)

        # Eval depth and motion on the camera coordinate.
        if dataset_key in ['kt']:
            (point_gaussian,
            point_rendered,
            motion_gaussian,
            motion_rendered) = self.transform_canonical_to_cam(
                batch["pred"]["extrinsics"],
                point_gaussian,
                point_rendered,
                motion_gaussian,
                motion_rendered)

        if self.test_cfg.median_scale_eval:
            use_first_frame_for_median_scale = (dataset_key in ['kt'])
            (point_gaussian,
            point_rendered,
            motion_gaussian,
            motion_rendered) = self.align_scale_of_prediction_to_target(
                batch=batch,
                target=batch["target"]["pointmap"],
                pred=point_rendered,
                use_first_frame_for_median_scale=use_first_frame_for_median_scale,
                output_list=[point_gaussian, point_rendered, motion_gaussian, motion_rendered]
            )

        # Rendered images, eval on two views.
        rgb_pred = rearrange(image_rendered, "b v c h w -> (b v) c h w")
        rgb_gt = rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w")
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        # lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        # ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        metrics_dict[f'psnr'] += psnr.cpu().numpy() * b
        # metrics_dict[f'lpips'] += lpips.cpu().numpy()
        # metrics_dict[f'ssim'] += ssim.cpu().numpy()

        # Gaussian's center.
        if "pointmap" in batch["target"]:
            target_point = batch["target"]["pointmap"]
            pred_point = point_gaussian
            mask = self.get_valid_mask(batch, key='valid_mask_depth', view=self.n_views)

            val = compute_epe(target_point, pred_point, mask).cpu().numpy()
            for v in eval_view_indices:
                metrics_dict[f'point_epe/{v}'] += val[:, v].mean() * b
            metrics_dict[f'point_epe/all'] += val[:, eval_view_indices].mean() * b

        # Rendered point.
        if "pointmap" in batch["target"]:
            target_point = batch["target"]["pointmap"]
            pred_point = point_rendered
            mask = self.get_valid_mask(batch, key='valid_mask_depth', view=self.n_views)

            val = compute_epe(target_point, pred_point, mask).cpu().numpy()
            for v in eval_view_indices:
                metrics_dict[f'point_r_epe/{v}'] += val[:, v].mean() * b
            metrics_dict[f'point_r_epe/all'] += val[:, eval_view_indices].mean() * b

        # Depth render, always eval on two views
        if "depth" in batch["target"]:
            target_depth = batch["target"]["depth"]
            pred_depth = point_rendered[:, :, 2:3]
            mask = self.get_valid_mask_depth_eval(
                batch=batch,
                mask_key='valid_mask_depth',
                target_depth=target_depth,
                view=self.n_views).float()

            depth_metric_eval_keys = ['abs_rel']
            val_dict = compute_depth_eval(
                ground_truth=target_depth,
                predicted=pred_depth,
                depth_metric_eval_keys=depth_metric_eval_keys,
                mask=mask)
            val_dict = {key: val_dict[key].cpu().numpy() for key in val_dict}

            for key in depth_metric_eval_keys:
                # Accumulate per view.
                for v in eval_view_indices:
                    metrics_dict[f'depth_{key}/{v}'] += val_dict[key][:, v].mean() * b
                # Accumulate all.
                metrics_dict[f'depth_{key}/all'] += val_dict[key][:, eval_view_indices].mean() * b

        # Motion of rendered and gaussian's center
        if "motion3d" in batch["target"]:
            target_motion = batch["target"]["motion3d"]  # b v 3 h w
            pred_motion_guass = motion_gaussian  # b v 3 h w
            pred_motion_render = motion_rendered  # b v 3 h w

            mask = self.get_valid_mask(batch, key='valid_mask_motion', view=self.n_views)
            val_g = compute_epe(target_motion, pred_motion_guass, mask).cpu().numpy()
            val_r = compute_epe(target_motion, pred_motion_render, mask).cpu().numpy()

            for v in eval_view_indices:
                metrics_dict[f'motion_g_epe/{v}'] += val_g[:, v].mean() * b
                metrics_dict[f'motion_r_epe/{v}'] += val_r[:, v].mean() * b
                
            metrics_dict[f'motion_g_epe/all'] += val_g[:, eval_view_indices].mean() * b
            metrics_dict[f'motion_r_epe/all'] += val_r[:, eval_view_indices].mean() * b

        # Focal length eval.
        focal_length_abs_err = abs(batch["target"]["intrinsics"][:, :, 0, 0] - batch["pred"]["intrinsics"][:, :, 0, 0]).mean()
        metrics_dict[f'focal_abs'] += focal_length_abs_err.cpu().numpy() * b

        return b


    def on_test_end(self) -> None:
        # Test end
        for n in self.list_dataloader_key:
            if self.metrics_count[n] == 0:
                continue
            if self.global_rank == 0:
                average_metrics = {}
                for k, v in self.metrics_accumulator[n].items():
                    average_metrics[k] = v / self.metrics_count[n]
                    self.logger.experiment.add_scalar(f"test/{n}/{k}", average_metrics[k])

                metric_list = [f'{k} {v:.3f}' for k, v in average_metrics.items() if k not in self.key_to_skip_print]
                metric_str = '; '.join(metric_list)
                print(
                    f"[{n}] {metric_str}"
                )
            # Free up the recorded validation metrics.
            self.init_metrics(n)


    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        # Render Gaussians.
        visualization_dump = {}
        encoder_dict = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )

        # Postprocess.
        batch = self.postprocess(batch, encoder_dict)

        # Rendering camera model
        render_intrinsics = batch["pred"]["intrinsics"]
        render_extrinsics = batch["pred"]["extrinsics"]

        # Preparing gaussians
        (render_gaussian, render_intrinsics, render_extrinsics) = self.prepare_rendering(
            batch["pred"]["gaussians"],
            render_intrinsics,
            render_extrinsics,
            intp_type='standard')

        _, _, _, h, w = batch["context"]["image"].shape
        output = self.decoder.forward(
            gaussians=render_gaussian,
            extrinsics=render_extrinsics,
            intrinsics=render_intrinsics,
            near=batch["target"]["near"],
            far=batch["target"]["far"],
            image_shape=(h, w),
        )

        # Get metrics.
        dataset_key = batch["dataset_alias"][0]
        eval_count = self.get_metrics(
            batch=batch,
            gaussians=batch["pred"]["gaussians"],
            output=output,
            metrics_dict=self.metrics_accumulator[dataset_key],
            dataset_key=dataset_key,
            mode='val')
        self.metrics_count[dataset_key] += eval_count

        if batch_idx <= 32 and batch_idx % 4 == 0:
            self.visualize_output(
                batch=batch,
                visualization_dump=visualization_dump,
                output=output,
                mode='val',
                batch_idx=batch_idx)


    @rank_zero_only
    def on_validation_epoch_end(self):

        for n in self.list_dataloader_key:
            if self.metrics_count[n] == 0:
                continue
            if self.global_rank == 0:
                average_metrics = {}
                for k, v in self.metrics_accumulator[n].items():
                    average_metrics[k] = v / self.metrics_count[n]
                    self.log(f"val/{n}/{k}", average_metrics[k])

                metric_list = [f'{k} {v:.3f}' for k, v in average_metrics.items() if k not in self.key_to_skip_print]
                metric_str = '; '.join(metric_list)
                print(
                    f"[{n}] validation step {self.global_step}; {metric_str}"
                )
            # Free up the recorded validation metrics.
            self.init_metrics(n)


    @rank_zero_only
    @torch.no_grad()
    def visualize_output(self, batch, visualization_dump, output, mode, batch_idx) -> None:

        b, n_view, c, context_h, context_w = batch["context"]["image"].shape
        _, n_view_render, _, target_h, target_w = batch["target"]["image"].shape
        depth_diff_viz_max = 1.0

        if (context_h != target_h) or (context_w != target_w):
            resize_output = True
        else:
            resize_output = False

        # Prep for visualization, (b, v, c, h, w)
        assert visualization_dump["means"].shape[4] == 1
        point_gaussian = rearrange(
            visualization_dump["means"],
            "b v h w s xyz -> b (v s) xyz h w",
            v=n_view, h=context_h, w=context_w)  # s = 1
        if self.test_cfg.use_point_for_eval:
            point_rendered = point_gaussian
        else:
            point_rendered = output.point_rendered
        motion_gaussian = rearrange(
            visualization_dump["motion"],
            "b v h w xyz -> b v xyz h w",
            v=n_view, h=context_h, w=context_w)
        if self.test_cfg.use_point_for_eval:
            motion_rendered = motion_gaussian
        else:
            motion_rendered = output.motion_rendered
        image_rendered = output.color 

        if resize_output:
            point_gaussian = self.interpolate_output(point_gaussian, target_h, target_w)
            point_rendered = self.interpolate_output(point_rendered, target_h, target_w)
            motion_gaussian = self.interpolate_output(motion_gaussian, target_h, target_w)
            motion_rendered = self.interpolate_output(motion_rendered, target_h, target_w)
            image_rendered = self.interpolate_output(image_rendered, target_h, target_w)

        # Eval depth and motion on the camera coordinate.
        dataset_key = batch["dataset_alias"][0]
        if dataset_key in ['kt']:
            (point_gaussian,
            point_rendered,
            motion_gaussian,
            motion_rendered) = self.transform_canonical_to_cam(
                batch["pred"]["extrinsics"],
                point_gaussian,
                point_rendered,
                motion_gaussian,
                motion_rendered)

        if self.test_cfg.median_scale_eval:
            use_first_frame_for_median_scale = (dataset_key in ['kt'])
            (point_gaussian,
            point_rendered,
            motion_gaussian,
            motion_rendered) = self.align_scale_of_prediction_to_target(
                batch=batch,
                target=batch["target"]["pointmap"],
                pred=point_rendered,
                use_first_frame_for_median_scale=use_first_frame_for_median_scale,
                output_list=[point_gaussian, point_rendered, motion_gaussian, motion_rendered]
            )

        # Construct comparison image.
        target_img = batch["target"]["image"][0]  # v, 3, h, w
        rendered_img = image_rendered[0]  # v, 3, h, w
        img_diff = torch.clamp(torch.abs(target_img - rendered_img).sum(dim=1, keepdim=True) * 3, 0., 1.)
        img_diff = img_diff.repeat(1, 3, 1, 1)

        img_comparison = hcat(
            add_label(vcat(*target_img), "GT image"),
            add_label(vcat(*rendered_img), "Rendered image"),
            add_label(vcat(*img_diff), "Diff"),
        )
        
        # Visualzation of the depth of Gaussian's center
        pred_depth_rendered = point_rendered[0, :, 2]  # v, h, w
        pred_depth_rendered_viz = vis_depth_map(pred_depth_rendered)

        if "depth" in batch["target"]:
            target_depth = batch["target"]["depth"][0, :, 0]  # v, h, w
            target_depth_viz = vis_depth_map(target_depth)
            depth_rendered_diff = torch.clamp(
                torch.abs(target_depth-pred_depth_rendered)/depth_diff_viz_max, 0., 1.)
            depth_rendered_diff = depth_rendered_diff.unsqueeze(1).repeat(1, 3, 1, 1)

            # Masking if valid_mask_depth exists.
            if 'valid_mask_depth' in batch["target"]:
                mask = batch["target"]["valid_mask_depth"][0].repeat(1, 3, 1, 1).bool()  # 1, 3, h, w
                target_depth_viz[~mask] = 0.
                depth_rendered_diff[~mask] = 0.

            depth_comparison = hcat(
                add_label(vcat(*target_depth_viz), "GT depth"),
                add_label(vcat(*pred_depth_rendered_viz), "Point depth rendered"),
                add_label(vcat(*depth_rendered_diff), "Point depth rendered diff"),
            )
        else:
            depth_comparison = hcat(
                add_label(vcat(*pred_depth_rendered_viz), "Point depth rendered"),
            )

        # Gaussian visualization.
        scale_map = visualization_dump["scales"][0, :, :, :, 0, 0, :] / 0.3  # (v, h, w, 3), [0, 0.3]
        scale_map_viz = scale_map.permute(0, 3, 1, 2)
        opacities_map = visualization_dump["opacities"][0, :, :, :, 0, 0]  # (v, h, w), [0, 1]
        opacities_map_viz = vis_normalized_map(opacities_map, min_value=0., max_value=1.)

        # motion visualization. 3D scene flow in optical flow viz
        motion_rendered_viz = vis_3dmotion_map(
            motion_rendered[0, :n_view],
            point_rendered[0, :n_view, 2:3],
            batch["target"]["intrinsics"][0,:n_view])  # v, 3, h, w

        if "motion3d" in batch["target"]:
            target_motion = batch["target"]["motion3d"][0,:n_view]
            target_motion_viz = vis_3dmotion_map(
                target_motion,
                batch["target"]["depth"][0,:n_view],
                batch["target"]["intrinsics"][0,:n_view])  # v, 3, h, w

            epe_map_rendered = torch.sqrt(torch.sum((target_motion - motion_rendered[0, :n_view]) ** 2, dim=1))
            motion_rendered_diff_viz = torch.clamp(epe_map_rendered * 10, 0., 1.).unsqueeze(1).repeat(1, 3, 1, 1)
            
            # Display only views to predict.
            target_motion_viz = target_motion_viz[:n_view]
            motion_rendered_viz = motion_rendered_viz[:n_view]
            motion_rendered_diff_viz = motion_rendered_diff_viz[:n_view]

            scale_map_viz = scale_map_viz[:n_view]
            opacities_map_viz = opacities_map_viz[:n_view]

            # Masking if valid_mask_motion exists.
            if 'valid_mask_motion' in batch["target"]:
                mask = batch["target"]["valid_mask_motion"][0, :n_view].repeat(1, 3, 1, 1).bool()  # 1, 3, h, w
                target_motion_viz[~mask] = 0.
                motion_rendered_diff_viz[~mask] = 0.

            motion_comparison = hcat(
                add_label(vcat(*target_motion_viz), "GT motion"),
                add_label(vcat(*motion_rendered_viz), "Pred motion rendered"),
                add_label(vcat(*motion_rendered_diff_viz), "Motion rendered diff"),
                # add_label(vcat(*scale_map_viz), "Gaussian scale"),
                # add_label(vcat(*opacities_map_viz), "Gaussian opacity"),
            )
        else:
            motion_rendered_viz = motion_rendered_viz[:1]
            motion_comparison = hcat(
                add_label(vcat(*pred_motion_viz), "Pred motion"),
                # add_label(vcat(*scale_map_viz), "Gaussian scale"),
                # add_label(vcat(*opacities_map_viz), "Gaussian opacity"),
            )

        viz_image_list = [img_comparison, depth_comparison, motion_comparison] # warp_comparison
        viz_image_list = [prep_image(add_border(img)) for img in viz_image_list]

        logging_mode = get_cfg()['logging']['mode']
        assert logging_mode in ['wandb', 'tensorboard', 'local']
        if logging_mode in ['local', 'wandb']:
            self.logger.log_image(
                "comparison",
                viz_image_list,
                step=self.global_step,
                mode=mode,
                batch_idx=batch_idx,
                caption=batch["scene"],
            )
        elif logging_mode == 'tensorboard':
            prefix_dict = {0: 'image', 1: 'point_depth', 2: 'motion', 3: 'warp'}
            for index, image in enumerate(viz_image_list):
                image = (torch.tensor(image) / 255.).permute(2, 0, 1)  # c, h, w
                # image = vutils.make_grid(image, normalize=False)
                self.logger.experiment.add_image(
                    f'{mode}_{dataset_key}_b{batch_idx}_{prefix_dict[index]}',
                    image,
                    self.global_step)
            
            # Pointcloud logging
            def tensorboard_vis_point3d(image, point, suffix):
                # for b in range(image.shape[0]):
                b = 0
                image_b = rearrange(image[b], "v c h w -> (v h w) c")
                point_b = rearrange(point[b], "v c h w -> (v h w) c")
                assert image_b.shape[-1] == 3, f'{image_b.shape}'
                assert point_b.shape[-1] == 3, f'{point_b.shape}'
                mask = (point_b[:, -1] > 0.1) & (point_b[:, -1] < 20.)
                self.logger.experiment.add_mesh(
                    f'{mode}_{dataset_key}_b{batch_idx}_{b}_point3d_{suffix}',
                    vertices=point_b[mask][None, ...] * -1,
                    colors=(image_b[mask][None, ...] * 255).to(torch.uint8),
                    global_step=self.global_step)
            
            def tensorboard_vis_point3d_overlay(target, pred, sampling_ratio):
                # for b in range(pred.shape[0]):
                b = 0
                pred_point = rearrange(pred[b], "v c h w -> (v h w) c")
                target_point = rearrange(target[b], "v c h w -> (v h w) c")
                mask = (target_point[:, -1] > 0.05) * (target_point[:, -1] < 80.)
                target_point = target_point[mask]

                def get_color(point, color_dim):
                    assert len(point.shape) == 2, f'{point.shape}'
                    ones = torch.ones_like(point)[:, 0]
                    color = torch.zeros_like(point)
                    color[:, color_dim] = ones
                    return color
                
                pred_color = get_color(pred_point, color_dim=2)
                target_color = get_color(target_point, color_dim=0)

                def sample_rand(p, c, ratio=0.1):
                    n_total = p.shape[0]
                    n_select = int(n_total * ratio)
                    random_indices = torch.randperm(n_total)[:n_select]
                    return p[random_indices], c[random_indices]
                pred_point, pred_color = sample_rand(pred_point, pred_color, sampling_ratio)

                points = torch.cat([pred_point, target_point], dim=0)
                colors = torch.cat([pred_color, target_color], dim=0)

                point_size_config = {
                    'cls': 'PointsMaterial',
                    'size': 0.05
                }

                self.logger.experiment.add_mesh(
                    f'{mode}_{dataset_key}_b{batch_idx}_{b}_point3d_comparision',
                    vertices=points[None, ...] * -1,
                    colors=(colors[None, ...] * 255).to(torch.uint8),
                    config_dict={"material": point_size_config},
                    global_step=self.global_step)

            if mode in ['val', 'test']:
                tensorboard_vis_point3d_overlay(
                    batch["target"]["pointmap"],
                    point_rendered,
                    sampling_ratio=0.05)

    def configure_optimizers(self):
        new_params_key = [
            'gaussian_param_head',
            'motion_head',
            'downstream_head',
            'pose_head',
            'intrinsic_encoder']
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if any(key in name for key in new_params_key):
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
