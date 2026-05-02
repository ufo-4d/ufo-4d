"""Smoothness loss class."""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from typing import Tuple

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians, DynamicGaussians
from .loss import Loss


@dataclass
class LossSmoothCfg:
    weight_point: float
    weight_motion: float
    scale_invariant: bool
    scale_invariant_apply_after: int

@dataclass
class LossSmoothCfgWrapper:
    smooth: LossSmoothCfg


class LossSmooth(Loss[LossSmoothCfg, LossSmoothCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | DynamicGaussians,
        global_step: int,
    ) -> Tuple[Tensor, Tensor]:

        target = batch["target"]["pointmap"]
        b = target.shape[0]

        # Smoothness loss
        def get_smooth_loss(pred, image, is_norm):
            """Computes the edge-aware smoothness, guided by image
            """
            grad_norm_x = (
                pred[:, :, :, :, :-1] - pred[:, :, :, :, 1:]
                ).norm(2, dim=2, keepdim=True)
            grad_norm_y = (
                pred[:, :, :, :-1, :] - pred[:, :, :, 1:, :]
                ).norm(2, dim=2, keepdim=True)

            if is_norm:
                grad_norm_x = grad_norm_x / (pred[:, :, 2:3, :, :-1] + pred[:, :, 2:3, :, 1:])
                grad_norm_y = grad_norm_y / (pred[:, :, 2:3, :-1, :] + pred[:, :, 2:3, 1:, :])

            grad_img_x = torch.mean(
                torch.abs(image[:, :, :, :, :-1] - image[:, :, :, :, 1:]), 2, keepdim=True)
            grad_img_y = torch.mean(
                torch.abs(image[:, :, :, :-1, :] - image[:, :, :, 1:, :]), 2, keepdim=True)

            grad_norm_x = grad_norm_x * torch.exp(-grad_img_x)
            grad_norm_y = grad_norm_y * torch.exp(-grad_img_y)

            def get_finite_mean(val):
                finite_mask = torch.isfinite(val)
                finite_val = val[finite_mask]
                if finite_val.numel() > 0:
                    return torch.mean(finite_val)
                else:
                    return torch.tensor(0.0, device=val.device)
            return get_finite_mean(grad_norm_x) + get_finite_mean(grad_norm_y)
    

        pred_points = prediction.point_rendered
        pred_motion = prediction.motion_rendered

        if self.cfg.scale_invariant and self.cfg.scale_invariant_apply_after < global_step:
            assert "pointmap_scale" in batch["pred"]
            pred_points = pred_points / (batch["pred"]["pointmap_scale"].reshape(b, 1, 1, 1, 1).detach())
            pred_motion = pred_motion / (batch["pred"]["pointmap_scale"].reshape(b, 1, 1, 1, 1).detach())

        # Smoothness on rendered point, both view
        loss_point = get_smooth_loss(
            pred=pred_points,
            image=batch["target"]["image"],
            is_norm=True)

        # Smoothness on motion.
        loss_motion = get_smooth_loss(
            pred=pred_motion,
            image=batch["target"]["image"],
            is_norm=False)

        loss = loss_point + loss_motion
        weighted_loss = self.cfg.weight_point * loss_point + self.cfg.weight_motion * loss_motion
        
        return weighted_loss, loss
