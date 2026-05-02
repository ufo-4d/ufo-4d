"""Pose loss class."""

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
class LossPoseCfg:
    weight: float
    scale_invariant: bool
    scale_invariant_apply_after: int


@dataclass
class LossPoseCfgWrapper:
    pose: LossPoseCfg


class LossPose(Loss[LossPoseCfg, LossPoseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | DynamicGaussians,
        global_step: int,
    ) -> Tuple[Tensor, Tensor]:

        assert "extrinsics_quat" in batch["pred"], 'extrinsics_quat is not found in pred batch.'
        
        # (b, 7), taking the 2nd view's pose. 1st view's pose is always identity.
        target = batch["target"]["extrinsics_quat"][:, 1, :]
        pred = batch["pred"]["extrinsics_quat"]  # (b, 7)

        target_trans = target[:, :3]
        target_quat = target[:, 3:]
        pred_trans = pred[:, :3]
        pred_quat = pred[:, 3:]

        if self.cfg.scale_invariant and self.cfg.scale_invariant_apply_after < global_step:
            b = target.shape[0]
            assert "pointmap_scale" in batch["target"]
            assert "pointmap_scale" in batch["pred"]
            target_trans = target_trans / (batch["target"]["pointmap_scale"].reshape(b, 1))
            pred_trans = pred_trans / (batch["pred"]["pointmap_scale"].reshape(b, 1))

        loss_trans = torch.norm((target_trans-pred_trans), dim=1).mean()
        loss_quat = torch.norm((target_quat-pred_quat), dim=1).mean()
        loss = loss_quat + loss_trans

        return self.cfg.weight * loss, loss
