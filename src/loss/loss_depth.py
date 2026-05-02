"""Depth loss class."""

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
class LossDepthCfg:
    weight: float
    scale_invariant: bool


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | DynamicGaussians,
        global_step: int,
    ) -> Tuple[Tensor, Tensor]:

        b, v, _, h, w = batch["target"]["depth"].shape
        target = batch["target"]["depth"][:, :, 0]
        
        # Rendered depth loss on two views
        rendered_depth = prediction.depth
        assert rendered_depth.size() == target.size()

        if self.cfg.scale_invariant:
            assert "pointmap_scale" in batch["target"]
            assert "pointmap_scale" in batch["pred"]
            target = target / batch["target"]["pointmap_scale"]
            rendered_depth = rendered_depth / batch["pred"]["pointmap_scale"]

        render_depth_loss = torch.abs(target - rendered_depth)

        if "valid_mask_depth" in batch["target"]:
            mask = batch["target"]["valid_mask_depth"][:, :, 0]
            render_depth_loss = self.masked_loss(pred=render_depth_loss, mask=mask)
        else:
            render_depth_loss = render_depth_loss.mean()
        loss = render_depth_loss

        return self.cfg.weight * loss, loss
