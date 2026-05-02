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

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from typing import Tuple

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians, DynamicGaussians
from .loss import Loss


@dataclass
class LossPointmapCfg:
    weight: float
    scale_invariant: bool
    scale_invariant_apply_after: int
    view_mode: str


@dataclass
class LossPointmapCfgWrapper:
    pointmap: LossPointmapCfg


class LossPointmap(Loss[LossPointmapCfg, LossPointmapCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | DynamicGaussians,
        global_step: int,
    ) -> Tuple[Tensor, Tensor]:

        target = batch["target"]["pointmap"]
        mask = batch["target"]["valid_mask_depth"]
        b, _, _, h, w = target.shape
        v_model = 1 if self.cfg.view_mode == 'single_view' else 2

        pred_points = rearrange(
            gaussians.means, "b (v h w) c -> b v c h w", v=v_model, h=h, w=w)

        # Scale invariant estimation
        if self.cfg.scale_invariant and self.cfg.scale_invariant_apply_after < global_step:
            assert "pointmap_scale" in batch["target"]
            assert "pointmap_scale" in batch["pred"]
            target = target / (batch["target"]["pointmap_scale"].reshape(b, 1, 1, 1, 1))
            pred_points = pred_points / (batch["pred"]["pointmap_scale"].reshape(b, 1, 1, 1, 1))

        target = target[:, :v_model]
        mask = mask[:, :v_model]
        assert pred_points.size() == target.size()  # b v c h w

        epe = torch.norm(pred_points - target, dim=2, keepdim=True)
        loss = self.masked_loss(pred=epe, mask=mask)

        return self.cfg.weight * loss, loss
