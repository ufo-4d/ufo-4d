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
from jaxtyping import Float
from torch import Tensor
from typing import Tuple
from einops import rearrange

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians, DynamicGaussians
from .loss import Loss


@dataclass
class LossMotionCfg:
    weight: float
    scale_invariant: bool
    scale_invariant_apply_after: int
    view_mode: str


@dataclass
class LossMotionCfgWrapper:
    motion: LossMotionCfg


class LossMotion(Loss[LossMotionCfg, LossMotionCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | DynamicGaussians,
        global_step: int,
    ) -> Tuple[Tensor, Tensor]:

        target = batch["target"]["motion3d"]
        mask = batch["target"]["valid_mask_motion"]
        b, _, _, h, w = target.size()
        v_model = 1 if self.cfg.view_mode == 'single_view' else 2

        assert gaussians.motions is not None
        motion = gaussians.motions  #  b (v g) xyz
        motion = rearrange(motion, "b (v h w) c -> b v c h w", v=v_model, h=h, w=w)

        # Scale invariant estimation
        if self.cfg.scale_invariant and self.cfg.scale_invariant_apply_after < global_step:
            assert "pointmap_scale" in batch["target"]
            assert "pointmap_scale" in batch["pred"]
            target = target / (batch["target"]["pointmap_scale"].reshape(b, 1, 1, 1, 1))
            motion = motion / (batch["pred"]["pointmap_scale"].reshape(b, 1, 1, 1, 1))

        target = target[:, :v_model]
        mask = mask[:, :v_model]
        assert motion.size() == target.size()  # b v c h w
    
        epe = torch.sqrt(((motion - target) ** 2).sum(2, keepdim=True))
        loss = self.masked_loss(pred=epe, mask=mask)

        return self.cfg.weight * loss, loss
