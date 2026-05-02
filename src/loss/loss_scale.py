"""Scale loss class."""

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
class LossScaleCfg:
    weight: float
    scale_invariant_apply_after: int


@dataclass
class LossScaleCfgWrapper:
    scale: LossScaleCfg


class LossScale(Loss[LossScaleCfg, LossScaleCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | DynamicGaussians,
        global_step: int,
    ) -> Tuple[Tensor, Tensor]:

        # Encourage the scale of predicted pointmap to be the same as target's scale.
        # It's for stabilizing the training. Without it, pointmap_scale goes to zero.
        loss = (batch["target"]["pointmap_scale"] - batch["pred"]["pointmap_scale"]).abs().mean()
        if self.cfg.scale_invariant_apply_after > global_step:
            loss = torch.zeros_like(loss)
        return self.cfg.weight * loss, loss
