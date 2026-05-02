"""L1 loss class."""

from dataclasses import dataclass

from typing import Tuple
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians, DynamicGaussians
from .loss import Loss


@dataclass
class LossL1Cfg:
    weight: float


@dataclass
class LossL1CfgWrapper:
    l1: LossL1Cfg


class LossL1(Loss[LossL1Cfg, LossL1CfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | DynamicGaussians,
        global_step: int,
    ) -> Tuple[Tensor, Tensor]:
        delta = prediction.color - batch["target"]["image"]

        loss = abs(delta).mean()
        return self.cfg.weight * loss, loss
