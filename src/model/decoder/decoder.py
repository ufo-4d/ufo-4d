"""Portions Copyright (c) [2025] [NoPoSplat]. Licensed under MIT."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..types import Gaussians, DynamicGaussians


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    point_rendered: Float[Tensor, "batch view 3 height width"] | None
    motion_rendered: Float[Tensor, "batch view 3 height width"] | None


T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians | DynamicGaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
    ) -> DecoderOutput:
        pass
