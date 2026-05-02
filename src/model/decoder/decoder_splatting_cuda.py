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
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians, DynamicGaussians
from .cuda_splatting import render_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool
    use_sh: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.use_sh = cfg.use_sh
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians | DynamicGaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        h, w = image_shape

        color, point_rendered, motion_rendered = render_cuda(
            extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
            intrinsics=rearrange(intrinsics, "b v i j -> (b v) i j"),
            near=rearrange(near, "b v -> (b v)"),
            far=rearrange(far, "b v -> (b v)"),
            image_shape=image_shape,
            background_color=repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            gaussian_means=rearrange(gaussians.means, "b (v g) xyz -> (b v) g xyz", v=v),
            gaussian_covariances=rearrange(gaussians.covariances, "b (v g) i j -> (b v) g i j", v=v),
            gaussian_sh_coefficients=rearrange(gaussians.harmonics, "b (v g) c d_sh -> (b v) g c d_sh", v=v),
            gaussian_opacities=rearrange(gaussians.opacities, "b (v g) -> (b v) g", v=v),
            gaussian_motions=rearrange(gaussians.motions, "b (v g) xyz -> (b v) g xyz", v=v),
            scale_invariant=self.make_scale_invariant,
            use_sh=self.use_sh,
            cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i") if cam_rot_delta is not None else None,
            cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i") if cam_trans_delta is not None else None,
        )

        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        point_rendered = rearrange(point_rendered, "(b v) c h w -> b v c h w", b=b, v=v)
        motion_rendered = rearrange(motion_rendered, "(b v) c h w -> b v c h w", b=b, v=v)
        
        return DecoderOutput(color, point_rendered, motion_rendered)
