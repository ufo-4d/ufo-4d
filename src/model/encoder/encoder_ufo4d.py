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

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.types import BatchedExample, DataShim
from ..types import DynamicGaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg


inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderUFO4DCfg:
    name: Literal["ufo4d"]
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    num_surfaces: int

    # Head types
    gs_params_head_type: str
    motion_head_type: str
    pose_head_type: str = ""

    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    use_colors_precomp: bool = False


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderUFO4D(Encoder[EncoderUFO4DCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderUFO4DCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)
        self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)

        self.use_colors_precomp = cfg.use_colors_precomp
        # If using give colors for rendering, it shouldn't use spherical harmonics.
        # Change the config correctly.
        if self.use_colors_precomp:
            assert not cfg.gaussian_adapter.use_sh
        # If not using spherical harmonics, sh_degree should be 0.
        # i.e., direct estimation of color.
        if not cfg.gaussian_adapter.use_sh:
            assert cfg.gaussian_adapter.sh_degree == 0
        if cfg.gaussian_adapter.sh_degree == 0:
            assert not cfg.gaussian_adapter.use_sh

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity
        self.motion_dim = 3

        self.gs_params_head_type = cfg.gs_params_head_type
        self.motion_head_type = cfg.motion_head_type
        self.pose_head_type = cfg.pose_head_type
        
        # For xyz points
        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                           depth_mode=('exp', -inf, inf), conf_mode=None,)
        # For GaussianSplat
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        # For motion
        self.set_motion_head(cfg, cfg.motion_head_type)

        # For pose
        self.pose_dim = 7  # quaternion + translation
        self.set_pose_head(cfg, cfg.pose_head_type)


    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)


    def set_gs_params_head(self, cfg, head_type):
        assert head_type == 'dpt_gs_v1', f"unexpected {head_type=}"
        self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
        self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)


    def set_motion_head(self, cfg, head_type):
        assert head_type == 'dpt_m_v1', f"unexpected {head_type=}"
        self.motion_head = head_factory(head_type, 'motion', self.backbone, has_conf=False, out_nchan=self.motion_dim)
        self.motion_head2 = head_factory(head_type, 'motion', self.backbone, has_conf=False, out_nchan=self.motion_dim)


    def set_pose_head(self, cfg, head_type):
        assert head_type == 'mlp_p_v1', f"unexpected {head_type=}"
        self.pose_head = head_factory(head_type, 'pose', self.backbone, has_conf=False, out_nchan=self.pose_dim)


    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> DynamicGaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True)

        # Pose token
        dec1_pose_token = dec1[-1][:, -1:]  # At last feature, last token
        del dec1_pose_token  # dec1_pose_token not used. Always the identity matrix.
        dec2_pose_token = dec2[-1][:, -1:]  # At last feature, last token
        dec1 = [tok[:, :-1] for tok in dec1]
        dec2 = [tok[:, :-1] for tok in dec2]

        with torch.amp.autocast(device_type='cuda', enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

            if self.gs_params_head_type in ['dpt_gs_v1']:
                GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3], shape1[0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
                GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
            else:
                raise ValueError(f"not supported headtype {self.gs_params_head_type}")

            # Two view estimation
            if self.motion_head_type == 'dpt_m_v1':
                motion_res1 = self.motion_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
                motion_res1 = rearrange(motion_res1, "b d h w -> b (h w) d")
                motion_res2 = self.motion_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
                motion_res2 = rearrange(motion_res2, "b d h w -> b (h w) d")
            else:
                raise ValueError(f"not supported headtype {self.gs_params_head_type}")

            # Pose head
            if self.pose_head_type == 'mlp_p_v1':
                pose_res2 = self.pose_head(dec2_pose_token)[:, 0, :]  # (b, 7)
            else:
                raise ValueError(f'Invalid pose {self.pose_head_type}.')

        # Merge results
        twoview_mode = ['dpt_gs_v1']

        if self.gs_params_head_type in twoview_mode:
            pts3d1 = rearrange(res1['pts3d'], "b h w d -> b (h w) d")
            pts3d2 = rearrange(res2['pts3d'], "b h w d -> b (h w) d")
            pts_all = torch.stack([pts3d1, pts3d2], dim=1)
            pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces
            depths = pts_all[..., -1].unsqueeze(-1)
            gaussians = torch.stack([GS_res1, GS_res2], dim=1)
            gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
            densities = gaussians[..., 0].sigmoid().unsqueeze(-1)
        else:
            raise ValueError(f'invalid header type {self.gs_params_head_type}')

        # Convert the features and depths into Gaussians.
        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            depths,
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
        )

        # Motion post-process
        if self.motion_head_type in ['dpt_m_v1']:
            motion = torch.stack([motion_res1, motion_res2], dim=1)
        else:
            raise ValueError(f'invalid motion head type {self.motion_head_type}')

        # Save the input rgbs to the harmonics variables.
        if self.use_colors_precomp:
            b, v, r, srf, spp, xyz = gaussians.means.shape
            assert context["image"].shape[2] == 3
            gaussians.harmonics = rearrange(
                context["image"], "b v (srf spp c d_sh) h w -> b v (h w) srf spp c d_sh",
                srf=1, spp=1, c=3, d_sh=1)

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v (h w) srf spp xyz -> b v h w srf spp xyz", h=h, w=w
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v (h w) srf spp xyzw -> b v h w srf spp xyzw", h=h, w=w
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            )
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump['motion'] = rearrange(
                motion, "b v (h w) xyz -> b v h w xyz", h=h, w=w
            )

        output_dynamic_gaussian = DynamicGaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            rearrange(
                motion,
                "b v r dim -> b (v r) dim",
            ),
        )

        return {
            'gaussians': output_dynamic_gaussian,
            'pose': pose_res2,  # Only output the pose for the second view
        }
         

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
