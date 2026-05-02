"""Loss class."""

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Generic, TypeVar, Tuple

from jaxtyping import Float
from torch import Tensor, nn
import torch

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians, DynamicGaussians

from einops import rearrange

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

        if self.name in ['point', 'motion', 'motion_rendered']:
            assert self.cfg.view_mode in ['single_view', 'two_view']

    @abstractmethod
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | DynamicGaussians,
        global_step: int,
    ) -> Tuple[Tensor, Tensor]:
        pass

    def masked_loss(
        self,
        pred,
        mask,
        drop_k=False,
    ):
        assert pred.shape == mask.shape, f"{pred.shape=} {mask.shape=}."
        assert len(mask.shape) == 5, f"{mask.shape=}"
        assert pred.shape[2] == 1, f"{pred.shape=}"

        if drop_k:
            b, v, c, h, w = pred.shape
            pred = rearrange(pred, "b v c h w -> b v (h w c)", h=h, w=w)
            mask = rearrange(mask, "b v c h w -> b v (h w c)", h=h, w=w)

            percentage = 0.07
            num_select = max(int(percentage * (h * w)), 1)

            val, _ = torch.topk(pred, k=num_select, dim=2)
            threshold = val[:, :, -1:]
            threshold_mask = pred < threshold

            mask = mask * threshold_mask

        return (pred * mask).sum() / (mask.sum() + 1e-8)
    
    def huber_loss(
        self,
        pred,
        target,
        mask,
        delta=1.0,
    ):
        assert pred.shape == target.shape, f"{pred.shape=} {target.shape=}."
        assert len(mask.shape) == 5, f"{mask.shape=}"

        b, v, c, h, w = pred.shape

        l1_loss = torch.abs(pred - target).mean(dim=2, keepdim=True)
        l2_loss = ((pred - target) ** 2).mean(dim=2, keepdim=True)

        delta_mask = (l1_loss < delta).detach().float()

        l2_mask = mask * delta_mask
        l1_mask = mask * (1 - delta_mask)

        loss_l2 = 0.5 * (l2_loss * l2_mask).sum() / (l2_mask.sum() + 1e-8)
        loss_l1 = (delta * (l1_loss - 0.5 * delta) * l1_mask).sum() / (l1_mask.sum() + 1e-8)

        return loss_l1 + loss_l2

