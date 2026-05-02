"""Loss helpers."""

from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_pointmap import LossPointmap, LossPointmapCfgWrapper
from .loss_pointmap_rendered import LossPointmapRendered, LossPointmapRenderedCfgWrapper
from .loss_motion import LossMotion, LossMotionCfgWrapper
from .loss_motion_rendered import LossMotionRendered, LossMotionRenderedCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_pose import LossPose, LossPoseCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_l1 import LossL1, LossL1CfgWrapper
from .loss_smooth import LossSmooth, LossSmoothCfgWrapper
from .loss_scale import LossScale, LossScaleCfgWrapper


LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossPointmapCfgWrapper: LossPointmap,
    LossPointmapRenderedCfgWrapper: LossPointmapRendered,
    LossMotionCfgWrapper: LossMotion,
    LossMotionRenderedCfgWrapper: LossMotionRendered,
    LossLpipsCfgWrapper: LossLpips,
    LossPoseCfgWrapper: LossPose,
    LossMseCfgWrapper: LossMse,
    LossL1CfgWrapper: LossL1,
    LossSmoothCfgWrapper: LossSmooth,
    LossScaleCfgWrapper: LossScale,
}

LossCfgWrapper = (
    LossDepthCfgWrapper | 
    LossLpipsCfgWrapper | 
    LossMseCfgWrapper | 
    LossMotionCfgWrapper | 
    LossPointmapCfgWrapper | 
    LossL1CfgWrapper | 
    LossPointmapRenderedCfgWrapper | 
    LossMotionRenderedCfgWrapper | 
    LossPoseCfgWrapper | 
    LossSmoothCfgWrapper | 
    LossScaleCfgWrapper
)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
