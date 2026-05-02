# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# --------------------------------------------------------
import torch
from ....misc.cam_utils import standardize_quaternion

def postprocess(out, depth_mode, conf_mode):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))

    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    return res


def reg_dense_depth(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    # assert no_bounds

    if mode == 'range':
        xyz = xyz.sigmoid()
        xyz = (1 - xyz) * vmin + xyz * vmax
        return xyz

    if mode == 'linear':
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    if mode == 'exp_direct':
        xyz = xyz.expm1()
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == 'square':
        return xyz * d.square()

    if mode == 'exp':
        exp_d = d.expm1()
        if not no_bounds:
            exp_d = exp_d.clip(min=vmin, max=vmax)
        xyz = xyz * exp_d
        return xyz

    raise ValueError(f'bad {mode=}')


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'opacity':
        return x.sigmoid()
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')


def postprocess_pose(out):
    """
    extract pose from prediction head output
    """
    trans, quats = out[..., 0:3], out[..., 3:7]

    d = trans.norm(dim=-1, keepdim=True)
    scale = torch.expm1(d) / d.clip(min=1e-8)

    trans = trans * scale
    quats = standardize_quaternion(quats)
    return torch.cat([trans, quats], dim=-1)


def postprocess_pose_conf(out):
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,1
    return dict(pose_conf=torch.sigmoid(fmap))
