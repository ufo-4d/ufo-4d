# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory
# --------------------------------------------------------
from .dpt_gs_head import create_gs_dpt_head
from .linear_head import LinearPts3d
from .dpt_head import create_dpt_head
from .mlp_head import create_pose_mlp_head
from .postprocess import postprocess_pose

def head_factory(head_type, output_mode, net, has_conf=False, out_nchan=3):
    """" build a prediction head for the decoder
    """
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf)
    elif head_type == 'dpt' and output_mode == 'pts3d':
        return create_dpt_head(net, has_conf=has_conf)
    elif head_type == 'dpt' and output_mode == 'gs_params':
        return create_dpt_head(net, has_conf=False, out_nchan=out_nchan, postprocess_func=None)
    # Motion head
    elif head_type in ['dpt_m_v1'] and output_mode == 'motion':
        # Initialized with the weights from the Point head.
        return create_dpt_head(net, has_conf=False, out_nchan=out_nchan, postprocess_func=None)
    # Gaussian Splatting head
    elif head_type in ['dpt_gs_v1'] and output_mode =="" 'gs_params':
        return create_gs_dpt_head(net, has_conf=has_conf, out_nchan=out_nchan, postprocess_func=None)
    elif head_type in ['mlp_p_v1'] and output_mode == 'pose':
        return create_pose_mlp_head(net, has_conf=False, out_nchan=out_nchan, postprocess_func=postprocess_pose)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
