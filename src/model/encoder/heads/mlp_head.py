# --------------------------------------------------------
# MLP head implementation
# Downstream heads assume inputs of size B x 1 x C (where 1 is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
import torch.nn as nn
from .postprocess import postprocess
from ..backbone.croco.blocks import Mlp


class PosemoduleWithMLP(nn.Module):
    """ MLP module for pose."""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        drop=0,
        postprocess=None,
        ):
        super(PosemoduleWithMLP, self).__init__()

        self.postprocess = postprocess
        self.mlp = Mlp(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            drop=drop,
        )

    def forward(self, x):
        out = self.mlp(x)
        if self.postprocess:
            out = self.postprocess(out)
        return out


def create_pose_mlp_head(net, has_conf=False, out_nchan=7, hidden_size=768, mlp_ratio=4, postprocess_func=postprocess):
    """
    return MLP for given net params.
    """
    return PosemoduleWithMLP(
        in_features=hidden_size,
        hidden_features=int(hidden_size * mlp_ratio),
        out_features=out_nchan,
        drop=0,
        postprocess=postprocess_func,
        )