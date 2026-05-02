"""Utils for visualization."""

import torch

from src.visualization.color_map import apply_color_map_to_image
from src.visualization.motion import flow_to_png_middlebury
from src.misc.geometry import sceneflow2opticalflow, backward_warp_flow
from src.misc.cam_utils import unnormalize_intrinsics


def inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.mul(std).add(mean)


# Color-map the result.
def vis_normalized_map(result, min_value=0., max_value=1.):
    result = torch.clamp(result, min_value, max_value)
    result = 1 - (result - min_value) / (max_value - min_value)
    return apply_color_map_to_image(result, "turbo")


# Color-map the result.
def vis_depth_map(result, near_depth=1., far_depth=40.):
    far = torch.tensor(far_depth).log().to(device=result.device)
    near = torch.tensor(near_depth).log().to(device=result.device)
    result = torch.clamp(result, near_depth, far_depth)
    result = result.log()
    result = 1 - (result - near) / (far - near)

    return apply_color_map_to_image(result, "turbo")


def vis_3dmotion_map(sceneflow, depth, intrinsics):

    # unnormalize_intrinsics
    v, _, h, w = sceneflow.shape
    intrinsics = unnormalize_intrinsics(intrinsics, h, w)
    optical_flow = sceneflow2opticalflow(sceneflow, depth, intrinsics, use_numpy=False)
    results = []
    for flow in optical_flow:
        flow_np = flow.data.cpu().numpy()
        flow_viz = torch.from_numpy(flow_to_png_middlebury(flow_np))
        flow_viz = torch.clamp(flow_viz / 255., 0., 1.)
        results.append(flow_viz.permute(2, 0, 1))
    return torch.stack(results, dim=0).float()


def vis_warped_target_map(sceneflow, depth, intrinsics, source_image, target_image):

    # unnormalize_intrinsics
    v, _, h, w = sceneflow.shape
    intrinsics = unnormalize_intrinsics(intrinsics, h, w)
    optical_flow = sceneflow2opticalflow(sceneflow, depth, intrinsics, use_numpy=False)

    warped_target = backward_warp_flow(target_image, optical_flow)
    diff_image = torch.abs(warped_target - source_image)

    return warped_target, diff_image
