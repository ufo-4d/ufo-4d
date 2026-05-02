"""Utils for 2D/3D geometry."""

import einops
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange


def get_pixelgrid(b, h, w, use_numpy=True):
    '''
    Get 2D pixel grid given the shape of batch, height, and width.
    '''
    if use_numpy:
        grid_h = np.linspace(0.0, w - 1, w).reshape(1, 1, 1, w).repeat(h, axis=2)
        grid_v = np.linspace(0.0, h - 1, h).reshape(1, 1, h, 1).repeat(w, axis=3)
        ones = np.ones_like(grid_h)
        pixelgrid = np.concatenate([grid_h, grid_v, ones], axis=1)
    else:
        grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
        grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)
        ones = torch.ones_like(grid_h)
        pixelgrid = torch.cat((grid_h, grid_v, ones), dim=1).float()
    return pixelgrid


def pixelgrid2point(pixelgrid, depth, intrinsics, use_numpy=True):
    '''
    Given pixelgrid, depth, and intrinsics, get 3D point map.
    '''
    b, _, h, w = depth.shape

    if use_numpy:
        depth_mat = depth.reshape(b, 1, -1)
        pixel_mat = pixelgrid.reshape(b, 3, -1)
        pts_mat = np.matmul(np.linalg.inv(intrinsics), pixel_mat) * depth_mat
        pts = pts_mat.reshape(b, 3, h, w)
    else:
        depth_mat = depth.view(b, 1, -1)
        pixel_mat = pixelgrid.view(b, 3, -1)
        inverse_intrinsics = torch.inverse(intrinsics.cpu()).float().to(device=pixel_mat.device)
        pts_mat = torch.matmul(inverse_intrinsics, pixel_mat) * depth_mat
        pts = pts_mat.view(b, 3, h, w)
        
    assert pts.shape == (b, 3, h, w), f'{pts.shape=}'

    return pts


def pixel2point(intrinsics, depth, use_numpy=True):
    '''
    Given camera intrinsics and a depth map, get a pointmap.
    '''
    
    assert len(depth.shape) == 4, f"{depth.shape=}"
    assert intrinsics.shape[1:] == (3, 3), f"{intrinsics.shape=}"
    b, _, h, w = depth.shape
    assert intrinsics.shape[0] == b, f"{intrinsics.shape=} != {b=}"

    pixelgrid = get_pixelgrid(b, h, w, use_numpy=use_numpy)
    if not use_numpy:
        pixelgrid = pixelgrid.to(depth.device)
    pts = pixelgrid2point(pixelgrid, depth, intrinsics, use_numpy=use_numpy)

    return pts


def point2pixel(pts, intrinsics, use_numpy=True):
    '''
    Project 3D points into image space. pts is a pointmap: each coordinate (i, j)
    has a 3D point (X, Y, Z). After projection, (i, j) will have (x, y) which is a
    projected value of (X, Y, Z) with given intrinsics.
    '''
    if use_numpy:
        b, _, h, w = pts.shape
        proj_pts = np.matmul(intrinsics, pts.reshape(b, 3, -1))
        pixels_mat = (proj_pts / (proj_pts[:, 2:3, :] + 1e-8))[:, :2, :]
        pixels_mat = pixels_mat.reshape(b, 2, h, w)
    else:
        b, _, h, w = pts.size()
        proj_pts = torch.matmul(intrinsics.float(), pts.view(b, 3, -1))
        pixels_mat = proj_pts.div(proj_pts[:, 2:3, :] + 1e-8)[:, :2, :]
        pixels_mat = pixels_mat.view(b, 2, h, w)

    return pixels_mat


def sceneflow2opticalflow(sceneflow, depth, intrinsics, use_numpy=True):
    '''
    Convert scene flow into optical flow, given depth and intrinsics.
    '''
    assert len(depth.shape) == len(sceneflow.shape) == 4
    assert len(intrinsics.shape) == 3

    b, _, h, w = depth.shape
    pixel_src = get_pixelgrid(b, h, w, use_numpy=use_numpy)[:, :2, :, :]
    if not use_numpy:
        pixel_src = pixel_src.to(depth.device)
    points_src = pixel2point(intrinsics, depth, use_numpy=use_numpy)
    points_tgt = points_src + sceneflow
    pixel_tgt = point2pixel(points_tgt, intrinsics, use_numpy=use_numpy)

    return pixel_tgt - pixel_src


def get_3d_motion_vector(depth, intrinsics, point_traj_3d, use_numpy=True):
    """
    Get 3D motion vector from depth, intrinsics, and point trajectory 3D
    point_traj_3d consists of optical flow and depth at the 2nd image.

    Args:
        depth: (b, 1, h, w)
        intrinsics: (b, 3, 3)
        point_traj_3d: (b, 3, h, w)

    Returns:
        sceneflow: (b, 3, h, w)
    """

    assert len(depth.shape) == 4 and len(point_traj_3d.shape) == 4, f"{depth.shape=} and {point_traj_3d.shape=}"
    assert depth[:, 0].shape == point_traj_3d[:, 0].shape, f"{depth.shape=} and {point_traj_3d.shape=}"
    assert depth.shape[1] == 1 and point_traj_3d.shape[1] == 3, f"{depth.shape=} and {point_traj_3d.shape=}"
    
    # Compute source points
    points_src = pixel2point(intrinsics, depth, use_numpy=use_numpy)

    # Compute target points
    b, _, h, w = depth.shape
    optical_flow, depth_tgt = point_traj_3d[:, :2, ...], point_traj_3d[:, 2:, ...]

    pixelgrid_tgt = get_pixelgrid(b, h, w, use_numpy=use_numpy)
    if not use_numpy:
        pixelgrid_tgt = pixelgrid_tgt.to(depth.device)

    if use_numpy:
        pixelgrid_tgt = (
            pixelgrid_tgt 
            + np.concatenate([optical_flow, np.zeros((b, 1, h, w))], axis=1)
        )
    else:
        zero_tensor = torch.zeros((b, 1, h, w), device=optical_flow.device)
        pixelgrid_tgt = (
            pixelgrid_tgt 
            + torch.cat([optical_flow, zero_tensor], dim=1)
        )

    points_tgt = pixelgrid2point(pixelgrid_tgt, depth_tgt, intrinsics, use_numpy=use_numpy)

    sceneflow = points_tgt - points_src

    # # for sanity check
    # optical_flow_proj = sceneflow2opticalflow(sceneflow, depth, intrinsics, use_numpy=use_numpy)
    # err = ((optical_flow_proj - point_traj_3d[:, :2, :, :]) ** 2).mean()
    # print(err)

    return sceneflow


def my_grid_sample(inputs, grid):
    return F.grid_sample(inputs, grid, align_corners=True)


def get_grid(x):
    assert x.dim() == 4
    b, _, h, w = x.size()
    grid_H = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    grid_V = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, 1, h, w).to(device=x.device, dtype=x.dtype)
    grids = torch.cat([grid_H, grid_V], dim=1).requires_grad_(False)
    
    return grids


def backward_warp_flow(x, flow):
    assert x.dim() == 4
    assert flow.dim() == 4

    flo_list = []
    flo_w = flow[:, 0] * 2 / max(x.size(3) - 1, 1)
    flo_h = flow[:, 1] * 2 / max(x.size(2) - 1, 1)
    flo_list.append(flo_w)
    flo_list.append(flo_h)
    flow_for_grid = torch.stack(flo_list).transpose(0, 1)
    grid = torch.add(get_grid(x), flow_for_grid).permute(0, 2, 3, 1)
    x_warp = my_grid_sample(x, grid)

    mask = torch.ones_like(x, requires_grad=False)
    mask = my_grid_sample(mask, grid)
    mask = (mask > 0.999).to(dtype=x_warp.dtype)

    return x_warp * mask


def estimate_pointmap_scale(est_pointmap, target_pointmap, mask):

    assert est_pointmap.shape == target_pointmap.shape, f'{est_pointmap.shape} =/= {target_pointmap.shape}.'
    assert len(est_pointmap.shape) == 5, f'{est_pointmap.shape}'

    pred = rearrange(est_pointmap, "b v c h w -> b (v h w) c")
    # No upperbound. Unnormalized point can be over 80 meters.
    pred_mask = (0.1 < pred[:, :, -1])

    target = rearrange(target_pointmap, "b v c h w -> b (v h w) c")
    target_mask = rearrange(mask, "b v h w -> b (v h w)")

    def get_scale(point, valid_mask):
        assert point.shape[2] == 3, f'{point.shape=}'
        assert point.shape[:-1] == valid_mask.shape, f'{point.shape=} {valid_mask.shape=}'

        scales = []
        for b in range(point.shape[0]):
            point_norm = point[b].norm(dim=-1)
            # if mask_sum is 0, initialize the scale with 1.
            if mask[b].sum() == 0:
                point_scale = torch.ones_like(point_norm.mean())
            else:
                point_scale = point_norm[mask[b].bool()].mean()
            scales.append(point_scale)
        return torch.stack(scales)

    mask = target_mask * pred_mask
    scale_target = get_scale(target, mask)
    scale_pred = get_scale(pred, mask)

    return scale_pred, scale_target


def transform_points(pose, pts):
    '''
    Apply the 4x4 pose transform to the input point.
    '''
    assert pose.shape == (4, 4), f'{pose.shape=}'
    assert len(pts.shape) in [3, 4], f'{pts.shape=}'
    
    input_pts_shape = len(pts.shape)
    if input_pts_shape == 4:
        b, c, h, w = pts.shape
        assert c == 3
        pts = pts.reshape(b, c, h*w)
    elif input_pts_shape == 3:
        b, c, n = pts.shape

    pose = pose[None, ...].repeat(b, 1, 1)
    ones = torch.ones_like(pts)[:, :1, :]
    pts = torch.cat([pts, ones], dim=1)

    tformed_pts = (pose @ pts)[:, :3]
    if input_pts_shape == 4:
        return tformed_pts.reshape(b, c, h, w)
    elif input_pts_shape == 3:
        return tformed_pts.reshape(b, c, n)
    return None



