"""Camera related utils."""

import cv2
import copy
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

inf = float("inf")


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)
        ], dim=1)


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):
    # [1, 4, 4], [N, 4, 4]

    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32, device=pivotal_pose.device)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)

    return poses


####### Pose update from delta

def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(cam_trans_delta: Float[Tensor, "batch 3"],
                cam_rot_delta: Float[Tensor, "batch 3"],
                extrinsics: Float[Tensor, "batch 4 4"],
                # original_rot: Float[Tensor, "batch 3 3"],
                # original_trans: Float[Tensor, "batch 3"],
                # converged_threshold: float = 1e-4
                ):
    # extrinsics is c2w, here we need w2c as input, so we need to invert it
    bs = cam_trans_delta.shape[0]

    tau = torch.cat([cam_trans_delta, cam_rot_delta], dim=-1)
    T_w2c = extrinsics.inverse()

    new_w2c_list = []
    for i in range(bs):
        new_w2c = SE3_exp(tau[i]) @ T_w2c[i]
        new_w2c_list.append(new_w2c)

    new_w2c = torch.stack(new_w2c_list, dim=0)
    return new_w2c.inverse()

    # converged = tau.norm() < converged_threshold
    # camera.update_RT(new_R, new_T)
    #
    # camera.cam_rot_delta.data.fill_(0)
    # camera.cam_trans_delta.data.fill_(0)
    # return converged


#######  Pose estimation
def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def get_pnp_pose(pts3d, opacity, K, H, W, opacity_threshold=0.3):
    pixels = np.mgrid[:W, :H].T.astype(np.float32)
    pts3d = pts3d.cpu().numpy()
    opacity = opacity.cpu().numpy()
    K = K.cpu().numpy()

    K[0, :] = K[0, :] * W
    K[1, :] = K[1, :] * H

    mask = opacity > opacity_threshold

    res = cv2.solvePnPRansac(pts3d[mask], pixels[mask], K, None,
                             iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
    success, R, T, inliers = res

    assert success

    R = cv2.Rodrigues(R)[0]  # world to cam
    pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world

    return torch.from_numpy(pose.astype(np.float32))


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


####### Camera intrinsics

def adjust_intrinsics_with_crop(intrinsics, orig_h, orig_w, crop_h, crop_w, normed):

    assert len(intrinsics.shape) == 3

    intrinsics = copy.deepcopy(intrinsics)
    
    if normed:
        intrinsics[:, 0, 0] *= orig_w / crop_w
        intrinsics[:, 1, 1] *= orig_h / crop_h
    else:
        h_diff = orig_h - crop_h
        w_diff = orig_w - crop_w
        intrinsics[:, 0, 2] -= (w_diff / 2.)
        intrinsics[:, 1, 2] -= (h_diff / 2.)

    return intrinsics


def adjust_intrinsics_with_resize(intrinsics, orig_h, orig_w, resize_h, resize_w, normed):

    assert len(intrinsics.shape) == 3
    if normed:
        return intrinsics
    else:
        raise ValueError('Not implemented yet.')


def normalize_intrinsics(
    intrinsics: np.ndarray | torch.Tensor, # v, 3, 3
    h,
    w):
    
    assert len(intrinsics.shape) == 3, f'{intrinsics.shape=}'

    intrinsics = copy.deepcopy(intrinsics)
    intrinsics[:, 0, :3] /= w
    intrinsics[:, 1, :3] /= h
    return intrinsics


def unnormalize_intrinsics(
    intrinsics: np.ndarray | torch.Tensor, # v, 3, 3
    h,
    w):

    assert len(intrinsics.shape) == 3

    intrinsics = copy.deepcopy(intrinsics)
    intrinsics[:, 0, :3] *= w
    intrinsics[:, 1, :3] *= h
    return intrinsics


####### Quaternion utils

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def camera_to_pose_encoding(
    camera,
    pose_encoding_type="absT_quaR",
):
    """
    Inverse to pose_encoding_to_camera
    camera: opencv, cam2world
    """
    if pose_encoding_type == "absT_quaR":

        quaternion_R = matrix_to_quaternion(camera[:, :3, :3])

        pose_encoding = torch.cat([camera[:, :3, 3], quaternion_R], dim=-1)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR",
):
    """
    Args:
        pose_encoding: A tensor of shape `BxC`, containing a batch of
                        `B` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
    """

    if pose_encoding_type == "absT_quaR":

        abs_T = pose_encoding[:, :3]
        quaternion_R = pose_encoding[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    c2w_mats = torch.eye(4, 4).to(R.dtype).to(R.device)
    c2w_mats = c2w_mats[None].repeat(len(R), 1, 1)
    c2w_mats[:, :3, :3] = R
    c2w_mats[:, :3, 3] = abs_T

    return c2w_mats


def quaternion_conjugate(q):
    """Compute the conjugate of quaternion q (w, x, y, z)."""

    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return q_conj


def quaternion_multiply(q1, q2):
    """Multiply two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    q_vec = q[..., 1:]
    q_w = q[..., :1]

    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    v_rot = v + q_w * t + torch.cross(q_vec, t, dim=-1)
    return v_rot


def relative_pose_absT_quatR(t1, q1, t2, q2):
    """Compute the relative translation and quaternion between two poses."""

    q1_inv = quaternion_conjugate(q1)

    q_rel = quaternion_multiply(q1_inv, q2)

    delta_t = t2 - t1
    t_rel = rotate_vector(q1_inv, delta_t)
    return t_rel, q_rel