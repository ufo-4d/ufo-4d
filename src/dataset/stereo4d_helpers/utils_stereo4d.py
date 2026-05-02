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

import numpy as np
import os.path as osp
from typing import Optional, List
import pickle
import mediapy as media
import math
import matplotlib.pyplot as plt

from src.misc.geometry import sceneflow2opticalflow, backward_warp_flow
from src.visualization.motion import flow_to_png_middlebury


class CameraAZ:
    def __init__(
        self,
        parallax_camera=None,
        sstable_camera=None,
        from_json=None,
        from_jaxcam=None,
        ):
        if parallax_camera is not None:
            self.parallax_camera = parallax_camera
            self._init_from_parallax_camera(parallax_camera)
        elif sstable_camera is not None:
            self.sstable_camera = sstable_camera
            self._init_from_sstable_camera(sstable_camera)
        elif from_json is not None:
            self.extr = from_json['extr']
            self.intr_normalized = from_json['intr_normalized']
        elif from_jaxcam is not None:
            self._init_from_jaxcam(from_jaxcam)
        else:
            raise NotImplementedError()

    def __str__(self):
        return f"extr: \n{self.extr}\n intr_normalized: \n{self.intr_normalized}"

    def _init_from_jaxcam(self, jax_camera):
        self.extr = np.asarray(jax_camera.world_to_camera_matrix[:3])
        self.intr_normalized = {
            'fx': (
                jax_camera.intrinsic_matrix[0][0] / jax_camera.image_size_x
            ).item(),
            'fy': (
                jax_camera.intrinsic_matrix[1][1] / jax_camera.image_size_y
            ).item(),
            'cx': (
                jax_camera.intrinsic_matrix[0][2] / jax_camera.image_size_x
            ).item(),
            'cy': (
                jax_camera.intrinsic_matrix[1][2] / jax_camera.image_size_y
            ).item(),
            'k1': 0,
            'k2': 0,
        }

    def _init_from_sstable_camera(self, sstable_camera):
        orientation_1 = np.reshape(np.asarray(sstable_camera.orientation.m), (3, 3))
        position_1 = np.reshape(np.asarray(sstable_camera.position.v), (3, 1))

        f_x = (1 / 2.0) / math.tan(
            math.radians(sstable_camera.field_of_view_x / 2.0)
        )
        f_y = (1 / 2.0) / math.tan(
            math.radians(sstable_camera.field_of_view_y / 2.0)
        )
        c2w = np.concatenate(
            (
                np.concatenate((orientation_1.T, position_1), axis=-1),
                np.array([[0, 0, 0, 1]]),
            ),
            axis=0,
        )
        self.extr = np.linalg.inv(c2w)[:3, :]
        self.intr_normalized = {
            'fx': f_x,
            'fy': f_y,
            'cx': sstable_camera.principal_point_x,
            'cy': sstable_camera.principal_point_y,
            'k1': 0,
            'k2': 0,
        }

    def to_json_format(self):
        return {
            'extr': self.extr,
            'intr_normalized': self.intr_normalized,
        }

    def get_w2c(self):
        return np.concatenate((self.extr, np.array([[0, 0, 0, 1]])), axis=0)

    def get_c2w(self):
        w2c = np.concatenate((self.extr, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        return c2w

    def get_hfov_deg(self):
        return math.degrees(2 * np.arctan(0.5 / self.intr_normalized['fx']))

    def _init_from_parallax_camera(self, line):
        # reference https://google.github.io/realestate10k/download.html
        for i in range(len(line)):
            line[i] = float(line[i])
        fx, fy, px, py, k1, k2 = line[:6]
        self.extr = np.array(line[6:]).reshape(3, 4)
        self.intr_normalized = {
            'fx': fx,
            'fy': fy,
            'cx': px,
            'cy': py,
            'k1': k1,
            'k2': k2,
        }

    def get_intri_matrix(self, imh, imw):
        return np.array([
            [self.intr_normalized['fx'] * imw, 0, self.intr_normalized['cx'] * imw],
            [0., self.intr_normalized['fy'] * imh, self.intr_normalized['cy'] * imh],
            [0., 0., 1.],
        ])

    def get_intri_matrix_norm(self):
        return np.array([
            [self.intr_normalized['fx'], 0, self.intr_normalized['cx']],
            [0., self.intr_normalized['fy'], self.intr_normalized['cy']],
            [0., 0., 1.],
        ])

    def pix_2_world_np(
        self,
        xy: np.ndarray,
        depth: np.ndarray,
        valid_depth_min: float,
        valid_depth_max: float,
        ):
        """unproject points from ndc from to world frame.

        depth: h x w xy definition:

            xy.shape [:, 2]
            left to right: [0, w]
            top to bottom: [0, h]
        """

        _, dim = xy.shape
        assert dim == 2
        imh, imw = depth.shape

        valid_mask = (
            (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < imw) & (xy[:, 1] < imh)
        )

        x_cam = (
            xy[..., 0] / imw - self.intr_normalized['cx']
        ) / self.intr_normalized['fx']
        y_cam = (
            xy[..., 1] / imh - self.intr_normalized['cy']
        ) / self.intr_normalized['fy']
        z_cam = np.ones_like(xy[..., 0])
        xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        x_query = np.clip(np.round(xy[:, 0]).astype(int), 0, imw - 1)
        y_query = np.clip(np.round(xy[:, 1]).astype(int), 0, imh - 1)
        depth_values = depth[y_query, x_query]

        valid_mask = (
            valid_mask
            & (depth_values > valid_depth_min)
            & (depth_values < valid_depth_max)
        )

        xyz_cam = depth_values[:, None] * xyz_cam

        xyz_world = (self.extr[:3, :3].T @ (xyz_cam - self.extr[:3, 3]).T).T
        return xyz_world, valid_mask

    def world_2_pix_np(
        self, xyz_world: np.ndarray, imh: int, imw: int, min_depth: float = 0.01
        ):
        """project points from world frame to screen space.

        xyz_world: [:, 3] array of points in world frame.
        """
        xyz_world_hom = np.concatenate(
            (xyz_world, np.ones_like(xyz_world[:, :1])), axis=-1
        )
        xyz_homo = (
            self.get_intri_matrix(imh, imw) @ self.extr @ xyz_world_hom.T
        ).T  # npt, 3
        depth = xyz_homo[:, 2]
        xy = xyz_homo[:, :2] / xyz_homo[:, 2:]
        valid_mask = (
            (xy[:, 0] >= 0.5)
            & (xy[:, 1] >= 0.5)
            & (xy[:, 0] < imw - 0.5)
            & (xy[:, 1] < imh - 0.5)
            & (depth > min_depth)
        )
        return xy, valid_mask, depth

    def world_2_cam_np(
        self, xyz_world: np.ndarray, imh: int, imw: int, min_depth: float = 0.01
        ):
        """Transform points from world frame to camera space.

        xyz_world: [:, 3] array of points in world frame.
        """
        xyz_world_hom = np.concatenate(
            (xyz_world, np.ones_like(xyz_world[:, :1])), axis=-1
        )
        xyz_homo = (
            self.extr @ xyz_world_hom.T
        ).T  # npt, 3?
        return xyz_homo



class Track3d:
    def __init__(
        self,
        tracks: Optional[np.ndarray] = None,
        visibles: Optional[np.ndarray] = None,
        depths: Optional[np.ndarray] = None,
        cameras: Optional[List[CameraAZ]] = None,
        video: Optional[np.ndarray] = None,
        query_points: Optional[np.ndarray] = None,
        valid_depth_min=0,
        valid_depth_max=20,
        load_from_json=None,
        track3d: Optional[np.ndarray] = None,
        visible_list: Optional[np.ndarray] = None,
        color_values: Optional[np.ndarray] = None,
        ):
        """tracks: npt x nframe x d2

        visibles: npt x nframe
        depths: nframe x imh x imw
        cameras: nframe x CameraAZ
        video: nframe x imh x imw
        query_points: npt x 3(t, h, w)
        valid_depth_min: float, min value for valid depth
        valid_depth_max: float, max value for valid depth
        """
        if load_from_json is not None:
            self._load_from_json(load_from_json)
        else:
            # sanity check
            if tracks is not None:
                npt, nframe, d2 = tracks.shape
                assert d2 == 2, 'tracks dimension should be 2'
                assert (npt, nframe) == visibles.shape, f'Wrong shape visibles.shape {visibles.shape}, expected {npt, nframe}'  # pytype: disable=attribute-error
                assert depths.shape[0] == nframe, (  # pytype: disable=attribute-error
                    f'Wrong shape depths.shape[0] {depths.shape[0]}, expected nframe'  # pytype: disable=attribute-error
                    f' {nframe}'
                )
                _, imh, imw = depths.shape  # pytype: disable=attribute-error
                assert (
                    len(cameras) == nframe
                ), f'Wrong shape cameras.shape {len(cameras)}, expected nframe {nframe}'
                assert (nframe, imh, imw, 3) == video.shape, f'Wrong shape video.shape {video.shape}, expected (nframe, imh, imw, 3) {nframe, imh, imw, 3}'  # pytype: disable=attribute-error
                if query_points is not None:
                    assert query_points.shape == (
                        npt,
                        3,
                    ), f'query_points should be npt x 3, got {query_points.shape}'

                # unproject track
                track3d = []
                visible_list = []
                for t in range(len(video)):
                    xyz_world, valid_mask = cameras[t].pix_2_world_np(
                        tracks[:, t],
                        depths[t],
                        valid_depth_min,
                        valid_depth_max,
                    )
                    visible_list.append(visibles[:, t] & valid_mask)
                    track3d.append(xyz_world)
                track3d = einops.rearrange(
                    np.stack(track3d, axis=0), 't npt d3->npt t d3'
                )
                visible_list = einops.rearrange(
                    np.stack(visible_list, axis=0), 't npt->npt t'
                )
            elif track3d is not None:
                npt, nframe = track3d.shape[:2]
                assert (
                    track3d.shape[2] == 3
                ), f'track3d should be npt x nframe x 3, got {track3d.shape}'
                assert (npt, nframe) == visible_list.shape, f'Wrong shape visible_list.shape {visible_list.shape}, expected {npt, nframe}'  # pytype: disable=attribute-error
                assert (
                    len(cameras) == nframe
                ), f'Wrong shape cameras.shape {len(cameras)}, expected nframe {nframe}'
                assert nframe == video.shape[0], f'Wrong shape video.shape[0] {video.shape[0]}, expected nframe {nframe}'  # pytype: disable=attribute-error
                _, imh, imw, _ = video.shape  # pytype: disable=attribute-error
            else:
                raise NotImplementedError
            if color_values is not None:
                pass
            elif query_points is not None:
                # get point color
                color_values = video[
                    query_points[:, 0],
                    query_points[:, 1].astype(int),
                    query_points[:, 2].astype(int),
                ]  # npt, 3
            else:
                color_values = None

            self.cameras = cameras
            self.track3d = track3d
            self.imh = imh
            self.imw = imw
            self.visible_list = visible_list
            self.color_values = color_values
            self.video = video

    def _load_from_json(self, load_from_json):
        if load_from_json['cameras'] is not None:
            self.cameras = [
                CameraAZ(from_json=camera)
                for camera in load_from_json['cameras']  # FIXME
                # for camera in load_from_json['camera']
            ]
        self.track3d = np.stack(load_from_json['track3d'], axis=0)
        self.imh = load_from_json['imh']
        self.imw = load_from_json['imw']
        self.visible_list = load_from_json['visible_list']
        self.color_values = load_from_json['color_values']
        self.video = load_from_json['video']

    def to_json_format(self, save_video=False, save_camera=True):
        return {
            'cameras': (
                [camera.to_json_format() for camera in self.cameras]
                if save_camera
                else None
            ),
            'track3d': self.track3d,
            'imh': self.imh,
            'imw': self.imw,
            'visible_list': self.visible_list,
            'color_values': self.color_values,
            'video': self.video if save_video else None,
        }

    def get_new_track(self, track_mask=None, percentage=None | float):
        if track_mask is None:
            track_mask = np.random.uniform(size=self.track3d.shape[0]) < percentage
        new_track = copy.deepcopy(self)
        new_track.track3d = new_track.track3d[track_mask]
        new_track.visible_list = new_track.visible_list[track_mask]
        if new_track.color_values is not None:
            new_track.color_values = new_track.color_values[track_mask]
        return new_track


def get_scene_motion_2d_displacement(
    track3d: Track3d,
    tracks_leave_trace=16,
    ):
    """Get 2D point trajectories of scene motion.

    Returns max 2D displacement of 3D points over tracks_leave_trace frames as if
    the camera is static. This measurement decouples camera motion.

    Args:
        track3d: An instance of Track3d containing 3D tracks and visibility info.
        tracks_leave_trace: Number of frames over which to compute displacement.

    Returns:
        displacement: A (npt, nframe) array of max 2D displacements over
        tracks_leave_trace frames.
    """
    all_points = track3d.track3d  # Shape: (npt, nframe, 3)
    npt, nframe, _ = all_points.shape
    displacement = np.zeros_like(track3d.visible_list, dtype=np.float32)
    for t in tqdm.tqdm(range(nframe)):
        s_start = max(0, t - tracks_leave_trace)
        s_end = t + 1  # Include current frame
        s_list = np.arange(s_start, s_end)  # Shape: (L,)
        L = len(s_list)
        if L < 2:
            # Not enough frames to compute displacement
            continue
        # Extract positions and visibilities for relevant frames
        positions = all_points[:, s_list, :]  # Shape: (npt, L, 3)
        visibilities = track3d.visible_list[:, s_list]  # Shape: (npt, L)
        # Flatten positions for projection
        positions_flat = positions.reshape(-1, 3)
        # Project all positions using the camera at frame t
        points_2d_flat, valid_mask_flat, _ = track3d.cameras[t].world_2_pix_np(
            positions_flat,
            track3d.imh,
            track3d.imw,
        )
        # Reshape back to (npt, L, 2)
        points_2d = points_2d_flat.reshape(npt, L, -1)
        valid_mask = valid_mask_flat.reshape(npt, L)

        # Extract positions and masks at time t
        points_2d_t = points_2d[:, -1, :]  # Shape: (npt, 2)
        valid_mask_t = valid_mask[:, -1]
        visibilities_t = visibilities[:, -1]

        # Compute displacements to previous frames
        deltas = (
            points_2d[:, :-1, :] - points_2d_t[:, None, :]
        )  # Shape: (npt, L-1, 2)
        distances = np.linalg.norm(deltas, axis=2)  # Shape: (npt, L-1)
        # print("distances: ", distances.shape)
        # Validity mask
        valid = (
            valid_mask[:, :-1]
            & valid_mask_t[:, None]
            & visibilities[:, :-1]
            & visibilities_t[:, None]
        )
        # Apply validity mask
        distances[~valid] = 0
        # Compute maximum displacement
        max_displacement = np.max(distances, axis=1)  # Shape: (npt,)
        displacement[:, t] = max_displacement
    return displacement


def plot_3d_tracks_plt(
    video: np.ndarray,
    track3d: Track3d,
    tracks_leave_trace=16,
    point_size: int = 10,
    ):
    """Visualize 2D point trajectories.

    The trail shows where the previous 3D point in the current camera frame. This
    visualization decouples camera motion
    """
    num_points, num_frames = track3d.track3d.shape[:2]
    figure_dpi = 64

    # Precompute colormap for points
    color_map = matplotlib.colormaps.get_cmap('hsv')
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

    # sort points by height of their first appearance
    point_colors = np.zeros((num_points, 3))
    for i in range(num_points):
        point_colors[i] = np.array(color_map(cmap_norm(i)))[:3]

    disp = []
    for t in range(num_frames):
        frame = video[t].copy()

        # Draw tracks on the frame
        all_points = track3d.track3d  # npt, nframe, xyz
        npt, nframe, _ = all_points.shape
        all_points = einops.rearrange(
            all_points, 'npt nframe xyz->(npt nframe) xyz'
        )

        points_at_frame, valid_mask, _ = track3d.cameras[t].world_2_pix_np(
            all_points,
            track3d.imh,
            track3d.imw,
        )
        points_at_frame = einops.rearrange(
            points_at_frame,
            '(npt nframe) xyz->nframe npt xyz',
            npt=npt,
            nframe=nframe,
        )
        valid_mask = einops.rearrange(
            valid_mask, '(npt nframe)-> npt nframe', npt=npt, nframe=nframe
        )
        valid_mask = valid_mask & track3d.visible_list
        valid_mask = valid_mask.transpose(1, 0)
        line_tracks = points_at_frame[max(0, t - tracks_leave_trace) : t + 1]
        line_visibles = valid_mask[max(0, t - tracks_leave_trace) : t + 1]
        fig = plt.figure(
            figsize=(frame.shape[1] / figure_dpi, frame.shape[0] / figure_dpi),
            dpi=figure_dpi,
            frameon=False,
            facecolor='w',
        )
        ax = fig.add_subplot()
        ax.axis('off')
        ax.imshow(frame / 255.0)

        for s in range(line_tracks.shape[0] - 1):
            # Collect lines and colors for the track
            visible_line_mask = (
                line_visibles[s] & line_visibles[s + 1] & line_visibles[-1]
            )
            pt1 = line_tracks[s, visible_line_mask]
            pt2 = line_tracks[s + 1, visible_line_mask]
            lines = np.concatenate([pt1, pt2], axis=1)
            lines = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in lines]
            c = point_colors[visible_line_mask]
            alpha = (s + 1) / (line_tracks.shape[0] - 1)
            c = np.concatenate([c, np.ones_like(c[..., :1]) * alpha], axis=1)
            lc = LineCollection(lines, colors=c, linewidths=1)
            ax.add_collection(lc)
        visibles_mask = valid_mask[t].astype(bool)
        colalpha = point_colors[visibles_mask]
        plt.scatter(
            points_at_frame[t, visibles_mask, 0],
            points_at_frame[t, visibles_mask, 1],
            s=point_size,
            c=colalpha,
        )

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]  # pytype: disable=attribute-error
        disp.append(np.copy(img))
        plt.close(fig)
        del fig, ax

    disp = np.stack(disp, axis=0)
    return disp


def load_rgbd_cam_from_pkl(video_id, root_dir, hfov=120, new_imw=1, new_imh=1):
    """load rgb, depth, and camera from webvis folder"""
    input_dict = {'left': {'camera': [], 'depth': [], 'video': []}}
    # Load camera
    cam_pkl = osp.join(root_dir, video_id, video_id + '-left_camera_c2w.pkl')
    with open(cam_pkl, 'rb') as f:
        cameras = pickle.load(f)
    cameras = np.concatenate(
        [cameras, np.tile(np.array([0, 0, 0, 1]), (len(cameras), 1, 1))], axis=1
    )
    nfr = len(cameras)
    input_dict['nfr'] = nfr
    for fid in range(nfr):
        intr_normalized = {
            'fx': (1 / 2.0) / math.tan(math.radians(hfov / 2.0)),
            'fy': (
                (1 / 2.0) / math.tan(math.radians(hfov / 2.0)) * new_imw / new_imh
            ),
            'cx': 0.5,
            'cy': 0.5,
            'k1': 0,
            'k2': 0,
        }
        input_dict['left']['camera'].append(
            CameraAZ(
                from_json={
                    'extr': np.linalg.inv(cameras[fid])[:3, :],
                    'intr_normalized': intr_normalized,
                }
            )
        )
    video_path = osp.join(
        root_dir,
        video_id,
        video_id + '-left_rectified.mp4',
    )
    rgbs = media.read_video(video_path)
    depth_path = osp.join(
        root_dir,
        video_id,
        video_id + '-left_depth.pkl',
    )
    with open(depth_path, 'rb') as f:
        depth = pickle.load(f)
    # with open(osp.join(root, vid, vid + '-flows_left_right.pkl'), 'rb') as f:
    #     raw_disparity = pickle.load(f)
    input_dict['left']['video'] = rgbs
    input_dict['left']['depth'] = depth
    # input_dict['left']['raw_disparity'] = raw_disparity
    return input_dict


def colored_depthmap(
    depth: np.ndarray,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    invalid_value: Optional[float] = None,
    colormap='Spectral',
    ) -> np.ndarray:
    """Converts a depth map to a colored image using a plasma colormap.

    Args:
        depth: The depth map (numpy array).
        d_min: Minimum depth value (float, optional). Defaults to None (minimum in
        depth).
        d_max: Maximum depth value (float, optional). Defaults to None (maximum in
        depth).

    Returns:
        The colored depth map as a numpy array of uint8 representing RGB channels.
    """
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    cmap = getattr(plt.cm, colormap, None)
    if cmap is not None:
        depth_colored = 255 * cmap(depth_relative)[:, :, :3]  # H, W, C
    else:
        raise ValueError(f"Colormap '{colormap}' is not recognized")

    depth_colored = depth_colored.astype(np.uint8)
    depth_colored[depth == invalid_value] = 0
    return depth_colored

def plot_depth_prism(depthmaps):
    depthmaps[depthmaps == 0] = np.nan

    vmin = np.nanpercentile(depthmaps, 5)
    vmax = np.nanpercentile(depthmaps, 95)
    colored_d = np.stack([colored_depthmap(1/d, 1/vmax, 1/vmin, np.nan, 'turbo') for d in depthmaps], axis=0)
    return colored_d

def draw_camera_pose(M, ax, al=1, fov=60, aspect=1.0, near=0.0, far=0.1):
    """
    Draws the camera pose with axes and frustum in the 3D plot.

    Parameters:
    - M: (4x4 array) The camera's extrinsic matrix.
    - ax: The matplotlib 3D axis to draw on.
    - al: (float) The length scaling factor for the axes.
    - orig_color: (str) The color of the camera origin point.
    - fov: (float) Field of view in degrees.
    - aspect: (float) Aspect ratio of the camera (width / height).
    - near: (float) Near clipping plane distance.
    - far: (float) Far clipping plane distance.
    """
    camera_orig = M[:3, 3]
    rot_mat = M[:3, :3]
    # Draw frustum
    fov_rad = np.deg2rad(fov / 2)
    h_near = 2 * np.tan(fov_rad) * near
    w_near = h_near * aspect
    h_far = 2 * np.tan(fov_rad) * far
    w_far = h_far * aspect

    # Define frustum corners in camera space
    near_corners = np.array([
        [ w_near / 2,  h_near / 2, near],
        [-w_near / 2,  h_near / 2, near],
        [-w_near / 2, -h_near / 2, near],
        [ w_near / 2, -h_near / 2, near],
    ])

    far_corners = np.array([
        [ w_far / 2,  h_far / 2, far],
        [-w_far / 2,  h_far / 2, far],
        [-w_far / 2, -h_far / 2, far],
        [ w_far / 2, -h_far / 2, far],
    ])

    # Combine near and far corners
    frustum_corners = np.vstack((near_corners, far_corners))

    # Transform corners to world space
    frustum_corners_world = (rot_mat @ frustum_corners.T).T + camera_orig

    # Define lines to draw the frustum edges
    frustum_lines = [
        # Lines from camera origin to near corners
        [camera_orig, frustum_corners_world[i]] for i in range(4)
    ] + [
        # Lines from camera origin to far corners
        [camera_orig, frustum_corners_world[i + 4]] for i in range(4)
    ] + [
        # Edges of the near plane
        [frustum_corners_world[i], frustum_corners_world[(i + 1) % 4]] for i in range(4)
    ] + [
        # Edges of the far plane
        [frustum_corners_world[i + 4], frustum_corners_world[((i + 1) % 4) + 4]] for i in range(4)
    ] + [
        # Edges connecting near and far planes
        [frustum_corners_world[i], frustum_corners_world[i + 4]] for i in range(4)
    ]

    # Convert to numpy array and adjust axes for plotting
    frustum_lines = np.array(frustum_lines)
    frustum_lines = frustum_lines[:, :, [0, 2, 1]]  # Swap y and z axes

    # Plot frustum using Line3DCollection
    frustum_collection = Line3DCollection(frustum_lines, colors=[0.0, 0.0, 0.7], linewidths=1)
    ax.add_collection3d(frustum_collection)


def world_points(camera, depth, rgb, stride=1):
    (height, width) = depth.shape
    # Pixel centers
    x = np.arange(0, width, stride) + 0.5
    y = np.arange(0, height, stride) + 0.5
    xv, yv = np.meshgrid(x, y)
    xy = np.stack([xv, yv], axis=-1)
    points, valid = camera.pix_2_world_np(
        xy.reshape(-1, 2),
        depth,
        valid_depth_min=0,
        valid_depth_max=np.inf
    )
    colors = np.reshape(rgb[::stride, ::stride], [-1, 3]) / 255.0
    alphas = np.ones_like(colors[..., 0:1])
    return points[..., [0,2,1]], valid, np.concat((colors, alphas), axis=1)


def plot_3d_tracks(track3d: Track3d, depth, rgb, axes=None, tracks_leave_trace=32):
    """Visualize 3D point trajectories with camera trajectory."""
    points = track3d.track3d.transpose(1,0,2)
    visibles = track3d.visible_list.transpose(1,0)
    M_list = [c.get_c2w()[:3] for c in track3d.cameras]

    num_frames, num_points = points.shape[0:2]
    points = points[..., [0,2,1]]

    # Colormap for points
    point_color_map = matplotlib.colormaps.get_cmap('hsv')
    point_cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

    # Colormap for camera trajectory
    camera_color_map = matplotlib.colormaps.get_cmap('Greys')

    if axes and 'x' in axes:
        x_min, x_max = axes['x']
    else:
        x_min, x_max = np.nanpercentile(points[visibles, 0], 5), np.nanpercentile(points[visibles, 0], 95)

    if axes and 'y' in axes:
        y_min, y_max = axes['y']
    else:
        y_min, y_max = np.nanpercentile(points[visibles, 1], 5), np.nanpercentile(points[visibles, 1], 95)

    if axes and 'z' in axes:
        z_min, z_max = axes['z']
    else:
        z_min, z_max = np.nanpercentile(points[visibles, 2], 5), np.nanpercentile(points[visibles, 2], 95)

    # Adjust axis limits
    interval = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) * 1.11
    # print(interval)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    x_min, x_max = x_mid - interval / 2, x_mid + interval / 2
    y_min, y_max = y_mid - interval / 2, y_mid + interval / 2
    z_min, z_max = z_mid - interval / 2, z_mid + interval / 2

    print(f'Axes: [{x_min}, {x_max}], [{y_min}, {y_max}], [{z_min}, {z_max}]')

    # Precompute camera positions and segments
    if M_list is not None:
        camera_positions = np.array([M[:3, 3] for M in M_list])  # Shape: (num_frames, 3)
        camera_positions = camera_positions[:, [0, 2, 1]]
        camera_segments = np.stack([camera_positions[:-1], camera_positions[1:]], axis=1)  # Shape: (num_frames - 1, 2, 3)

    frames = []
    for t in tqdm.tqdm(range(num_frames)):
        fig = Figure(figsize=(5.12, 5.12), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.invert_zaxis()
        ax.view_init()

        # Plot background points
        background_points, valid, colors = world_points(
            track3d.cameras[t], depth[t], rgb[t], stride=2)
        background = ax.scatter(
            background_points[..., 0],
            background_points[..., 1],
            background_points[..., 2],
            c=colors, s=16, depthshade=False, edgecolors='none')

        # Plot camera trajectory up to frame t
        if M_list is not None:
            if False and t > 0:
                segments_to_plot = camera_segments[:t]
                camera_cmap_norm = matplotlib.colors.Normalize(vmin=-1, vmax=t)
                camera_colors = camera_color_map(camera_cmap_norm(np.arange(t-1)))
                colors_to_plot = camera_colors[:t]
                camera_trajectory = Line3DCollection(segments_to_plot, colors=colors_to_plot, linewidths=1)
                ax.add_collection3d(camera_trajectory)

            path_pos = camera_positions[0:np.min((t+20, num_frames))]
            ax.plot(path_pos[..., 0], path_pos[..., 1], path_pos[..., 2],
                    color=[0.0, 0.0, 0.7],
                    linestyle='dashed')
            # Plot current camera pose
            draw_camera_pose(M_list[t], ax, al=interval/10, far=interval/10)

        # Compute mask for visible points
        mask = visibles[t, :]
        indices = np.where(mask)[0]

        if indices.size > 0:
            start_t = max(0, t - tracks_leave_trace)

            # Extract lines in a vectorized manner
            lines = points[start_t:t+1, indices]  # Shape: (line_length, num_visible_points, 3)
            lines = np.transpose(lines, (1, 0, 2))  # Shape: (num_visible_points, line_length, 3)

            # Prepare colors
            colors = point_color_map(point_cmap_norm(indices))

            # Plot lines using Line3DCollection
            line_collection = Line3DCollection(lines, colors=colors, linewidths=1)
            ax.add_collection3d(line_collection)

            # Plot scatter points
            end_points = points[t, indices]
            ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2], c=colors, s=3)

        fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
        # Draw including background
        fig.canvas.draw()
        with_background = np.array(canvas.buffer_rgba(), dtype=np.float32) / 255.
        background.set_visible(False)
        fig.canvas.draw()
        without_background = np.array(canvas.buffer_rgba(), dtype=np.float32) / 255.
        k = .4
        blend = with_background * k + without_background * (1-k)
        frames.append(blend)
        plt.close(fig)  # Close the figure to free memory

    return np.array(frames)[..., :3]


if __name__ == '__main__':
    
    # Video info.
    fov = 60
    
    root = '/home/junhwahur_google_com/dataset/stereo4d_small'
    vid = '-9SUMp35t30-clip6'

    # Load input dictionary.
    input_dict = load_rgbd_cam_from_pkl(vid, root, hfov=fov, new_imw=1, new_imh=1)

    # Load optimized tracks.
    with open(osp.join(root, vid, vid + '-optimized_tracks_filtered_v2.pkl'), 'rb') as f:
        track_optimized = pickle.load(f)

    track3d_optimized = Track3d(
        video=input_dict['left']['video'],
        color_values=track_optimized['color_values'],
        visible_list=track_optimized['visible_list'],
        cameras=input_dict['left']['camera'],
        track3d=track_optimized['track3d'],
    )

    

    # Frame index
    index_src = 0
    index_tgt = 190


    # Frame
    if True:
        frame_src = input_dict['left']['video'][index_src]
        frame_tgt = input_dict['left']['video'][index_tgt]
        # media.show_images([frame_src, frame_tgt])
        media.write_image("temp_frame_src.png", frame_src)
        media.write_image("temp_frame_tgt.png", frame_tgt)


    # Camera intrinsics
    if True:
        # Intrinsic normalized.
        print(input_dict['left']['camera'][index_src].intr_normalized)
        # Intrinsic matrix.
        imh, imw = input_dict['left']['video'][index_src].shape[:2]
        print(
            'intrinsics:\n',
            input_dict['left']['camera'][0].get_intri_matrix(imh, imw),
        )


    # Depth
    if True:
        # Reading from /cns/jn-d, loading speed may vary based on your kernel location.
        # disparity_raw = np.stack([-flow['fwd'][..., 0] for flow in input_dict['left']['raw_disparity']], axis=0)
        depth_src = input_dict['left']['depth'][index_src][None, ...]  # 1, h, w
        depth_tgt = input_dict['left']['depth'][index_tgt][None, ...]  # 1, h, w
        depth_src_vis = plot_depth_prism(depth_src)[0]
        depth_tgt_vis = plot_depth_prism(depth_tgt)[0]
        # media.show_images([depth_src_vis, depth_tgt_vis])
        media.write_image("temp_depth_src_vis.png", depth_src_vis)
        media.write_image("temp_depth_tgt_vis.png", depth_tgt_vis)


    # 3D motion trajectory
    # 3D points are in the world coordinate. Need to transfrom to the camera coordinate with extrinsics.
    # Refer world_2_pix_np().
    if True:
        # Given source and target index, select valid trajectories via 2D projection and visiblity flag.
        # points_at_frame_src saves their 2D projected pixel location.
        points_at_frame_src, valid_mask_src, depth_src = input_dict['left']['camera'][
            index_src
        ].world_2_pix_np(
            track3d_optimized.track3d[:, index_src],
            track3d_optimized.imh,
            track3d_optimized.imw,
        )
        valid_mask_src = valid_mask_src & track3d_optimized.visible_list[:, index_src]

        points_at_frame_tgt, valid_mask_tgt, depth_tgt = input_dict['left']['camera'][
            index_tgt
        ].world_2_pix_np(
            track3d_optimized.track3d[:, index_tgt],
            track3d_optimized.imh,
            track3d_optimized.imw,
        )
        valid_mask_tgt = valid_mask_tgt & track3d_optimized.visible_list[:, index_tgt]
        valid_mask = valid_mask_src & valid_mask_tgt

        # Compute their 3D position at source and target index via transformation with camera extrinsics.
        points_at_cam_src = input_dict['left']['camera'][index_src].world_2_cam_np(
            track3d_optimized.track3d[:, index_src],
            track3d_optimized.imh,
            track3d_optimized.imw,
        )
        points_at_cam_tgt = input_dict['left']['camera'][index_tgt].world_2_cam_np(
            track3d_optimized.track3d[:, index_tgt],
            track3d_optimized.imh,
            track3d_optimized.imw,
        )

        # Save depth and scene flow in the source index.
        n, h, w = input_dict['left']['depth'].shape
        depth_optimized = np.zeros((2, h, w))
        sceneflow_optimized = np.zeros((h, w, 3))

        points_at_frame_src = points_at_frame_src[valid_mask]
        points_at_frame_tgt = points_at_frame_tgt[valid_mask]
        depth_src = depth_src[valid_mask]
        depth_tgt = depth_tgt[valid_mask]

        depth_optimized[
            0,
            points_at_frame_src[:, 1].astype(int),
            points_at_frame_src[:, 0].astype(int),
        ] = depth_src
        depth_optimized[
            1,
            points_at_frame_tgt[:, 1].astype(int),
            points_at_frame_tgt[:, 0].astype(int),
        ] = depth_tgt

        sceneflow = (points_at_cam_tgt - points_at_cam_src)[valid_mask]
        sceneflow_optimized[
            points_at_frame_src[:, 1].astype(int),
            points_at_frame_src[:, 0].astype(int),
        ] = sceneflow

        valid_mask_2d_src = depth_optimized[0] > 0.0001
        valid_mask_2d_tgt = depth_optimized[1] > 0.0001

    

    # Visualize depth and mask
    depth_opt_vis = plot_depth_prism(depth_optimized)
    media.write_image("temp_depth_optimized_src.png", depth_opt_vis[0])
    media.write_image("temp_depth_optimized_tgt.png", depth_opt_vis[1])
    media.write_image("temp_valid_mask_2d_src.png", valid_mask_2d_src)
    media.write_image("temp_valid_mask_2d_tgt.png", valid_mask_2d_tgt)


    # Get optical flow from depth and scene flow
    depth_src = depth_optimized[None, :1]
    scene_flow = sceneflow_optimized.transpose(2, 0, 1)[None, ...]
    imh, imw = input_dict['left']['video'][index_src].shape[:2]
    intrinsics = input_dict['left']['camera'][0].get_intri_matrix(imh, imw)[None, ...]

    optical_flow = sceneflow2opticalflow(scene_flow, depth_src, intrinsics)[0]
    optical_flow_valid_mask = np.stack([valid_mask_2d_src, valid_mask_2d_src], axis=0)
    optical_flow = optical_flow * optical_flow_valid_mask
    optical_flow_viz = flow_to_png_middlebury(optical_flow)
    media.write_image("temp_optical_flow_viz.png", optical_flow_viz)

    # Warping - need GPU
    import torch
    frame_to_warp = torch.from_numpy(frame_tgt.transpose(2, 0, 1)[None, ...]).to("cuda:0") / 255.
    flow_to_warp = torch.from_numpy(optical_flow[None, ...]).to("cuda:0")
    warped_x = backward_warp_flow(frame_to_warp.float(), flow_to_warp.float()).cpu().numpy()
    media.write_image("temp_warped_x.png", warped_x.transpose(0, 2, 3, 1)[0])


