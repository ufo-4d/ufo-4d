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

import os
import hydra
import torch
import copy
import numpy as np
from jaxtyping import install_import_hook
from omegaconf import DictConfig
import mediapy as media
from einops import rearrange, repeat
from src.model import model_wrapper
from src.misc.weight_modify import checkpoint_filter_fn

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.misc.cam_utils import pose_encoding_to_camera
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder

from src.misc.utils import vis_depth_map, vis_3dmotion_map, vis_normalized_map
from src.misc.estimate_intrinsics import estimate_intrinsics
from scipy.spatial.transform import Rotation, Slerp


def get_interpolated_pose(pose0, pose1, t):
    assert 0 <= t <= 1, f'{t} is not in [0, 1]'
    # Initialize output matrix.
    pose = np.eye(4)

    # Linear Interpolation for Translation.
    pose[:3, 3] = (1 - t) * pose0[:3, 3] + t * pose1[:3, 3]

    # SLERP for Rotation
    # Create rotation object containing both start and end rotations.
    rots = Rotation.from_matrix([pose0[:3, :3], pose1[:3, :3]])
    # Interpolate at time t (assuming t is between 0 and 1).
    pose[:3, :3] = Slerp([0, 1], rots)(t).as_matrix()
    
    return pose


def load_models(cfg, device):
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    ckpt_weights = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'model' in ckpt_weights:
        ckpt_weights = ckpt_weights['model']
        ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
        missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
    elif 'state_dict' in ckpt_weights:
        ckpt_weights = ckpt_weights['state_dict']
        ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
        missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
    return encoder, decoder


def concat_decoder_outputs(output_list):
    """Concatenates list of single-view outputs into one multi-view output."""
    base = copy.copy(output_list[0])
    for k in ['color', 'point_rendered', 'motion_rendered']:
        if hasattr(base, k):
            tensors = [getattr(o, k) for o in output_list]
            setattr(base, k, torch.cat(tensors, dim=1))
    return base


def save_media(output_dir, list_images, name, save_video=True, fps_out=20):
    if save_video:
        media.write_video(f'{output_dir}/{name}.gif', list_images, fps=fps_out, codec='gif')
    else:
        for f_id, img in enumerate(list_images):
            media.write_image(f'{output_dir}/{name}_{f_id}.jpg', img)


def save_multiview_visualization(batch, output_list, output_dir, view_names, fps_out=20, save_video=True):
    """
    Saves visualization for multiple views into separate subfolders.
    Includes RGB, Depth, Motion, and Opacity.
    """
    b, _, _, h, w = batch["context"]["image"].shape
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input reference video
    if save_video:
        images = (batch["context"]["image"].detach().cpu().numpy() + 1)/2.
        images = images[0].transpose(0, 2, 3, 1)  # v, h, w, c
        media.write_video(f'{output_dir}/input.gif', [images[0], images[1]], fps=1, codec='gif')

    # Save Opacity
    opacities = batch["pred"]["gaussians"].opacities.reshape(1, 2, h, w)
    opacities_map_viz = vis_normalized_map(opacities, min_value=0., max_value=1.)
    opacities_map_viz = opacities_map_viz.detach().cpu().numpy()[0].transpose(0, 2, 3, 1)
    for i in range(2):
        media.write_image(f'{output_dir}/opacity{i}.jpg', opacities_map_viz[i])

    # Project 3D motion into optical flow
    intrinsics = batch["pred"]["intrinsics"][:, :1].repeat(1, len(view_names), 1, 1)
    motion_viz = [
        vis_3dmotion_map(o.motion_rendered[0], o.point_rendered[0, :, 2:3], intrinsics[0]) 
        for o in output_list
    ]

    # Helper to process and save a specific view index
    def process_view(view_idx, view_name):
        sub_dir = os.path.join(output_dir, view_name)
        os.makedirs(sub_dir, exist_ok=True)
        
        # RGB Video
        rgb_frames = [o.color[0, view_idx].detach().cpu().numpy().transpose(1, 2, 0) for o in output_list]
        save_media(sub_dir, rgb_frames, 'rgb', save_video, fps_out)

        # Depth Video
        depth_raw = [o.point_rendered[0, view_idx, -1] for o in output_list]
        depth_viz = [
            vis_depth_map(d, near_depth=0.01, far_depth=100.).cpu().numpy().transpose(1, 2, 0) 
            for d in depth_raw
        ]
        save_media(sub_dir, depth_viz, 'depth', save_video, fps_out)

        # Motion Video
        motion_frames = [
            m[view_idx].detach().cpu().numpy().transpose(1, 2, 0) 
            for m in motion_viz
        ]
        save_media(sub_dir, motion_frames, 'motion', save_video, fps_out)
        
    # Loop over the requested views
    for idx, name in enumerate(view_names):
        process_view(idx, name)


def load_batch(image0_name, image1_name, device, model_w=512):
    
    image0 = media.read_image(image0_name)
    image1 = media.read_image(image1_name)
    h, w, _ = image1.shape
    model_h = int((model_w / w * h) // 16 * 16)
    
    def image_resize(img):
        h, w, _ = img.shape
        if h != model_h or w != model_w:
            img = media.resize_image(img, (model_h, model_w))
        return img

    image0 = image_resize(image0)[None, ...]
    image1 = image_resize(image1)[None, ...]
    image0 = torch.tensor(image0).float().permute(0, 3, 1, 2) / 255.
    image1 = torch.tensor(image1).float().permute(0, 3, 1, 2) / 255.

    # Normalized camera intrinsics, a placeholder value.
    fc = 400 / model_w
    cy, cx = 0.5, 0.5
    camera_intrinsics = torch.tensor(
        [[fc, 0, cx],
        [0, fc, cy],
        [0, 0, 1.]]
    ).float()

    return {
        "context": {
            "image": torch.stack([image0, image1], axis=1).to(device) * 2 - 1,
            "intrinsics": camera_intrinsics.to(device).view(1, 1, 3, 3).repeat(1, 2, 1, 1),  # placeholder
            "near": torch.tensor(0.1).float().view(1, 1).to(device).repeat(1, 2),  # placeholder
            "far": torch.tensor(100).float().view(1, 1).to(device).repeat(1, 2),  # placeholder
        }
    }


def postprocess(
    batch,
    encoder_dict):

    pred_dict = dict()

    b, v, _, h, w = batch["context"]["image"].shape

    gaussians = encoder_dict['gaussians']
    pred_dict = {
        'gaussians': gaussians
    }

    # Infer intrinsics from pointmap.
    pred_intrinsics = estimate_intrinsics(
        rearrange(
            gaussians.means,
            "b (v h w) xyz -> b v h w xyz",
            v=v, h=h, w=w),
        focal_mode='weiszfeld')  # b v c d

    for b_idx in range(b):
        for v_idx in range(v):
            det = abs(torch.linalg.det(pred_intrinsics[b_idx, v_idx]))
            if det < 1e-6:
                raise ValueError('Estimated intrinsics is singular.')
    pred_dict['intrinsics'] = pred_intrinsics

    # Prepare pose for rendering. Convert quaternion to pose matrix.
    # The pose of the first frame is always the identity matrix.
    # The pose for the 2nd frame is estimated.
    pose_est_view2 = pose_encoding_to_camera(
        encoder_dict['pose'], pose_encoding_type="absT_quaR")
    pose_est_view1 = torch.eye(4).to(device=pose_est_view2.device)
    pose_est_view1 = pose_est_view1.unsqueeze(0).expand(pose_est_view2.shape[0], -1, -1)
    pose_est = torch.stack([pose_est_view1, pose_est_view2], dim=1)
    pred_dict["extrinsics"] = pose_est

    # Save the pred_dict in batch.
    batch["pred"] = pred_dict

    return batch 


def prepare_rendering(
    gaussians,
    t: float,
    view_render=1,
    ):

    # Our model only supports two-view estimation.
    view_model_output = 2 

    # All estimations are at the canonical coordinate.
    # The current model estimates Gaussians and motion for the two views.
    means = rearrange(gaussians.means, "b (v g) xyz -> b v g xyz", v=view_model_output)
    motions = rearrange(gaussians.motions, "b (v g) xyz -> b v g xyz", v=view_model_output)
    means_view0, means_view1 = means.split([1,1], dim=1)  # b 1 g xyz
    motions_view0, motions_view1 = motions.split([1,1], dim=1)  # b 1 g xyz
    
    # means_view0 and motions_view0 are from image0.
    # means_view1 and motions_view1 are from image1.
    # Merge them by:
    # translating means_view1 by using its estimated motion, motions_view1.
    means_view1 = (means_view1 + motions_view1).clone()
    # reversing motions_view1 so that it's now t0 -> t1.
    motions_view1 = (motions_view1 * -1).clone()
    # Translate each 3D Gaussians by fraction t with its motion.
    means_t = torch.cat([means_view0 + t * motions_view0, means_view1 + t * motions_view1], dim=2)  # b 1 gg xyz
    motion_t = torch.cat([t * motions_view0, t * motions_view1], dim=2)  # b 1 gg xyz

    # Given a set of 3D Gaussians, replicate it by the number of views we want to render.
    # We can render multiple views at the same time through the decoder. 
    means_t = means_t.repeat(1, view_render, 1, 1)
    motion_t = motion_t.repeat(1, view_render, 1, 1)
    covariances = repeat(gaussians.covariances, "b gg i j -> b v gg i j", v=view_render)
    harmonics = repeat(gaussians.harmonics, "b gg c d_sh -> b v gg c d_sh", v=view_render)
    opacities = repeat(gaussians.opacities, "b gg -> b v gg", v=view_render)

    # Rearrange.
    gaussians.means = rearrange(means_t, "b v gg xyz -> b (v gg) xyz")
    gaussians.covariances = rearrange(covariances, "b v gg i j -> b (v gg) i j")
    gaussians.harmonics = rearrange(harmonics, "b v gg c d_sh -> b (v gg) c d_sh")
    gaussians.opacities = rearrange(opacities, "b v gg -> b (v gg)")
    gaussians.motions = rearrange(motion_t, "b v gg xyz -> b (v gg) xyz")

    return gaussians


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def main(cfg_dict: DictConfig):

    image0_name = 'demo_example/image0.jpg'
    image1_name = 'demo_example/image1.jpg'

    model_w = 512
    interpolation_steps = 20
    view_render = 1

    device = torch.device("cuda")

    # Load model.
    cfg = load_typed_root_config(cfg_dict)
    encoder, decoder = load_models(cfg, device)
    encoder.eval()

    # Load batch.
    alias = "/".join(cfg.checkpointing.load.split("/")[-4:-2])
    batch = load_batch(
        image0_name=image0_name,
        image1_name=image1_name,
        device=device,
        model_w=model_w)


    # Run inference
    _, _, _, model_h, model_w = batch["context"]["image"].shape
    visualization_dump = {}
    with torch.no_grad():

        # Predict 3D Gaussians in the canonical camera coordinate.
        encoder_dict = encoder(batch["context"], 0, visualization_dump)

        # Postprocess.
        batch = postprocess(batch, encoder_dict)

        # Use estimated intrinsics for rendering.
        render_intrinsics = batch["pred"]["intrinsics"][:, :1].repeat(1, view_render, 1, 1)

        # Prepare extrinsics to render.
        pose_view0 = batch["pred"]["extrinsics"][:, :1]
        pose_view1 = batch["pred"]["extrinsics"][:, 1:]
        pose0_np = pose_view0[0, 0].cpu().numpy()
        pose1_np = pose_view1[0, 0].cpu().numpy()
        
        list_pose_intp = [
            torch.from_numpy(get_interpolated_pose(pose0_np, pose1_np, t / interpolation_steps))
            .float().to(device).view(1, 1, 4, 4)
            for t in range(interpolation_steps + 1)
        ]

        # 4D interpolation.
        output_list = []
        for t in range(interpolation_steps+1):

            t_fraction = t / float(interpolation_steps)

            # Prepare 3D Gaussians for rendering. We render one view at a time here.
            curr_gaussian = copy.deepcopy(batch["pred"]["gaussians"])
            curr_gaussian = prepare_rendering(
                gaussians=curr_gaussian,
                t=t_fraction,
                view_render=view_render)
            
            # Prepare extrinsics to render
            poses_to_render = [pose_view0, pose_view1, list_pose_intp[t]]
            
            single_view_outputs = []
            for pose in poses_to_render:
                out_single = decoder.forward(
                    curr_gaussian,      # Same geometry
                    pose,               # Different extrinsics (== pose)
                    render_intrinsics,  # Same intrinsic
                    batch["context"]["near"],
                    batch["context"]["far"],
                    (model_h, model_w),
                )
                single_view_outputs.append(out_single)

            # Concatenate results to shape (B, 3, ...)
            combined_output = concat_decoder_outputs(single_view_outputs)
            output_list.append(combined_output)
            del curr_gaussian

        save_multiview_visualization(
            batch=batch,
            output_list=output_list,
            output_dir=f'demo_example/result/{alias}',
            view_names=['view0', 'view1', 'view_intp'],
            fps_out=10,
            save_video=True
        )

    del batch
    del output_list

if __name__ == "__main__":
    main()