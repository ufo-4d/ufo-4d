"""Data augmentation utils."""

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import cv2
import random
import torchvision.transforms.functional as F

from ..types import AnyExample, AnyViews
from src.misc.cam_utils import adjust_intrinsics_with_crop, adjust_intrinsics_with_resize
from src.misc.cam_utils import unnormalize_intrinsics


def apply_resize(
    example: dict,
    target_shape: tuple[int, int],
    input_only: bool = False
    ) -> AnyViews:
    """Apply simple resize.
    """

    # List of data to augment.
    list_data_keys = [
        'image',
        'depth',
        'motion3d',
        'pointmap',
        'valid_mask_depth',
        'valid_mask_motion',
        ]
    keys_for_nn_resize = [
        'depth',
        'motion3d',
        'pointmap',
        'valid_mask_depth',
        'valid_mask_motion',
    ]

    def resize_view(
        views,
        target_shape):

        # Augment data that are only in the list_keys.
        list_keys = [key for key in list_data_keys if key in views]

        b, _, h_in, w_in = views['image'].shape
        input_shape = (h_in, w_in)
        if input_shape == target_shape:
            return views

        # Resize to the target dimension.
        intrinsics = views['intrinsics']
        h_target, w_target = target_shape
        for key in list_keys:
            resize_mode = (
                F.InterpolationMode.NEAREST if key in keys_for_nn_resize
                else F.InterpolationMode.BILINEAR
            )
            views[key] = F.resize(
                img=views[key],
                size=target_shape,
                interpolation=resize_mode,
                antialias=True)
        # Rescale intrinsics
        intrinsics = adjust_intrinsics_with_resize(
            intrinsics, h_in, w_in, target_shape[0], target_shape[1], normed=True)

        # Overwrite into dictionary.
        views['intrinsics'] = intrinsics
        return views

    # Random scale and crop augmentation
    context_resize = resize_view(example["context"], target_shape=target_shape)
    target_resize = example["target"] if input_only else resize_view(example["target"], target_shape=target_shape)

    return {
        **example,
        "context": context_resize,
        "target": target_resize,
    }


def center_crop_resize(
    views: AnyViews,
    input_shape: tuple[int, int],
    target_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    ) -> AnyViews:
    """Center crop to the arbitrary resolution and the resize to the target_shape.
    This simulates diverse focal length.
    """

    # List of data to augment.
    list_data_keys = [
        'image',
        'depth',
        'motion3d',
        'pointmap',
        'valid_mask_depth',
        'valid_mask_motion',
        ]
    keys_for_nn_resize = [
        'depth',
        'motion3d',
        'pointmap',
        'valid_mask_depth',
        'valid_mask_motion',
    ]
    intrinsics = views['intrinsics']
    # Augment data that are only in the list_keys.
    list_keys = [key for key in list_data_keys if key in views]

    # Center crop
    top = (input_shape[0] - crop_shape[0]) // 2
    left = (input_shape[1] - crop_shape[1]) // 2
    def crop_image(im, top, left, crop_h, crop_w):
        return im[:, :, top:top+crop_h, left:left+crop_w]
    for key in list_keys:
        views[key] = crop_image(views[key], top, left, crop_shape[0], crop_shape[1])

    # Adjust the intrinsics to account for the cropping.
    # Dividing cropping ratio as it's normalized intrinsics. 
    intrinsics = adjust_intrinsics_with_crop(
        intrinsics,
        input_shape[0],
        input_shape[1],
        crop_shape[0],
        crop_shape[1],
        normed=True)

    # Resize to the target dimension.
    if crop_shape != target_shape:
        h_target, w_target = target_shape
        for key in list_keys:
            resize_mode = (
                F.InterpolationMode.NEAREST if key in keys_for_nn_resize
                else F.InterpolationMode.BILINEAR
            )
            views[key] = F.resize(
                img=views[key],
                size=target_shape,
                interpolation=resize_mode,
                antialias=True)
        # Rescale intrinsics
        intrinsics = adjust_intrinsics_with_resize(
            intrinsics, crop_shape[0], crop_shape[1], target_shape[0], target_shape[1], normed=True)

    # Overwrite into dictionary.
    views['intrinsics'] = intrinsics

    return views


def apply_center_crop(
    example: AnyExample,
    target_shape: tuple[int, int],
    ) -> AnyExample:
    """Apply random scale and crop augmentation.
    """

    # Assumes all inputs have the same size.
    b, _, h_in, w_in = example["context"]['image'].shape

    # If crop size is bigger than the input, we crop and upscale the image.
    if h_in < target_shape[0] or w_in < target_shape[1]:
        crop_size = min(w_in, h_in)
        crop_shape = (crop_size, crop_size)
    else:
        crop_shape = target_shape
    input_shape = (h_in, w_in)

    if input_shape == crop_shape:
        return example

    # Random scale and crop augmentation
    return {
        **example,
        "context": center_crop_resize(example["context"], input_shape=input_shape, target_shape=target_shape, crop_shape=crop_shape),
        "target": center_crop_resize(example["target"], input_shape=input_shape, target_shape=target_shape, crop_shape=crop_shape),
    }


def apply_random_scale_crop_resize(
    example: AnyExample,
    target_shape: tuple[int, int],
    min_scale: float = 0.7,
    max_scale: float = 1.0,
    ) -> AnyExample:
    """Apply random scale and crop augmentation.
    """

    # Assumes all inputs have the same size.

    # Generate augmentation parameters.
    rand_scale = min_scale + random.random() * (max_scale-min_scale)
    b, _, h_in, w_in = example["context"]['image'].shape
    crop_size = int(rand_scale * min(h_in, w_in))
    crop_shape = (crop_size, crop_size)
    input_shape = (h_in, w_in)

    if (input_shape == crop_shape) and (input_shape == target_shape):
        return example

    # Random scale and crop augmentation
    return {
        **example,
        "context": center_crop_resize(example["context"], input_shape=input_shape, target_shape=target_shape, crop_shape=crop_shape),
        "target": center_crop_resize(example["target"], input_shape=input_shape, target_shape=target_shape, crop_shape=crop_shape),
    }


def random_scale_crop_resize_aspect_ratio(
    example: AnyExample,
    target_shape: tuple[int, int],
    min_scale: float = 0.7,
    max_scale: float = 1.0,
    ) -> AnyExample:
    """Apply random scale and crop augmentation.
    """

    b, _, h_in, w_in = example["context"]['image'].shape

    # Find min and max width, based on input and target shape
    min_width = max(int(w_in * min_scale), target_shape[1])
    max_width = min(int(h_in / target_shape[0] * target_shape[1]), w_in)
    if min_width >= max_width:
        crop_width = max_width
    else:
        crop_width = int(min_width + random.random() * (max_width-min_width))

    crop_height = int(crop_width / target_shape[1] * target_shape[0])
    crop_shape = (crop_height, crop_width)
    input_shape = (h_in, w_in)

    if (input_shape == crop_shape) and (input_shape == target_shape):
        return example

    # Random scale and crop augmentation
    return {
        **example,
        "context": center_crop_resize(example["context"], input_shape=input_shape, target_shape=target_shape, crop_shape=crop_shape),
        "target": center_crop_resize(example["target"], input_shape=input_shape, target_shape=target_shape, crop_shape=crop_shape),
    }


def apply_geometric_augmentation(
    example: AnyExample,
    target_shape: tuple[int, int],
    augmentation_type: str=None
    ) -> AnyExample:
    """Apply geometric augmentation.
    """

    if augmentation_type == "random_scale_crop_resize":
        return apply_random_scale_crop_resize(
            example,
            target_shape,
            min_scale=0.7,
            max_scale=1.0,
            )
    elif augmentation_type == "crop_resize":
        return apply_random_scale_crop_resize(
            example,
            target_shape,
            min_scale=1.0,
            max_scale=1.0,
            )
    elif augmentation_type == "random_scale_crop_resize_aspect_ratio":
        return random_scale_crop_resize_aspect_ratio(
            example,
            target_shape,
            min_scale=0.7,
            max_scale=1.0,
            )
    elif augmentation_type == "resize":
        return apply_resize(
            example,
            target_shape,
            )
    elif augmentation_type == "center_crop":
        return apply_center_crop(
            example,
            target_shape,
            )
    elif augmentation_type == "resize_test":
        return apply_resize(
            example,
            target_shape,
            input_only=True,
            )
    elif augmentation_type == "no_augmentation":
        return example
    else:
        raise ValueError(f'{augmentation_type} not defined.')