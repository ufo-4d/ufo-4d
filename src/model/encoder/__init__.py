"""Portions Copyright (c) [2025] [NoPoSplat]. Licensed under MIT."""

from typing import Optional

from .encoder import Encoder
from .encoder_ufo4d import EncoderUFO4DCfg, EncoderUFO4D
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "ufo4d": (EncoderUFO4D, None),
}

EncoderCfg = EncoderUFO4DCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
