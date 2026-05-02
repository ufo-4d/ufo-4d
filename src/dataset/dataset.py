"""Dataset config class."""

from dataclasses import dataclass

@dataclass
class DatasetCfgCommon:
    original_image_shape: list[int]
    test_image_shape: list[int]
    background_color: list[float]
    probability: float
