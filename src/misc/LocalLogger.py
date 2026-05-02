"""Logger class."""

import os
import shutil
from pathlib import Path
from typing import Any, Optional

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image


class LocalLogger(Logger):
    def __init__(self, log_path) -> None:
        super().__init__()
        self.experiment = None
        self.log_path = log_path
        self.log_file = open(f'{log_path}/main.text', 'a') 

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            self.log_file.write(f'Step {step+1}:\t{k}={v:.4f}\n')
        self.log_file.flush() 

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        mode: str="",
        batch_idx: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        if not batch_idx:
            batch_idx = ""

        for index, image in enumerate(images):
            path = self.log_path / f"{key}/{mode}/{step:0>6}_{batch_idx}_{index:0>2}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
