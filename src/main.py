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

import glob
import os
import collections
from pathlib import Path

import hydra
import torch
import wandb
import signal
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

from src.misc.weight_modify import checkpoint_filter_fn
from src.model import model_wrapper
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def load_weights_motionhead(cfg, ckpt_weights):

    # Initialize motion head's weight with pointmap's head.
    print('')
    print(f'loading pretrained weights for {cfg.model.encoder.motion_head_type}.')
    if not any(k.startswith('motion_head.') for k in ckpt_weights):
        motion_head_weights = {}
        for key, value in ckpt_weights.items():
            if key.startswith('downstream_head1.'):
                motion_head_weights[key.replace('downstream_head1.', 'motion_head.')] = value
                motion_head_weights[key.replace('downstream_head1.', 'motion_head2.')] = value
        ckpt_weights.update(motion_head_weights)

    return ckpt_weights


def load_weights_combined(cfg, ckpt_mast3r_path, ckpt_noposplat_path, encoder):

    ckpt_mast3r_weights = torch.load(
        ckpt_mast3r_path, map_location='cpu', weights_only=False)
    ckpt_noposplat_weights = torch.load(
        ckpt_noposplat_path, map_location='cpu', weights_only=False)

    # Extract the actual state dictionaries from the loaded files.
    mast3r_weights = ckpt_mast3r_weights.get('model', {})
    noposplat_weights = ckpt_noposplat_weights.get('state_dict', {})
    combined_weights = collections.OrderedDict()

    # Define the prefixes for the layers to be loaded from the noposplat checkpoint
    noposplat_prefixes = [
        'backbone.intrinsic_encoder',
        'gaussian_param_head',
    ]
    noposplat_prefixes.append('gaussian_param_head2')

    # Load weights from the noposplat checkpoint for the specified layers[].
    noposplat_weights = {k[8:]: v for k, v in noposplat_weights.items() if k.startswith('encoder.')}
    for key, value in noposplat_weights.items():
        if any(key.startswith(p) for p in noposplat_prefixes):
            combined_weights[key] = value
    print(f"Loaded {len(combined_weights)} keys from noposplat checkpoint.")

    # Load the remaining weights from the MASt3R checkpoint.
    mast3r_weights = checkpoint_filter_fn(mast3r_weights, encoder)
    for key, value in mast3r_weights.items():
        # Only add the key if it wasn't already loaded from the noposplat checkpoint.
        if key not in combined_weights:
            combined_weights[key] = value
    
    # Load motion weights
    combined_weights = load_weights_motionhead(cfg, combined_weights)

    return combined_weights


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    assert cfg_dict.logging.mode in ['wandb', 'tensorboard', 'local']

    if cfg_dict.logging.mode == "wandb":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))
        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    elif cfg_dict.logging.mode == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=output_dir,
            name="tensorboard"
        )
    elif cfg_dict.logging.mode == "local":
        logger = LocalLogger(log_path=output_dir)
    else:
        raise ValueError(f'Invalid logging mode: {cfg_dict.logging.mode}')

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        max_steps=cfg.trainer.max_steps,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":

        ckpt_mast3r_path = "./pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        ckpt_noposplat_path = "./pretrained_weights/mixRe10kDl3dv_512x512.ckpt"
        
        # Load weights.
        final_weights = load_weights_combined(cfg, ckpt_mast3r_path, ckpt_noposplat_path, encoder)
        missing_keys_expected = [5]
        unexpected_keys_expected = [16]

        # Load weights to the model
        missing_keys, unexpected_keys = encoder.load_state_dict(final_weights, strict=False)
        print(f'missing_keys: {len(missing_keys)}')
        print(f'unexpected_keys: {len(unexpected_keys)}')
        assert len(missing_keys) in missing_keys_expected, f"{len(missing_keys)} != {missing_keys_expected}"
        assert len(unexpected_keys) in unexpected_keys_expected, f"{len(unexpected_keys)} != {unexpected_keys_expected}"

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
        train_width=cfg.train.train_width,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
        # Overwrite the checkpoint path for the test after training finishes.
        checkpoint_path = sorted(glob.glob(str(output_dir / "checkpoints/*")))[-1]

    # Run test as default.
    trainer.test(
        model_wrapper,
        datamodule=data_module,
        ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":
    train()
