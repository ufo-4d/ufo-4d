"""Data module for training and validation."""

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate

from ..misc.step_tracker import StepTracker
from . import DatasetCfgWrapper, get_dataset
from .types import DataShim, Stage
from .validation_wrapper import ValidationWrapper
from .shims.crop_shim import apply_geometric_augmentation

def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class CombinedDatasetWithProbs(IterableDataset):
    
    def __init__(
        self,
        datasets: dict[str, IterableDataset],
        batch_size: int,
        probabilities: list[float],
        train_width: int = 256):
        super().__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.dataset_names = list(datasets.keys())
        self.probabilities = probabilities
        self.iterators = {}
        self.dataset_cfg = {
            name: self.datasets[name].cfg for name in self.dataset_names
            }

        # Multi aspect ratio augmentation.
        # Input needs to be a multiple of patch size (16).
        self.train_height_dict = {
            256: [16*i for i in range(5, 16+1)],
            512: [16*i for i in range(10, 32+1)]
        }
        # Validate target_width once during initialization
        if train_width not in self.train_height_dict:
            raise ValueError(f"target_width must be one of {list(self.train_height_dict.keys())}, got {train_width}")
        self.train_width = train_width

    def _get_target_shape(self) -> tuple:
        train_height = random.choice(self.train_height_dict[self.train_width])
        return (train_height, self.train_width)


    def _get_next_batch(self) -> list:
        batch = []
        target_shape = self._get_target_shape()

        dataset_names = random.choices(self.dataset_names, weights=self.probabilities, k=self.batch_size)
        for dataset_name in dataset_names:
            try:
                sample = next(self.iterators[dataset_name])
                if '__key__' in sample:
                    del sample['__key__']
                # Data augmentation
                sample = apply_geometric_augmentation(
                    sample,
                    target_shape=target_shape,
                    augmentation_type=self.dataset_cfg[dataset_name].train_aug_type)
                batch.append(sample)
            except StopIteration:
                self.iterators[dataset_name] = iter(self.datasets[dataset_name])
                return self._get_next_batch()
        return batch

    def __iter__(self):
        self.iterators = {name: iter(ds) for name, ds in self.datasets.items()}
        while True:
            yield self._get_next_batch()


class DataModule(LightningDataModule):
    dataset_cfgs: list[DatasetCfgWrapper]
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfgs: list[DatasetCfgWrapper],
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
        train_width: int = 256,
    ) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank
        self.train_width = train_width

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        datasets = get_dataset(self.dataset_cfgs, "train", self.step_tracker, self.global_rank)

        datasets_dict = {ds.cfg.name: self.dataset_shim(ds, "train") for ds in datasets}
        batch_size = self.data_loader_cfg.train.batch_size
        probabilities = [ds.cfg.probability for ds in datasets]
        print(f'{probabilities=}')
        combined_sampler = CombinedDatasetWithProbs(
            datasets_dict,
            batch_size,
            probabilities,
            train_width=self.train_width)

        return DataLoader(
            combined_sampler,
            batch_size=None,  # Sampler yields complete batches
            sampler=None,  # Sampler handles the logic
            shuffle=False,
            collate_fn=default_collate,
            num_workers=self.data_loader_cfg.train.num_workers, 
            generator=self.get_generator(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )


    def val_dataloader(self):
        datasets = get_dataset(self.dataset_cfgs, "val", self.step_tracker, self.global_rank)
        data_loaders = []
        for i, dataset in enumerate(datasets):
            dataset = self.dataset_shim(dataset, "val")
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.data_loader_cfg.val.batch_size,
                    num_workers=self.data_loader_cfg.val.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.val),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(self.data_loader_cfg.val),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def test_dataloader(self):
        datasets = get_dataset(self.dataset_cfgs, "test", self.step_tracker, self.global_rank)
        data_loaders = []
        for i, dataset in enumerate(datasets):
            dataset = self.dataset_shim(dataset, "test")
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.data_loader_cfg.test.batch_size,
                    num_workers=self.data_loader_cfg.test.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.test),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(self.data_loader_cfg.test),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]
