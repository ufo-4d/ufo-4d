"""Dataset factory."""

from dataclasses import fields

from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_stereo4d import DatasetStereo4d, DatasetStereo4dCfg, DatasetStereo4dCfgWrapper
from .dataset_kitti2015 import DatasetKITTI2015, DatasetKITTI2015Cfg, DatasetKITTI2015CfgWrapper
from .dataset_pointodyssey import DatasetPointOdyssey, DatasetPointOdysseyCfg, DatasetPointOdysseyCfgWrapper
from .dataset_vkitti2 import DatasetVKITTI2, DatasetVKITTI2Cfg, DatasetVKITTI2CfgWrapper
from .dataset_bonn import DatasetBonn, DatasetBonnCfg, DatasetBonnCfgWrapper
from .dataset_spring import DatasetSpring, DatasetSpringCfg, DatasetSpringCfgWrapper
from .types import Stage


DATASETS: dict[str, Dataset] = {
    "bonn": DatasetBonn,
    "kitti2015": DatasetKITTI2015,
    "pointodyssey": DatasetPointOdyssey,
    "spring": DatasetSpring,
    "stereo4d": DatasetStereo4d,
    "vkitti2": DatasetVKITTI2,
}


DatasetCfgWrapper = (
    DatasetStereo4dCfgWrapper | 
    DatasetKITTI2015CfgWrapper | 
    DatasetPointOdysseyCfgWrapper | 
    DatasetVKITTI2CfgWrapper | 
    DatasetBonnCfgWrapper | 
    DatasetSpringCfgWrapper
)
DatasetCfg = (
    DatasetStereo4dCfg | 
    DatasetKITTI2015Cfg | 
    DatasetPointOdysseyCfg | 
    DatasetBonnCfg | 
    DatasetSpringCfg
)

def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
    global_rank: int | None,
) -> list[Dataset]:
    datasets = []
    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        # Skip datasets whose probability is zero.
        if stage in ['train']:
            if cfg.probability == 0:
                continue

        dataset = DATASETS[cfg.name](cfg, stage, global_rank)
        datasets.append(dataset)

    return datasets
