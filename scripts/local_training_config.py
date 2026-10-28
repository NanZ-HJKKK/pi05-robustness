"""Use verified local weights/statistics while preserving their exact contents."""
import dataclasses
from pathlib import Path


def localize(config):
    from openpi.training.weight_loaders import CheckpointWeightLoader
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/"cache/checkpoints/pi05_libero"
    assets=dataclasses.replace(config.data.assets,assets_dir=str(checkpoint/"assets"))
    return dataclasses.replace(config,data=dataclasses.replace(config.data,assets=assets),
                               weight_loader=CheckpointWeightLoader(str(checkpoint/"params")))
