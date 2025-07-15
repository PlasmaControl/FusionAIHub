import hydra
from omegaconf import DictConfig

from . import preprocess

@hydra.main(
    config_path="config", 
    config_name="config",
    version_base=None,
    )
def main(cfg: DictConfig):
    
    # TODO: Add hydra config to preprocess
    preprocess()