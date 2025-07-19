import hydra
from omegaconf import DictConfig

from .preprocess import preprocess


@hydra.main(
    config_path="config",
    config_name="default",
    version_base=None,
    )
def main(cfg: DictConfig):
    # Pass hydra config to preprocess
    preprocess(cfg)

if __name__ == "__main__":
    main()
