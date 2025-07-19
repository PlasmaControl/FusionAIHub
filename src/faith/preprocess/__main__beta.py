import logging

import hydra
from omegaconf import DictConfig

from .preprocess_beta import prepare_dataset


@hydra.main(
    config_path="config",
    config_name="default",
    version_base=None,
    )
def main(
    cfg: DictConfig,
):

    log_level = getattr(
        logging,
        cfg.get('log_level', 'INFO').upper(),
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Pass hydra config to preprocess
    prepare_dataset(cfg)

if __name__ == "__main__":
    main() # type: ignore
