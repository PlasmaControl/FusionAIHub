import numpy as np
import h5py
import hydra
import logging
from multiprocessing import Pool
from functools import partial
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

# ── hardcoded until video data is merged into the main data path ──
_VIDEO_DATA_PATH = Path("/scratch/gpfs/EKOLEMEN/big_d3d_data/d3d_image_data")


def _get_valid_shots(
    shot_list: list[int],
    input_data_path: Path,
    video_data_path: Path,
) -> list[int]:
    """Return only shots that have files in *both* the main data path and the
    video data path.  Expects ``{shot}.h5`` in input_data_path and
    ``{shot}_image.h5`` in video_data_path."""

    main_shots = {
        int(p.stem)
        for p in input_data_path.glob("*.h5")
        if p.stem.isdigit()
    }
    video_shots = {
        int(p.stem.replace("_image", ""))
        for p in video_data_path.glob("*_image.h5")
    }
    available = main_shots & video_shots
    requested = set(shot_list)
    valid = sorted(requested & available)

    n_missing = len(requested) - len(valid)
    if n_missing:
        log.warning(
            f"{n_missing}/{len(requested)} requested shots missing from one or "
            f"both data paths – skipped"
        )
    log.info(f"{len(valid)} shots available in both paths")
    return valid


def _process_shot(shot: int, cfg_dict: dict) -> str | None:
    """Worker function executed in a child process.

    Args:
        shot: Shot number.
        cfg_dict: Plain dict (not DictConfig – must be picklable).

    Returns:
        None on success, or an error message string on failure.
    """
    try:
        input_data_path = Path(cfg_dict["input_data_path"])
        video_data_path = Path(cfg_dict.get("video_data_path", str(_VIDEO_DATA_PATH)))
        output_data_path = Path(cfg_dict["output_data_path"])
        output_data_path.mkdir(parents=True, exist_ok=True)

        output_file = output_data_path / f"{shot}_processed.h5"

        signals = cfg_dict["signals"]

        # ── group signals by source ──
        source_to_signals: dict[str, list[tuple[str, dict]]] = {}
        for abbr, sig_cfg in signals.items():
            source = sig_cfg.get("source", "default")
            source_to_signals.setdefault(source, []).append((abbr, sig_cfg))

        # Map source key → input filename
        source_file_map = {
            "default": input_data_path / f"{shot}.h5",
            "video": video_data_path / f"{shot}_image.h5",
        }

        # ── read all signals ──
        read_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for source_key, sigs in source_to_signals.items():
            fpath = source_file_map.get(source_key)
            if fpath is None or not fpath.exists():
                continue

            with h5py.File(fpath, "r") as f:
                for abbr, sig_cfg in sigs:
                    grp_name = sig_cfg["input_group"]
                    if grp_name not in f:
                        continue

                    xdata = f[grp_name][sig_cfg["input_xkey"]][:]
                    ydata = f[grp_name][sig_cfg["input_ykey"]][:]

                    if sig_cfg.get("swap_axes") is not None:
                        ydata = ydata.swapaxes(*sig_cfg["swap_axes"])

                    read_data[abbr] = (xdata, ydata)

        if not read_data:
            return f"shot {shot}: no data read – skipped"

        # ── write processed file ──
        with h5py.File(output_file, "w") as f:
            for abbr, (xdata, ydata) in read_data.items():
                grp = f.create_group(abbr)
                grp.create_dataset("xdata", data=xdata)
                grp.create_dataset("ydata", data=ydata)

        return None  # success

    except Exception as e:
        return f"shot {shot}: {type(e).__name__}: {e}"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    mod_cfg = cfg.modalities
    input_data_path = Path(mod_cfg.input_data_path)
    video_data_path = Path(mod_cfg.get("video_data_path", str(_VIDEO_DATA_PATH)))
    num_workers = mod_cfg.get("num_workers", 8)

    # ── filter to shots that exist in both paths ──
    shots = _get_valid_shots(
        shot_list=list(cfg.shot_list.shots),
        input_data_path=input_data_path,
        video_data_path=video_data_path,
    )

    if not shots:
        log.error("No valid shots found – exiting.")
        return

    # Convert to plain dict so it's picklable for multiprocessing
    cfg_dict = OmegaConf.to_container(mod_cfg, resolve=True)

    log.info(f"Processing {len(shots)} shots with {num_workers} workers")

    worker = partial(_process_shot, cfg_dict=cfg_dict)

    errors = []
    
    with Pool(processes=num_workers) as pool:
        for i, err in enumerate(tqdm(pool.imap_unordered(worker, shots), total=len(shots))):
            if err is not None:
                log.error(err)
                errors.append(err)

    log.info(
        f"Done. {len(shots) - len(errors)}/{len(shots)} succeeded, "
        f"{len(errors)} failed."
    )


if __name__ == "__main__":
    # python -m tokamak_foundation_model.data.prepare_data
    main()