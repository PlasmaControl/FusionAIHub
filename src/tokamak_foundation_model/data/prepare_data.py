import numpy as np
import h5py
import hydra
import logging
from multiprocessing import Pool
from functools import partial
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
import os


log = logging.getLogger(__name__)

# ── hardcoded until video data is merged into the main data path ──
_VIDEO_DATA_PATH = Path("/scratch/gpfs/EKOLEMEN/big_d3d_data/d3d_image_data")


def _resample_time_series(data, time, target_frequency):
    """
    Resample non-uniformly sampled time series to uniform sampling.

    Parameters:
    -----------
    data : np.ndarray, shape (n_samples, ...)
        Time series data
    time : np.ndarray, shape (n_samples,)
        Time axis (can be non-uniform)
    target_frequency : float
        Desired sampling frequency in Hz

    Returns:
    --------
    resampled_data : np.ndarray
        Uniformly resampled data
    new_time : np.ndarray
        New uniform time axis
    """
    if len(data) <= 1:
        return time.copy(), data.copy()

    # Calculate target sampling period
    dt = 1.0 / target_frequency

    # Create uniform time grid
    n_samples = int(np.ceil((time[-1] - time[0]) / dt)) + 1
    new_time = time[0] + np.arange(n_samples) * dt

    # Handle multi-dimensional data
    original_shape = data.shape
    if data.ndim > 1:
        # Flatten all dimensions except the first (time)
        data_flat = data.reshape(data.shape[0], -1)
        resampled_flat = np.full((len(new_time), data_flat.shape[1]), np.nan)

        # Interpolate each channel, handling NaNs
        for i in range(data_flat.shape[1]):
            # Find valid (non-NaN) data points
            valid_mask = ~np.isnan(data_flat[:, i])

            if np.sum(valid_mask) >= 2:  # Need at least 2 points to interpolate
                valid_time = time[valid_mask]
                valid_data = data_flat[valid_mask, i]

                # Only interpolate within the range of valid data
                interpolator = interp1d(valid_time, valid_data, kind='linear',
                                        bounds_error=False, fill_value=np.nan)
                resampled_flat[:, i] = interpolator(new_time)
            # else: remains NaN (initialized above)

        # Reshape back to original dimensions (except time axis)
        new_shape = (len(new_time),) + original_shape[1:]
        resampled_data = resampled_flat.reshape(new_shape)
    else:
        # 1D case
        valid_mask = ~np.isnan(data)

        if np.sum(valid_mask) >= 2:
            valid_time = time[valid_mask]
            valid_data = data[valid_mask]

            interpolator = interp1d(valid_time, valid_data, kind='linear',
                                    bounds_error=False, fill_value=np.nan)
            resampled_data = interpolator(new_time)
        else:
            # Not enough valid data to interpolate
            resampled_data = np.full(len(new_time), np.nan)

    return new_time, resampled_data


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
            f"{n_missing}/{len(requested)} requested shots missing from one "
            f"or both data paths – skipped"
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
        video_data_path = Path(
            cfg_dict.get("video_data_path", str(_VIDEO_DATA_PATH)))
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

                    xdata, ydata = _resample_time_series(
                        data=ydata,
                        time=xdata / 1000,
                        target_frequency=sig_cfg["sampling_rate"])

                    read_data[abbr] = (xdata * 1000, ydata)

        if not read_data:
            return f"shot {shot}: no data read – skipped"

        # ── write processed file ──
        with h5py.File(output_file, "w") as f:
            for abbr, (xdata, ydata) in read_data.items():
                grp = f.create_group(abbr)
                grp.create_dataset("xdata", data=xdata, dtype='f8')
                grp.create_dataset("ydata", data=ydata, dtype='f8')

        os.chmod(output_file, 0o664)
        return None  # success

    except Exception as e:
        log.info(f"shot {shot}: {type(e).__name__}: {e}")
        return f"shot {shot}: {type(e).__name__}: {e}"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    mod_cfg = cfg.modalities
    input_data_path = Path(mod_cfg.input_data_path)
    video_data_path = Path(
        mod_cfg.get("video_data_path", str(_VIDEO_DATA_PATH)))
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
        for i, err in enumerate(
                tqdm(pool.imap_unordered(worker, shots), total=len(shots))):
            if err is not None:
                log.error(err)
                errors.append(err)

    log.info(
        f"Done. {len(shots) - len(errors)}/{len(shots)} succeeded, "
        f"{len(errors)} failed."
    )


if __name__ == "__main__":
    main()
