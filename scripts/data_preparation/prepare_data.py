import numpy as np
import h5py
import hydra
import logging
from multiprocessing import Pool
from functools import partial
from omegaconf import DictConfig, OmegaConf
from typing import Union
from pathlib import Path
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
import warnings
import os


log = logging.getLogger(__name__)


class SignalLoader:
    """Load grouped signals from MDSPlus HDF5 files."""

    def __init__(self, h5_file_path: str | Path, verbose: bool = True):
        """
        Initialize loader with HDF5 file path.

        Parameters
        ----------
        h5_file_path : str | Path
            Path to HDF5 file (e.g., '/path/to/200000.h5')
        verbose : bool, default=True
            Print warnings about missing signals.
        """
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.shot_number = Path(h5_file_path).stem
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5_file.close()

    def load_signal_data(
            self,
            tree: str,
            signal_path: str,
            data_key: str = 'data',
            time_key: str = 'dim0'
    ) -> dict[str, np.ndarray]:
        """
        Load a single signal's data and time arrays from HDF5.

        Parameters
        ----------
        tree : str
            Tree name (e.g., 'D3D', 'IRTV')
        signal_path : str
            Full signal path (e.g., '\\SPECTROSCOPY::FS02')
        data_key : str
            HDF5 dataset name for signal data (default: 'data')
        time_key : str
            HDF5 dataset name for time axis (default: 'dim0')

        Returns
        -------
        Dictionary with 'data' and 'time' keys, or empty dict if not found
        """
        try:
            # Access: f[shot_number][tree][signal_path]
            if self.shot_number not in self.h5_file:
                if self.verbose:
                    warnings.warn(f"Shot {self.shot_number} not in HDF5 file")
                return {}

            shot_group = self.h5_file[self.shot_number]

            if tree not in shot_group:
                if self.verbose:
                    warnings.warn(
                        f"Tree '{tree}' not found for shot {self.shot_number}")
                return {}

            tree_group = shot_group[tree]

            if signal_path not in tree_group:
                if self.verbose:
                    warnings.warn(
                        f"Signal '{signal_path}' not found in tree '{tree}'")
                return {}

            signal_group = tree_group[signal_path]

            # Load data and time
            result = {}

            if data_key in signal_group:
                result['data'] = signal_group[data_key][:]
            else:
                if self.verbose:
                    warnings.warn(f"Data key '{data_key}' not found "
                                  f"in {tree}/{signal_path}")
                result['data'] = np.array([])

            if time_key in signal_group:
                result['time'] = signal_group[time_key][:]
            else:
                # Time might not exist for all signals
                result['time'] = np.array([])

            return result

        except Exception as e:
            if self.verbose:
                warnings.warn(f"Error loading {tree}/{signal_path}: {e}")
            return {}

    def load_signal_group(
            self,
            tree: str,
            signal_paths: list[str],
            data_key: str = 'data',
            time_key: str = 'dim0'
    ) -> dict[str, Union[np.ndarray, list[np.ndarray]]]:
        """
        Load multiple signals from the same tree.

        Parameters
        ----------
        tree : str
            Tree name (e.g., 'D3D')
        signal_paths : str
            List of signal paths
        data_key : str
            HDF5 dataset name for signal data
        time_key : str
           HDF5 dataset name for time axis

        Returns
        -------
        Dictionary with:
          - 'data': Stacked array (channels x time) or list if shapes differ
          - 'time': Time array or list of time arrays
          - 'valid_indices': List of indices where data was successfully loaded
          - 'num_valid': Number of valid signals
        """
        data_list = []
        time_list = []
        valid_indices = []

        for idx, path in enumerate(signal_paths):
            signal_data = self.load_signal_data(tree, path, data_key, time_key)

            if signal_data and len(signal_data.get('data', [])) > 0:
                data_list.append(signal_data['data'])
                time_list.append(signal_data.get('time', np.array([])))
                valid_indices.append(idx)
            else:
                data_list.append(np.array([]))
                time_list.append(np.array([]))

        if not data_list:
            warnings.warn(f"No valid signals loaded from {len(signal_paths)} "
                          f"paths in tree {tree}")
            return {
                'data': np.array([]),
                'time': np.array([]),
                'valid_indices': [],
                'num_valid': 0
            }

        # Check if we can stack the data
        shapes = [d.shape for d in data_list]
        all_same_shape = len(set(shapes)) == 1

        result = {
            'valid_indices': valid_indices,
            'num_valid': len(valid_indices)
        }

        if all_same_shape:
            # Stack data into array (C x T) for 1D or (C x ...)
            result['data'] = np.stack(data_list, axis=0)
            if result['data'].shape[0] == 1:
                result['data'] = result['data'].squeeze(0)

            # Check if all time arrays are the same
            if time_list and len(time_list[0]) > 0:
                time_shapes = [t.shape for t in time_list if len(t) > 0]
                if len(set(time_shapes)) == 1:
                    result['time'] = time_list[0]  # Use first time array
                else:
                    # Stack different times
                    result['time'] = np.stack(time_list, axis=0)
            else:
                result['time'] = np.array([])
        else:
            # Keep as list - shapes don't match
            result['data'] = data_list
            result['time'] = time_list

            if self.verbose:
                warnings.warn(f"Signals have mismatched shapes: {shapes}")

        return result

    def load_from_config(self, config: dict) -> dict[str, dict]:
        """
        Load all signal groups from a processing config.

        Parameters
        ----------
        config : dict
            Dictionary with 'signals' key containing groups

        Returns
        -------
        Dictionary mapping group names to loaded data
        """
        results = {}

        if 'signals' not in config:
            raise ValueError("Config must have 'signals' key")

        for group_name, group_config in config['signals'].items():
            if self.verbose:
                print(f"\nLoading group: {group_name}")

            tree = group_config['tree']
            signal_paths = group_config['input_key']
            data_key = group_config.get('input_ykey', 'data')  # ykey is data
            time_key = group_config.get('input_xkey', 'dim0')  # xkey is time

            # Load signals
            loaded = self.load_signal_group(
                tree=tree,
                signal_paths=signal_paths,
                data_key=data_key,
                time_key=time_key
            )

            # Add config metadata
            loaded['config'] = group_config
            loaded['tree'] = tree

            results[group_name] = loaded

            # Print summary
            if (isinstance(loaded['data'], np.ndarray)
                    and loaded['data'].size > 0):
                print(f"Loaded {loaded['num_valid']}/"
                      f"{len(signal_paths)} channels")
                print(f"    Data shape: {loaded['data'].shape}")
                if (isinstance(loaded['time'], np.ndarray)
                        and len(loaded['time']) > 0):
                    print(f"    Time shape: {loaded['time'].shape}")
                    if loaded['time'].ndim == 1:
                        print(f"    Time range: {loaded['time'][0]:.3f} to "
                              f"{loaded['time'][-1]:.3f} s")
            elif isinstance(loaded['data'], list) and len(loaded['data']) > 0:
                print(f"    Loaded {len(loaded['data'])}/{len(signal_paths)}"
                      f" signals (unstacked)")
                print(
                    f"    Shapes: {[d.shape for d in loaded['data'][:3]]}...")
            else:
                print(f"    No valid data loaded")

        return results


def process_shot(
        h5_file: str | Path,
        config: dict,
        verbose: bool = True
) -> dict[str, dict]:
    """
    Load shot data from HDF5 using config.

    Parameters
    ----------
    h5_file : str | Path
        Path to HDF5 file
    config : dict
        Loaded config dictionary
    verbose : bool
        Print progress

    Returns
    -------
    Dictionary of loaded signal groups
    """
    with SignalLoader(h5_file, verbose=verbose) as loader:
        data = loader.load_from_config(config)

    return data


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
        return (np.asarray(time, dtype=float).copy(),
                np.asarray(data, dtype=float).copy())

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

            if np.sum(valid_mask) >= 2:  # Need at least 2 points
                valid_time = time[valid_mask]
                valid_data = data_flat[valid_mask, i]

                # Only interpolate within the range of valid data
                interpolator = interp1d(valid_time, valid_data, kind='linear',
                                        bounds_error=False, fill_value=np.nan)
                resampled_flat[:, i] = interpolator(new_time)

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


def resample_signal_groups(loaded_data: dict[str, dict]) -> dict[str, dict]:
    """
    Resample all signal groups to their target sampling frequencies.

    All signals within a group are resampled to the SAME time grid.

    Parameters
    ----------
    loaded_data : dict
        Dictionary from process_shot() containing signal groups

    Returns
    -------
    Dictionary with resampled data, same structure as input
    """
    resampled = {}

    for group_name, group_data in loaded_data.items():
        print(f"\nResampling group: {group_name}")

        data = group_data['data']
        time = group_data['time']
        target_freq = group_data['config']['sampling_rate']
        num_channels = group_data['config']['num_channels']

        # Skip if no valid data
        if isinstance(data, np.ndarray) and data.size == 0:
            print(f"  Skipping - no data")
            resampled[group_name] = group_data.copy()
            continue

        # Handle stacked array (channels x time) - all share same time axis
        # Standard 1D signals usually come in as (channels, time)
        # But we need to be careful not to catch video data here if it happens to match criteria
        # checking ndim=2 helps distinguish 1D signals from 3D video tensors
        if isinstance(data, np.ndarray) and time.ndim == 1 and data.ndim == 2:
            if time.size == 0:
                print(f"  Skipping - no time axis")
                resampled[group_name] = group_data.copy()
                continue

            pass

            # --- Robust General Processing ---
        print(f"  Processing signals with potentially different time axes")

        # Normalize inputs to lists
        if isinstance(data, np.ndarray):
            if data.ndim == 2:  # (Channels, Time)
                data_list = list(data)
            else:
                # For 3D+ data, it's likely (Channels, ...)
                # or if it's a single video volume, maybe it shouldn't be split yet?
                # But the loop below expects data_list to match num_channels.
                # If shape is (720, 240, 420), this is ONE signal (one channel).
                # If data is a list, it's a list of signals.
                data_list = [data[i] for i in range(data.shape[0])]
        else:
            data_list = list(data)

        if isinstance(time, np.ndarray):
            # shared time axis
            time_list = [time] * len(data_list)
        else:
            time_list = list(time)

        # Step 1: Find global time range across ALL signals
        t_min = np.inf
        t_max = -np.inf

        for t in time_list:
            if isinstance(t, np.ndarray) and len(t) > 0:
                t_min = min(t_min, t[0] / 1000)
                t_max = max(t_max, t[-1] / 1000)

        if np.isinf(t_min) or np.isinf(t_max):
            print(f"  No valid time data found")
            resampled[group_name] = group_data.copy()
            continue

        # Step 2: Create single uniform time grid for entire group
        dt = 1.0 / target_freq
        n_samples = int(np.ceil((t_max - t_min) / dt)) + 1
        common_time = t_min + np.arange(n_samples) * dt

        print(f"  Global time range: {t_min:.3f} to {t_max:.3f} s")
        print(f"  Common time grid: {len(common_time)} samples @ {target_freq} Hz")
        common_time = common_time * 1000  # Convert back to ms for interpolation

        # Step 3: Determine Spatial Shape and Prepare Output Array
        spatial_shape = None

        def fix_video_shape(d):
            # Force reshape for EDICAM video data if size matches
            # The user confirmed that reshaping to (-1, 240, 720) is correct.
            # 240*720 = 172800 pixels per frame.
            PIXELS_PER_FRAME = 240 * 720
            if d.size > 0 and d.size % PIXELS_PER_FRAME == 0:
                frames = d.size // PIXELS_PER_FRAME
                # Return shape (Time, Height, Width)
                return d.reshape(frames, 240, 720)
            return d

        # Scan for shape
        for d in data_list:
            d_fixed = fix_video_shape(d)
            # If it's a video, d_fixed will be (Time, 240, 720) -> ndim=3
            if isinstance(d_fixed, np.ndarray) and d_fixed.ndim > 1 and d_fixed.size > 0:
                # Standardize on (Time, H, W) -> Spatial is (H, W)
                if d_fixed.ndim == 3:
                    spatial_shape = d_fixed.shape[1:]
                    break

        # Allocate output array: (Channels, Time, H, W)
        # This is the PyTorch-friendly format we want to end up with.
        if spatial_shape is not None:
            resampled_data_array = np.full(
                (num_channels, len(common_time)) + spatial_shape, np.nan, dtype='f4')
        else:
            resampled_data_array = np.full((num_channels, len(common_time)), np.nan,
                                           dtype='f4')

        # Step 4: Resample
        for i, (signal_data, signal_time) in enumerate(zip(data_list, time_list)):
            if i >= num_channels: break

            signal_data = fix_video_shape(signal_data)

            if not isinstance(signal_data, np.ndarray) or signal_data.size == 0: continue
            if not isinstance(signal_time, np.ndarray) or signal_time.size == 0: continue

            if len(signal_time) < 2: continue

            # --- 1D Case ---
            if signal_data.ndim == 1:
                valid_mask = ~np.isnan(signal_data)
                if np.sum(valid_mask) >= 2:
                    f = interp1d(signal_time[valid_mask], signal_data[valid_mask],
                                 kind='linear', bounds_error=False, fill_value=np.nan)
                    resampled_data_array[i, :] = f(common_time)

            # --- Video / Multi-dim Case ---
            # We now expect (Time, H, W) from fix_video_shape
            elif signal_data.ndim == 3:
                # signal_data is (T, H, W)
                # We need to interpolate along axis 0 (Time)

                # Check if time dimension matches signal_time length
                if signal_data.shape[0] != len(signal_time):
                    print(
                        f"    Warning: Time dim {signal_data.shape[0]} != Time vec {len(signal_time)}")
                    # Try to transpose if it helps (e.g. if it came in as H,W,T)
                    if signal_data.shape[-1] == len(signal_time):
                        signal_data = np.moveaxis(signal_data, -1, 0)
                    else:
                        continue

                T_in, H, W = signal_data.shape

                # Flatten spatial dims: (T, H*W)
                flat_data = signal_data.reshape(T_in, -1)

                # Interpolate along axis 0
                f = interp1d(signal_time, flat_data, axis=0, kind='linear',
                             bounds_error=False, fill_value=np.nan)

                flat_resampled = f(common_time)

                # Reshape back to (NewTime, H, W)
                resampled_nd = flat_resampled.reshape(len(common_time), H, W)

                # Assign to output array (Channels, Time, H, W)
                # Since resampled_data_array is (C, T, H, W), we assign directly
                try:
                    resampled_data_array[i] = resampled_nd
                except ValueError:
                    print(
                        f"    Mismatch: Target {resampled_data_array[i].shape}, Got {resampled_nd.shape}")

            valid_samples = int(np.sum(~np.isnan(resampled_data_array[i])))
            print(f"    Channel {i}: {valid_samples} valid samples")

        resampled[group_name] = group_data.copy()
        resampled[group_name]['data'] = resampled_data_array
        resampled[group_name]['time'] = common_time / 1000.0
        print(f"    Final group shape: {resampled_data_array.shape}")

    return resampled


def write_resampled_data(
        resampled_data: dict[str, dict],
        output_file: str | Path,
) -> None:
    """
    Write resampled data to HDF5 file.

    Parameters
    ----------
    resampled_data : dict
        Dictionary from resample_signal_groups()
    output_file :  str | Path
        Path to output HDF5 file
    """
    print(f"\nWriting to {output_file}")

    with h5py.File(output_file, "w") as f:
        for group_name, group_data in resampled_data.items():
            data = group_data['data']
            time = group_data['time']
            num_channels = group_data['config']['num_channels']

            # Create HDF5 group for this signal group
            grp = f.create_group(group_name)

            # Handle stacked arrays
            if isinstance(data, np.ndarray):
                # If data is empty, create NaN array with expected shape
                if data.size == 0 or time.size == 0:
                    # Create minimal time axis (single point)
                    time_out = np.array([0.0])
                    data_out = np.full((num_channels, 1), np.nan, dtype='f8')
                    print(f"  ! {group_name}: "
                          f"No data, writing NaN array {data_out.shape}")
                else:
                    time_out = time

                    # Check if we have fewer channels than expected
                    if data.shape[0] < num_channels:
                        # Pad with NaN channels
                        missing_channels = num_channels - data.shape[0]
                        nan_channels = np.full(
                            (missing_channels, data.shape[1]),
                            np.nan,
                            dtype='f8')
                        data_out = np.vstack([data, nan_channels])
                        print(f"  ! {group_name}: "
                              f"Padded {missing_channels} NaN channels")
                    elif data.shape[0] > num_channels:
                        # Truncate extra channels (shouldn't happen)
                        data_out = data[:num_channels]
                        print(f"  ! {group_name}: "
                              f"Truncated to {num_channels} channels")
                    else:
                        data_out = data

                grp.create_dataset('xdata', data=time_out, dtype='f8')
                grp.create_dataset('ydata', data=data_out, dtype='f8')

                print(f"    {group_name}: "
                      f"{data_out.shape} @ {len(time_out)} samples")

            # Handle list of arrays
            elif isinstance(data, list):
                # Find the longest time axis to use as reference
                max_time_len = 1
                reference_time = np.array([0.0])

                for t in time:
                    if isinstance(t, np.ndarray) and len(t) > max_time_len:
                        max_time_len = len(t)
                        reference_time = t

                # Build full data array with NaN padding
                data_out = np.full(
                    (num_channels, max_time_len), np.nan, dtype='f8')

                for i, channel_data in enumerate(data):
                    if i >= num_channels:
                        break  # Don't exceed expected channels

                    if channel_data.size > 0:
                        # Copy available data
                        n_samples = min(len(channel_data), max_time_len)
                        data_out[i, :n_samples] = channel_data[:n_samples]

                grp.create_dataset('xdata', data=reference_time, dtype='f8')
                grp.create_dataset('ydata', data=data_out, dtype='f8')

                print(f"    {group_name}: {data_out.shape} "
                      f"@ {len(reference_time)} samples (from list)")

    # Set file permissions
    os.chmod(output_file, 0o664)


def process_and_write_shot(shot: int, cfg_dict: dict) -> str | None:
    """Worker function executed in a child process.

    Args:
        shot: Shot number.
        cfg_dict: Plain dict (not DictConfig – must be picklable).

    Returns:
        None on success, or an error message string on failure.
    """
    try:
        input_data_path = Path(cfg_dict["input_data_path"])
        output_data_path = Path(cfg_dict["output_data_path"])
        output_data_path.mkdir(parents=True, exist_ok=True)

        input_file = input_data_path / f"{shot}.h5"
        output_file = output_data_path / f"{shot}_processed.h5"

        data = process_shot(str(input_file), cfg_dict, verbose=True)

        resampled_data = resample_signal_groups(data)

        write_resampled_data(resampled_data, output_file)

        return None  # success

    except Exception as e:
        log.info(f"shot {shot}: {type(e).__name__}: {e}")
        return f"shot {shot}: {type(e).__name__}: {e}"


@hydra.main(version_base=None,
            config_path="../../src/tokamak_foundation_model/data/config",
            config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    mod_cfg = cfg.modalities
    num_workers = mod_cfg.get("num_workers", 8)

    # ── filter to shots that exist in both paths ──
    shots = list(cfg.shot_list.shots)

    if not shots:
        log.error("No valid shots found – exiting.")
        return

    # Convert to plain dict so it's picklable for multiprocessing
    cfg_dict = OmegaConf.to_container(mod_cfg, resolve=True)

    log.info(f"Processing {len(shots)} shots with {num_workers} workers")

    worker = partial(process_and_write_shot, cfg_dict=cfg_dict)

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
