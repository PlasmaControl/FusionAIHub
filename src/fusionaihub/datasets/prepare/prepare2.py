import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample, resample_poly
from sklearn.model_selection import train_test_split
from ...util.parmap import ParallelMapper
import logging
import joblib
import torch
from concurrent.futures import ProcessPoolExecutor
from typing import Optional


log = logging.getLogger(__name__)

sample_cfg = {
    "signal": [ # start with signals to be transformed. this is hacked on
        ("magnetics_high_resolution", "mhr", True),
        ("ece_cali", "ece", True),
        ("co2_density", "co2", True),
        ("gas", "gas", False),
        ("ech", "ech", False),
        ("p_inj", "pin", False),
        ("t_inj", "tin", False),
    ],
    "randomize_shots": True,
    "random_seed": 42,
    "num_shots": 50,
    "fs_khz": 500,
    "ip_threshold": 1e-1,
    "window_ms": 250,
    "hop_ms": 50,
    "remove_empty": True,
    "train_test_split": 0.2,
    "raw_data_dir": "/scratch/gpfs/EKOLEMEN/d3d_fusion_data",
    "output_dir": "/scratch/gpfs/nc1514/FusionAIHub/data/foundation_v2",
}


def resample_nearest(y: np.ndarray, new_len: int) -> np.ndarray:
    orig_len = len(y)
    gcd = np.gcd(orig_len, new_len)
    up = new_len // gcd
    down = orig_len // gcd
    # return resample_poly(y, up, down)
    resampled = resample(y, new_len)
    return np.asarray(resampled)
    # return interp1d(np.linspace(0, 1, len(y)), y, kind='cubic')(np.linspace(0, 1, new_len))


def extract(
    shot: int,
    directory: Path,
    signal: str,
) -> pd.DataFrame:
    
    path = (directory / str(shot)).with_suffix(".h5")
    df = pd.read_hdf(path, key=signal)
    
    return pd.DataFrame(df)

def running_time(
    directory: Path,
    shot: int,
    ip_threshold: float,
) -> tuple[float, float]:
    
    path = (directory / str(shot)).with_suffix(".h5")
    with pd.HDFStore(path, 'r') as store:
        df = store['ip']['ipsip']
    df = df.loc[df > ip_threshold]
    start_time = df.index[0]
    end_time = df.index[-1]
    return start_time, end_time

def align(
    df: pd.DataFrame,
    start_time: float,
    end_time: float,
    fs: float,
) -> pd.DataFrame:
    
    # get sampling frequency
    fs_raw = len(df) / (df.index[-1] - df.index[0])
    
    # crop time
    df = df.loc[(df.index >= start_time) & (df.index <= end_time)]
    
    # resample
    num = len(df)
    num = int(num * fs / fs_raw)
    
    df = pd.DataFrame(
        {col: resample(df[col].values, num) for col in df.columns},
        index=np.linspace(df.index[0], df.index[-1], num)
    )
    
    # mark on-off states
    start_nan = (df.index[0] - start_time) * fs
    end_nan = (end_time - df.index[-1]) * fs
    start_pad = pd.DataFrame(
        0, index=pd.RangeIndex(start=int(start_nan)), columns=df.columns)
    end_pad = pd.DataFrame(
        0, index=pd.RangeIndex(start=int(len(df) + start_nan), stop=int(len(df) + start_nan + end_nan)), columns=df.columns)
    
    df_state = pd.DataFrame(True, index=df.index, columns=df.columns)
    start_pad_state = pd.DataFrame(False, index=start_pad.index, columns=df.columns)
    end_pad_state = pd.DataFrame(False, index=end_pad.index, columns=df.columns)
    
    df = pd.concat([start_pad, df, end_pad], ignore_index=True)
    df_state = pd.concat([start_pad_state, df_state, end_pad_state], ignore_index=True)
    df_state.columns = [f"{col}_state" for col in df.columns]
    
    # combine data with state
    df = df.astype(np.float32)
    df_state = df_state.astype(np.bool)
    df = pd.concat([df, df_state], axis=1)

    # convert time to ms
    df = df.rename_axis("time")

    return df


def split(
    df: pd.DataFrame,
    window_ms: int,
    hop_ms: int,
    fs_khz: float,
) -> list[pd.DataFrame]:
    
    # Create sample indicies
    num_samples = int((window_ms) * fs_khz)
    hop_samples = int((hop_ms) * fs_khz)
    
    # Separate samples
    samples = []
    for start in range(
        0, len(df) - num_samples + 1, hop_samples
        ):
        end = start + num_samples
        sample = df.iloc[start:end]
        if len(sample) == num_samples:
            samples.append(sample)
            
    return samples

def transform_individual_sample(
    x: np.ndarray,
    ) -> np.ndarray:
    x_tensor = torch.from_numpy(x).float()
    y = torch.stft(
        x_tensor, 
        n_fft=1024, 
        hop_length=256, 
        window=torch.hann_window(1024), 
        return_complex=True
    )
    y = torch.log(torch.abs(y))
    # y = torch.clip(y, min=-10, max=5)
    return y.numpy()


def create_missing_signal_dataframes(
    cfg: dict,
    processed_signals: set,
    reference_df: pd.DataFrame
) -> list[pd.DataFrame]:
    """Create fully off dataframes for missing signals using reference dataframe structure."""
    
    missing_dfs = []
    
    for signal in cfg["signal"]:
        signal_name, signal_abbr, do_transform = signal
        if signal_abbr not in processed_signals:
            print(f"Creating fully off dataframe for missing signal {signal_name}")
            
            # Create off dataframe by copying structure and zeroing values
            off_df = reference_df.copy()
            
            # Get columns that belong to the reference signal (to replace with new signal columns)
            ref_signal_cols = [col for col in off_df.columns if not col.endswith('_state')]
            
            # Create new column names for the missing signal
            new_cols = {}
            new_state_cols = {}
            
            for i, col in enumerate(ref_signal_cols):
                new_col_name = f"{signal_abbr}col{i}"
                new_cols[col] = new_col_name
                new_state_cols[f"{col}_state"] = f"{new_col_name}_state"
            
            # Rename columns to match the missing signal
            off_df = off_df.rename(columns={**new_cols, **new_state_cols})
            
            # Set all data columns to 0 and all state columns to False
            for col in off_df.columns:
                if col.endswith('_state'):
                    off_df[col] = False
                else:
                    off_df[col] = 0.0
            
            missing_dfs.append(off_df)
    
    return missing_dfs


def transform_samples(
    samples: list[pd.DataFrame],
    directory: Path,
    signal_config: list[tuple],
    shot: int,
) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(samples)} samples for shot {shot}")
    samples_dict = []
    
    # Create mapping from signal abbreviation to whether it should be transformed
    transform_map = {}
    for signal_name, signal_abbr, should_transform in signal_config:
        transform_map[signal_abbr] = should_transform
    
    for i, sample in enumerate(samples):
        
        # Remove columns ending with '_state'
        sample_to_save = sample.loc[:, ~sample.columns.str.endswith('_state')]
        
        # Only save if not fully off (i.e., at least one True in any state col)
        state_cols = [col for col in sample.columns if col.endswith('_state')]
        if np.any(sample[state_cols].to_numpy()):
            
            # First pass: apply transformations and collect results
            sample_dict = {}
            transformed_sample = None
            original_time_length = len(sample_to_save)
            
            for col in sample_to_save.columns:
                # Convert each column to float32 numpy array
                col_array = sample_to_save[col].values.astype(np.float32)
                
                # Determine if this column should be transformed based on signal abbreviation
                should_transform = False
                for signal_abbr in transform_map.keys():
                    if col.startswith(signal_abbr):
                        should_transform = transform_map[signal_abbr]
                        break
                
                if should_transform:
                    transformed_array = transform_individual_sample(col_array)
                    sample_dict[col] = transformed_array
                    # Store an example transformed sample to get target dimensions
                    if transformed_sample is None:
                        transformed_sample = transformed_array
                        print(f"Reference transformed sample shape: {transformed_array.shape}")
                else:
                    # Store original array for now, will resample later
                    sample_dict[col] = col_array
            
            # Second pass: resample non-transformed samples to match transformed dimensions
            if transformed_sample is not None:
                target_width = transformed_sample.shape[-1]  # Last dimension is time
                # Calculate target sample frequency based on transformed sample
                target_fs = target_width / original_time_length
                print(f"Target frequency ratio: {target_fs:.4f} (target width: {target_width}, original length: {original_time_length})")
                
                for col in sample_dict.keys():
                    # Check if this column was transformed
                    should_transform = False
                    for signal_abbr in transform_map.keys():
                        if col.startswith(signal_abbr):
                            should_transform = transform_map[signal_abbr]
                            break
                    
                    if not should_transform:
                        # Resample non-transformed data to match target width
                        original_array = sample_dict[col]
                        resampled_array = resample_nearest(original_array, target_width)
                        
                        # Crop end if needed to ensure exact match
                        if len(resampled_array) > target_width:
                            resampled_array = resampled_array[:target_width]
                        elif len(resampled_array) < target_width:
                            # Pad with zeros if too short (shouldn't happen with resample_nearest)
                            pad_width = target_width - len(resampled_array)
                            resampled_array = np.pad(resampled_array, (0, pad_width), mode='constant')
                        
                        sample_dict[col] = resampled_array.astype(np.float32)
                        print(f"Resampled {col} from {len(original_array)} to {len(resampled_array)}")
            
            samples_dict.append(sample_dict)
            print(f"Sample {i} processed with {len(sample_dict)} signals")

    return samples_dict

def save_samples(
    samples: list[dict],
    directory: Path,
    shot: int
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(samples)} samples to {directory}")
    for i, sample in enumerate(samples):
        # Save using joblib
        joblib.dump(sample, directory / f"{shot}_{i}.pkl", compress=True)

def process_shot(
    shot: int,
    cfg: dict,
    out_dir: Path,
) -> None:
    
    dfs = []
    start_time = None
    end_time = None
    
    try:
        start_time, end_time = running_time(
            directory=Path(cfg["raw_data_dir"]),
                shot=shot,
                ip_threshold=cfg["ip_threshold"]
            )
        reference_len = int((end_time - start_time) * cfg["fs_khz"])
        print(f"Running time for shot {shot}: {start_time} to {end_time}")  
    except Exception as e:
        print(f"Error: Could not determine running time for shot {shot}: {e}")
        return
    
    # Process each signal and track which ones succeeded
    processed_signals = set()
    
    for signal in cfg["signal"]:
        signal_name, signal_abbr, is_transformed = signal
        df = None
        
        try:
            # Try to extract and process the signal
            df = extract(shot=shot, directory=Path(cfg["raw_data_dir"]), signal=signal_name)
            df.columns = [f"{signal_abbr}{col}" if col != "time" else col for col in df.columns]
            df = align(df, start_time, end_time, cfg["fs_khz"])
            processed_signals.add(signal_abbr)
            dfs.append(df)
            print(f"Successfully processed signal {signal_name} for shot {shot}")
            
        except Exception as e:
            print(f"Error processing signal {signal_name} for shot {shot}: {e}")
    
    if not dfs:
        print(f"Error: No dataframes created for shot {shot}")
        return

    # For missing signals, create "fully off" dataframes using the structure of the last dataframe
    if len(processed_signals) < len(cfg["signal"]):
        reference_df = dfs[-1]  # Use the last successfully processed dataframe as reference
        missing_dfs = create_missing_signal_dataframes(cfg, processed_signals, reference_df)
        dfs.extend(missing_dfs)
        
    df = pd.concat(dfs, axis=1, join='inner')

    num_samples = len(df)
    new_index = np.linspace(start_time, end_time, num_samples)
    df.index = new_index
    df.index = pd.to_timedelta(df.index, unit='ms')
    
    samples = [df] # no splitting for this dataset
    print(f"Shot {shot} has {len(samples)} samples after splitting.")
    samples_dict = transform_samples(samples, out_dir, cfg["signal"], shot)

    save_samples(samples_dict, out_dir, shot)
    print(f"Processed shot {shot} successfully with {len(cfg['signal'])} signals.")

    return


def index_dataset(out_dir: Path) -> None:
    
    files = list(out_dir.glob("*.pkl"))
    df_files = pd.DataFrame({'files': [str(file) for file in files]})
    df_files.to_pickle(out_dir / "index.pkl")

    print(f"Indexed {len(files)} files.")


def prepare_dataset(cfg: dict) -> None:
    
    raw_data_dir = Path(cfg["raw_data_dir"])
    cache_dir = Path(cfg["output_dir"]) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort all shot numbers
    print(f"Collecting shots from {raw_data_dir}...")
    all_shots = [
        int(p.stem) 
        for p in raw_data_dir.iterdir() 
        if p.suffix == ".h5"]
    all_shots.sort()
    
    # if cfg["randomize_shots"]:
    #     np.random.seed(cfg["random_seed"])
    #     all_shots = np.random.permutation(all_shots)
    # all_shots = all_shots[:cfg["num_shots"]]
    
    # print(f"Processing {len(all_shots)} shots into cache...")
    
    mapper = ParallelMapper()
    mapper(process_shot, [170000], cfg=cfg, out_dir=cache_dir)

    # for shot in tqdm(all_shots): # for debugging
    #     process_shot(shot, cfg, out_dir=cache_dir)
    #     break
    
    # Move cached files into train/test split
    print("Splitting dataset into train and valid sets...")
    all_files = list(cache_dir.glob("*.pkl"))
    all_files.sort()
    train_files, valid_files = train_test_split(
        all_files, 
        test_size=cfg.get("train_test_split", 0.2), 
        random_state=cfg["random_seed"])

    train_dir = Path(cfg["output_dir"]) / "train"
    valid_dir = Path(cfg["output_dir"]) / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    for f in train_files:
        f.rename(train_dir / f.name)
    for f in valid_files:
        f.rename(valid_dir / f.name)
    
    # Index the datasets
    index_dataset(train_dir)
    index_dataset(valid_dir)
        
    # Remove cache directory after splitting
    for f in cache_dir.glob("*"): f.unlink()
    cache_dir.rmdir()

    print("Dataset preparation complete.")

if __name__ == "__main__":
    cfg = sample_cfg.copy()
    prepare_dataset(cfg=cfg)