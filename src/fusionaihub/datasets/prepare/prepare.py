import re
import numpy as np
import pandas as pd
import polars as pl
import h5py
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import resample, resample_poly
from sklearn.model_selection import train_test_split
from datetime import time, timedelta
from ..utils.parmap import ParallelMapper
import logging
from tqdm.auto import tqdm
import pickle
import torch
from concurrent.futures import ProcessPoolExecutor


log = logging.getLogger(__name__)

sample_cfg = {
    "signal": [
        ("magnetics_high_resolution", "mhr"),
        ("ece_cali", "ece"),
        ("bes", "bes"),
        ("co2_density", "co2"),
    ],
    "randomize_shots": True,
    "random_seed": 42,
    "num_shots": 50,
    "fs_khz": 500,
    "start_ms": 0,
    "end_ms": 5000,
    "window_ms": 250,
    "hop_ms": 50,
    "remove_empty": True,
    "train_test_split": 0.2,
    "raw_data_dir": "/scratch/gpfs/EKOLEMEN/d3d_fusion_data",
    "output_dir": "/scratch/gpfs/nc1514/specseg/data/foundation_v1",
}


def resample_nearest(y: np.ndarray, new_len: int) -> np.ndarray:
    orig_len = len(y)
    gcd = np.gcd(orig_len, new_len)
    up = new_len // gcd
    down = orig_len // gcd
    return resample_poly(y, up, down)
    # return resample(y, new_len)
    # return interp1d(np.linspace(0, 1, len(y)), y, kind='cubic')(np.linspace(0, 1, new_len))


def extract(
    shot: int,
    directory: Path,
    signal: str,
) -> pd.DataFrame:
    
    path = (directory / str(shot)).with_suffix(".h5")
    df = pd.read_hdf(path, key=signal)
    
    return pd.DataFrame(df)


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
    df.index = pd.to_timedelta(df.index, unit='ms')

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
    x = torch.from_numpy(x).float()
    transformed = torch.stft(
        x, 
        n_fft=1024, 
        hop_length=256, 
        window=torch.hann_window(1024), 
        return_complex=True
    )
    transformed = torch.log(torch.abs(transformed).clamp(min=1e-10))
    transformed = torch.clip(transformed, min=-10, max=5).numpy()
    return transformed

def save_samples(
    samples: list[pd.DataFrame],
    directory: Path,
    shot: int
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        
        # Remove columns ending with '_state'
        sample_to_save = sample.loc[:, ~sample.columns.str.endswith('_state')]
        
        # Only save if not fully off (i.e., at least one True in any state col)
        state_cols = [col for col in sample.columns if col.endswith('_state')]
        if np.any(sample[state_cols].to_numpy()):
            sample_array = sample_to_save.to_numpy().T
            sample_array = sample_array.astype(np.float32)
            sample_array = transform_individual_sample(sample_array)
            with open(directory / f"{shot}_{i}.pkl", 'wb') as f:
                pickle.dump(sample_array, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_shot(
    shot: int,
    cfg: dict,
    out_dir: Path,
) -> None:
    
    dfs = []
    for signal in cfg["signal"]:
        signal_name, signal_abbr = signal
        
        try:
            df = extract(shot=shot, directory=Path(cfg["raw_data_dir"]), signal=signal_name)
        except FileNotFoundError:
            print(f"Missing {shot} -- {signal_name}")
            return
        
        try:
            df.columns = [f"{signal_abbr}{col}" if col != "time" else col for col in df.columns]
        except Exception as e:
            print(f"Error renaming columns for shot {shot} and signal {signal_name}: {e}")
            return
        
        try:
            df = align(df, cfg["start_ms"], cfg["end_ms"], cfg["fs_khz"])
        except ValueError:
            print(f"Error aligning data for shot {shot} and signal {signal_name}.")
            return
        
        dfs.append(df)
        
    df = pd.concat(dfs, axis=1)
    
    # Split into windows
    samples = split(df, cfg["window_ms"], cfg["hop_ms"], cfg["fs_khz"])
    print(f"Shot {shot} has {len(samples)} samples after splitting.")
    
    # Save to cache (only non-fully-off windows)
    save_samples(samples, out_dir, shot)
    print(f"Processed shot {shot} successfully.")

    return


def index_dataset(out_dir: Path) -> None:
    
    files = list(out_dir.glob("*.pkl"))
    df_files = pd.DataFrame({'files': [str(file) for file in files]})
    df_files.to_pickle(out_dir / "index.pkl")

    print(f"Indexed {len(files)} files.")


def prepare_dataset(cfg: dict) -> None:
    
    cfg["num_samples"] = int((cfg["end_ms"] - cfg["start_ms"]) * cfg["fs_khz"])
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