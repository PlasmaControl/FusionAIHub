# Fusion Dataset Preparation Pipeline

This module provides a modular, configurable pipeline for preparing fusion plasma datasets from HDF5 data files.

## Structure

The pipeline has been refactored into a modular structure:

```
src/fusionaihub/datasets/prepare/
├── config/
│   ├── default.yaml           # Default configuration file
│   └── raw.yaml              # Raw signal configuration
├── __init__.py               # Package initialization
├── signal_processing.py      # Signal resampling and transformation
├── data_extraction.py        # Data extraction and alignment
├── sample_processing.py      # Sample splitting and transformation
├── dataset_utils.py          # Dataset utilities and indexing
├── shot_processing.py        # Shot-level processing orchestration
├── prepare_dataset.py         # Main executable script
├── __main__.py               # Module entry point for direct execution
└── README.md                 # This file
```

## Core Modules

### `signal_processing.py`
- `resample_nearest()`: Resample signals to new length
- `transform_individual_sample()`: Apply STFT transformation

### `data_extraction.py`
- `extract()`: Extract signal data from HDF5 files
- `running_time()`: Determine plasma running time from current threshold
- `align()`: Align signals to common timebase and sampling frequency

### `sample_processing.py`
- `split()`: Split signals into overlapping time windows
- `transform_samples()`: Apply transformations and resample to match dimensions
- `save_samples()`: Save processed samples to disk

### `dataset_utils.py`
- `create_missing_signal_dataframes()`: Create placeholder data for missing signals
- `index_dataset()`: Create dataset index files

### `shot_processing.py`
- `process_shot()`: Main orchestration function for processing individual shots

## Configuration

The pipeline is configured using YAML files. The default configuration is in `config/default.yaml`:

```yaml
# Signal configuration - list of signals to process
signal:
  - ["magnetics_high_resolution", "mhr", true]
  - ["ece_cali", "ece", true]
  - ["co2_density", "co2", true]
  - ["gas", "gas", false]
  - ["ech", "ech", false]
  - ["p_inj", "pin", false]
  - ["t_inj", "tin", false]

# Processing parameters
randomize_shots: true
random_seed: 42
num_shots: 50
fs_khz: 500
ip_threshold: 0.1
window_ms: 250
hop_ms: 50

# Directory paths
raw_data_dir: "/scratch/gpfs/EKOLEMEN/d3d_fusion_data"
output_dir: "/scratch/gpfs/nc1514/FusionAIHub/data/foundation_v2"

# Additional parameters...
```

### Signal Configuration Format
Each signal is configured as a list: `[signal_name, abbreviation, should_transform]`
- `signal_name`: Name in the HDF5 file
- `abbreviation`: Short name for column prefixes
- `should_transform`: Whether to apply STFT transformation (boolean)

## Usage

### Command Line
```bash
# Use default configuration
python -m src.fusionaihub.datasets.prepare

# Use custom configuration file
python -m src.fusionaihub.datasets.prepare --config /path/to/config.yaml
```

### Programmatic Usage
```python
from src.fusionaihub.datasets.prepare.prepare_dataset import load_config, prepare_dataset

# Load configuration
cfg = load_config("path/to/config.yaml")

# Run dataset preparation
prepare_dataset(cfg)
```

### Customizing Configuration
Create a custom YAML file based on `config/default.yaml`:

```python
import yaml
from src.fusionaihub.datasets.prepare.prepare_dataset import prepare_dataset

# Load and modify configuration
with open("config/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Customize settings
cfg["num_shots"] = 100
cfg["raw_data_dir"] = "/path/to/your/data"
cfg["output_dir"] = "/path/to/output"

# Run with custom configuration
prepare_dataset(cfg)
```

## Output Structure

The pipeline produces the following output structure:

```
output_dir/
├── train/
│   ├── 170000_0.joblib          # Processed samples
│   ├── 170001_0.joblib
│   ├── ...
│   └── index.csv             # Dataset index
└── valid/
    ├── 170010_0.joblib
    ├── 170011_0.joblib
    ├── ...
    └── index.csv
```

Each `.joblib` file contains a dictionary with signal arrays, where transformed signals have STFT representation and non-transformed signals are resampled to match dimensions.

## Key Features

1. **Modular Design**: Each processing step is in a separate module for easier testing and modification
2. **YAML Configuration**: All parameters are configurable via YAML files
3. **Parallel Processing**: Uses `ParallelMapper` for efficient multi-shot processing
4. **Missing Signal Handling**: Automatically creates placeholder data for missing signals
5. **Flexible Transformations**: Configurable per-signal transformation (STFT or raw)
6. **Train/Validation Split**: Automatic dataset splitting with indexing
7. **Multiple Configurations**: Includes default and raw signal configurations

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- torch
- joblib
- PyYAML
- pathlib (built-in) 

# To-Do
Change YAML loading to hydra