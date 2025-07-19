"""Example usage of JoblibDataset with real file inspection and training."""

from torch.utils.data import DataLoader
import torch
from pathlib import Path

# Assuming your package structure
from faith.train.data.datasets.file_based import JoblibDataset
from faith.train.data.loaders.factory import worker_init_fn
from faith.train.models.autoencoder import BlockBasedAutoencoder
from faith.train.training import train_model


def collate_fn(
    data: list[tuple[torch.Tensor, ...]]
) -> tuple[torch.Tensor, ...]:
    """
    Custom collate function to remove the highest frequency bin of
    spectrograms.  # TODO list of dicts to dict of tensors

    Parameters
    ----------
    data : list of tuples
        List of tuples containing input and target tensors.

    Returns
    -------
    Tuples
        Processed list of tuples with the last frequency bin removed.
    """
    batch_size = len(data)
    if isinstance(data[0], dict):
        collated = {}
        for key in data[0].keys():
            values = [item[key] for item in data]

            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
    else:
        processed_inputs = torch.stack([d[0] for d in data])
        processed_targets = torch.stack([d[1] for d in data])

        processed_inputs = processed_inputs[:, :, :, :-1]
        processed_targets = processed_targets[:, :, :, :-1]

        return processed_inputs, processed_targets


def inspect_joblib_file(file_path: str):
    """Inspect a joblib file to understand its structure."""
    from joblib import load

    print(f"Inspecting file: {file_path}")
    print("=" * 50)

    # Load file to see structure
    data_dict = load(file_path, mmap_mode='r')

    print("Available keys:", list(data_dict.keys()))
    print()

    # Inspect each key
    for key, value in data_dict.items():
        print(f"Key: '{key}'")
        if hasattr(value, 'shape'):
            print(f"  Type: {type(value)}")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
        else:
            print(f"  Type: {type(value)}")
            print(f"  Value: {value}")
        print()

    # Store keys before cleaning up
    available_keys = list(data_dict.keys())

    # Clean up
    del data_dict

    return available_keys


def example_basic_usage():
    """Basic usage example with a single file."""
    print("EXAMPLE 1: Basic Usage")
    print("=" * 40)

    file_path = "171348_0.joblib"  # Replace with your actual file

    # First, inspect the file to understand its structure
    print("Step 1: Inspect file structure")
    available_keys = inspect_joblib_file(file_path)

    # Create dataset with auto-detection
    print("Step 2: Create dataset with auto-detection")
    dataset = JoblibDataset(
        file_paths=[file_path],
        subseq_len=128,  # Extract 128-sample subsequences
        auto_detect_keys=True,  # Let it figure out the keys
        validate_on_init=True
    )

    print("Dataset created successfully!")
    print(f"Total subsequences: {len(dataset)}")
    print(f"Is autoencoder mode: {dataset.is_autoencoder_mode}")
    print()

    # Test data loading without worker_init()
    print("Step 3: Inspect data shapes")
    input_shape, target_shape = dataset.get_sample_shape()
    print(f"Input shape: {input_shape}")
    print(f"Target shape: {target_shape}")

    # Get a sample for inspection
    sample_input, sample_target = dataset.peek_sample()
    print(f"Sample input dtype: {sample_input.dtype}")
    print(f"Sample target dtype: {sample_target.dtype}")
    print()

    # Create DataLoader with proper worker init function
    print("Step 4: Test DataLoader")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )

    # Test loading a batch
    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Target shape: {targets.shape}")
        print(f"  Input dtype: {inputs.dtype}")
        print(f"  Target dtype: {targets.dtype}")

        if batch_idx == 0:  # Only show first batch
            break

    print()
    return dataset


def example_custom_configuration():
    """Example with custom key configuration."""
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 40)

    file_path = "170797_0.joblib"  # Replace with your actual file

    # Create dataset with specific keys (adjust based on your file)
    dataset = JoblibDataset(
        file_paths=[file_path],
        subseq_len=256,  # Longer subsequences
        input_key=['co2', 'ece'],  # Specify your input keys as list
        target_key=None,  # Autoencoder mode
        chunking_strategy='sliding_window',  # Overlapping chunks
        overlap=64,  # 64-sample overlap
        validate_on_init=True
    )

    print("Dataset with custom config:")
    print(f"  Input key: {dataset.input_key}")
    print(f"  Target key: {dataset.target_key}")
    print(f"  Subsequence length: {dataset.subseq_len}")
    print(f"  Chunking strategy: {dataset.chunking_strategy}")
    print(f"  Is multi-input: {dataset.is_multi_input}")
    print(f"  Total subsequences: {len(dataset)}")
    print()

    # Show dataset summary
    summary = dataset.summary()
    print("Dataset summary:")
    for key, value in summary.items():
        if key not in ['file_metadata', 'file_paths']:  # Skip verbose fields
            print(f"  {key}: {value}")
    print()

    # Inspect multi-key data shapes
    print("Data shape inspection:")
    input_shapes, target_shapes = dataset.get_sample_shape()
    print(f"Input shapes: {input_shapes}")
    print(f"Target shapes: {target_shapes}")

    # Get sample data to inspect
    sample_inputs, sample_targets = dataset.peek_sample()
    print("Sample data types:")
    if isinstance(sample_inputs, dict):
        for key, tensor in sample_inputs.items():
            print(
                f"  Input '{key}': shape {tensor.shape}, dtype {tensor.dtype}")
    if isinstance(sample_targets, dict):
        for key, tensor in sample_targets.items():
            print(f"  Target '{key}': shape {tensor.shape}, "
                  f"dtype {tensor.dtype}")
    print()

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        worker_init_fn=worker_init_fn
    )

    # Test loading a batch
    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        if isinstance(inputs, dict):
            print(f"  Input keys: {list(inputs.keys())}")
            for key, val in inputs.items():
                print(f"    Input '{key}' shape: {val.shape}, "
                      f"dtype: {val.dtype}")
        if isinstance(targets, dict):
            print(f"  Target keys: {list(targets.keys())}")
            for key, val in targets.items():
                print(f"    Target '{key}' shape: {val.shape}, "
                      f"dtype: {val.dtype}")

        if batch_idx == 0:  # Only show first batch
            break

    return dataset


def example_multiple_files():
    """Example with multiple files and advanced features."""
    print("EXAMPLE 3: Multiple Files with Advanced Features")
    print("=" * 40)

    # Use glob pattern or directory to find multiple files
    file_pattern = "./*.joblib"  # Adjust to your path

    dataset = JoblibDataset(
        file_paths=file_pattern,  # Can be directory, glob pattern, or list
        subseq_len=128,
        input_key='ece',
        target_key=None,  # Autoencoder mode
        file_pattern="*.joblib",  # Pattern for file discovery
        max_files=10,  # Limit to 10 files
        sort_files=True,  # Sort files by name
        balance_files=True,  # Balance samples across files
        chunking_strategy='non_overlapping',
        validate_on_init=True
    )

    print("Multi-file dataset:")
    print(f"  Number of files: {dataset.num_files}")
    print(f"  Total subsequences: {len(dataset)}")
    print(f"  Balanced: {dataset.balance_files}")
    print()

    # Show file statistics
    print("File statistics:")
    file_stats = dataset.get_file_stats()
    for i, stats in enumerate(file_stats[:3]):  # Show first 3 files
        print(f"  File {i}: {Path(stats['file_path']).name}")
        print(f"    Subsequences: {stats['num_subsequences']}")
        print(f"    Sequence length: {stats['sequence_length']}")

    if len(file_stats) > 3:
        print(f"  ... and {len(file_stats) - 3} more files")
    print()

    # Inspect data from multiple files
    print("Inspecting data from different files:")
    for i in range(min(3, dataset.num_files)):
        try:
            input_shape, target_shape = dataset.get_sample_shape(file_idx=i)
            print(f"  File {i}: input {input_shape}, target {target_shape}")
        except Exception as e:
            print(f"  File {i}: Error - {e}")
    print()

    # Split dataset by files
    print("Splitting dataset by files...")
    train_ds, val_ds = dataset.split_by_files(
        train_ratio=0.8,
        val_ratio=0.2,
        random_seed=42
    )

    print("Split results:")
    print(f"  Train: {len(train_ds)} subsequences from {train_ds.num_files} "
          f"files")
    print(f"  Val: {len(val_ds)} subsequences from {val_ds.num_files} files")
    print()

    return train_ds, val_ds


def example_autoencoder_training():
    """Example: Train an autoencoder with JoblibDataset."""
    print("EXAMPLE 4: Autoencoder Training")
    print("=" * 40)

    # Create dataset for autoencoder training
    dataset = JoblibDataset(
        file_paths="171348_0.joblib",  # Adjust path
        subseq_len=128,
        input_key='co2',
        target_key=None,  # Autoencoder mode: input = target
        auto_detect_keys=True,
        validate_on_init=True
    )

    print(f"Autoencoder dataset: {len(dataset)} samples")

    # Get data shape for model configuration WITHOUT worker_init()
    input_shape, target_shape = dataset.get_sample_shape()
    print(f"Input shape: {input_shape}")
    print(f"Target shape: {target_shape}")

    # Alternative: Get a full sample for inspection
    sample_input, sample_target = dataset.peek_sample()
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Sample target shape: {sample_target.shape}")
    print(f"Sample input dtype: {sample_input.dtype}")

    # Get detailed sample information
    sample_info = dataset.get_sample_info(0)
    print(f"Sample info: {sample_info}")
    print()

    # Create model based on data shape
    model = BlockBasedAutoencoder(
        input_channels=sample_input.shape[0],  # Number of channels
        hidden_dim=64,
        activation='gelu'
    )

    print(f"Model created with {model.parameter_count:,} parameters")

    # Create DataLoader with proper worker init
    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
    )

    d = next(iter(train_loader))

    # Train the model
    print("Starting training...")
    lightning_model, trainer = train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=None,  # Using same data for demo
        max_epochs=5,
        learning_rate=1e-4,
        logger_type="tensorboard",
        project_name="joblib-autoencoder"
    )

    print("Training completed!")
    print("View logs with: tensorboard --logdir=./logs")
    print()


def example_multikey_usage():
    """Example showing multi-key input/target usage."""
    print("EXAMPLE 5: Multi-Key Input/Target Usage")
    print("=" * 40)

    # Create dataset with multiple input and target keys
    dataset = JoblibDataset(
        file_paths="171348_0.joblib",  # Adjust path
        subseq_len=128,
        input_key=['co2', 'ece'],  # Multiple input keys
        target_key=['co2', 'mhr'],
        # Multiple target keys (or None for autoencoder)
        validate_on_init=True
    )

    print("Multi-key dataset:")
    print(f"  Input keys: {dataset.input_key}")
    print(f"  Target keys: {dataset.target_key}")
    print(f"  Is multi-input: {dataset.is_multi_input}")
    print(f"  Is multi-target: {dataset.is_multi_target}")
    print(f"  Total subsequences: {len(dataset)}")
    print()

    # Inspect shapes for each key
    print("Shape inspection:")
    input_shapes, target_shapes = dataset.get_sample_shape()
    print("Input shapes by key:")
    for key, shape in input_shapes.items():
        print(f"  '{key}': {shape}")
    print("Target shapes by key:")
    for key, shape in target_shapes.items():
        print(f"  '{key}': {shape}")
    print()

    # Get sample data
    sample_inputs, sample_targets = dataset.peek_sample()
    print("Sample data inspection:")
    print(f"Input data keys: {list(sample_inputs.keys())}")
    print(f"Target data keys: {list(sample_targets.keys())}")

    for key, tensor in sample_inputs.items():
        print(f"  Input '{key}': shape {tensor.shape}, dtype {tensor.dtype}")
        print(f"    Min: {tensor.min().item():.4f}, "
              f"Max: {tensor.max().item():.4f}")

    for key, tensor in sample_targets.items():
        print(f"  Target '{key}': shape {tensor.shape}, dtype {tensor.dtype}")
        print(f"    Min: {tensor.min().item():.4f}, "
              f"Max: {tensor.max().item():.4f}")
    print()

    # Test DataLoader
    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=2,
                        worker_init_fn=worker_init_fn,
                        collate_fn=collate_fn)

    print("DataLoader test:")
    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Input batch keys: {list(inputs.keys())}")
        print(f"  Target batch keys: {list(targets.keys())}")

        for key, tensor in inputs.items():
            print(f"    Input '{key}' batch shape: {tensor.shape}")
        for key, tensor in targets.items():
            print(f"    Target '{key}' batch shape: {tensor.shape}")

        if batch_idx == 0:  # Only show first batch
            break

    return dataset


def example_error_handling():
    """Example showing error handling and debugging."""
    print("EXAMPLE 6: Error Handling and Debugging")
    print("=" * 40)

    # Test with mix of valid and invalid files
    test_files = [
        "171348_0.joblib",  # Replace with your actual valid file
        "nonexistent_file.joblib",  # This should fail
    ]

    print("Testing error handling with mixed file list:")
    for f in test_files:
        exists = "✓" if Path(f).exists() else "✗"
        print(f"  {exists} {f}")
    print()

    # First, let's validate files BEFORE creating the dataset
    print("Manual file validation:")
    valid_files = []

    for file_path in test_files:
        try:
            # Try to inspect the file directly
            from joblib import load
            data_dict = load(file_path, mmap_mode='r')
            available_keys = list(data_dict.keys())
            del data_dict  # Clean up

            print(f"  ✓ {Path(file_path).name} - Keys: {available_keys}")
            valid_files.append(file_path)

        except Exception as e:
            print(f"  ✗ {Path(file_path).name} - Error: {e}")
    print()

    if not valid_files:
        print("No valid files found! Cannot proceed with dataset examples.")
        return None

    # Create dataset with only valid files
    print(f"Creating dataset with {len(valid_files)} valid files...")
    try:
        dataset = JoblibDataset(
            file_paths=valid_files,  # Only use valid files
            subseq_len=128,
            auto_detect_keys=True,
            validate_on_init=True  # This should work with valid files
        )

        print("✓ Dataset created successfully!")
        print(f"  Total subsequences: {len(dataset)}")
        print(f"  Number of files: {dataset.num_files}")
        print(f"  Input key: {dataset.input_key}")
        print(f"  Target key: {dataset.target_key}")
        print()

        # Now test file access - this should work
        print("Testing file access on valid dataset:")
        for i in range(dataset.num_files):
            try:
                input_shape, target_shape = dataset.get_sample_shape(
                    file_idx=i)
                print(
                    f"  File {i}: ✓ Input {input_shape}, Target {target_shape}")
            except Exception as e:
                print(f"  File {i}: ✗ Error - {e}")
        print()

        # Test peek functionality
        print("Testing peek functionality:")
        try:
            sample_input, sample_target = dataset.peek_sample()
            print("  ✓ Peek successful!")
            s = sample_input.shape \
                if not isinstance(sample_input, dict) \
                else {k: v.shape for k, v in sample_input.items()}
            print(f"    Input shape: {s}")
            s = sample_target.shape \
                if not isinstance(sample_target, dict) \
                else {k: v.shape for k, v in sample_target.items()}
            print(f"    Target shape: {s}")
        except Exception as e:
            print(f"  ✗ Peek failed: {e}")
        print()

        # Show dataset info for debugging
        print("Dataset summary:")
        info = dataset.summary()
        for key, value in info.items():
            if key not in ['file_metadata', 'file_paths']:
                print(f"  {key}: {value}")
        print()

        return dataset

    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None


def example_error_handling_advanced():
    """More advanced error handling example."""
    print("EXAMPLE 6B: Advanced Error Handling")
    print("=" * 40)

    # Test the validate_files method on a dataset with mixed files
    test_files = [
        "171348_0.joblib",  # Valid file
        "nonexistent_file.joblib",  # Invalid file
        "another_invalid_file.joblib",  # Another invalid file
    ]

    try:
        # Create dataset without validation to test the validate_files method
        dataset = JoblibDataset(
            file_paths=test_files,
            subseq_len=128,
            auto_detect_keys=True,
            validate_on_init=False  # Don't validate during init
        )

        print("Testing validate_files() method:")
        validation_results = dataset.validate_files()

        valid_files = []
        for file_path, is_valid, error_msg in validation_results:
            status = "✓" if is_valid else "✗"
            print(f"  {status} {Path(file_path).name}")
            if error_msg:
                print(f"    Error: {error_msg}")
            if is_valid:
                valid_files.append(file_path)
        print()

        if valid_files:
            print(f"Found {len(valid_files)} valid files. "
                  f"Creating new dataset...")
            # Create a new dataset with only valid files
            valid_dataset = JoblibDataset(
                file_paths=valid_files,
                subseq_len=128,
                auto_detect_keys=True,
                validate_on_init=True
            )

            print(f"✓ Valid dataset created with {len(valid_dataset)} "
                  f"subsequences")

            # Test this dataset
            input_shape, target_shape = valid_dataset.get_sample_shape()
            print(f"  Input shape: {input_shape}")
            print(f"  Target shape: {target_shape}")

            return valid_dataset
        else:
            print("No valid files found!")
            return None

    except Exception as e:
        print(f"Error in advanced error handling: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all examples."""
    print("JoblibDataset Usage Examples")
    print("=" * 60)
    print()

    # Note: You'll need to replace file paths with actual files
    print("NOTE: Replace file paths with your actual joblib files in each "
          "example.")
    print()

    try:
        """
        # Basic usage
        dataset1 = example_basic_usage()

        # Custom configuration
        dataset2 = example_custom_configuration()

        # Multiple files (comment out if you don't have multiple files)
        train_ds, val_ds = example_multiple_files()

        # Autoencoder training (comment out if you don't want to train)
        example_autoencoder_training()
        """
        # Multi-key usage
        dataset3 = example_multikey_usage()

        # Error handling
        dataset4 = example_error_handling()
        dataset5 = example_error_handling_advanced()

        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure to:")
        print("1. Replace file paths with real joblib files")
        print("2. Install required dependencies: joblib, torch, "
              "pytorch-lightning")
        print("3. Ensure your files have the expected structure")
        print("4. Update the input_key and target_key based on your data")


if __name__ == "__main__":
    main()