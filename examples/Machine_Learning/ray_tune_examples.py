"""Examples demonstrating hyperparameter tuning with Ray Tune."""

import atexit
import os
import sys
from pathlib import Path

import torch
from ray import tune
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.faith.train.models.autoencoder import BlockBasedAutoencoder
from src.faith.train.tuning import (
    CustomSearchSpace,
    RayTuner,
    SearchSpaces,
    get_search_space,
)

# Ensure Ray cleanup on exit
try:
    from src import cleanup_ray

    atexit.register(cleanup_ray)
except ImportError:
    pass


def get_results_dir(name: str) -> str:
    """Get absolute path for results directory.

    Parameters
    ----------
    name : str
        Directory name.

    Returns
    -------
    str
        Absolute path.
    """
    results_dir = Path.cwd() / "tune_results" / name
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir)


def create_dummy_dataset(num_samples: int = 1000, batch_size: int = 32):
    """Create dummy dataset for tuning examples.

    Parameters
    ----------
    num_samples : int, optional
        Number of samples, by default 1000.
    batch_size : int, optional
        Batch size, by default 32.

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary with train and val dataloaders.
    """
    # Create random spectrogram-like data
    data = torch.randn(num_samples, 4, 100, 128)

    # Split into train/val
    train_size = int(0.8 * num_samples)
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Create datasets (autoencoder: input = target)
    train_dataset = TensorDataset(train_data, train_data)
    val_dataset = TensorDataset(val_data, val_data)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader}


def example_basic_tuning() -> None:
    """Basic hyperparameter tuning example."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Hyperparameter Tuning")
    print("=" * 60)

    # Create data
    data_loaders = create_dummy_dataset(num_samples=200, batch_size=16)
    print(f"Created dataset with {len(data_loaders['train'])} train batches")

    # Base model configuration
    model_base_config = {
        "input_channels": 4,
    }

    # Create tuner
    tuner = RayTuner(
        model_class=BlockBasedAutoencoder,
        model_base_config=model_base_config,
        data_loaders=data_loaders,
        num_samples=5,  # Small for demo
        max_epochs_per_trial=5,  # Increased to work with ASHA scheduler
        gpus_per_trial=0.0,  # CPU only for demo
        scheduler_type="asha",
        search_algorithm="optuna",
        storage_path=get_results_dir("basic_tuning"),
    )

    # Define search space
    search_space = get_search_space("basic", learning_rate_range=(1e-4, 1e-2))

    print("Search space:", search_space)

    # Run tuning
    print("\nStarting hyperparameter search...")
    analysis = tuner.tune(search_space, name="basic_autoencoder_tune")

    # Get best configuration
    best_config = tuner.get_best_config(analysis)
    print(f"\nBest configuration: {best_config}")

    # Train final model
    print("\nTraining final model with best hyperparameters...")
    tuner.train_best_model(analysis, max_epochs=5, save_path="best_basic_model.pth")
    print("Basic tuning completed!")


def example_architecture_search() -> None:
    """Architecture search example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Architecture Search")
    print("=" * 60)

    # Create data
    data_loaders = create_dummy_dataset(num_samples=150, batch_size=8)

    # Base configuration
    model_base_config = {
        "input_channels": 4,
    }

    # Create tuner focused on architecture
    tuner = RayTuner(
        model_class=BlockBasedAutoencoder,
        model_base_config=model_base_config,
        data_loaders=data_loaders,
        num_samples=4,
        max_epochs_per_trial=4,  # Increased for ASHA compatibility
        gpus_per_trial=0.0,
        scheduler_type="fifo",  # Simple scheduler for architecture search
        search_algorithm="random",
    )

    # Architecture-focused search space
    search_space = get_search_space(
        "architecture",
        learning_rate=1e-3,  # Fixed
        layer_choices=[2, 3],
        width_choices=[32, 64],
    )

    print("Architecture search space:", search_space)

    # Run tuning
    analysis = tuner.tune(search_space, name="architecture_search")

    # Get results
    best_config = tuner.get_best_config(analysis)
    print(f"\nBest architecture: {best_config}")

    # Compare all trials
    df = analysis.dataframe()
    print("\nAll trial results:")
    print(df[["config/num_layers", "val_loss"]].head())


def example_custom_search_space() -> None:
    """Custom search space example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Search Space")
    print("=" * 60)

    # Create data
    data_loaders = create_dummy_dataset(num_samples=100, batch_size=16)

    # Model configuration
    model_base_config = {
        "input_channels": 4,
    }

    # Create custom search space using builder
    custom_space = (
        CustomSearchSpace()
        .add_continuous("learning_rate", 1e-5, 1e-2, log_scale=True)
        .add_discrete("activation", ["relu", "gelu"])
        .add_continuous("dropout", 0.0, 0.3)
        .add_fixed("weight_decay", 1e-5)
        .build()
    )

    print("Custom search space:", custom_space)

    # Create tuner
    tuner = RayTuner(
        model_class=BlockBasedAutoencoder,
        model_base_config=model_base_config,
        data_loaders=data_loaders,
        num_samples=3,
        max_epochs_per_trial=4,  # Increased for scheduler compatibility
        gpus_per_trial=0.0,
    )

    # Run tuning
    analysis = tuner.tune(custom_space, name="custom_search")

    # Results
    best_config = tuner.get_best_config(analysis)
    print(f"\nBest custom configuration: {best_config}")


def example_quick_parameter_test() -> None:
    """Quick single parameter testing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Quick Parameter Testing")
    print("=" * 60)

    # Create data
    data_loaders = create_dummy_dataset(num_samples=80, batch_size=8)

    # Test different learning rates quickly
    model_base_config = {"input_channels": 4}

    tuner = RayTuner(
        model_class=BlockBasedAutoencoder,
        model_base_config=model_base_config,
        data_loaders=data_loaders,
        num_samples=3,
        max_epochs_per_trial=4,  # Minimum for ASHA
        gpus_per_trial=0.0,
    )

    # Quick search for learning rate only
    search_space = SearchSpaces.quick_search(
        param_name="learning_rate",
        param_choices=[1e-4, 5e-4, 1e-3],
        base_config={"weight_decay": 1e-5},
    )

    print("Quick search space:", search_space)

    # Run tuning
    analysis = tuner.tune(search_space, name="quick_lr_test")

    # Show all results
    print("\nLearning rate comparison:")
    df = analysis.dataframe()
    for _, row in df.iterrows():
        lr = row["config/learning_rate"]
        val_loss = row["val_loss"]
        print(f"LR: {lr:.2e} -> Val Loss: {val_loss:.4f}")


def example_advanced_configuration() -> None:
    """Advanced tuning configuration example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Advanced Configuration")
    print("=" * 60)

    # Create data
    data_loaders = create_dummy_dataset(num_samples=120, batch_size=12)

    # Model configuration for block-based autoencoder
    model_base_config = {
        "input_channels": 4,
    }

    # Advanced tuner configuration
    tuner = RayTuner(
        model_class=BlockBasedAutoencoder,
        model_base_config=model_base_config,
        data_loaders=data_loaders,
        num_samples=4,
        max_epochs_per_trial=6,  # Sufficient for PBT
        gpus_per_trial=0.0,
        scheduler_type="pbt",  # Population Based Training
        search_algorithm="optuna",
        metric="val_loss",
        mode="min",
        storage_path=get_results_dir("advanced_tuning"),
    )

    # Block-based autoencoder search space
    # (parameters that actually work with the model)
    search_space = {
        # Training hyperparameters
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "scheduler_type": tune.choice(["cosine", "linear", "none"]),
        # Model architecture
        "activation": tune.choice(["relu", "gelu", "swish", "leaky_relu"]),
        # Skip architecture parameters that require custom block_configs
        # as they are more complex to implement in this example
    }

    print("Advanced search space:", search_space)

    # Run tuning with custom name
    analysis = tuner.tune(search_space, name="advanced_block_autoencoder", resume=False)

    # Detailed analysis
    best_trial = analysis.get_best_trial("val_loss", "min")
    print("\nBest trial:")
    print(f"  Config: {best_trial.config}")
    print(f"  Final val_loss: {best_trial.last_result['val_loss']:.4f}")
    print(f"  Training time: {best_trial.last_result.get('time_total_s', 0):.1f}s")

    # Train final model with more epochs
    print("\nTraining final model...")
    tuner.train_best_model(analysis, max_epochs=5, save_path="advanced_best_model.pth")

    print("Advanced tuning completed!")
    print(f"Results saved to: {tuner.storage_path}")


def example_model_comparison() -> None:
    """Compare different model configurations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Model Configuration Comparison")
    print("=" * 60)

    # Create data
    data_loaders = create_dummy_dataset(num_samples=100, batch_size=16)

    # Test different model configurations
    configs_to_test = [
        {
            "name": "small_model",
            "config": {"input_channels": 4},
            "search_space": {"learning_rate": 1e-3},
        },
        {
            "name": "medium_model",
            "config": {"input_channels": 4},
            "search_space": {"learning_rate": 1e-3},
        },
        {
            "name": "large_model",
            "config": {"input_channels": 4},
            "search_space": {"learning_rate": 5e-4},
        },
    ]

    results = {}

    for config_info in configs_to_test:
        print(f"\nTesting {config_info['name']}...")

        tuner = RayTuner(
            model_class=BlockBasedAutoencoder,
            model_base_config=config_info["config"],
            data_loaders=data_loaders,
            num_samples=1,  # Single trial per config
            max_epochs_per_trial=3,
            gpus_per_trial=0.0,
        )

        analysis = tuner.tune(config_info["search_space"], name=config_info["name"])

        best_trial = analysis.get_best_trial("val_loss", "min")
        results[config_info["name"]] = {
            "val_loss": best_trial.last_result["val_loss"],
            "params": sum(
                p.numel()
                for p in BlockBasedAutoencoder(**config_info["config"]).parameters()
            ),
        }

    # Compare results
    print("\n" + "=" * 40)
    print("MODEL COMPARISON RESULTS")
    print("=" * 40)
    for name, result in results.items():
        print(
            f"{name:12} | Val Loss: {result['val_loss']:.4f} | "
            f"Params: {result['params']:,}"
        )

    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]["val_loss"])
    print(f"\nBest model: {best_model[0]} (val_loss: {best_model[1]['val_loss']:.4f})")


def run_all_examples() -> None:
    """Run all tuning examples."""
    print("Running Ray Tune Examples...")
    print("Note: These are minimal examples for demonstration.")
    print("For real tuning, use larger datasets and more trials.\n")

    try:
        example_basic_tuning()
        example_architecture_search()
        example_custom_search_space()
        example_quick_parameter_test()
        example_advanced_configuration()
        example_model_comparison()

        print("\n" + "=" * 60)
        print("ALL TUNING EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nCheck the following directories for results:")
        print("  - ./ray_results/")
        print("  - ./advanced_tune_results/")
        print("  - ./final_training_logs/")

    except ImportError as e:
        print(f"\nSkipping examples due to missing dependency: {e}")
        print("Install Ray Tune with: pip install ray[tune] optuna hyperopt")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run examples
    run_all_examples()
