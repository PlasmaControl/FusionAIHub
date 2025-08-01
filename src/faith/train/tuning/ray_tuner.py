"""Ray Tune integration for hyperparameter optimization."""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.optuna import OptunaSearch

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn(
        "Ray Tune not available. Install with: pip install ray[tune] optuna hyperopt",
        stacklevel=2,
    )

from ..training import LightningTrainer


def _resolve_path(path: str) -> str:
    """Resolve path to absolute path for Ray Tune compatibility.

    Parameters
    ----------
    path : str
        Input path (relative or absolute).

    Returns
    -------
    str
        Absolute path.
    """
    return str(Path(path).resolve())


def _safe_ray_init(**kwargs: Any) -> None:
    """Safely initialize Ray with error handling.

    Parameters
    ----------
    **kwargs
        Arguments passed to ray.init().
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray not available")

    if ray.is_initialized():
        return

    try:
        # Default Ray init arguments for stability
        default_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,  # Disable dashboard for stability
            "configure_logging": False,  # Avoid logging conflicts
        }
        default_args.update(kwargs)

        ray.init(**default_args)
    except Exception as e:
        warnings.warn(
            f"Ray initialization failed: {e}. Some features may not work.",
            stacklevel=2,
        )


def cleanup_ray() -> None:
    """Safely shutdown Ray."""
    if RAY_AVAILABLE and ray.is_initialized():
        try:
            ray.shutdown()
        except Exception as e:
            warnings.warn(f"Ray shutdown failed: {e}", stacklevel=2)


class RayTuner:
    """Ray Tune integration for hyperparameter optimization.

    This class provides a simple interface to tune hyperparameters of your
    autoencoder models using Ray Tune's optimization algorithms.
    """

    def __init__(
        self,
        model_class: type,
        model_base_config: dict[str, Any],
        data_loaders: dict[str, Any],
        num_samples: int = 20,
        max_epochs_per_trial: int = 10,
        gpus_per_trial: float = 0.25,
        cpus_per_trial: int = 1,
        scheduler_type: str = "asha",
        search_algorithm: str = "optuna",
        metric: str = "val_loss",
        mode: str = "min",
        storage_path: Optional[str] = None,
    ) -> None:
        """Initialize Ray Tuner.

        Parameters
        ----------
        model_class : type
            Model class to instantiate (e.g., BlockBasedAutoencoder).
        model_base_config : Dict[str, Any]
            Base configuration for model instantiation.
        data_loaders : Dict[str, Any]
            Dictionary containing 'train' and optionally 'val' dataloaders.
        num_samples : int, optional
            Number of hyperparameter configurations to try, by default 20.
        max_epochs_per_trial : int, optional
            Maximum epochs per trial, by default 10.
        gpus_per_trial : float, optional
            GPU fraction per trial, by default 0.25.
        cpus_per_trial : int, optional
            CPU cores per trial, by default 1.
        scheduler_type : str, optional
            Scheduler type ("asha", "pbt", "fifo"), by default "asha".
        search_algorithm : str, optional
            Search algorithm ("optuna", "hyperopt", "random"),
            by default "optuna".
        metric : str, optional
            Metric to optimize, by default "val_loss".
        mode : str, optional
            Optimization mode ("min" or "max"), by default "min".
        storage_path : Optional[str], optional
            Directory for Ray Tune results,
            by default None (uses current dir + ray_results).
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray Tune is required for hyperparameter tuning. "
                "Install with: pip install ray[tune] optuna hyperopt"
            )

        self.model_class = model_class
        self.model_base_config = model_base_config
        self.data_loaders = data_loaders
        self.num_samples = num_samples
        self.max_epochs_per_trial = max_epochs_per_trial
        self.gpus_per_trial = gpus_per_trial
        self.cpus_per_trial = cpus_per_trial
        self.scheduler_type = scheduler_type
        self.search_algorithm = search_algorithm
        self.metric = metric
        self.mode = mode

        # Resolve storage_path to absolute path
        if storage_path is None:
            storage_path = os.path.join(os.getcwd(), "ray_results")
        self.storage_path = _resolve_path(storage_path)

        # Ensure directory exists
        os.makedirs(self.storage_path, exist_ok=True)

        # Initialize Ray if not already done
        _safe_ray_init()

    def _validate_scheduler_params(self) -> None:
        """Validate and adjust scheduler parameters for consistency.

        Raises
        ------
        ValueError
            If configuration is invalid and cannot be automatically fixed.
        """
        if self.scheduler_type == "asha":
            if self.max_epochs_per_trial < 2:
                raise ValueError(
                    f"ASHA scheduler requires max_epochs_per_trial >= 2, "
                    f"got {self.max_epochs_per_trial}. "
                    f"Use scheduler_type='fifo' for single epoch trials."
                )
        elif self.scheduler_type == "pbt":
            if self.max_epochs_per_trial < 3:
                warnings.warn(
                    f"PBT scheduler works best with max_epochs_per_trial >= 3"
                    f", got {self.max_epochs_per_trial}. "
                    f"Consider using 'asha' or 'fifo'.",
                    stacklevel=2,
                )
            if self.num_samples < 4:
                warnings.warn(
                    f"PBT scheduler works best with num_samples >= 4, got "
                    f"{self.num_samples}. Consider using 'asha' scheduler.",
                    stacklevel=2,
                )
        elif self.scheduler_type == "fifo":
            return SchedulerType.FIFO  # Explicitly represent FIFO
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def _create_scheduler(self) -> tune.schedulers.TrialScheduler | None:
        """Create scheduler based on configuration.

        Returns
        -------
        ray.tune.schedulers.TrialScheduler
            Configured scheduler.
        """
        # Validate parameters first
        self._validate_scheduler_params()

        if self.scheduler_type == "asha":
            # Ensure grace_period is not greater than max_t
            grace_period = min(3, max(1, self.max_epochs_per_trial // 2))
            max_t = self.max_epochs_per_trial

            return ASHAScheduler(
                metric=self.metric,
                mode=self.mode,
                max_t=max_t,
                grace_period=grace_period,
                reduction_factor=2,
            )
        elif self.scheduler_type == "pbt":
            # For PBT, ensure perturbation_interval is reasonable
            perturbation_interval = min(2, max(1, self.max_epochs_per_trial // 3))

            return PopulationBasedTraining(
                time_attr="training_iteration",
                metric=self.metric,
                mode=self.mode,
                perturbation_interval=perturbation_interval,
                hyperparam_mutations={
                    "learning_rate": lambda: tune.loguniform(1e-5, 1e-2).sample(),
                    "weight_decay": lambda: tune.loguniform(1e-6, 1e-3).sample(),
                },
            )
        elif self.scheduler_type == "fifo":
            return None  # FIFO is default
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def _has_search_space(self, config: dict[str, Any]) -> bool:
        """Check if config contains actual search space or just fixed values.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to check.

        Returns
        -------
        bool
            True if config contains search space objects,
            False if all fixed values.
        """
        if not RAY_AVAILABLE:
            return False

        for value in config.values():
            # Check if any value is a Ray Tune search space object
            if hasattr(value, "sample") or str(type(value)).startswith(
                "<class 'ray.tune"
            ):
                return True
        return False

    def _create_search_algorithm(
        self,
        search_space: dict[str, Any],
    ) -> tune.search.Searcher | None:
        """Create search algorithm based on configuration.

        Parameters
        ----------
        search_space : Dict[str, Any]
            Search space configuration.

        Returns
        -------
        ray.tune.search.Searcher
            Configured search algorithm.
        """
        # If all values are fixed - random search (no search algorithm needed)
        if not self._has_search_space(search_space):
            return None  # Use default random search

        if self.search_algorithm == "optuna":
            return OptunaSearch(metric=self.metric, mode=self.mode)
        elif self.search_algorithm == "hyperopt":
            return HyperOptSearch(metric=self.metric, mode=self.mode)
        elif self.search_algorithm == "random":
            return None  # Random search is default
        else:
            raise ValueError(f"Unknown search algorithm: {self.search_algorithm}")

    def _training_function(self, config: dict[str, Any]) -> None:
        """Training function for Ray Tune trials.

        Parameters
        ----------
        config : Dict[str, Any]
            Hyperparameter configuration for this trial.
        """
        # Merge base config with trial config
        model_config = {**self.model_base_config}

        # Extract model parameters
        model_params = {}
        training_params = {}

        for key, value in config.items():
            if key in [
                "learning_rate",
                "weight_decay",
                "batch_size",
                "scheduler_type",
                "warmup_epochs",
            ]:
                training_params[key] = value
            else:
                model_params[key] = value

        # Update model config
        model_config.update(model_params)

        # Create model
        model = self.model_class(**model_config)

        # Create Lightning trainer
        lightning_model = LightningTrainer(
            model=model,
            max_epochs=self.max_epochs_per_trial,
            **training_params,
        )

        # Create temporary directory for this trial
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup Lightning trainer
            trainer = Trainer(
                max_epochs=self.max_epochs_per_trial,
                accelerator="gpu" if self.gpus_per_trial > 0 else "cpu",
                devices=1,  # Always use 1 device (GPU or CPU)
                precision="16-mixed" if self.gpus_per_trial > 0 else "32",
                default_root_dir=temp_dir,
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=False,
                callbacks=[
                    RayTuneReportCallback(
                        metrics={
                            "loss": "train_loss",
                            "val_loss": "val_loss",
                            "epoch": "epoch",
                        },
                        on="validation_end",
                    )
                ],
            )

            # Train
            trainer.fit(
                lightning_model,
                self.data_loaders["train"],
                self.data_loaders.get("val", None),
            )

    def tune(
        self,
        search_space: dict[str, Any],
        name: str = "autoencoder_tune",
        resume: bool = False,
    ) -> Any:
        """Run hyperparameter tuning.

        Parameters
        ----------
        search_space : Dict[str, Any]
            Search space configuration using Ray Tune syntax.
        name : str, optional
            Name for this tuning experiment, by default "autoencoder_tune".
        resume : bool, optional
            Whether to resume from previous run, by default False.

        Returns
        -------
        ray.tune.ExperimentAnalysis
            Results of the tuning experiment.
        """
        # Check if we have a real search space
        has_search_space = self._has_search_space(search_space)

        # Warn if using advanced search algorithm with fixed values
        if not has_search_space and self.search_algorithm in [
            "optuna",
            "hyperopt",
        ]:
            warnings.warn(
                f"Using {self.search_algorithm} with fixed values. "
                f"Consider using search_algorithm='random' "
                f"for fixed configurations.",
                stacklevel=2,
            )

        # Create scheduler and search algorithm
        scheduler = self._create_scheduler()
        search_alg = self._create_search_algorithm(search_space)

        # Configure reporter
        reporter = CLIReporter(
            parameter_columns=list(search_space.keys())[:4],
            # Show first 4 params
            metric_columns=[self.metric, "training_iteration"],
        )

        # Run tuning
        analysis = tune.run(
            self._training_function,
            config=search_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            search_alg=search_alg if self.scheduler_type != "pbt" else None,
            progress_reporter=reporter,
            name=name,
            storage_path=self.storage_path,
            resources_per_trial={
                "cpu": self.cpus_per_trial,
                "gpu": self.gpus_per_trial,
            },
            resume=resume,
            raise_on_failed_trial=False,
        )

        return analysis

    def get_best_config(self, analysis: Any) -> dict[str, Any]:
        """Get best configuration from tuning results.

        Parameters
        ----------
        analysis : ray.tune.ExperimentAnalysis
            Results from tune.run().

        Returns
        -------
        Dict[str, Any]
            Best hyperparameter configuration.
        """
        best_trial = analysis.get_best_trial(self.metric, self.mode)
        return best_trial.config

    def train_best_model(
        self,
        analysis: Any,
        max_epochs: int = 100,
        save_path: Optional[str] = None,
    ) -> LightningTrainer:
        """Train final model with best hyperparameters.

        Parameters
        ----------
        analysis : ray.tune.ExperimentAnalysis
            Results from tune.run().
        max_epochs : int, optional
            Number of epochs for final training, by default 100.
        save_path : Optional[str], optional
            Path to save final model, by default None.

        Returns
        -------
        LightningTrainer
            Trained Lightning model with best configuration.
        """
        best_config = self.get_best_config(analysis)

        # Separate model and training parameters
        model_config = {**self.model_base_config}
        training_params = {}

        for key, value in best_config.items():
            if key in [
                "learning_rate",
                "weight_decay",
                "batch_size",
                "scheduler_type",
                "warmup_epochs",
            ]:
                training_params[key] = value
            else:
                model_config[key] = value

        # Create final model
        model = self.model_class(**model_config)

        # Create Lightning trainer
        lightning_model = LightningTrainer(
            model=model, max_epochs=max_epochs, **training_params
        )

        # Setup final trainer
        callbacks = [ModelCheckpoint(monitor=self.metric, save_top_k=1)]

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            precision="16-mixed" if torch.cuda.is_available() else "32",
            callbacks=callbacks,
            logger=TensorBoardLogger("./final_training_logs", name="best_model"),
        )

        # Train final model
        trainer.fit(
            lightning_model,
            self.data_loaders["train"],
            self.data_loaders.get("val", None),
        )

        # Save model if requested
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to: {save_path}")

        return lightning_model


class RayTuneReportCallback(pl.Callback):
    """Callback to report metrics to Ray Tune."""

    def __init__(
        self, metrics: dict[str, str] = None, on: str = "validation_end"
    ) -> None:
        """Initialize callback.

        Parameters
        ----------
        metrics : Dict[str, str], optional
            Mapping from Ray Tune metric names to Lightning metric names.
            If None, uses default mapping.
        on : str, optional
            When to report ("validation_end" or "epoch_end"),
            by default "validation_end".
        """
        if metrics is None:
            # Default metrics mapping
            metrics = {
                "loss": "train_loss",
                "val_loss": "val_loss",
                "epoch": "epoch",
            }

        self.metrics = metrics
        self.on = on

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Report metrics after validation."""
        if self.on == "validation_end":
            self._report_metrics(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Report metrics after training epoch."""
        if self.on == "epoch_end":
            self._report_metrics(trainer, pl_module)

    def _report_metrics(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Report metrics to Ray Tune."""
        # Import here to avoid issues if Ray not available
        try:
            from ray import train
        except ImportError:
            return

        metrics_to_report = {}

        # Get logged metrics from trainer
        logged_metrics = trainer.logged_metrics
        callback_metrics = trainer.callback_metrics

        # Combine both metric sources
        all_metrics = {**logged_metrics, **callback_metrics}

        for tune_name, lightning_name in self.metrics.items():
            if lightning_name in all_metrics:
                value = all_metrics[lightning_name]
                if isinstance(value, torch.Tensor):
                    value = float(value.detach().cpu())
                metrics_to_report[tune_name] = value

        # Always include epoch information
        metrics_to_report["training_iteration"] = trainer.current_epoch

        # Report to Ray Tune using positional argument
        if metrics_to_report:
            # Use the correct Ray Train reporting format
            try:
                train.report(metrics_to_report)
            except Exception as e:
                warnings.warn(
                    f"Failed to report metrics to Ray Tune: {metrics_to_report}. Exception: {e}",
                    stacklevel=2,
                )


def suggest_scheduler_config(
    max_epochs_per_trial: int,
    num_samples: int,
    training_time_per_epoch: str = "medium",
) -> dict[str, str]:
    """Suggest appropriate scheduler configuration.

    Parameters
    ----------
    max_epochs_per_trial : int
        Maximum epochs per trial.
    num_samples : int
        Number of samples/trials.
    training_time_per_epoch : str, optional
        Expected training time ("fast", "medium", "slow"), by default "medium".

    Returns
    -------
    Dict[str, str]
        Suggested configuration with reasoning.
    """
    suggestions = {}

    if max_epochs_per_trial < 2:
        suggestions["scheduler_type"] = "fifo"
        suggestions["reason"] = "FIFO recommended for single epoch trials"
    elif max_epochs_per_trial < 4:
        suggestions["scheduler_type"] = "fifo"
        suggestions["reason"] = "FIFO recommended for very short trials (< 4 epochs)"
    elif max_epochs_per_trial >= 10 and num_samples >= 8:
        if training_time_per_epoch in ["fast", "medium"]:
            suggestions["scheduler_type"] = "asha"
            suggestions["reason"] = "ASHA recommended for efficient early stopping"
        else:
            suggestions["scheduler_type"] = "pbt"
            suggestions["reason"] = (
                "PBT recommended for longer training with population evolution"
            )
    elif num_samples >= 4 and max_epochs_per_trial >= 6:
        suggestions["scheduler_type"] = "pbt"
        suggestions["reason"] = "PBT recommended for medium-scale experiments"
    else:
        suggestions["scheduler_type"] = "asha"
        suggestions["reason"] = "ASHA recommended as general-purpose scheduler"

    return suggestions


def create_basic_search_space(
    learning_rate_range: tuple = (1e-5, 1e-2),
    weight_decay_range: tuple = (1e-6, 1e-3),
    batch_size_choices: list = None,
) -> dict[str, Any]:
    """Create a basic search space for autoencoder tuning.

    Parameters
    ----------
    learning_rate_range : tuple, optional
        Learning rate range (min, max), by default (1e-5, 1e-2).
    weight_decay_range : tuple, optional
        Weight decay range (min, max), by default (1e-6, 1e-3).
    batch_size_choices : list, optional
        Batch size choices, by default [16, 32, 64].

    Returns
    -------
    Dict[str, Any]
        Search space configuration.
    """
    if batch_size_choices is None:
        batch_size_choices = [16, 32, 64]
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune is required to create search spaces")

    return {
        "learning_rate": tune.loguniform(*learning_rate_range),
        "weight_decay": tune.loguniform(*weight_decay_range),
        "batch_size": tune.choice(batch_size_choices),
        "scheduler_type": tune.choice(["cosine", "linear"]),
        "warmup_epochs": tune.choice([0, 3, 5]),
    }
