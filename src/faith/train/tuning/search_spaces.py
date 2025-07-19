"""Predefined search spaces for common hyperparameter tuning scenarios."""

from typing import Any

try:
    from ray import tune


    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class SearchSpaces:
    """Collection of predefined search spaces for different scenarios."""

    @staticmethod
    def basic_autoencoder(
            learning_rate_range: tuple[float, float] = (1e-5, 1e-2),
            weight_decay_range: tuple[float, float] = (1e-6, 1e-3),
            activation_choices: list[str] = None
    ) -> dict[str, Any]:
        """Basic autoencoder search space.

        Parameters
        ----------
        learning_rate_range : Tuple[float, float], optional
            Learning rate range, by default (1e-5, 1e-2).
        weight_decay_range : Tuple[float, float], optional
            Weight decay range, by default (1e-6, 1e-3).
        activation_choices : List[str], optional
            Activation function choices, by default ["relu", "gelu", "swish"].

        Returns
        -------
        Dict[str, Any]
            Search space configuration.
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune required for search spaces")

        if activation_choices is None:
            activation_choices = ["relu", "gelu", "swish"]

        return {
            "learning_rate": tune.loguniform(*learning_rate_range),
            "weight_decay": tune.loguniform(*weight_decay_range),
            "activation": tune.choice(activation_choices),
            "scheduler_type": tune.choice(["cosine", "linear"]),
            "warmup_epochs": tune.choice([0, 3, 5, 10])
        }

    @staticmethod
    def block_based_autoencoder(
            learning_rate_range: tuple[float, float] = (1e-5, 1e-2),
            dropout_range: tuple[float, float] = (0.0, 0.5),
            num_blocks_range: tuple[int, int] = (2, 6)
    ) -> dict[str, Any]:
        """Search space for BlockBasedAutoencoder architecture.

        Parameters
        ----------
        learning_rate_range : Tuple[float, float], optional
            Learning rate range, by default (1e-5, 1e-2).
        dropout_range : Tuple[float, float], optional
            Dropout range, by default (0.0, 0.5).
        num_blocks_range : Tuple[int, int], optional
            Number of blocks range, by default (2, 6).

        Returns
        -------
        Dict[str, Any]
            Search space configuration.
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune required for search spaces")

        return {
            # Training hyperparameters
            "learning_rate": tune.loguniform(*learning_rate_range),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "scheduler_type": tune.choice(["cosine", "linear", "none"]),

            # Model architecture
            "activation": tune.choice(["relu", "gelu", "swish", "leaky_relu"]),
            "dropout": tune.uniform(*dropout_range),

            # Block configuration (for custom block_configs)
            "base_channels": tune.choice([32, 64, 128]),
            "channel_multiplier": tune.choice([1.5, 2.0, 2.5]),
            "pool_size": tune.choice([(1, 2), (2, 2), (1, 4)]),
        }

    @staticmethod
    def quick_search(
            param_name: str,
            param_choices: list[Any],
            base_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Quick search space for testing single parameters.

        Parameters
        ----------
        param_name : str
            Name of parameter to search.
        param_choices : List[Any]
            Choices for the parameter.
        base_config : Dict[str, Any], optional
            Base configuration to extend, by default None.

        Returns
        -------
        Dict[str, Any]
            Search space configuration.
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune required for search spaces")

        if base_config is None:
            base_config = {"learning_rate": 1e-4}

        search_space = base_config.copy()
        search_space[param_name] = tune.choice(param_choices)

        return search_space

    @staticmethod
    def regularization_focused(
            base_lr: float = 1e-4,
            dropout_range: tuple[float, float] = (0.0, 0.5),
            weight_decay_range: tuple[float, float] = (1e-6, 1e-2)
    ) -> dict[str, Any]:
        """Search space focused on regularization parameters.

        Parameters
        ----------
        base_lr : float, optional
            Fixed learning rate, by default 1e-4.
        dropout_range : Tuple[float, float], optional
            Dropout range, by default (0.0, 0.5).
        weight_decay_range : Tuple[float, float], optional
            Weight decay range, by default (1e-6, 1e-2).

        Returns
        -------
        Dict[str, Any]
            Search space configuration.
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune required for search spaces")

        return {
            "learning_rate": base_lr,  # Fixed
            "weight_decay": tune.loguniform(*weight_decay_range),
            "dropout": tune.uniform(*dropout_range),

            # Regularization techniques
            "label_smoothing": tune.uniform(0.0, 0.2),
            "gradient_clip_val": tune.choice([0.5, 1.0, 2.0, None]),

            # Data augmentation (if applicable)
            "noise_factor": tune.uniform(0.0, 0.1),
            "mixup_alpha": tune.uniform(0.0, 0.4),
        }

    @staticmethod
    def architecture_search(
            learning_rate: float = 1e-4,
            layer_choices: list[int] = None,
            width_choices: list[int] = None
    ) -> dict[str, Any]:
        """Search space focused on architecture parameters.

        Parameters
        ----------
        learning_rate : float, optional
            Fixed learning rate, by default 1e-4.
        layer_choices : List[int], optional
            Number of layers choices, by default [2, 3, 4, 5].
        width_choices : List[int], optional
            Layer width choices, by default [64, 128, 256, 512].

        Returns
        -------
        Dict[str, Any]
            Search space configuration.
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune required for search spaces")

        if layer_choices is None:
            layer_choices = [2, 3, 4, 5]
        if width_choices is None:
            width_choices = [64, 128, 256, 512]

        return {
            "learning_rate": learning_rate,  # Fixed
            "weight_decay": 1e-5,  # Fixed

            # Architecture parameters
            "num_layers": tune.choice(layer_choices),
            "bottleneck_dim": tune.choice([16, 32, 64, 128]),

            # Activation and normalization
            "activation": tune.choice(["relu", "gelu", "swish", "leaky_relu"]),
            "use_batch_norm": tune.choice([True, False]),
            "use_layer_norm": tune.choice([True, False]),

            # Skip connections
            "use_skip_connections": tune.choice([True, False]),
            "skip_type": tune.choice(["add", "concat"]),
        }


class CustomSearchSpace:
    """Builder for custom search spaces."""

    def __init__(self):
        """Initialize custom search space builder."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune required for custom search spaces")

        self.space = {}

    def add_continuous(
            self,
            name: str,
            low: float,
            high: float,
            log_scale: bool = False
    ) -> 'CustomSearchSpace':
        """Add continuous parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        low : float
            Lower bound.
        high : float
            Upper bound.
        log_scale : bool, optional
            Use log scale, by default False.

        Returns
        -------
        CustomSearchSpace
            Self for chaining.
        """
        if log_scale:
            self.space[name] = tune.loguniform(low, high)
        else:
            self.space[name] = tune.uniform(low, high)
        return self

    def add_discrete(self, name: str, choices: list[Any]) \
            -> 'CustomSearchSpace':
        """Add discrete parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        choices : List[Any]
            List of choices.

        Returns
        -------
        CustomSearchSpace
            Self for chaining.
        """
        self.space[name] = tune.choice(choices)
        return self

    def add_integer(self, name: str, low: int,
                    high: int) -> 'CustomSearchSpace':
        """Add integer parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        low : int
            Lower bound.
        high : int
            Upper bound.

        Returns
        -------
        CustomSearchSpace
            Self for chaining.
        """
        self.space[name] = tune.randint(low, high + 1)
        return self

    def add_fixed(self, name: str, value: Any) -> 'CustomSearchSpace':
        """Add fixed parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        value : Any
            Fixed value.

        Returns
        -------
        CustomSearchSpace
            Self for chaining.
        """
        self.space[name] = value
        return self

    def build(self) -> dict[str, Any]:
        """Build the search space.

        Returns
        -------
        Dict[str, Any]
            Complete search space configuration.
        """
        return self.space.copy()


# Convenience function for quick access
def get_search_space(name: str, **kwargs) -> dict[str, Any]:
    """Get predefined search space by name.

    Parameters
    ----------
    name : str
        Search space name ("basic", "block_based", "multimodal",
        "regularization", "architecture").
    **kwargs
        Additional parameters for the search space.

    Returns
    -------
    Dict[str, Any]
        Search space configuration.

    Examples
    --------
    >>> space = get_search_space("basic", learning_rate_range=(1e-4, 1e-2))
    >>> space = get_search_space("architecture", layer_choices=[2, 3, 4])
    """
    spaces = {
        "basic": SearchSpaces.basic_autoencoder,
        "block_based": SearchSpaces.block_based_autoencoder,
        "regularization": SearchSpaces.regularization_focused,
        "architecture": SearchSpaces.architecture_search,
    }

    if name not in spaces:
        available = ", ".join(spaces.keys())
        raise ValueError(
            f"Unknown search space '{name}'. Available: {available}")

    return spaces[name](**kwargs)
