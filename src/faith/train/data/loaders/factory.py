"""DataLoader factory and worker utilities for lazy file datasets."""

import warnings
from torch.utils.data import DataLoader, get_worker_info


def worker_init_fn(worker_id: int) -> None:
    """Initialize dataset in worker process.

    Parameters
    ----------
    worker_id : int
        ID of the current worker process.
    """
    # Get the dataset from worker info
    worker_info = get_worker_info()
    if worker_info is not None:
        worker_dataset = worker_info.dataset

        # Check if it's a lazy file dataset that needs initialization
        if hasattr(worker_dataset, 'worker_init'):
            try:
                worker_dataset.worker_init()
            except Exception as e:
                warnings.warn(f"Failed to initialize dataset in worker "
                              f"{worker_id}: {e}")
        else:
            # Not a lazy dataset, no initialization needed
            pass
    else:
        # No worker info available (single-threaded mode)
        pass


def create_dataloader(
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
):
    """
    Create a DataLoader with automatic worker initialization for lazy datasets.

    This function automatically handles worker initialization for lazy file
    datasets while providing sensible defaults for other DataLoader parameters.

    Parameters
    ----------
    dataset : Dataset
        PyTorch dataset to load from.
    batch_size : int, optional
        Number of samples per batch, by default 32.
    shuffle : bool, optional
        Whether to shuffle data, by default True.
    num_workers : int, optional
        Number of worker processes, by default 4.
    pin_memory : bool, optional
        Whether to pin memory for GPU transfer, by default True.
    drop_last : bool, optional
        Whether to drop last incomplete batch, by default False.
    **kwargs
        Additional arguments passed to DataLoader.

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader.

    Examples
    --------
    >>> from src import JoblibDataset
    >>> dataset = JoblibDataset(['file1.joblib'], subseq_len=128)
    >>> loader = create_dataloader(dataset, batch_size=16, num_workers=2)
    >>>
    >>> # Use in training
    >>> for batch in loader:
    ...     inputs, targets = batch
    ...     # Training code here
    """
    # Determine if we need worker initialization
    worker_init_fn = None
    if hasattr(dataset, 'worker_init'):
        worker_init_fn = create_worker_init_fn(dataset)

    # Auto-adjust num_workers based on dataset type
    if hasattr(dataset, 'worker_init') and num_workers == 0:
        warnings.warn(
            "Using num_workers=0 with lazy file dataset. "
            "Consider using num_workers>=1 for better performance."
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        **kwargs
    )


def create_train_val_loaders(
        train_dataset,
        val_dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
):
    """Create train and validation DataLoaders with consistent configuration.

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.
    val_dataset : Dataset
        Validation dataset.
    batch_size : int, optional
        Batch size for both loaders, by default 32.
    num_workers : int, optional
        Number of workers for both loaders, by default 4.
    pin_memory : bool, optional
        Whether to pin memory, by default True.
    **kwargs
        Additional arguments passed to both DataLoaders.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Train and validation DataLoaders.

    Examples
    --------
    >>> train_ds = JoblibDataset(train_files, subseq_len=128)
    >>> val_ds = JoblibDataset(val_files, subseq_len=128)
    >>> train_loader, val_loader = create_train_val_loaders(
    ...     train_ds, val_ds, batch_size=64
    ... )
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Always shuffle training
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop validation samples
        **kwargs
    )

    return train_loader, val_loader


def create_dataloaders_from_config(config: dict):
    """Create DataLoaders from a configuration dictionary.

    This function provides a high-level interface for creating datasets and
    DataLoaders from configuration, useful for experiments and hyperparameter
    tuning.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing dataset and loader parameters.
        Expected keys:
        - 'dataset_type': str ('joblib', 'hdf5', 'numpy')
        - 'file_paths': str or list
        - 'subseq_len': int
        - 'batch_size': int (optional)
        - 'num_workers': int (optional)
        - Other dataset-specific parameters

    Returns
    -------
    DataLoader or tuple[DataLoader, DataLoader]
        Single DataLoader or tuple of (train_loader, val_loader) if split is
        requested.

    Examples
    --------
    >>> config = {
    ...     'dataset_type': 'joblib',
    ...     'file_paths': '/data/*.joblib',
    ...     'subseq_len': 128,
    ...     'batch_size': 32,
    ...     'split_by_files': True,
    ...     'train_ratio': 0.8
    ... }
    >>> train_loader, val_loader = create_dataloaders_from_config(config)
    """
    # Import here to avoid circular imports
    from ..datasets.file_based import JoblibDataset, HDF5Dataset, NumpyDataset

    # Dataset type mapping
    dataset_classes = {
        'joblib': JoblibDataset,
        'hdf5': HDF5Dataset,
        'numpy': NumpyDataset
    }

    # Extract configuration
    dataset_type = config['dataset_type']
    file_paths = config['file_paths']
    subseq_len = config['subseq_len']

    # DataLoader configuration
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)

    # Dataset-specific configuration
    dataset_config = {k: v for k, v in config.items()
                      if k not in ['dataset_type', 'batch_size', 'num_workers',
                                   'split_by_files', 'train_ratio',
                                   'val_ratio', 'test_ratio']}

    # Create dataset
    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataset_class = dataset_classes[dataset_type]
    dataset = dataset_class(file_paths, subseq_len, **dataset_config)

    # Handle splitting
    if config.get('split_by_files', False):
        train_ratio = config.get('train_ratio', 0.8)
        val_ratio = config.get('val_ratio', 0.1)
        test_ratio = config.get('test_ratio', 0.1)

        train_ds, val_ds, test_ds = dataset.split_by_files(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=config.get('random_seed', 42)
        )

        train_loader, val_loader = create_train_val_loaders(
            train_ds, val_ds, batch_size=batch_size, num_workers=num_workers
        )

        if test_ratio > 0:
            test_loader = create_dataloader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers
            )
            return train_loader, val_loader, test_loader
        else:
            return train_loader, val_loader
    else:
        # Single dataset
        return create_dataloader(
            dataset, batch_size=batch_size, num_workers=num_workers
        )
