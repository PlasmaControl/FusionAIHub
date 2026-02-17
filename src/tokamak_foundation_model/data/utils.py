import torch


def worker_init_fn(worker_id):
    """Each worker needs to open its own file handle."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Force re-open file for this worker
        if hasattr(dataset, 'datasets'):  # ConcatDataset
            for ds in dataset.datasets:
                ds.h5_file = None
                ds._open_hdf5()
        else:
            dataset.h5_file = None
            dataset._open_hdf5()
