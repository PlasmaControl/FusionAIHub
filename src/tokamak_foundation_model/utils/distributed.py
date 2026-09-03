import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class DistributedManager:

    def __init__(self):
        if os.environ.get('MASTER_PORT'):
            # torchrun sets RANK; fall back to SLURM_PROCID for srun launches
            self.rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ['WORLD_SIZE'])

            # On Frontier with --gpus-per-task=1, each rank sees exactly one
            # GPU (masked by ROCR_VISIBLE_DEVICES); the device index is 0
            # regardless of LOCAL_RANK. Clamp accordingly.
            visible = torch.cuda.device_count()
            self.device_index = self.local_rank if visible > 1 else 0

            self.distributed = True
            # 90-min collective timeout (default is 10 min). Raised from 30 →
            # 90 (2026-06-27): on a COLD lengths-cache rebuild, rank 0 scans all
            # video-present files (~4430) in _load_or_compute_lengths while ranks
            # 1..N wait at the dist.broadcast_object_list — a ~55-min single-rank
            # HDF5 metadata scan that blew past the 30-min watchdog (job 4909383
            # crash). 90 min covers the one-time rebuild; steady-state collectives
            # are sub-second so this only matters during startup scans. (Also
            # covers the extended Stage 2 K=80 post-val rank-skew, smoke 4793237.)
            dist.init_process_group(
                'nccl',
                rank=self.rank,
                world_size=self.world_size,
                device_id=torch.device("cuda", self.device_index),
                timeout=timedelta(minutes=90),
            )
            torch.cuda.set_device(self.device_index)
        else:
            # Single-process (plain python)
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.device_index = 0
            self.distributed = False
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device_index)
        self.barrier()

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.device_index)
        return torch.device("cpu")

    def wrap(
        self, model: torch.nn.Module, find_unused_parameters: bool = False,
    ) -> torch.nn.Module:
        """Wrap model with DDP if distributed, otherwise return as-is.

        Default ``find_unused_parameters=False`` relies on every parameter
        being touched in every step. The video / spectrogram tokenizers
        always run ``_encode`` and reference ``missing_token`` regardless
        of the per-batch validity mask, so the autograd graph is
        data-independent and DDP's reducer can use static buckets. This
        avoids RCCL bucket-rebuild faults observed on Frontier.

        Override to ``True`` only as a debugging escape hatch — it incurs
        a per-step unused-param scan and was previously observed to
        trigger GPU memory faults via RCCL on this stack.
        """
        if self.distributed:
            return DistributedDataParallel(
                model,
                device_ids=[self.device_index],
                find_unused_parameters=find_unused_parameters,
            )
        return model

    def unwrap(self, model: torch.nn.Module):
        if self.distributed and hasattr(model, 'module'):
            return model.module
        return model

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def gather_concat(self, x: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return x
        x_list = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(x_list, x)
        return torch.cat(x_list)

    def __del__(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
