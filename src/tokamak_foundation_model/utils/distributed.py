import os
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

            self.distributed = True
            dist.init_process_group(
                'nccl',
                rank=self.rank,
                world_size=self.world_size,
                device_id=torch.device("cuda", self.local_rank),
            )
            torch.cuda.set_device(self.local_rank)
        else:
            # Single-process (plain python)
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.distributed = False
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
        self.barrier()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def wrap_ddp(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model with DDP if distributed, otherwise return as-is."""
        if self.distributed:
            return DistributedDataParallel(model, device_ids=[self.local_rank])
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
