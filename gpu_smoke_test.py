import os
import socket

import torch


def main() -> None:
    hostname = socket.gethostname()
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    proc_id = int(os.environ.get("SLURM_PROCID", 0))

    print(f"[{hostname} rank={proc_id} local={local_rank}] torch={torch.__version__}")
    print(
        f"[{hostname} rank={proc_id}] cuda.is_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} hip={getattr(torch.version, 'hip', None)}"
    )

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch")

    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(device)
    print(f"[{hostname} rank={proc_id}] using {device} ({name})")

    a = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    c = a @ b
    torch.cuda.synchronize(device)
    print(
        f"[{hostname} rank={proc_id}] matmul ok: shape={tuple(c.shape)} "
        f"mean={c.mean().item():.4f}"
    )


if __name__ == "__main__":
    main()
