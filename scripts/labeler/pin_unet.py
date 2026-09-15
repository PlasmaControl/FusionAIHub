"""Pin the vendored TokEye U-Net: hash, parameter count and a golden output.

`src/labeler/events/unet.py` is a copy of TokEye's `big_tf_unet` forward
pass. A copy is only trustworthy if somebody has checked it against the
original, and only *stays* trustworthy if that check is repeated by the test
suite - so this script does the checking once, against the real `tokeye`
package, and writes what it found into
`tests/labeler/data/unet_golden.npz`, which
`tests/labeler/test_events_unet.py` then reads on every run.

What it does:

1. prints the checkpoint's sha256 and the model's parameter count, which are
   the two constants `events/unet.py` pins;
2. builds a deterministic input, `default_rng(0).standard_normal(...)` at
   `(1, 1, 128, 128)` float32 - noise, not a spectrogram, because the point
   is arithmetic fidelity and noise exercises every kernel;
3. runs the **vendored** module on it (CPU, fp32, one thread, `no_grad`);
4. runs the **original** `tokeye` module on the same input and records
   `max_abs_diff_vs_tokeye`. `tokeye` is put on `sys.path` here and nowhere
   else: no library code in labeler may import it;
5. writes the golden.

    PYTHONPATH=$PWD/src \\
        pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \\
        -e labelmaker python scripts/labeler/pin_unet.py

Run it in the environment the **test suite** uses, so the golden's
`max_abs_diff` is 0 there. Re-run it only when the checkpoint or the vendored
code deliberately changes; `--out` writes elsewhere for a dry run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Imported after the `sys.path` line above, deliberately: this script runs
# from a checkout rather than an installed package.
import torch

from labeler.config import sha256_of
from labeler.events import unet

#: Read only here. Library code resolves the checkpoint through
#: `unet.default_checkpoint_path()` and never mentions this path.
TOKEYE_SRC = Path("/scratch/gpfs/nc1514/tokeye/src")

GOLDEN = REPO / "tests" / "labeler" / "data" / "unet_golden.npz"


def tokeye_probabilities(checkpoint: Path, x: torch.Tensor,
                         tokeye_src: Path) -> np.ndarray:
    """The same masks, from the upstream package rather than from our copy."""
    sys.path.insert(0, str(tokeye_src))
    from tokeye.models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
    from tokeye.models.big_tf_unet.model_big_tf_unet import BigTFUNetModel

    model = BigTFUNetModel(BigTFUNetConfig())
    model.load_state_dict(
        torch.load(checkpoint, weights_only=True, map_location="cpu"), strict=True
    )
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x)[0]).numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path,
                        default=unet.default_checkpoint_path(),
                        help="the pinned U-Net checkpoint")
    parser.add_argument("--tokeye-src", type=Path, default=TOKEYE_SRC,
                        help="upstream's src/ directory, for the comparison")
    parser.add_argument("--out", type=Path, default=GOLDEN,
                        help="where to write the golden .npz")
    args = parser.parse_args(argv)

    # One thread, and one thread in the test too: a multi-threaded reduction
    # is free to reassociate, and then "equal to the last bit" is not a thing
    # a test can ask for.
    torch.set_num_threads(1)

    # Both pinned constants are reported - and both "NOTE:" lines given
    # their chance - BEFORE `load_unet`, whose two guards would otherwise
    # raise on exactly the runs this script exists for. The parameter count
    # is a property of the vendored code alone, so counting it needs no
    # checkpoint; the hash decides whether the hash guard is asked for at
    # all, so a deliberate re-pin of a NEW checkpoint gets past it.
    sha = sha256_of(args.checkpoint)
    n_params = sum(
        p.numel()
        for p in unet.BigTFUNetModel(unet.BigTFUNetConfig()).parameters()
    )
    print(f"checkpoint    {args.checkpoint}")
    print(f"sha256        {sha}")
    print(f"n_params      {n_params}")
    print(f"torch         {torch.__version__}")
    if sha != unet.CHECKPOINT_SHA256:
        print(f"NOTE: unet.CHECKPOINT_SHA256 is {unet.CHECKPOINT_SHA256}")
    if n_params != unet.N_PARAMS:
        print(f"NOTE: unet.N_PARAMS is {unet.N_PARAMS}")
    model = unet.load_unet(
        args.checkpoint, verify_sha256=(sha == unet.CHECKPOINT_SHA256)
    )

    x = np.random.default_rng(0).standard_normal((1, 1, 128, 128)).astype(np.float32)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        y = unet.probabilities(model, xt).numpy()
    assert y.shape == (1, 2, 128, 128) and y.dtype == np.float32

    want = tokeye_probabilities(args.checkpoint, xt, args.tokeye_src)
    max_abs_diff = float(np.abs(y - want).max())
    print(f"vendored vs tokeye   max_abs_diff {max_abs_diff!r}")
    print(f"output p  min {y.min():.6f}  mean {y.mean():.6f}  max {y.max():.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=x,
        y=y[0],
        max_abs_diff_vs_tokeye=np.float64(max_abs_diff),
        sha256=np.str_(sha),
        n_params=np.int64(n_params),
        torch_version=np.str_(torch.__version__),
    )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
