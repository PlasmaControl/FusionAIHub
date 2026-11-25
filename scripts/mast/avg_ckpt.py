"""Uniform average of the model weights of several IGNITE dynamics snapshots.

A post-hoc stand-in for a decayed learning rate (dev-peter's trainer keeps no EMA
weights). Floating-point tensors are averaged in float64 and cast back; every other
tensor must be equal in all snapshots. The other checkpoint keys come from the last
snapshot, the optimizer state is dropped, and `averaged_from` lists the inputs. The
result loads like any snapshot in `port_rollout_plots.py` / `port_recon_plots.py`.

Used 2026-11-25 on run 6 (steps 3,000, 3,500, 4,000):
docs/models/ignite-mast-spectrogram-quality.md, "Averaging the weights".

Usage: scripts/mast/py.sh scripts/mast/avg_ckpt.py OUT.pt IN1.pt IN2.pt ...
"""
import sys

import torch


def main(out, ins):
    acc, last = None, None
    for p in ins:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"]
        if acc is None:
            acc = {k: v.double().clone() if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            assert sd.keys() == acc.keys(), p
            for k, v in sd.items():
                if v.is_floating_point():
                    acc[k] += v.double()
                else:
                    assert torch.equal(v, acc[k]), k
        last = ck
    for k, v in last["model"].items():
        if v.is_floating_point():
            acc[k] = (acc[k] / len(ins)).to(v.dtype)
    res = {k: v for k, v in last.items() if k != "opt"}
    res["model"] = acc
    res["averaged_from"] = list(ins)
    torch.save(res, out)
    print("wrote", out, "from", len(ins), "snapshots, step", res["step"])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
