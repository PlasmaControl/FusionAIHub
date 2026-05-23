"""Capture the G3 reference fixture for Step 5 byte-identical guards.

Builds a small TS+actuator-only :class:`E2EFoundationModel`, runs one
forward pass on a fixed-seed input, and saves to
``tests/e2e/fixtures/no_video_forward.pt``:

* ``input`` — ``diag_inputs``, ``act_inputs``, ``step_index``,
  ``time_offset_s`` tensors
* ``output`` — the dict returned by ``model.forward(...)``
* ``state_dict_keys`` — sorted list of every key in
  ``model.state_dict()``
* ``config`` — the dataclasses used to build the model, plus the
  seed and ``d_model`` / ``n_layers``

The fixture is consumed by ``tests/e2e/test_video_integration.py``:

* G2 (``test_no_video_state_dict_keys_identical``) compares the
  current model's ``state_dict()`` keys against the saved set.
* G3 (``test_no_video_forward_bitwise_identical``) rebuilds the same
  model with the same seed, feeds the saved input, and asserts the
  output matches the saved tensors byte-for-byte.

WHEN TO REGENERATE
==================
Re-run this script to regenerate the fixture **only** after an
intentional change to the time-series forward path of
:class:`E2EFoundationModel` — e.g. a new TS/actuator tokenizer
architecture, a backbone-block change, a new key in ``state_dict()``
that is part of the TS path. **Do NOT** regenerate to "make the test
pass" after a Phase C / video edit — that defeats the purpose of the
fixture: silent perturbations to the TS forward path are exactly
what G3 is meant to catch.

Run on CPU. CUDA non-determinism (cuDNN algorithm choice etc.) can
break byte-identical comparisons across machines; CPU forward is
fully deterministic given the seed.

Usage::

    pixi run python scripts/capture_no_video_fixture.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)


# ── Fixture configuration (kept small for fast tests + small file) ──────


SEED = 0
BATCH = 2
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
MLP_RATIO = 4.0
DROPOUT = 0.0

# Three modality kinds covered: slow_ts (linear-per-channel),
# fast_ts (Conv1d patching), and one actuator. This exercises the
# three branches of E2EFoundationModel.__init__ that Step 5 will
# extend with a "video" branch.
DIAGNOSTICS = [
    DiagnosticConfig(
        name="slow_a", kind="slow_ts", n_channels=4, window_samples=5
    ),
    DiagnosticConfig(
        name="fast_a", kind="fast_ts",
        n_channels=2, window_samples=20, patch_size=10,
    ),
]
ACTUATORS = [
    ActuatorConfig(
        name="act_a", n_channels=3, window_samples=20, n_tokens=5,
    ),
]


def build_model() -> E2EFoundationModel:
    torch.manual_seed(SEED)
    return E2EFoundationModel(
        diagnostics=DIAGNOSTICS,
        actuators=ACTUATORS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    )


def build_input() -> dict:
    g = torch.Generator().manual_seed(SEED + 1)
    diag_inputs = {
        "slow_a": torch.randn(BATCH, 4, 5, generator=g),
        "fast_a": torch.randn(BATCH, 2, 20, generator=g),
    }
    act_inputs = {
        "act_a": torch.randn(BATCH, 3, 20, generator=g),
    }
    step_index = torch.tensor([0, 1], dtype=torch.long)
    time_offset_s = torch.tensor([0.0, 0.05], dtype=torch.float32)
    return dict(
        diag_inputs=diag_inputs,
        act_inputs=act_inputs,
        step_index=step_index,
        time_offset_s=time_offset_s,
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "tests" / "e2e" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "no_video_forward.pt"

    model = build_model().eval()
    inp = build_input()

    with torch.no_grad():
        output = model(
            inp["diag_inputs"],
            inp["act_inputs"],
            inp["step_index"],
            inp["time_offset_s"],
        )

    fixture = {
        "seed": SEED,
        "config": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "mlp_ratio": MLP_RATIO,
            "dropout": DROPOUT,
            "diagnostics": [asdict(d) for d in DIAGNOSTICS],
            "actuators": [asdict(a) for a in ACTUATORS],
            "batch": BATCH,
        },
        "input": inp,
        "output": output,
        "state_dict_keys": sorted(model.state_dict().keys()),
    }
    torch.save(fixture, out_path)

    print(f"Saved {out_path}")
    print(f"  total state_dict keys: {len(fixture['state_dict_keys'])}")
    print(f"  output modalities: {sorted(output.keys())}")
    for name, t in output.items():
        print(f"    {name}: shape={tuple(t.shape)}, dtype={t.dtype}")
    print(f"  total backbone tokens: {model.n_total_tokens}")


if __name__ == "__main__":
    main()
