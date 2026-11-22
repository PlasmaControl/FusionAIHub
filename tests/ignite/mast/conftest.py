"""Shared fixtures for the IGNITE-on-MAST tests. Later tasks add them here."""

import importlib.util

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import paths, tasks


@pytest.fixture(autouse=True)
def _skip_real_data_without_data(request):
    """`real_data` tests need TokaMark's Zarr store and the `tokamark` / `MAST_tools` packages.

    The pixi environment sees the store but has no `tokamark`, so both are checked.
    """
    if request.node.get_closest_marker("real_data") is None:
        return
    if not paths.tokamark_zarr_root().exists():
        pytest.skip(f"TokaMark data not found at {paths.tokamark_zarr_root()}")
    for module in ("tokamark", "MAST_tools"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not importable in this environment")


@pytest.fixture
def toy_task() -> tasks.TaskSpec:
    """A hand-built markovian task that needs no TokaMark: 5 ms in, 25 ms out, delta 0.

    Means and standard deviations are TokaMark's, read from
    `tokamark.tasks.get_signals_metadata()` on 2026-11-22, pasted here so that the
    fixture works in the pixi environment, which has no `tokamark`.
    """
    ip = tasks.SignalSpec("summary-ip", 0.00025, (1,), 352977.58724943904, 301642.02514258376)
    flux = tasks.SignalSpec("magnetics-flux_loop_flux", 0.0002, (15,),
                            -0.25546626077270007, 0.382853499608823)
    coil = tasks.SignalSpec("pf_active-coil_voltage", 0.00025, (4,),
                            -6.899173983623492, 3117.527841332377)
    return tasks.TaskSpec(
        name="task_toy", markovian=True,
        input_length=0.005, output_length=0.025, delta=0.0, stride=0.001,
        inputs=(ip, flux), actuators=(coil,), outputs=(ip,),
        steps={
            "input": {"summary-ip": 20, "magnetics-flux_loop_flux": 25},
            "actuator": {"pf_active-coil_voltage": 120},
            "output": {"summary-ip": 100},
        },
    )


@pytest.fixture
def toy_batch(toy_task):
    """Factory ``toy_batch(n, seed)``: a `WindowBatch` of random finite float32 windows of `toy_task`.

    Values are mean + std * standard normal (physical units). `last_valid` is the last input
    sample, every (shot, window) pair is distinct. Needs no `tokamark`.
    """
    from tokamak_foundation_model.ignite.mast.harness import WindowBatch

    def make(n: int, seed: int = 0) -> WindowBatch:
        rng = np.random.default_rng(seed)

        def draw(specs, length):
            return {
                s.name: (s.mean + s.std * rng.standard_normal((n, *s.values_shape, length(s)))).astype(
                    np.float32
                )
                for s in specs
            }

        inputs = draw(toy_task.inputs, toy_task.n_in)
        return WindowBatch(
            shot_ids=(1000 + np.arange(n) // 4).astype(np.int64),
            window_indices=(np.arange(n) % 4).astype(np.int64),
            inputs=inputs,
            actuators=draw(toy_task.actuators, toy_task.n_act),
            targets=draw(toy_task.outputs, toy_task.n_out),
            last_valid={name: a[..., -1].copy() for name, a in inputs.items()},
        )

    return make
