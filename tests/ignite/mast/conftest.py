"""Shared fixtures for the IGNITE-on-MAST tests. Later tasks add them here."""

import pytest

from tokamak_foundation_model.ignite.mast import tasks


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
