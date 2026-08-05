"""Shot-level signal-presence filter (fix (a) for co2 codec collapse).

co2 is recorded on only a subset of shots; when absent, preprocessing writes a length-1
``(C, 1)`` placeholder ``ydata`` (which floors to -10 everywhere). Whole co2-absent shots
swamp the codec batch with silence, and within-shot activity stratification cannot rescue
them (no active window to re-draw to). ``filter_signal_present_files`` drops those shots
up front, keyed on ``ydata.shape[-1] > min_len`` (a pure metadata check).
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from tokamak_foundation_model.data.multi_file_dataset import filter_signal_present_files


def _write_shot(path: Path, signal: str, n_samples: int, channels: int = 4) -> None:
    """Write a {signal}/xdata + {signal}/ydata shot with n_samples along the last axis.

    n_samples == 1 mirrors the co2-absent placeholder ``(C, 1)``; n_samples >= 2 is a real
    recording.
    """
    with h5py.File(path, "w") as h5:
        grp = h5.create_group(signal)
        t = np.arange(max(1, n_samples)) / 500_000.0
        grp.create_dataset("xdata", data=t.astype(np.float64))
        grp.create_dataset(
            "ydata", data=np.random.randn(channels, max(1, n_samples)).astype(np.float32)
        )


def test_keeps_present_drops_placeholder_and_missing(tmp_path):
    present = tmp_path / "200049_processed.h5"
    _write_shot(present, "co2", 5000)
    placeholder = tmp_path / "190000_processed.h5"
    _write_shot(placeholder, "co2", 1)  # (4, 1) placeholder -> absent
    no_group = tmp_path / "180000_processed.h5"
    with h5py.File(no_group, "w") as h5:
        h5.create_group("ece")  # has some other signal, but no co2

    kept = filter_signal_present_files([present, placeholder, no_group], "co2")
    assert kept == [present]


def test_order_preserved(tmp_path):
    a = tmp_path / "200100_processed.h5"
    _write_shot(a, "co2", 3000)
    b = tmp_path / "200200_processed.h5"
    _write_shot(b, "co2", 1)  # absent
    c = tmp_path / "200300_processed.h5"
    _write_shot(c, "co2", 3000)

    kept = filter_signal_present_files([a, b, c], "co2")
    assert kept == [a, c]


def test_min_len_threshold(tmp_path):
    # A 1-sample recording is a placeholder; require >= 2 real samples.
    short = tmp_path / "200400_processed.h5"
    _write_shot(short, "co2", 1)
    ok = tmp_path / "200500_processed.h5"
    _write_shot(ok, "co2", 2)
    kept = filter_signal_present_files([short, ok], "co2")
    assert kept == [ok]


def test_cache_round_trip_and_avoids_rescan(tmp_path):
    a = tmp_path / "200100_processed.h5"
    _write_shot(a, "co2", 3000)
    b = tmp_path / "190000_processed.h5"
    _write_shot(b, "co2", 1)  # absent at first scan
    cache = tmp_path / "codec_co2_present.pt"

    kept1 = filter_signal_present_files([a, b], "co2", cache_path=cache)
    assert kept1 == [a]
    assert cache.exists()

    # Give b REAL data now. A cache hit must return the stale [a] (proving no rescan);
    # a fresh scan would have returned [a, b].
    _write_shot(b, "co2", 3000)
    kept2 = filter_signal_present_files([a, b], "co2", cache_path=cache)
    assert kept2 == [a]


def test_cache_key_invalidates_on_path_change(tmp_path):
    a = tmp_path / "200100_processed.h5"
    _write_shot(a, "co2", 3000)
    b = tmp_path / "200200_processed.h5"
    _write_shot(b, "co2", 3000)
    cache = tmp_path / "codec_co2_present.pt"

    _ = filter_signal_present_files([a], "co2", cache_path=cache)
    # Different path list -> cache miss -> rescan -> both kept.
    kept = filter_signal_present_files([a, b], "co2", cache_path=cache)
    assert kept == [a, b]


def test_co2_in_presence_filter_signals():
    import tokamak_foundation_model.ignite.train_codec as tc

    assert "co2" in tc._PRESENCE_FILTER_SIGNALS
    # the already-working spectros must NOT be filtered (byte-identical behaviour)
    for m in ("ece", "bes", "mhr"):
        assert m not in tc._PRESENCE_FILTER_SIGNALS


def test_wiring_maps_paths_back_to_shot_ids(tmp_path):
    """Replicates main()'s glue: _shot_paths -> filter -> name.split('_')[0] -> id set."""
    import tokamak_foundation_model.ignite.train_codec as tc

    for sid, n in [("200049", 5000), ("190000", 1), ("200810", 5000)]:
        _write_shot(tmp_path / f"{sid}_processed.h5", "co2", n)
    all_shots = ["200049", "190000", "200810", "999999"]  # 999999 = missing file

    paths = tc._shot_paths(all_shots, tmp_path)
    kept = {
        p.name.split("_")[0]
        for p in filter_signal_present_files(paths, "co2")
    }
    filtered = [s for s in all_shots if str(s) in kept]
    # placeholder (190000) and missing (999999) dropped; order preserved
    assert filtered == ["200049", "200810"]


def test_signal_name_scoped(tmp_path):
    # A shot with real ece but placeholder co2 is present for ece, absent for co2.
    p = tmp_path / "200600_processed.h5"
    with h5py.File(p, "w") as h5:
        ece = h5.create_group("ece")
        ece.create_dataset("ydata", data=np.random.randn(40, 4000).astype(np.float32))
        co2 = h5.create_group("co2")
        co2.create_dataset("ydata", data=np.zeros((4, 1), dtype=np.float32))
    assert filter_signal_present_files([p], "ece") == [p]
    assert filter_signal_present_files([p], "co2") == []
