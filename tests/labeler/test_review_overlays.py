"""TokEye overlays (`labeler.events.review.overlays`): files, states, the request queue
and its drain, and the producer on the CPU with stand-in networks (no GPU in tests)."""

from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import pytest
import torch

from labeler.ae import full, gpu
from labeler.ae import labels as ae_labels
from labeler.ae.seg import pseudo, whole
from labeler.ae.transform import STD_EPS
from labeler.config import Paths
from labeler.events import masks, raw, unet
from labeler.events.review import overlays
from labeler.events.review import rows as review_rows
from labeler.events.review.rows import Grid, ImageRow, TraceRow

SHOT = 170815
EVENT = "mhd_n1"


class FakeSlurm:
    """squeue and sbatch as the drain sees them; anything else fails the test."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.jobs: dict[str, tuple[str, str]] = {}
        self.next_job = 4242
        self.fail: str | None = None

    def __call__(self, command, **kwargs):
        command = [str(part) for part in command]
        self.calls.append(command)
        if command[0] == "squeue":
            if self.fail == "squeue":
                raise FileNotFoundError("squeue")
            out = "".join(f"{j}|{name}|{state}\n" for j, (name, state) in self.jobs.items())
            return subprocess.CompletedProcess(command, 0, out, "")
        if command[0] == "sbatch":
            if self.fail == "sbatch":
                raise subprocess.CalledProcessError(1, command, "", "sbatch: error: QOS")
            job = str(self.next_job)
            self.next_job += 1
            self.jobs[job] = ("overlays-queue", "PENDING")
            return subprocess.CompletedProcess(command, 0, f"{job};stellar\n", "")
        raise AssertionError(f"unexpected command {command}")


@pytest.fixture(autouse=True)
def slurm(monkeypatch):
    fake = FakeSlurm()
    monkeypatch.setattr(overlays, "RUN", fake)
    overlays.forget_jobs()
    yield fake
    overlays.forget_jobs()


@pytest.fixture
def paths(tmp_path):
    return Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus", text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl", label_tables=tmp_path / "events",
        raw_cache=tmp_path / "raw", config_dir=tmp_path / "config",
    )


class Pointwise(torch.nn.Module):
    """A stand-in for the U-Net that sees each pixel alone: coherent logits rise with
    the input, transient ones only far above it. Tiling cannot change its answer."""

    def __init__(self, coherent=1.0, transient=4.0):
        super().__init__()
        self.coherent, self.transient = coherent, transient

    def forward(self, x):
        return (torch.cat([3 * (x - self.coherent), 3 * (x - self.transient)], dim=1),)


class Ramp(Pointwise):
    """Adds a ramp across each tile's columns, so overlapping tiles disagree and the
    stitch's averaging shows."""

    def forward(self, x):
        ramp = torch.linspace(-2, 2, x.shape[-1], dtype=x.dtype)
        return (torch.cat([3 * x + ramp, -3 * x - ramp - 4], dim=1),)


def _codes(n_y, n, seed=0, gap=100):
    """A uint8 spectrogram with a drifting line on noise, no data in its first columns."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(20, 120, size=(n_y, n)).astype(np.uint8)
    line = (n_y // 3 + 5 * np.sin(np.arange(n) / 50)).astype(int)
    codes[line, np.arange(n)] = 240
    codes[:, :gap] = 0
    return codes


def _store(paths, event=EVENT, shot=SHOT, *, n=1500, n_y=20, seed=0, dt=1.0):
    rows = [
        ImageRow("p0", "Mirnov power", _codes(n_y, n, seed), y0=0.0, dy=0.5,
                 y_units="kHz", z_lo=0.0, z_hi=30.0, z_units="dB"),
        ImageRow("p1", "toroidal n", np.zeros((n_y, n), dtype=np.uint8), y0=0.0, dy=0.5,
                 y_units="kHz", z_lo=0.0, z_hi=1.0, z_units="", modes={"n": [1]}),
        ImageRow("p2", "ECE Te map", np.ones((8, n), dtype=np.uint8), y0=1.6, dy=0.1,
                 y_units="R (m)", z_lo=0.0, z_hi=5.0, z_units="keV"),
        TraceRow("p3", "Density", np.zeros((2, 1, n), dtype=np.float32)),
    ]
    path = paths.spectrogram_file(event, shot)
    review_rows.write(path, Grid(0.0, dt, n), rows, event=event)
    return path


def _meta(paths, event=EVENT, shot=SHOT):
    return overlays.store_rows_meta(paths, event, shot)


def _reference_input(codes):
    """`prepare_row` in numpy: clip to the data's own percentiles, z-score over it."""
    valid = (codes > 0).any(0)
    lo, hi = np.percentile(codes[:, valid], [1, 99])
    x = np.clip(codes.astype(np.float64), lo, hi)
    mean, std = x[:, valid].mean(), x[:, valid].std()
    z = (x - mean) / (std + STD_EPS)
    z[:, ~valid] = (lo - mean) / (std + STD_EPS)
    return z.astype(np.float32), valid


def _unpack(layer):
    import base64

    packed = np.frombuffer(base64.b64decode(layer["bits"]), dtype=np.uint8)
    packed = packed.reshape(layer["n_y"], -1)
    return np.unpackbits(packed, axis=1, count=layer["grid"]["n"]).astype(bool)


# --- files and rows -------------------------------------------------------------------


def test_the_overlays_are_filed_under_the_pinned_model(paths):
    assert overlays.MODEL_SHA256 == unet.CHECKPOINT_SHA256
    assert overlays.LAYER == "tokeye-4afc3948"
    assert overlays.layer_file(paths, "mhd_n2", SHOT) == (
        paths.root / "overlays/tokeye/4afc3948/mhd_n1/170815.npz"
    )
    assert overlays.request_file(paths, EVENT, SHOT).name == "mhd_n1__170815.req"


def test_only_spectrogram_rows_are_applicable_and_each_reads_its_own_level():
    assert overlays.applicable({"kind": "image", "y_units": "kHz"})
    assert not overlays.applicable({"kind": "image", "y_units": "kHz", "modes": {}})
    assert not overlays.applicable({"kind": "image", "y_units": "R (m)"})
    assert not overlays.applicable({"kind": "trace", "y_units": "kHz"})
    # The coarsest level with columns of at most 1 ms.
    assert overlays.level_for(0.05) == 8 and overlays.level_for(0.1) == 8
    assert overlays.level_for(0.256) == 1 and overlays.level_for(2.56) == 1
    assert overlays.level_for(0.01) == 64


# --- states, requests and the drain ---------------------------------------------------


def test_a_shot_with_nothing_yet_is_not_computed_and_traces_are_not_applicable(paths):
    _store(paths)
    states = overlays.row_states(paths, EVENT, SHOT, _meta(paths))
    assert states["p0"] == {"state": "not_computed", "reason": "No TokEye yet"}
    for name in ("p1", "p2", "p3"):
        assert states[name] == {"state": "not_applicable", "reason": "not a spectrogram"}


def test_requests_queue_once_each_and_keep_their_place_in_line(paths, slurm):
    _store(paths)
    _store(paths, shot=SHOT + 1)
    assert overlays.request(paths, EVENT, SHOT) == {"state": "queued", "queue_position": 1}
    os.utime(overlays.request_file(paths, EVENT, SHOT), (1, 1))  # the older request
    assert overlays.request(paths, EVENT, SHOT + 1) == {
        "state": "queued", "queue_position": 2,
    }
    again = overlays.request(paths, EVENT, SHOT)
    assert again == {"state": "queued", "queue_position": 1}
    assert len(overlays.pending(paths)) == 2
    state = overlays.row_states(paths, EVENT, SHOT + 1, _meta(paths))["p0"]
    assert state["state"] == "queued" and state["queue_position"] == 2
    assert state["reason"] == "waiting for a GPU job" and state["elapsed_s"] >= 0
    assert slurm.calls == []  # queueing never calls SLURM


def test_a_shot_without_spectrograms_or_store_is_refused(paths):
    review_rows.write(paths.spectrogram_file(EVENT, SHOT), Grid(0.0, 1.0, 10),
                      [TraceRow("p3", "Density", np.zeros((2, 1, 10), dtype=np.float32))])
    assert overlays.request(paths, EVENT, SHOT) == {
        "state": "not_applicable", "queue_position": None,
    }
    with pytest.raises(FileNotFoundError):
        overlays.request(paths, EVENT, SHOT + 5)
    assert overlays.pending(paths) == []


def test_the_drain_submits_one_job_with_the_shots_in_a_file(paths, slurm):
    _store(paths)
    overlays.request(paths, EVENT, SHOT)
    found = overlays.drain(paths)
    assert found["submitted"] == "4242" and found["shots"] == 1
    assert overlays.pending(paths) == []
    requests = next(overlays.batches_dir(paths).glob("*/requests.txt"))
    assert requests.read_text() == f"mhd_n1 {SHOT}\n"
    assert slurm.calls[-1] == [
        "sbatch", "--parsable", "--job-name=overlays-queue", "--time=00:05:00",
        "--cpus-per-task=4", "--mem=8G", f"--export=ALL,REPO={overlays.REPO}",
        str(overlays.SBATCH), "--requests", str(requests),
    ]
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state["state"] == "queued" and state["reason"] == "GPU job 4242 is pending"
    assert state["queue_position"] == 1
    # The job starts but has not reached the shot yet.
    slurm.jobs = {"4242": ("overlays-queue", "RUNNING")}
    overlays.forget_jobs()
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state["state"] == "queued"
    assert state["reason"] == "GPU job 4242 is running; this shot is waiting in it"
    # The job runs, then leaves squeue without a result: the shot failed.
    record = json.loads(overlays.state_file(paths, EVENT, SHOT).read_text())
    overlays.write_record(paths, EVENT, SHOT, "computing", "GPU job 4242 is running",
                          job="4242")
    slurm.jobs = {"4242": ("overlays-queue", "RUNNING")}
    overlays.forget_jobs()
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state["state"] == "computing" and state["reason"] == "GPU job 4242 is running"
    assert record["batch"] == requests.parent.name
    path = overlays.state_file(paths, EVENT, SHOT)
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "since": time.time() - 60}))
    slurm.jobs = {}
    overlays.forget_jobs()
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state["state"] == "failed"
    assert state["reason"] == ("the GPU job 4242 ended without a result: see "
                               "runs/slurm/4242.out")
    # A failed shot can be asked for again.
    assert overlays.request(paths, EVENT, SHOT)["state"] == "queued"


def test_the_drain_waits_for_its_own_job_and_a_minute_between_submissions(paths, slurm):
    _store(paths)
    _store(paths, shot=SHOT + 1)
    overlays.request(paths, EVENT, SHOT)
    assert overlays.drain(paths)["submitted"] == "4242"
    overlays.request(paths, EVENT, SHOT + 1)
    calls = len(slurm.calls)
    assert overlays.drain(paths)["reason"].startswith("next submission in")
    assert len(slurm.calls) == calls  # the minute's guard asks nothing of SLURM
    last = overlays.queue_dir(paths) / "last_submit"
    os.utime(last, (time.time() - 120, time.time() - 120))
    assert overlays.drain(paths) == {"submitted": None, "reason": "job 4242 is pending"}
    assert len(overlays.pending(paths)) == 1
    # The second request waits behind the first job's shot.
    state = overlays.row_states(paths, EVENT, SHOT + 1, _meta(paths))["p0"]
    assert state["queue_position"] == 2
    slurm.jobs = {"77": ("some-other-job", "RUNNING")}
    found = overlays.drain(paths)
    assert found["submitted"] == "4243" and overlays.pending(paths) == []


def test_a_queue_job_is_small_short_and_capped(paths, slurm, monkeypatch):
    monkeypatch.setattr(overlays, "QUEUE_MAX_SHOTS", 3)
    shots = [SHOT + i for i in range(5)]
    for i, shot in enumerate(shots):
        _store(paths, shot=shot, n=600)
        overlays.request(paths, EVENT, shot)
        os.utime(overlays.request_file(paths, EVENT, shot), (100 + i, 100 + i))
    found = overlays.drain(paths)
    assert found["shots"] == 3
    requests = next(overlays.batches_dir(paths).glob("*/requests.txt"))
    assert requests.read_text() == "".join(f"mhd_n1 {shot}\n" for shot in shots[:3])
    assert [p.name for p in overlays.pending(paths)] == [
        overlays.request_file(paths, EVENT, shot).name for shot in shots[3:]]
    argv = slurm.calls[-1]
    script = argv.index(str(overlays.SBATCH))
    options = argv[:script]
    assert {"--time=00:05:00", "--cpus-per-task=4", "--mem=8G"} <= set(options)
    assert f"--export=ALL,REPO={overlays.REPO}" in options
    assert argv[script + 1:] == ["--requests", str(requests)]
    # The two left are in line behind the job's three.
    state = overlays.row_states(paths, EVENT, shots[4], _meta(paths, shot=shots[4]))["p0"]
    assert state["state"] == "queued" and state["queue_position"] == 5


def test_a_job_killed_at_its_limit_stops_holding_places_in_line(paths, slurm, monkeypatch):
    monkeypatch.setattr(overlays, "QUEUE_MAX_SHOTS", 1)
    _store(paths)
    _store(paths, shot=SHOT + 1)
    overlays.request(paths, EVENT, SHOT)
    os.utime(overlays.request_file(paths, EVENT, SHOT), (1, 1))  # the older request
    overlays.request(paths, EVENT, SHOT + 1)
    assert overlays.drain(paths)["submitted"] == "4242"
    meta = _meta(paths, shot=SHOT + 1)
    assert overlays.row_states(paths, EVENT, SHOT + 1, meta)["p0"]["queue_position"] == 2
    # Job 4242 is killed at its time limit: it writes no finished.json and leaves
    # squeue, so its shot no longer counts ahead of the waiting one.
    submitted = next(overlays.batches_dir(paths).glob("*/job.json"))
    os.utime(submitted, (time.time() - 600, time.time() - 600))
    slurm.jobs = {}
    overlays.forget_jobs()
    assert overlays.line(paths) == [(EVENT, SHOT + 1)]
    assert overlays.row_states(paths, EVENT, SHOT + 1, meta)["p0"]["queue_position"] == 1
    # A batch that sbatch never answered for closes after the grace period too.
    stuck = overlays.batches_dir(paths) / "20261016T000000000000"
    stuck.mkdir()
    (stuck / "requests.txt").write_text(f"{EVENT} {SHOT}\n")
    overlays.write_record(paths, EVENT, SHOT, "queued", "submitting a GPU job", job=None)
    assert overlays.line(paths)[0] == (EVENT, SHOT)
    old = time.time() - overlays.SUBMIT_GRACE_S - 10
    os.utime(stuck / "requests.txt", (old, old))
    assert overlays.line(paths) == [(EVENT, SHOT + 1)]


def test_a_failed_submission_fails_its_shots_and_squeue_silence_waits(paths, slurm):
    _store(paths)
    overlays.request(paths, EVENT, SHOT)
    slurm.fail = "squeue"
    assert overlays.drain(paths) == {"submitted": None, "reason": "squeue did not answer"}
    assert len(overlays.pending(paths)) == 1
    slurm.fail = "sbatch"
    found = overlays.drain(paths)
    assert found["submitted"] is None and found["reason"].startswith("sbatch failed")
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state["state"] == "failed" and "QOS" in state["reason"]


def test_a_malformed_request_file_is_dropped_not_submitted(paths, slurm):
    _store(paths)
    overlays.request(paths, EVENT, SHOT)
    bad = overlays.queue_dir(paths) / "x__1.req"
    bad.write_text(json.dumps({"store_event": "mhd_n1; rm -rf /", "shot": "1 2"}))
    found = overlays.drain(paths)
    assert found["shots"] == 1
    requests = next(overlays.batches_dir(paths).glob("*/requests.txt"))
    assert requests.read_text() == f"mhd_n1 {SHOT}\n"
    with pytest.raises(ValueError):
        overlays.request(paths, "Bad Event", SHOT)
    # A drain that claims nothing but a malformed request leaves no batch behind.
    overlays.queue_dir(paths).joinpath("last_submit").unlink()
    slurm.jobs = {}
    bad.write_text("[1, 2]")
    before = set(overlays.batches_dir(paths).iterdir())
    assert overlays.drain(paths) == {"submitted": None, "reason": "nothing is queued"}
    assert set(overlays.batches_dir(paths).iterdir()) == before
    assert overlays.pending(paths) == []


def test_a_malformed_record_reads_as_nothing_not_an_error(paths):
    _store(paths)
    path = overlays.state_file(paths, EVENT, SHOT)
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2]")
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state == {"state": "not_computed", "reason": "No TokEye yet"}
    stamp = overlays._stamp(paths.spectrogram_file(EVENT, SHOT))
    path.write_text(json.dumps({"state": "done", "store_stamp": stamp, "rows": [1],
                                "since": "soon"}))
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state == {"state": "not_computed", "reason": "No TokEye yet for this row"}
    path.write_text(json.dumps({"state": "done", "store_stamp": stamp,
                                "rows": {"p0": {"state": "done"}}}))
    assert overlays.layers(paths, EVENT, SHOT, _meta(paths)) == {}
    path.write_text(json.dumps({"state": "queued", "since": "x", "requested_at": None}))
    assert overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]["state"] == "queued"


def test_a_run_outside_slurm_reads_as_computing_not_never_submitted(paths):
    _store(paths)
    overlays.write_record(paths, EVENT, SHOT, "computing", "TokEye is running", job=None)
    path = overlays.state_file(paths, EVENT, SHOT)
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "since": time.time() - 400}))
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state["state"] == "computing" and state["reason"] == "TokEye is running"
    overlays.write_record(paths, EVENT, SHOT, "queued", "submitting a GPU job", job=None)
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "since": time.time() - 400}))
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state == {"state": "failed",
                     "reason": "the GPU job for this request was never submitted"}


# --- the producer, on the CPU ---------------------------------------------------------


def test_the_producer_lights_each_spectrogram_row_and_files_it_as_done(paths):
    _store(paths)
    summary = overlays.build(paths, [(EVENT, SHOT)], device="cpu", workers=0,
                             model=Pointwise())
    assert summary["done"] == 1 and summary["failed"] == 0
    assert set(summary["stages"]) >= {"read_wait", "prep", "network", "post", "write"}
    states = overlays.row_states(paths, EVENT, SHOT, _meta(paths))
    assert states["p0"]["state"] == "done" and states["p0"]["layer"] == overlays.LAYER
    assert states["p1"]["state"] == "not_applicable"
    found = overlays.layers(paths, EVENT, SHOT, _meta(paths))
    assert list(found) == ["p0"]
    layer = found["p0"]
    assert layer["grid"] == {"t0_ms": 0.0, "dt_ms": 1.0, "n": 1500}
    assert (layer["y0_khz"], layer["dy_khz"], layer["n_y"]) == (0.0, 0.5, 20)
    assert (layer["y0"], layer["dy"], layer["row"]) == (0.0, 0.5, "p0")
    z, valid = _reference_input(_codes(20, 1500))
    p = torch.sigmoid(torch.from_numpy(3 * (z - 1.0))).numpy()
    q = torch.sigmoid(torch.from_numpy(3 * (z - 4.0))).numpy()
    expected = ((p >= 0.2) | (q >= 0.2)) & valid[None, :]
    lit = _unpack(layer)
    assert lit.any() and not lit[:, :100].any()  # nothing where the row has no data
    assert np.array_equal(lit, expected)


def test_the_producer_skips_done_shots_and_redoes_a_rebuilt_store(paths):
    _store(paths)
    overlays.build(paths, [(EVENT, SHOT)], device="cpu", workers=0, model=Pointwise())
    again = overlays.build(paths, [(EVENT, SHOT), (EVENT, SHOT + 9)], device="cpu",
                           workers=0, model=Pointwise())
    assert again["done"] == 0 and again["skipped"] == {"no_store": 1, "done": 1}
    time.sleep(0.01)
    _store(paths, seed=3, n=1600)
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state["state"] == "stale"
    assert state["reason"] == "the store was rebuilt after TokEye ran"
    assert overlays.layers(paths, EVENT, SHOT, _meta(paths)) == {}
    redone = overlays.build(paths, [(EVENT, SHOT)], device="cpu", workers=0,
                            model=Pointwise())
    assert redone["done"] == 1
    assert overlays.layers(paths, EVENT, SHOT, _meta(paths))["p0"]["grid"]["n"] == 1600


def test_an_older_models_layer_reads_as_stale(paths):
    _store(paths)
    older = overlays.model_dir(paths, "00000000") / EVENT / f"{SHOT}.json"
    older.parent.mkdir(parents=True)
    older.write_text("{}")
    state = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    assert state == {"state": "stale", "reason": "TokEye from an older model"}


def test_overlapping_tiles_are_averaged_as_masks_stitches_them(paths):
    """A 512-bin row: the reference is masks.tile and masks.stitch themselves."""
    codes = _codes(512, 1200, seed=1)
    path = paths.spectrogram_file(EVENT, SHOT)
    review_rows.write(path, Grid(0.0, 1.0, 1200), [
        ImageRow("p0", "Mirnov power", codes, y0=0.0, dy=0.5, y_units="kHz",
                 z_lo=0.0, z_hi=30.0, z_units="dB"),
    ], event=EVENT)
    overlays.build(paths, [(EVENT, SHOT)], device="cpu", workers=0, model=Ramp(), batch=1)
    lit = _unpack(overlays.layers(paths, EVENT, SHOT, _meta(paths))["p0"])
    z, valid = _reference_input(codes)
    tiles, meta = masks.tile(z)
    with torch.inference_mode():
        probs = unet.probabilities(Ramp(), torch.from_numpy(tiles)[:, None]).numpy()
    mean = masks.stitch(probs, meta)
    expected = ((mean[0] >= 0.2) | (mean[1] >= 0.2)) & valid[None, :]
    assert meta["n_tiles"] == 3
    assert np.array_equal(lit, expected)


def test_one_bad_shot_fails_alone(paths, monkeypatch):
    _store(paths)
    _store(paths, shot=SHOT + 1)
    real = overlays.prepare_row

    def broken(values):
        if values.shape[1] == 1500 and int(values[0, -1]) == 0:
            raise RuntimeError("no luck")
        return real(values)

    codes = _codes(20, 1500, seed=0)
    codes[:, -1] = 0  # this shot's last column is empty: the stand-in refuses it
    review_rows.write(paths.spectrogram_file(EVENT, SHOT + 1), Grid(0.0, 1.0, 1500), [
        ImageRow("p0", "Mirnov power", codes, y0=0.0, dy=0.5, y_units="kHz",
                 z_lo=0.0, z_hi=30.0, z_units="dB"),
    ], event=EVENT)
    monkeypatch.setattr(overlays, "prepare_row", broken)
    summary = overlays.build(paths, [(EVENT, SHOT), (EVENT, SHOT + 1)], device="cpu",
                             workers=0, model=Pointwise())
    assert summary["done"] == 1 and summary["failed"] == 1
    state = overlays.row_states(paths, EVENT, SHOT + 1, _meta(paths, shot=SHOT + 1))["p0"]
    assert state["state"] == "failed" and state["reason"] == "RuntimeError: no luck"


class Picky(Pointwise):
    """Fails any pass that holds a tile of the sentinel value."""

    def forward(self, x):
        if (x > 1e5).any():
            raise RuntimeError("a bad tile")
        return super().forward(x)


def test_a_failed_forward_pass_fails_only_its_shot(paths, monkeypatch):
    _store(paths)
    _store(paths, shot=SHOT + 1, seed=4)
    real = overlays.prepare_row
    flagged = torch.as_tensor(_codes(20, 1500, seed=4))

    def flag(values):
        x, valid = real(values)
        if torch.equal(values.cpu(), flagged):
            x = torch.full_like(x, 1e6)
        return x, valid

    monkeypatch.setattr(overlays, "prepare_row", flag)
    summary = overlays.build(paths, [(EVENT, SHOT), (EVENT, SHOT + 1)], device="cpu",
                             workers=0, model=Picky())
    assert summary["done"] == 1 and summary["failed"] == 1, summary
    good = overlays.row_states(paths, EVENT, SHOT, _meta(paths))["p0"]
    bad = overlays.row_states(paths, EVENT, SHOT + 1, _meta(paths, shot=SHOT + 1))["p0"]
    assert good["state"] == "done"
    assert bad == {"state": "failed", "reason": "RuntimeError: a bad tile",
                   "updated": bad["updated"]}


# --- the GPU helpers against the CPU definitions --------------------------------------


@pytest.mark.parametrize("n", [5, 19, 20, 21, 300])
def test_persist_open_matches_labels(n):
    rng = np.random.default_rng(n)
    mask = rng.random((3, 7, n)) < 0.8
    mask[0, 0, :] = True
    got = overlays.persist_open(torch.from_numpy(mask), 20).numpy()
    assert np.array_equal(got, ae_labels.persist_open(mask, 20))


def test_notch_and_the_union_on_the_grid_match_whole_build():
    rng = np.random.default_rng(7)
    clean = rng.random((4, 512, 900)) < 0.3
    clean[:, 40, :] = True  # a line lit on every frame: notched
    t_ms = 0.13 + np.arange(900) * 0.256
    grid = Grid(-1.0, 2.048, 120)
    expected = whole.build(grid, 250, (t_ms, ae_labels.apply_notch(
        clean, ae_labels.notch_bins(ae_labels.bin_active_fraction(clean), 0.8)), None))
    kept = overlays.notch(torch.from_numpy(clean))
    got = overlays.union_on_grid(kept, t_ms, grid, 250).numpy()
    assert np.array_equal(got, expected)
    assert np.array_equal(
        got, pseudo.pool_columns(pseudo.tokeye_rows(kept.numpy()), t_ms, grid)[:250]
    )


@pytest.mark.parametrize("seed", range(4))
def test_percentiles_of_codes_match_numpy(seed):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, size=(int(rng.integers(1, 40)), int(rng.integers(1, 90))))
    codes = codes.astype(np.uint8)
    got = overlays.percentiles_u8(torch.from_numpy(codes))
    assert got == tuple(np.percentile(codes, [1, 99]))


def test_ae_cuts_keep_every_frame_once_in_tiles_of_one_width():
    for n in (100, 2304, 2305, 9000):
        cuts = overlays.ae_cuts(n)
        kept = np.concatenate([np.arange(k0, k1) for _, _, k0, k1 in cuts])
        assert np.array_equal(kept, np.arange(n))
        widths = {hi - lo for lo, hi, _, _ in cuts}
        assert len(widths) == 1 and all(0 <= lo <= k0 and k1 <= hi <= n
                                        for lo, hi, k0, k1 in cuts)


# --- the AE union ---------------------------------------------------------------------


def _ae_store(paths, n=782, shot=SHOT):
    rows = [ImageRow(name, name, np.full((257, n), 9, dtype=np.uint8), y0=0.0,
                     dy=0.9765625, y_units="kHz", z_lo=-3.0, z_hi=27.0, z_units="dB")
            for name in ("R0xV1", "R0xV2", "R0xV3")]
    review_rows.write(paths.spectrogram_file("alfven_eigenmode", shot),
                      Grid(0.0, 0.256, n), rows, event="alfven_eigenmode")


def _co2(end_ms=200.0):
    """Four chords at 500 kHz from -10 ms: noise and a 100 kHz line from 50 to 150 ms."""
    rng = np.random.default_rng(11)
    t = -10.0 + np.arange(int((end_ms + 20) * 500)) / 500
    tone = np.where((t > 50) & (t < 150), np.sin(2 * np.pi * 100.0 * t), 0.0)
    chords = (rng.normal(0, 0.05, (4, t.size)) + tone).astype(np.float32)
    keep = t <= end_ms + full.MARGIN_MS
    return {"shot": SHOT, "chords": np.ascontiguousarray(chords[:, keep]),
            "n_samples": int(keep.sum()), "first_ms": float(t[0]),
            "last_ms": float(t[keep][-1]), "end_ms": end_ms}


def test_the_ae_union_is_whole_build_over_the_whole_record(paths, monkeypatch):
    _ae_store(paths)
    window = _co2()
    monkeypatch.setattr(gpu, "read_co2", lambda shot, paths, end_ms: dict(window))
    monkeypatch.setattr(overlays, "AE_KEEP", 256)  # several tiles on a short record
    monkeypatch.setattr(overlays, "AE_MARGIN", 32)
    model = Pointwise(coherent=1.5, transient=6.0)
    summary = overlays.build(paths, [("alfven_eigenmode", SHOT)], device="cpu",
                             workers=0, model=model)
    assert summary["done"] == 1, summary
    layer = overlays.ae_view(paths, SHOT)
    assert layer["layer"] == overlays.LAYER and layer["grid"]["n"] == 98

    signals, t0_ms, fs_hz = gpu.input_signals(window)
    spec = gpu.standardise(gpu.spectrogram(signals))
    with torch.inference_mode():
        probs = unet.probabilities(model, spec[:, None]).numpy()
    codes = np.floor(np.clip(probs, 0, 1) * 255)
    clean = ae_labels.clean_mask(codes[:, 0], codes[:, 1], threshold=51)
    clean = ae_labels.persist_open(clean, ae_labels.PERSIST_FRAMES)
    t_ms = full.frame_times(signals.shape[1], t0_ms, fs_hz)
    expected = whole.build(Grid(0.0, 0.256 * 8, 98), 257, (t_ms, ae_labels.apply_notch(
        clean, ae_labels.notch_bins(ae_labels.bin_active_fraction(clean), 0.8)), None))
    lit = _unpack(layer)
    assert expected[100:105].any()  # the 100 kHz line
    assert np.array_equal(lit, expected)
    states = overlays.row_states(paths, "alfven_eigenmode", SHOT,
                                 _meta(paths, "alfven_eigenmode"))
    assert {s["state"] for s in states.values()} == {"done"}

    # A tokeye-full layer appears: the page keeps it until the gate passes.
    whole.save(whole.layer_file(paths, SHOT), Grid(0.0, 2.048, 98), 0.0, 0.9765625, lit)
    assert overlays.ae_view(paths, SHOT) == whole.view(paths, SHOT)
    # The gate: against a tokeye-full layer equal to it, every IoU is 1.
    found = overlays.compare(paths)
    assert found["passed"] and found["n"] == 1 and found["bar"]["passed"]
    assert found["shots"][0]["iou"] == {"0-250": 1.0, "80-250": 1.0}
    gate = json.loads(overlays.gate_file(paths).read_text())
    assert gate["passed"] is True and gate["shots_compared"] == [SHOT]
    assert gate["model_sha256"] == overlays.MODEL_SHA256 and gate["excluded"] == []
    assert overlays.ae_view(paths, SHOT)["layer"] == overlays.LAYER


def _ae_built(paths, monkeypatch, shot=SHOT):
    """An AE store and this model's union for `shot` (a stand-in network on the CPU)."""
    _ae_store(paths, shot=shot)
    window = _co2()
    monkeypatch.setattr(gpu, "read_co2", lambda shot, paths, end_ms: dict(window))
    summary = overlays.build(paths, [("alfven_eigenmode", shot)], device="cpu",
                             workers=0, model=Pointwise(coherent=1.5, transient=6.0))
    assert summary["done"] == 1, summary


def _ae_layer_names(paths, shot=SHOT):
    states = overlays.row_states(paths, "alfven_eigenmode", shot,
                                 _meta(paths, "alfven_eigenmode", shot))
    return overlays.ae_view(paths, shot)["layer"], {s["layer"] for s in states.values()}


def test_the_ae_page_keeps_tokeye_full_until_the_gate_passes(paths, monkeypatch):
    _ae_built(paths, monkeypatch)
    other = np.zeros((257, 98), dtype=bool)
    other[30:40, 50:60] = True  # nothing like the new union: the gate fails
    whole.save(whole.layer_file(paths, SHOT), Grid(0.0, 2.048, 98), 0.0, 0.9765625, other)
    legacy = ("tokeye-full", {"tokeye-full"})
    assert _ae_layer_names(paths) == legacy  # no gate yet
    found = overlays.compare(paths)
    assert found["n"] == 1 and not found["passed"] and not found["bar"]["passed"]
    assert json.loads(overlays.gate_file(paths).read_text())["passed"] is False
    assert _ae_layer_names(paths) == legacy  # a failed gate
    assert overlays.request(paths, "alfven_eigenmode", SHOT)["state"] == "done"
    gate = overlays.gate_file(paths)
    gate.write_text(json.dumps({"passed": True, "model_sha256": "0" * 64}))
    assert _ae_layer_names(paths) == legacy  # another model's pass
    gate.write_text("not json")
    assert _ae_layer_names(paths) == legacy  # an unreadable gate
    gate.write_text(json.dumps({"passed": True, "model_sha256": overlays.MODEL_SHA256}))
    assert _ae_layer_names(paths) == (overlays.LAYER, {overlays.LAYER})  # a passed gate


def test_the_gate_fails_unless_every_tokeye_full_shot_is_compared(paths, monkeypatch):
    _ae_built(paths, monkeypatch)
    lit = _unpack(overlays.ae_view(paths, SHOT))
    for shot in (SHOT, SHOT + 1):
        whole.save(whole.layer_file(paths, shot), Grid(0.0, 2.048, 98), 0.0, 0.9765625,
                   lit)
    found = overlays.compare(paths)
    assert found["n"] == 1 and found["n_shots"] == 2 and found["bar"]["passed"]
    assert not found["passed"]
    assert found["excluded"] == [{"shot": SHOT + 1,
                                  "reason": "the new layer is not_computed: No TokEye yet"}]
    assert json.loads(overlays.gate_file(paths).read_text())["passed"] is False
    # A subset is a look, not a gate: it neither passes nor writes one.
    before = overlays.gate_file(paths).read_text()
    subset = overlays.compare(paths, [SHOT])
    assert subset["subset"] and not subset["passed"] and subset["gate"] is None
    assert overlays.gate_file(paths).read_text() == before
    # The command exits 1 on a failed gate (on this test's root, never the real one).
    monkeypatch.setenv(raw.NO_FETCH_ENV, "1")
    monkeypatch.setattr(overlays.Paths, "from_env", staticmethod(lambda *a, **k: paths))
    assert overlays.main(["compare"]) == 1


def test_the_ae_page_falls_back_to_the_tokeye_full_layer(paths):
    _ae_store(paths)
    assert overlays.ae_view(paths, SHOT) is None
    lit = np.zeros((257, 98), dtype=bool)
    lit[100, 10:20] = True
    whole.save(whole.layer_file(paths, SHOT), Grid(0.0, 2.048, 98), 0.0, 0.9765625, lit)
    assert overlays.ae_view(paths, SHOT) == whole.view(paths, SHOT)
    states = overlays.row_states(paths, "alfven_eigenmode", SHOT,
                                 _meta(paths, "alfven_eigenmode"))
    assert states["R0xV1"] == {"state": "done", "layer": "tokeye-full",
                               "reason": "TokEye from the tokeye-full layer (0-6 s)"}
    assert overlays.request(paths, "alfven_eigenmode", SHOT)["state"] == "done"
