"""Plan D.8's twenty-one D2 checks, review D2B's dock width at 1280 (22, 23), D2C's colour
bar and value readout (24) and plan D.B's dock checks in a real browser: views,
the dock, pins, the Values readout, the cut margins, added signals with their fetch and
error states, saved views, the narrow overlay, and on detachment the dock that opens with
the TangTV inversion, the EFIT equilibrium and the cameras; and round R2's page (R1-R9): the
shot box's search, the dock tab, the slices of a shot without camera frames, row select and
Delete and Pop out, q in the dock by default on mhd_n1, the Add row box (a pointname finds
its catalog entry), one selection for rows and annotations, a failed popped-out row's chip,
and the row undo kept to its layout. The scenario is
review_workspace.mjs, run on review_browser.mjs's transport as test_review_page_d1_browser.py
runs its own, against an api-11 workspace server whose fetch worker is a stand-in: a
PTDATA point is queued, then fetching, then answered, and nothing leaves the machine."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import uvicorn

from labeler.config import Paths
from labeler.events.raw import NO_FETCH_ENV
from labeler.events.review import build, labels
from labeler.events.review import catalog as cat
from labeler.events.review import rows as review_rows
from labeler.events.review.fetch_worker import Fdp, Job
from labeler.events.review.rows import Grid, ImageRow, TraceRow
from labeler.events.ui import workspace_routes
from labeler.events.ui.app import create_app

from .test_review_browser import DRIVER, NODE, SHELLS, SOURCE, TOKEN, _store
from .test_review_detachment_browser import detachment_store
from .test_review_masks_browser import _layer, _pseudo
from .test_review_slices import write_equilibrium, write_inversion

SCENARIO = Path(__file__).with_name("review_workspace.mjs")
REPO_CONFIG = Path(__file__).parents[2] / "configs" / "labeler"
AE, MHD, EMPTY = "alfven_eigenmode", "mhd_n1", "disruption"
MHD_SHOTS = (180001, 180002)
#: The detachment shot's slices: the video frames fall at 0, 40, ... 240 ms, so a seek
#: to 95 ms shows frame 80 (inversion 100, EFIT 80) and one to 170 ms frame 160
#: (inversion 150, EFIT 190). review_workspace.mjs computes the same from S.video.
INVERSION_MS = (0, 50, 100, 150, 200, 240)
EFIT_MS = (0, 30, 80, 110, 190, 240)
HEADER = "shot,tier,holdout,reviewers,verified_on,notes\n"
CATALOG = """
signals:
  - {id: betan, label: beta_N, group: Equilibrium, units: "", kind: trace,
     events: [mhd_n1], sources: [{corpus: betan, rows: 0}]}
  - {id: ece_map, label: "ECE Te, channels 1-40", group: Core, units: keV, kind: trace,
     draw: map, events: [mhd_n1], sources: [{corpus: ece, rows: "0-39"}]}
  - {id: q_profile, label: q, group: Profiles, units: "", kind: profile, axis: psi_n,
     events: [mhd_n1], sources: [{recipe: park_q}]}
  - {id: ts_core_te, label: Thomson core Te, group: Profiles, units: keV,
     kind: profile, axis: R_m, events: [mhd_n1], sources: [{recipe: thomson_te}]}
"""

pytestmark = [
    pytest.mark.usefixtures("detachment_inputs"),
    pytest.mark.skipif(
        NODE is None or not SHELLS, reason="needs node and a headless Chromium"
    ),
]


class ProgressFetcher:
    """The fetch worker's interface without a worker: a submitted job is queued; the
    next look finds it fetching; the one after hands `on_result` a sine over 0-4000 ms
    and the job leaves, so /api/signal answers 202 queued, 202 fetching, then 200."""

    def __init__(self):
        self.jobs: dict = {}
        self.looks: dict = {}

    def submit(self, key, request, *, shot, ident, where, limit_s, on_result):
        return self.jobs.setdefault(
            key, Job(key, request, shot, ident, where, limit_s, on_result)
        )

    def job(self, key):
        job = self.jobs.get(key)
        if job is None:
            return None
        self.looks[key] = self.looks.get(key, 0) + 1
        if self.looks[key] == 1:
            job.state, job.stage = "running", "fetching"
            return job
        t = np.arange(4000, dtype=np.float64)
        job.on_result({"t_ms": t, "y": np.sin(t / 250)[None].astype(np.float32),
                       "units": "V"})
        del self.jobs[key]
        return None

    def take_error(self, key):
        return None

    def position(self, job):
        return 1

    def summary(self):
        return {"queue": len(self.jobs), "running": None}


class Watch:
    """An fdp login that is valid until 2030."""

    def get(self):
        return Fdp(True, "2030-01-01T00:00:00Z", "ok")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _mhd_store(paths: Paths, shot: int) -> None:
    """The mhd_n1 rows by their real titles over 0-4000 ms at 1 ms: a power spectrogram
    (128 bins of 2 kHz, a line near 80 kHz), the toroidal-n modes row (n = 1 before
    2000 ms, n = 2 after, 40-100 kHz) and a three-channel amplitude trace."""
    t = np.arange(4000.0)
    bins = np.arange(128)[:, None]
    line = 40 + 10 * np.sin(t / 400)
    power = (30 + 210 * np.exp(-((bins - line) ** 2) / 8)).astype("uint8")
    mode = (t >= 2000).astype(np.int64)
    lit = (bins >= 20) & (bins < 50)
    modes = np.where(lit, 100 * 2 + mode[None, :], 0).astype("uint8")
    amplitude = np.stack([np.sin(t / (150 * k)) for k in (1, 2, 3)])
    rows = [
        ImageRow("power", "MPI66M322D power", power, y0=0.0, dy=2.0, y_units="kHz",
                 z_lo=-40.0, z_hi=0.0, z_units="dB"),
        ImageRow("modes", "toroidal n, MPI66M", modes, y0=0.0, dy=2.0, y_units="kHz",
                 z_lo=-40.0, z_hi=0.0, z_units="dB",
                 modes={"n": [1, 2], "levels": 128, "colours": ["#ff3b30", "#3d8bff"]}),
        TraceRow("amplitude", "n=1..3 amplitude", np.stack([amplitude, amplitude]),
                 y_units="G", legend=["n=1", "n=2", "n=3"]),
    ]
    review_rows.write(paths.spectrogram_file(MHD, shot), Grid(0.0, 1.0, 4000), rows,
                      event=MHD, panel_version=build.PANEL_VERSIONS[MHD])


def _mhd_inputs(paths: Paths, shot: int) -> None:
    """The corpus (ECE 40 channels, beta_N, core Thomson on uneven times) and an EFIT02
    park file on 16 uneven times whose q runs from 0.9 to 3.5 and above."""
    seconds = np.arange(4000) / 1e3
    channel = np.arange(40)[:, None]
    ece = 3 * (1 - channel / 45) + 0.4 * np.sin(2 * np.pi * 3 * seconds + channel / 5)
    rng = np.random.default_rng(shot)
    ts_t = np.sort(rng.uniform(0.05, 3.95, 60))
    chord = np.arange(44)[:, None]
    ts = 1000 * (2.2 - chord / 25) + 150 * np.sin(ts_t * 4)[None, :]
    path = paths.corpus_file(shot)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, x, y in (("ece", seconds, ece),
                           ("betan", seconds, 1.5 + 0.5 * np.sin(seconds / 0.7)[None]),
                           ("ts_core_temp", ts_t, ts)):
            f[f"{name}/xdata"] = x
            f[f"{name}/ydata"] = np.asarray(y, dtype=np.float32)
    times = np.array([50, 120, 300, 310, 600, 900, 1000, 1400, 1450, 1900, 2300, 2600,
                      3000, 3300, 3700, 3950], dtype=np.float64)
    psin = np.linspace(0, 1, 33)
    qpsi = 0.9 + 2.6 * psin[None, :] ** 2 + 0.02 * np.arange(len(times))[:, None]
    park = paths.root / "mhd" / "efit02"
    park.mkdir(parents=True, exist_ok=True)
    np.savez(park / f"{shot}.npz", times=times, qpsi=qpsi, fpol=np.ones_like(qpsi),
             psirz=np.zeros((len(times), 9, 9)), r=np.linspace(1, 2, 9),
             z=np.linspace(-1, 1, 9), ssimag=np.zeros(len(times)),
             ssibry=np.ones(len(times)), rmaxis=np.full(len(times), 1.7))


def workspace_inputs(tmp_path: Path, monkeypatch) -> Paths:
    """Every event the scenario opens, all on disk.

    alfven_eigenmode: 170815 (the pseudo-mask and a TokEye layer) and 170816. mhd_n1:
    180001 and 180002 with current stores, the corpus and EFIT02 park files the catalog
    and the profiles read. detachment: 170815 as test_review_detachment_browser builds
    it, with the EFIT equilibria and TangTV inversions the dock's slice panel reads.
    disruption: a header-only roster. The config holds the fixture catalog and the
    repository's presets ("q and Te" on mhd_n1); there is no events.yaml (an empty
    taxonomy, as in the other browser tests). The detachment environment variables are
    set through `monkeypatch`; the caller provides `detachment_inputs`'s files."""
    paths = Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus",
        text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tmp_path / "tables", raw_cache=tmp_path / "raw",
        config_dir=tmp_path / "config",
    )
    _write(paths.config_dir / cat.CATALOG_FILE, CATALOG)
    shutil.copy(REPO_CONFIG / "views.yaml", paths.config_dir / "views.yaml")
    tables = paths.label_tables
    _write(tables / AE / "shots.csv",
           HEADER + "170815,gold,false,,,\n170816,unverified,false,,,\n")
    _write(tables / AE / f"format/{AE}_format_2026_v1.csv", SOURCE)
    for shot in (170815, 170816):
        _store(paths, shot)
    _pseudo(paths, 170815)
    _layer(paths, 170815)
    _write(tables / MHD / "shots.csv",
           HEADER + "".join(f"{shot},unverified,false,,,\n" for shot in MHD_SHOTS))
    for shot in MHD_SHOTS:
        _mhd_store(paths, shot)
        _mhd_inputs(paths, shot)
    _write(tables / EMPTY / "shots.csv", HEADER)
    _write(tables / EMPTY / f"format/{EMPTY}_format_2026_v1.csv", SOURCE.splitlines()[0])
    detachment_store(paths, tmp_path, monkeypatch)
    write_equilibrium(paths, 170815, times=EFIT_MS)
    write_inversion(paths, 170815, times=INVERSION_MS)
    return paths


def workspace_server(paths: Paths, monkeypatch):
    """The app with fetching on, the stand-in worker and a valid fdp login."""
    monkeypatch.delenv(NO_FETCH_ENV, raising=False)
    monkeypatch.setattr(workspace_routes, "WATCH", Watch())
    app = create_app(paths=paths, token=TOKEN)
    app.state.workspace.fetcher = ProgressFetcher()
    app.state.workspace.watch = Watch()
    return uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0,
                                         log_level="warning"))


@pytest.fixture
def served(tmp_path, monkeypatch):
    paths = workspace_inputs(tmp_path, monkeypatch)
    server = workspace_server(paths, monkeypatch)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", paths
    server.should_exit = True
    thread.join(10)


def run_scenario(base: str, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """review_workspace.mjs on the driver's browser transport."""
    script = tmp_path / "review_workspace.mjs"
    prefix = DRIVER.read_text().split("const opened =", 1)[0]
    script.write_text(prefix + SCENARIO.read_text())
    return subprocess.run(
        [NODE, str(script), base, TOKEN, str(SHELLS[-1]), str(tmp_path / "profile"),
         *args],
        capture_output=True, text=True, timeout=420, check=False,
    )


def test_the_d2_workspace_in_a_browser(served, tmp_path):
    base, paths = served
    result = run_scenario(base, tmp_path, "checks")
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-4000:]
    checks = json.loads(result.stdout.splitlines()[-1])
    print(json.dumps(checks, indent=1))  # the report's numbers (pytest -rP)
    assert [c for c in checks if not c["ok"]] == [], checks
    names = [c["name"].split(".")[0] for c in checks]
    assert names == ([str(n) for n in range(1, 25)] + [f"B{n}" for n in range(1, 6)]
                     + [f"R{n}" for n in range(1, 10)] + ["E"])
    spans = labels.read_saved(paths.label_tables / AE)[170815].intervals
    assert {(1240, 1360), (500, 600)} <= {(a, b) for a, b, *_ in spans}, spans
