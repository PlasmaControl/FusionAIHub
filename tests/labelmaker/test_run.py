"""End-to-end through the CLI on synthetic data, in a temp root."""
import json
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

from labelmaker import run
from labelmaker.features import namespace as ns
from labelmaker.features.store import is_complete, write_features
from labelmaker.labels.store import labelled, read_label
from labelmaker.models.base import (
    InputField,
    InputSpec,
    ModelAdapter,
    OutputField,
    OutputSpec,
)

SLUG = "fake_model"
NAMES = ["bt", "ip", "pinj_total", "ne_zipfit"]


def _archive(tmp_path, shots=(190000, 190001), n=240):
    p = tmp_path / "archive.h5"
    with h5py.File(p, "w") as f:
        for shot in shots:
            g = f.create_group(str(shot))
            g.create_dataset("bt", data=np.full(n, 2.0))
            g.create_dataset("ip", data=np.full(n, 1.0e6))
            g.create_dataset(
                "zipfit_edensfit_rho", data=np.tile(np.linspace(4.0, 1.0, 33), (n, 1))
            )
            # pinj deliberately absent: it must fall through to the corpus
    return p


def _corpus(tmp_path, shots=(190000, 190001)):
    d = tmp_path / "corpus"
    d.mkdir()
    n, t = 2000, np.linspace(0.0, 2.0, 2000, dtype=np.float32)
    for shot in shots:
        with h5py.File(d / f"{shot}_processed.h5", "w") as f:
            g = f.create_group("pinj")
            g.create_dataset("xdata", data=t)
            g.create_dataset("ydata", data=np.full((8, n), 1.0e6, dtype=np.float32))
    return d


def _predict(built):
    # two "members": the scalar sum, and the profile mean
    a = built.scalars.sum(axis=1)
    b = built.profiles.mean(axis=(1, 2))
    return np.stack([a[:, None], b[:, None]])


def _adapter(fields, ensemble_n=2):
    return ModelAdapter(
        slug=SLUG, card_id="test/fake-model", framework="none", time_step_ms=25.0,
        artifacts=(), upstream="none",
        input_spec=InputSpec(fields=fields, dt_s=0.025),
        output_spec=OutputSpec(
            fields=(OutputField("score", "binary", column=0, activation="sigmoid"),)
        ),
        load=lambda model_dir: _predict,
        ensemble_n=ensemble_n,
    )


def _fake_adapter():
    return _adapter((
        InputField("bt", "bt", lag="t+dt"),
        InputField("ip", "ip", lag="t+dt"),
        InputField("pinj", "pinj_total", lag="t+dt"),
        InputField("ne", "ne_zipfit", lag="t"),
    ))


def _archive_only_adapter():
    """A model every one of whose features the archive can serve alone."""
    return _adapter((
        InputField("bt", "bt", lag="t+dt"),
        InputField("ip", "ip", lag="t+dt"),
    ))


@pytest.fixture
def wired(tmp_path, monkeypatch):
    from labelmaker.models import registry

    monkeypatch.setattr(registry, "load_adapter", lambda slug: _fake_adapter())
    monkeypatch.setattr(registry, "verify_artifacts", lambda slug, d: None)
    monkeypatch.setattr(
        registry, "read_card",
        lambda slug: {"labelmaker": {"upstream": {"sha256": {"fake.h5": "00"}}}},
    )
    return {
        "archive": _archive(tmp_path),
        "corpus": _corpus(tmp_path),
        "root": tmp_path / "out",
    }


def _argv(wired, stage, *extra, shots=("190000", "190001")):
    return [
        stage, "--models", SLUG, "--shots", *shots,
        "--root", str(wired["root"]), "--corpus-dir", str(wired["corpus"]),
        "--archive", str(wired["archive"]), "--workers", "1", *extra,
    ]


def test_features_stage_writes_one_file_per_shot(wired):
    assert run.main(_argv(wired, "features")) == 0
    for shot in (190000, 190001):
        p = wired["root"] / "features" / f"{shot}_features.h5"
        assert p.exists()
        with h5py.File(p, "r") as f:
            assert set(f.keys()) == {"bt", "ip", "pinj_total", "ne_zipfit"}
            assert f["bt"].attrs["resolver"] == "archive"
            assert f["pinj_total"].attrs["resolver"] == "corpus"
            assert json.loads(f.attrs["missing"]) == {}


def test_features_stage_skips_complete_shots_and_force_overrides(wired, capsys):
    run.main(_argv(wired, "features"))
    run.main(_argv(wired, "features"))
    assert "skipped" in capsys.readouterr().out
    assert run.main(_argv(wired, "features", "--force")) == 0


def test_every_status_logs_the_same_row_shape(wired):
    # `skipped` carries the shot's provenance and misses too, so nothing
    # reading log.txt has to special-case a status to learn where a shot's
    # features came from.
    run.main(_argv(wired, "features"))
    run.main(_argv(wired, "features"))
    keys = {"shot", "status", "resolved", "missing", "resolvers", "sources",
            "mixed", "seconds"}
    for run_dir in (wired["root"] / "runs").iterdir():
        for line in (run_dir / "log.txt").read_text().splitlines():
            row = json.loads(line)
            assert set(row) == keys, row["status"]
            assert row["resolved"] == 4


def test_infer_stage_writes_labels_and_the_index(wired):
    run.main(_argv(wired, "features"))
    assert run.main(_argv(wired, "infer")) == 0
    p = wired["root"] / "labels" / "190000_labels.h5"
    assert labelled(p) == {f"{SLUG}/score"}
    got = read_label(p, SLUG, "score")
    assert got.y.shape == (1, 240)
    assert ((got.y >= 0) & (got.y <= 1)).all()          # sigmoid applied
    assert got.attrs["task"] == "binary"
    df = pd.read_parquet(wired["root"] / "labels_index.parquet")
    assert set(df["shot"]) == {190000, 190001}
    assert set(df["label"]) == {"score"}
    assert (df["n_total"] == 240).all()


def test_all_chains_the_stages(wired):
    assert run.main(_argv(wired, "all")) == 0
    assert (wired["root"] / "labels" / "190001_labels.h5").exists()


def test_infer_refuses_to_run_against_unverified_weights(tmp_path, monkeypatch):
    # The existing tests monkeypatch verify_artifacts to a no-op, so nothing
    # otherwise asserts the pipeline reaches its only weight guard. Here the
    # card names a digest for a file that is absent, so the run must abort
    # before writing any label rather than failing every shot in turn.
    from labelmaker.models import registry

    monkeypatch.setattr(registry, "load_adapter", lambda slug: _fake_adapter())
    monkeypatch.setattr(
        registry, "read_card",
        lambda slug: {"labelmaker": {"upstream": {"sha256": {"absent.h5": "00" * 32}}}},
    )
    archive, corpus, root = _archive(tmp_path), _corpus(tmp_path), tmp_path / "out"
    argv = [
        "infer", "--models", SLUG, "--shots", "190000",
        "--root", str(root), "--corpus-dir", str(corpus),
        "--archive", str(archive), "--workers", "1",
    ]
    assert run.main(argv) == 3
    assert not (root / "labels").exists() or not list((root / "labels").iterdir())


def test_the_weight_guard_runs_before_the_features_stage(tmp_path, monkeypatch):
    # `all` must not spend the whole features pass on a run whose labels
    # could never be trusted. The guard is therefore before either pool
    # forks, not merely before the infer pool: with a digest the artifact
    # cannot match, nothing at all is written.
    from labelmaker.models import registry

    monkeypatch.setattr(registry, "load_adapter", lambda slug: _fake_adapter())
    monkeypatch.setattr(
        registry, "read_card",
        lambda slug: {"labelmaker": {"upstream": {"sha256": {"absent.h5": "00" * 32}}}},
    )
    archive, corpus, root = _archive(tmp_path), _corpus(tmp_path), tmp_path / "out"
    argv = [
        "all", "--models", SLUG, "--shots", "190000", "190001",
        "--root", str(root), "--corpus-dir", str(corpus),
        "--archive", str(archive), "--workers", "1",
    ]
    assert run.main(argv) == 3
    assert list((root / "features").iterdir()) == []
    assert list((root / "labels").iterdir()) == []
    assert list((root / "runs").iterdir()) == []


def test_the_weight_guard_is_called_for_every_model_before_any_shot(
    tmp_path, monkeypatch
):
    # Proves the guard is *reached*, not merely that a broken artifact
    # happens to fail: it records the call and what the run had written by
    # the time it happened.
    from labelmaker.models import registry

    seen = []

    def recording_verify(slug, model_dir):
        seen.append((slug, sorted(p.name for p in (tmp_path / "out" / "features")
                                  .iterdir())))

    monkeypatch.setattr(registry, "load_adapter", lambda slug: _fake_adapter())
    monkeypatch.setattr(registry, "verify_artifacts", recording_verify)
    monkeypatch.setattr(
        registry, "read_card",
        lambda slug: {"labelmaker": {"upstream": {"sha256": {"fake.h5": "00"}}}},
    )
    archive, corpus, root = _archive(tmp_path), _corpus(tmp_path), tmp_path / "out"
    argv = [
        "all", "--models", SLUG, "--shots", "190000",
        "--root", str(root), "--corpus-dir", str(corpus),
        "--archive", str(archive), "--workers", "1",
    ]
    assert run.main(argv) == 0
    # First call is the parent's up-front guard, with nothing written yet.
    assert seen[0] == (SLUG, [])
    # The worker re-checks too, which is what catches an artifact swapped
    # out mid-run.
    assert len(seen) >= 2
    assert {slug for slug, _ in seen} == {SLUG}


def test_a_model_that_cannot_be_loaded_is_a_run_level_fault(tmp_path, monkeypatch):
    # A scaffold spec raises NotImplementedError at import. That is not a
    # per-shot error to be repeated N times.
    from labelmaker.models import registry

    def boom(slug):
        raise NotImplementedError(f"{slug} is a scaffold")

    monkeypatch.setattr(registry, "load_adapter", boom)
    root = tmp_path / "out"
    assert run.main([
        "features", "--models", SLUG, "--shots", "190000", "--root", str(root),
        "--workers", "1",
    ]) == 4
    assert list((root / "features").iterdir()) == []


def test_a_broken_shot_does_not_stop_the_run(wired):
    bad = wired["corpus"] / "190001_processed.h5"
    bad.write_bytes(b"not an hdf5 file")
    assert run.main(_argv(wired, "all")) == 0
    assert (wired["root"] / "labels" / "190000_labels.h5").exists()
    with h5py.File(wired["root"] / "features" / "190001_features.h5", "r") as f:
        assert "pinj_total" in json.loads(f.attrs["missing"])


def test_every_run_writes_a_manifest(wired):
    run.main(_argv(wired, "features"))
    runs = sorted((wired["root"] / "runs").iterdir())
    assert len(runs) == 1
    manifest = json.loads((runs[0] / "manifest.json").read_text())
    assert manifest["stage"] == "features"
    assert manifest["models"] == [SLUG]
    assert manifest["shots"] == [190000, 190001]
    assert manifest["git_sha"] and manifest["labelmaker_version"]
    assert manifest["hostname"]
    assert (runs[0] / "log.txt").exists()


def test_shot_selection_from_a_file(wired, tmp_path):
    f = tmp_path / "shots.txt"
    f.write_text("190001\n# comment\n")
    argv = [
        "features", "--models", SLUG, "--shot-file", str(f),
        "--root", str(wired["root"]), "--corpus-dir", str(wired["corpus"]),
        "--archive", str(wired["archive"]), "--workers", "1",
    ]
    assert run.main(argv) == 0
    assert (wired["root"] / "features" / "190001_features.h5").exists()
    assert not (wired["root"] / "features" / "190000_features.h5").exists()


def test_a_shot_selection_is_required(wired):
    with pytest.raises(SystemExit):
        run.main(["features", "--models", SLUG, "--root", str(wired["root"])])


def test_time_limit_raises_rather_than_hanging():
    import time

    with pytest.raises(run.StageTimeout), run.time_limit(1):
        time.sleep(3)


# --------------------------------------------------------------------------
# Source mixing: the property Task 15 needs and nothing else records.
# --------------------------------------------------------------------------


def test_a_mixed_source_shot_is_flagged_in_the_log_and_the_summary(wired, capsys):
    # The fixture is deliberately a mixed build: the archive serves bt, ip
    # and the density, and the corpus serves pinj. Those do not share a time
    # base - the archive stamps its rows one whole dt late - so a run has to
    # say which shots are affected instead of leaving the mix implicit in
    # per-group attributes nobody reads.
    assert run.main(_argv(wired, "all")) == 0
    out = capsys.readouterr().out
    assert "mixed sources" in out
    assert "25 ms" in out
    run_dir = next(iter((wired["root"] / "runs").iterdir()))
    summary = json.loads((run_dir / "summary.json").read_text())
    features = next(s for s in summary["stages"] if s["stage"] == "features")
    assert features["mixed_source_shots"] == [190000, 190001]
    assert features["source_combinations"] == {"archive+corpus": 2}
    assert "3.9e-08" in summary["mixed_source_note"]
    logged = [
        json.loads(line)
        for line in (run_dir / "log.txt").read_text().splitlines()
    ]
    by_shot = {(r["shot"], r["status"]): r for r in logged}
    row = by_shot[(190000, "ok")]
    assert row["mixed"] is True
    assert row["resolvers"] == {
        "bt": "archive", "ip": "archive",
        "ne_zipfit": "archive", "pinj_total": "corpus",
    }
    # The infer stage carries the same verdict, as the model saw it.
    infer = next(s for s in summary["stages"] if s["stage"].startswith("infer"))
    assert infer["mixed_source_shots"] == [190000, 190001]


def test_a_single_source_shot_is_not_flagged(wired, monkeypatch):
    # The other half of the previous test: `mixed` has to be able to say no,
    # or flagging everything would be indistinguishable from flagging the
    # right thing.
    from labelmaker.models import registry

    monkeypatch.setattr(registry, "load_adapter", lambda slug: _archive_only_adapter())
    assert run.main(_argv(wired, "features")) == 0
    run_dir = next(iter((wired["root"] / "runs").iterdir()))
    summary = json.loads((run_dir / "summary.json").read_text())
    features = summary["stages"][0]
    assert features["mixed_source_shots"] == []
    assert features["source_combinations"] == {"archive": 2}


# --------------------------------------------------------------------------
# Fork safety. The pool is forked before any fetch, so nothing fork-unsafe
# may be imported in the parent - and toksearch_d3d's ptserver reader is
# fork-unsafe.
# --------------------------------------------------------------------------


def _import_probe(module: str) -> list[str]:
    """Names in `sys.modules` matching toksearch after importing `module`.

    A subprocess, not this process: by the time this test runs, another test
    or a conftest may already have imported anything, so an in-process check
    would be answering a different question.
    """
    code = (
        "import sys\n"
        f"import {module}\n"
        "print(repr(sorted(m for m in sys.modules if 'toksearch' in m "
        "or m == 'labelmaker.features.resolve_fdp')))\n"
    )
    # Fixed argv, no shell, no caller input.
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    # `eval` of our own `repr` of a list of str, printed by the line above.
    return eval(out.stdout.strip())


def test_importing_the_runner_imports_neither_toksearch_nor_the_fdp_resolver():
    # Two probes, because the first alone would be vacuous: `resolve_fdp`
    # already defers its own toksearch imports into function bodies, so
    # hoisting run.py's `from .features import resolve_fdp` to module scope
    # would NOT make toksearch appear. The second probe is the one that goes
    # red for that mutation; the first covers the case where resolve_fdp
    # stops deferring.
    assert _import_probe("labelmaker.run") == []
    assert _import_probe("labelmaker.features.resolve_fdp") == [
        "labelmaker.features.resolve_fdp"
    ]


# --------------------------------------------------------------------------
# Resume behaviour: which recorded misses a rerun asks for again.
# --------------------------------------------------------------------------


def _recording_resolver(monkeypatch):
    calls = []

    def fake(source, shot, want, ctx):
        calls.append((source, list(want)))
        return {}, {}

    monkeypatch.setattr(run, "_resolve_one_source", fake)
    return calls


def test_a_rerun_asks_for_nothing_when_every_miss_is_permanent(wired, monkeypatch):
    path = wired["root"] / "features" / "190000_features.h5"
    write_features(path, 190000, {}, {
        "bt": "archive:KeyError",
        "ip": "archive:KeyError,fdp:MdsException",
        "pinj_total": "corpus:SignalAbsent",
        "ne_zipfit": "archive:KeyError,fdp:MdsException",
    })
    calls = _recording_resolver(monkeypatch)
    assert run.main(_argv(wired, "features", shots=("190000",))) == 0
    assert calls == []


def test_a_rerun_asks_only_for_the_transiently_missed_feature(wired, monkeypatch):
    # The point of the transient/permanent split: one timed-out fetch must
    # not drag the shot's three settled misses back through the resolvers,
    # and must not be settled itself.
    path = wired["root"] / "features" / "190000_features.h5"
    write_features(path, 190000, {}, {
        "bt": "archive:KeyError",
        "ip": "archive:KeyError,fdp:TimeoutError",
        "pinj_total": "corpus:SignalAbsent",
        "ne_zipfit": "archive:KeyError,fdp:MdsException",
    })
    calls = _recording_resolver(monkeypatch)
    assert run.main(_argv(wired, "features", shots=("190000",))) == 0
    # `ip` alone, offered to both of its sources in preference order.
    assert calls == [("archive", ["ip"]), ("fdp", ["ip"])]


def test_force_retries_everything_including_permanent_misses(wired, monkeypatch):
    path = wired["root"] / "features" / "190000_features.h5"
    write_features(path, 190000, {}, dict.fromkeys(NAMES, "archive:KeyError"))
    calls = _recording_resolver(monkeypatch)
    assert run.main(_argv(wired, "features", "--force", shots=("190000",))) == 0
    asked = {n for _, want in calls for n in want}
    assert asked == set(NAMES)


# --------------------------------------------------------------------------
# The parallel path. The serial path (`--workers 1`) skips the pool
# entirely, so every other test above exercises only half the runner.
# --------------------------------------------------------------------------


def test_the_pool_path_produces_the_same_files(wired):
    argv = _argv(wired, "all")
    argv[argv.index("--workers") + 1] = "2"
    assert run.main(argv) == 0
    for shot in (190000, 190001):
        assert (wired["root"] / "features" / f"{shot}_features.h5").exists()
        assert (wired["root"] / "labels" / f"{shot}_labels.h5").exists()
    df = pd.read_parquet(wired["root"] / "labels_index.parquet")
    assert set(df["shot"]) == {190000, 190001}


def test_a_shot_that_exceeds_its_budget_stops_and_the_run_continues(
    wired, monkeypatch
):
    # The per-shot SIGALRM must not be absorbed into a per-source miss.
    # `calls` is the load-bearing assertion: SIGALRM is one-shot, so if
    # `features_for_shot` catches StageTimeout with its generic per-source
    # handler, the shot goes on to the remaining sources with no budget left
    # and sleeps again for each of them. Without the dedicated clause this
    # test sees three calls and a `partial` status.
    calls = []

    def slow(source, shot, want, ctx):
        calls.append((shot, source))
        if shot == 190001:
            import time

            time.sleep(3)
        return {}, dict.fromkeys(want, "Nope")

    monkeypatch.setattr(run, "_resolve_one_source", slow)
    argv = _argv(wired, "features", "--timeout", "1")
    assert run.main(argv) == 0
    assert [s for shot, s in calls if shot == 190001] == ["archive"]
    run_dir = next(iter((wired["root"] / "runs").iterdir()))
    logged = [
        json.loads(line)
        for line in (run_dir / "log.txt").read_text().splitlines()
    ]
    by_shot = {r["shot"]: r for r in logged}
    assert by_shot[190001]["status"] == "timeout"
    # A timeout is a property of the attempt, so the next run retries it.
    assert by_shot[190001]["missing"]["bt"] == "archive:StageTimeout"
    assert not is_complete(
        wired["root"] / "features" / "190001_features.h5", NAMES
    )
    # The other shot is untouched by its neighbour's budget.
    assert by_shot[190000]["status"] == "partial"


def test_ns_grid_is_the_label_time_base(wired):
    # The labels' x axis is the canonical 25 ms grid, not whatever the
    # features happened to carry.
    run.main(_argv(wired, "all", shots=("190000",)))
    got = read_label(wired["root"] / "labels" / "190000_labels.h5", SLUG, "score")
    np.testing.assert_allclose(got.x, ns.GRID_S)
