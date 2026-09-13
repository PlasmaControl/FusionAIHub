"""What may become a diagnostic feature, and what may not.

The iteration-0 critic reproduced two contaminations on REAL records: shot
185980's logbook line "Updated ELM detector tuning." became an ELM at
`t = 0` with `elm_rate_hz = 2.94` in the window `(0.0, 0.34)`, and shot
193348's "Sawtooth piggyback: Density higher than desired but good shot."
became `n_sawtooth = 1` in the same window. A schema-valid
`label_forecast` ELM row at `t = 1` became `elm_rate_hz = 2.94` in
`(0.9, 1.24)`. All three are `windows.EventTable.from_frame` selecting a
family by `phenomenon` and ignoring `evidence_kind` and `source`.

The two logbook lines here are the REAL ones, transcribed from
`$LABELMAKER_ROOT/text/logs_subset.jsonl`, and they reach the frame
through the real `text_weak.text_events` and the real lexicon - so this
file pins the actual reproduction and not a hand-written row that
resembles it. Everything is written into `tmp_path` in the layout
`text_weak` reads; nothing here opens `/scratch/gpfs/EKOLEMEN`.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from labelmaker.config import Paths
from labelmaker.events import lexicon as lx
from labelmaker.events import schema, windows
from labelmaker.events import text_weak as tw
from labelmaker.features import resolve_events

from .test_events_windows import SHOT, _block, _t_grid, _write_masks

#: The two real logbook entries, verbatim from the prepared subset. Shot
#: 185980's is a note about the DETECTOR, not about an ELM; 193348's names
#: a piggyback experiment, not a crash that was observed at `t = 0`.
ELM_TUNING_SHOT = 185980
ELM_TUNING_ENTRY = (
    "### [PHYSICS_OPERATOR] eldond 2021-04-20 10:37:00\n"
    "Updated ELM detector tuning.\n"
)
SAWTOOTH_PIGGYBACK_SHOT = 193348
SAWTOOTH_PIGGYBACK_ENTRY = (
    "### [SESSION_LEADER] heidbrin 2022-12-07 11:23:55\n"
    "35 keV for Dus low-voltage beams.\n"
    "Sawtooth piggyback: Density higher than desired but good shot.\n"
)

#: The critic's two windows, and the rate one spurious point event in a
#: 0.34 s window produces.
EARLY_WINDOW = (0.0, 0.34)
FORECAST_WINDOW = (0.9, 1.24)
ONE_EVENT_RATE_HZ = 1.0 / windows.WINDOW_S

RUN = "test-evidence"


@pytest.fixture
def paths(tmp_path):
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "bundles",
        logs_jsonl=tmp_path / "logs.jsonl",
    )


@pytest.fixture
def lex():
    return lx.load_lexicon()


def _real_text_events(paths, shot: int, entry: str, lex) -> list[schema.Event]:
    """The real `text_weak` rows for one real logbook entry.

    No bundle, so `shot_span_s` is `(0.0, 0.0)` and the row is a point at
    zero - which is exactly the shape the critic reported, and the reason
    an untimed mention lands at shot start.
    """
    paths.logs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    record = {"shot": int(shot), "run": "r", "log_text": entry}
    paths.logs_jsonl.write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    return tw.text_events(int(shot), lex, paths=paths)


def _forecast(shot: int, phenomenon: str, t: float) -> schema.Event:
    """A schema-valid `label_forecast` row: a prediction, not an observation."""
    return schema.Event(
        shot=int(shot), source="label_forecast", evidence_kind="forecast",
        phenomenon=phenomenon, t0_s=float(t), t1_s=float(t),
        confidence=0.9, horizon_s=0.05, t_cov0_s=0.0, t_cov1_s=2.0,
    )


def _detected_elm(shot: int, t: float) -> schema.Event:
    """What the ELM clock's detector actually writes."""
    return schema.Event(
        shot=int(shot), source="tokeye_transient", evidence_kind="detector",
        phenomenon="elm", t0_s=float(t), t1_s=float(t), confidence=0.8,
        diag="mhr", channel=0, pass_name="wide", t_cov0_s=0.0, t_cov1_s=2.0,
    )


def _elm_free(shot: int, t0: float, t1: float) -> schema.Event:
    """What `elm_clock` actually writes: the quiet, not the crashes."""
    return schema.Event(
        shot=int(shot), source="elm_clock", evidence_kind="heuristic",
        phenomenon="elm_free", t0_s=float(t0), t1_s=float(t1),
        diag="mhr", channel=0, pass_name="wide", t_cov0_s=0.0, t_cov1_s=2.0,
    )


def _detected_sawtooth(shot: int, t: float) -> schema.Event:
    return schema.Event(
        shot=int(shot), source="ece_sawtooth", evidence_kind="heuristic",
        phenomenon="sawtooth", t0_s=float(t), t1_s=float(t), confidence=0.7,
        diag="ece", t_cov0_s=0.0, t_cov1_s=2.0,
    )


def _frame(tmp_path, shot: int, events):
    """The rows through the real writer, so dtypes are the stored ones."""
    path = tmp_path / f"{shot}_events.parquet"
    return schema.write_events(path, int(shot), list(events), run_id=RUN)


def _f(vec, name: str) -> float:
    return float(np.asarray(vec)[windows.FEATURE_NAMES.index(name)])


def _features(frame, window):
    return windows.window_features(
        window, blocks={}, events=frame, cov={"mhr": (0.0, 2.0)}
    )


# ------------------------------------------------------------- the policy

def test_the_evidence_policy_is_two_named_constants():
    # The point of naming them: `resolve_events` and any future consumer
    # import the SAME policy rather than restating a `phenomenon` test.
    assert windows.DIAGNOSTIC_EVIDENCE == ("detector", "heuristic")
    for kind in ("forecast", "text", "human", "database", "model"):
        assert kind not in windows.DIAGNOSTIC_EVIDENCE
        assert kind in schema.EVIDENCE_KINDS
    assert windows.FAMILY_SOURCES == {
        "coherent_mode": ("tokeye_track",),
        "pickup": ("tokeye_track",),
        "elm": ("tokeye_transient",),
        "elm_free": ("elm_clock",),
        "sawtooth": ("ece_sawtooth",),
        "lh_transition": ("dalpha_lh",),
    }


def test_a_phenomenon_no_feature_names_reaches_no_feature(tmp_path):
    # `hl_transition`, `nbi_on` and `qh` are real rows of a real events
    # file and no feature is computed from any of them; the policy
    # excludes them by the same test that excludes a text row.
    frame = _frame(tmp_path, SHOT, [
        schema.Event(shot=SHOT, source="dalpha_lh", evidence_kind="heuristic",
                     phenomenon="hl_transition", t0_s=0.1, t1_s=0.1,
                     t_cov0_s=0.0, t_cov1_s=2.0),
        schema.Event(shot=SHOT, source="actuator", evidence_kind="heuristic",
                     phenomenon="nbi_on", t0_s=0.0, t1_s=1.0,
                     t_cov0_s=0.0, t_cov1_s=2.0),
    ])
    assert not windows.diagnostic_mask(frame).any()
    assert windows.diagnostic_rows(frame).empty


# ------------------------------------- the critic's shot 185980 reproduction

def test_the_real_elm_tuning_note_is_not_an_elm(paths, lex, tmp_path):
    text = _real_text_events(paths, ELM_TUNING_SHOT, ELM_TUNING_ENTRY, lex)
    # The note IS a positive ELM mention and the text row is right to
    # exist: what is wrong is counting it as a crash.
    assert [(e.phenomenon, e.evidence_kind, e.source) for e in text] == [
        ("elm", "text", "text")
    ]
    assert (text[0].t0_s, text[0].t1_s) == (0.0, 0.0)
    assert "elm detector tuning" in text[0].attrs["snippet"]

    frame = _frame(tmp_path, ELM_TUNING_SHOT, [
        *text,                                     # (a) the real note
        _forecast(ELM_TUNING_SHOT, "elm", 1.0),    # (b) a forecast at 1 s
        _elm_free(ELM_TUNING_SHOT, 0.0, 0.4),      # (c) a genuine elm_clock
        _detected_elm(ELM_TUNING_SHOT, 0.60),      # (d) a genuine detector
    ])
    assert len(frame) == 4                         # all four ARE in the file

    early = _features(frame, EARLY_WINDOW)
    assert _f(early, "elm_rate_hz") == 0.0
    # No ELM before the window centre at all, so the age is the cap and
    # not the 0.17 s the text row produced.
    assert _f(early, "time_since_last_elm_s") == windows.ELM_AGE_CAP_S

    ahead = _features(frame, FORECAST_WINDOW)
    assert _f(ahead, "elm_rate_hz") == 0.0

    # Not vacuous: the genuine detector row at 0.60 s still counts, and the
    # genuine `elm_clock` interval still fills `elm_free_frac`.
    real = _features(frame, (0.51, 0.85))
    assert _f(real, "elm_rate_hz") == pytest.approx(1.0 / 0.34, rel=1e-5)
    assert _f(_features(frame, (0.0, 0.4)), "elm_free_frac") == 1.0


def test_selecting_by_phenomenon_alone_is_the_defect_this_pins(
    paths, lex, tmp_path,
):
    # The critic's number, reproduced from the same frame with the policy
    # removed: without the evidence test the text row and the forecast are
    # each worth 2.941 Hz. If this ever stops reproducing, the test above
    # has stopped testing anything.
    text = _real_text_events(paths, ELM_TUNING_SHOT, ELM_TUNING_ENTRY, lex)
    frame = _frame(tmp_path, ELM_TUNING_SHOT, [
        *text, _forecast(ELM_TUNING_SHOT, "elm", 1.0),
    ])
    for window in (EARLY_WINDOW, FORECAST_WINDOW):
        start, end = window
        t = frame.loc[frame["phenomenon"] == "elm", "t0_s"].to_numpy()
        by_phenomenon = int(((t >= start) & (t < end)).sum()) / (end - start)
        assert by_phenomenon == pytest.approx(ONE_EVENT_RATE_HZ, rel=1e-5)
        assert _f(_features(frame, window), "elm_rate_hz") == 0.0


# ------------------------------------- the critic's shot 193348 reproduction

def test_the_real_sawtooth_piggyback_note_is_not_a_crash(paths, lex, tmp_path):
    text = _real_text_events(
        paths, SAWTOOTH_PIGGYBACK_SHOT, SAWTOOTH_PIGGYBACK_ENTRY, lex
    )
    assert [e.phenomenon for e in text] == ["sawtooth"]
    assert text[0].evidence_kind == "text"
    assert "sawtooth piggyback" in text[0].attrs["snippet"].lower()

    frame = _frame(tmp_path, SAWTOOTH_PIGGYBACK_SHOT, [
        *text,
        _forecast(SAWTOOTH_PIGGYBACK_SHOT, "sawtooth", 1.0),
        _detected_sawtooth(SAWTOOTH_PIGGYBACK_SHOT, 0.50),
        _detected_sawtooth(SAWTOOTH_PIGGYBACK_SHOT, 0.57),
    ])
    assert _f(_features(frame, EARLY_WINDOW), "n_sawtooth") == 0.0
    assert _f(_features(frame, FORECAST_WINDOW), "n_sawtooth") == 0.0
    # The two genuine crashes 70 ms apart are still a train.
    got = _features(frame, (0.45, 0.79))
    assert _f(got, "n_sawtooth") == 2.0
    assert _f(got, "sawtooth_period_ms") == pytest.approx(70.0, abs=1e-6)


# --------------------------------------- the other kinds, and the tracks

@pytest.mark.parametrize("kind,source", [
    ("human", "database"), ("database", "database"), ("model", "model"),
])
def test_a_human_database_or_model_row_is_not_a_detection(tmp_path, kind, source):
    frame = _frame(tmp_path, SHOT, [
        schema.Event(shot=SHOT, source=source, evidence_kind=kind,
                     phenomenon="elm", t0_s=0.1, t1_s=0.1,
                     t_cov0_s=0.0, t_cov1_s=2.0),
    ])
    assert _f(_features(frame, EARLY_WINDOW), "elm_rate_hz") == 0.0


def test_a_model_row_wearing_the_trackers_name_is_not_a_track(tmp_path):
    # Source alone is not enough either: `evidence_kind` is what says the
    # row is a measurement, and a `model` row may name any source it likes.
    rows = []
    for kind in ("detector", "model"):
        rows.append(schema.Event(
            shot=SHOT, source="tokeye_track", evidence_kind=kind,
            phenomenon="pickup", t0_s=0.1 if kind == "detector" else 1.0,
            t1_s=0.2 if kind == "detector" else 1.1, diag="mhr", channel=0,
            pass_name="zoom", attrs={"pickup": True}, t_cov0_s=0.0,
            t_cov1_s=2.0,
        ))
    frame = _frame(tmp_path, SHOT, rows)
    assert _f(_features(frame, (0.0, 0.34)), "pickup_flag") == 1.0
    assert _f(_features(frame, (0.9, 1.24)), "pickup_flag") == 0.0


# ------------------------------------------------- a shot with only text

def test_a_text_only_shot_resolves_to_zero_events_and_honest_coverage(
    tmp_path, paths, lex,
):
    """The whole path: masks + a text-only events file -> the 46 numbers."""
    paths.mkdirs()
    t_s = _t_grid(0.0, 1.0)
    _write_masks(paths, [_block("mhr", 0, "wide", t_s)])
    text = (
        _real_text_events(paths, ELM_TUNING_SHOT, ELM_TUNING_ENTRY, lex)
        + _real_text_events(
            paths, SAWTOOTH_PIGGYBACK_SHOT, SAWTOOTH_PIGGYBACK_ENTRY, lex
        )
    )
    # Both notes, restated for this shot: an events file that holds text
    # and nothing else, which is what 380 of the 500 selected shots have.
    schema.write_events(
        paths.events_file(SHOT), SHOT,
        [
            schema.Event(
                shot=SHOT, source="text", evidence_kind="text",
                phenomenon=e.phenomenon, t0_s=0.0, t1_s=0.0,
                confidence=e.confidence, attrs=dict(e.attrs),
                t_cov0_s=0.0, t_cov1_s=0.0,
            )
            for e in text
        ],
        run_id=RUN,
    )

    arrays, missing = resolve_events.resolve(
        SHOT, [resolve_events.NAME], paths=paths
    )
    assert missing == {}
    got = arrays[resolve_events.NAME]
    rows = {name: got.y[i] for i, name in enumerate(windows.FEATURE_NAMES)}

    valid = np.isfinite(rows["cov_frac_mhr"]) & (rows["cov_frac_mhr"] > 0.0)
    assert valid.any()
    for name in ("elm_rate_hz", "elm_free_frac", "n_sawtooth",
                 "sawtooth_period_ms", "lh_recent", "pickup_flag"):
        assert np.all(rows[name][valid] == 0.0), name
    # "No ELM" is the CAP, not zero: an uncapped age would scale with the
    # shot length and a zero would read as "an ELM just now".
    assert np.all(
        rows["time_since_last_elm_s"][valid] == windows.ELM_AGE_CAP_S
    )
    # Coverage is still honest - the mask says mhr was looked at for the
    # whole second, and nothing about the text changes that.
    assert rows["cov_frac_mhr"][valid].max() == pytest.approx(1.0)
    assert np.all(rows["cov_frac_co2"][valid] == 0.0)
    # And the policy is on the array, so a stored feature can be re-checked.
    assert got.attrs["evidence_kinds"] == "detector|heuristic"
    assert "elm<-tokeye_transient" in got.attrs["evidence_sources"]
