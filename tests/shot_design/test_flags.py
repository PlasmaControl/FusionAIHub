"""Operating-limit rules: the triples, the skips, and the numbers actually shipped.

Two halves. The first runs on rule sets written inline, so the mechanism (globs, aliases,
derived quantities, envelopes, ordering) is tested independently of whatever limits happen to be
in configs/shot_design/flags.yaml. The second runs the shipped config over the real database and
asserts the calibration claim the config's comments make -- a limit that fires on most good shots
is a wrong limit, and that has to fail a test rather than be noticed on a demo screen.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from shot_design import config
from shot_design.flags import rules

DB_DIR = Path("/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/db")


def write_rules(tmp_path: Path, doc: dict, name: str = "flags.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def cfg_from(tmp_path: Path, doc: dict) -> dict:
    return rules.load_rules(write_rules(tmp_path, doc))


def ids(flags) -> list[str]:
    return [f.rule_id for f in flags]


# --------------------------------------------------------------------------------- mechanism


def test_a_rule_is_a_triple_and_a_violation_names_both_sides(tmp_path):
    cfg = cfg_from(
        tmp_path,
        {
            "rules": [
                {
                    "id": "q95_floor",
                    "field": "q95_min",
                    "op": "ge",
                    "limit": 2.0,
                    "severity": "error",
                    "message": "too low",
                }
            ]
        },
    )
    (flag,) = rules.evaluate_flags({"q95_min": 1.8}, None, cfg)
    assert flag.rule_id == "q95_floor"
    assert flag.severity == "error"
    assert flag.value == 1.8
    assert flag.limit == 2.0
    assert "1.8 < 2" in flag.message and "too low" in flag.message


def test_a_rule_that_passes_says_nothing(tmp_path):
    cfg = cfg_from(
        tmp_path, {"rules": [{"id": "q95_floor", "field": "q95_min", "op": "ge", "limit": 2.0}]}
    )
    assert rules.evaluate_flags({"q95_min": 3.5}, None, cfg) == []


@pytest.mark.parametrize(
    ("op", "value", "trips"),
    [
        ("ge", 2.0, False),
        ("ge", 1.99, True),
        ("gt", 2.0, True),
        ("le", 2.0, False),
        ("le", 2.01, True),
        ("lt", 2.0, True),
    ],
)
def test_op_limit_names_the_allowed_side(tmp_path, op, value, trips):
    cfg = cfg_from(tmp_path, {"rules": [{"id": "r", "field": "x", "op": op, "limit": 2.0}]})
    assert bool(rules.evaluate_flags({"x": value}, None, cfg)) is trips


def test_a_missing_field_is_an_info_flag_that_names_the_field(tmp_path):
    cfg = cfg_from(
        tmp_path, {"rules": [{"id": "q95_floor", "field": "q95_min", "op": "ge", "limit": 2.0}]}
    )
    (flag,) = rules.evaluate_flags({"ip_mean": 1.0e6}, None, cfg)
    assert flag.severity == "info"
    assert flag.message.startswith("rule q95_floor skipped: missing q95_min")
    assert flag.value is None


def test_nan_is_missing_not_a_value(tmp_path):
    """A Parquet round trip turns an unrecorded scalar into NaN, and NaN violates no comparison --
    so without this it would read as a clean pass instead of a skip."""
    cfg = cfg_from(
        tmp_path, {"rules": [{"id": "q95_floor", "field": "q95_min", "op": "ge", "limit": 2.0}]}
    )
    (flag,) = rules.evaluate_flags({"q95_min": float("nan")}, None, cfg)
    assert flag.severity == "info" and "missing q95_min" in flag.message


def test_provisional_is_visible_in_the_message(tmp_path):
    cfg = cfg_from(
        tmp_path,
        {
            "rules": [
                {"id": "a", "field": "x", "op": "le", "limit": 1.0, "provisional": True},
                {"id": "b", "field": "x", "op": "le", "limit": 1.0, "provisional": False},
            ]
        },
    )
    got = {f.rule_id: f.message for f in rules.evaluate_flags({"x": 2.0}, None, cfg)}
    assert got["a"].endswith("[provisional limit]")
    assert "provisional" not in got["b"]


def test_a_glob_field_reports_every_member_that_is_over(tmp_path):
    cfg = cfg_from(
        tmp_path,
        {"rules": [{"id": "icoil", "field": "irmp_I*_peak", "op": "le", "limit": 7.0e3}]},
    )
    flags = rules.evaluate_flags(
        {"irmp_IU30_peak": 7100.0, "irmp_IU90_peak": 300.0, "irmp_IL90_peak": 7050.0}, None, cfg
    )
    assert ids(flags) == ["icoil", "icoil"]
    assert sorted(f.value for f in flags) == [7050.0, 7100.0]
    assert "irmp_IL90_peak" in flags[0].message


def test_a_glob_that_matches_nothing_skips_once(tmp_path):
    cfg = cfg_from(
        tmp_path, {"rules": [{"id": "icoil", "field": "irmp_I*_peak", "op": "le", "limit": 7.0e3}]}
    )
    (flag,) = rules.evaluate_flags({"ip_mean": 1.0e6}, None, cfg)
    assert flag.severity == "info" and "missing irmp_I*_peak" in flag.message


def test_exclude_keeps_a_member_cap_off_the_system_total(tmp_path):
    cfg = cfg_from(
        tmp_path,
        {
            "rules": [
                {
                    "id": "cap",
                    "field": "pnbi_*_peak",
                    "exclude": ["pnbi_total_*"],
                    "op": "le",
                    "limit": 3.0e6,
                }
            ]
        },
    )
    flags = rules.evaluate_flags(
        {"pnbi_total_peak": 1.1e7, "pnbi_15L_peak": 3.5e6, "pnbi_30L_peak": 2.0e6}, None, cfg
    )
    assert [f.value for f in flags] == [3.5e6]


def test_flags_come_back_errors_first(tmp_path):
    cfg = cfg_from(
        tmp_path,
        {
            "rules": [
                {"id": "info_one", "field": "absent", "op": "le", "limit": 1.0},
                {"id": "warn_one", "field": "x", "op": "le", "limit": 1.0, "severity": "warn"},
                {"id": "error_one", "field": "x", "op": "le", "limit": 1.0, "severity": "error"},
            ]
        },
    )
    assert ids(rules.evaluate_flags({"x": 5.0}, None, cfg)) == ["error_one", "warn_one", "info_one"]


# ----------------------------------------------------------------------------------- aliases


def test_a_bare_name_may_stand_for_several_columns(tmp_path):
    """Someone who types q95 = 1.8 means it of the whole flat top, so it is checked against both
    the mean rule and the minimum rule."""
    cfg = cfg_from(
        tmp_path,
        {
            "aliases": {"q95": ["q95_mean", "q95_min"]},
            "rules": [
                {"id": "floor_min", "field": "q95_min", "op": "ge", "limit": 2.0},
                {"id": "floor_mean", "field": "q95_mean", "op": "ge", "limit": 2.0},
            ],
        },
    )
    assert sorted(ids(rules.evaluate_flags({"q95": 1.8}, None, cfg))) == ["floor_mean", "floor_min"]


def test_a_scalar_alias_is_one_column_not_a_list_of_characters(tmp_path):
    """`aliases: {q95: q95_min}` is what a YAML author writes for a single column. `list()` on
    that string gave ['q', '9', '5', '_', 'm', 'i', 'n'], and the rule then only ever reported
    itself skipped."""
    cfg = cfg_from(
        tmp_path,
        {
            "aliases": {"q95": "q95_min"},
            "rules": [{"id": "floor", "field": "q95_min", "op": "ge", "limit": 2.0}],
        },
    )
    assert cfg["aliases"] == {"q95": ["q95_min"]}
    (flag,) = rules.evaluate_flags({"q95": 1.8}, None, cfg)
    assert flag.severity == "warn" and flag.value == 1.8


def test_an_alias_that_is_neither_a_name_nor_a_list_is_refused(tmp_path):
    with pytest.raises(ValueError, match="alias 'q95'"):
        cfg_from(tmp_path, {"aliases": {"q95": {"col": "q95_min"}}, "rules": []})


def test_actuator_keys_come_from_the_registry_not_the_flags_config(tmp_path):
    """`nbi.total` and `ech.LUKE` are the shapes QueryState.actuators uses. They are expanded from
    configs/shot_design/actuators.yaml, so adding a gyrotron never needs an edit here."""
    cfg = cfg_from(
        tmp_path,
        {
            "rules": [
                {"id": "nbi", "field": "pnbi_total_peak", "op": "le", "limit": 2.0e7},
                {"id": "luke", "field": "pech_LUKE_peak", "op": "le", "limit": 1.0e6},
            ]
        },
    )
    flags = rules.evaluate_flags({"nbi.total": 2.5e7, "ech.LUKE": 1.5e6}, None, cfg)
    assert sorted(ids(flags)) == ["luke", "nbi"]
    assert cfg["actuator_keys"]["nbi.15L"] == ["pnbi_15L_peak"]
    # the one resolver the retrieval channels read as well: one key, one column, the peak stat
    cols = rules.actuator_columns()
    assert cols["nbi.15L"] == "pnbi_15L_peak" and cols["nbi.total"] == "pnbi_total_peak"
    assert cols["ech.LUKE"] == "pech_LUKE_peak"
    assert all(v == [cols[k]] for k, v in cfg["actuator_keys"].items())


def test_an_unknown_key_is_carried_through_not_dropped(tmp_path):
    cfg = cfg_from(
        tmp_path, {"rules": [{"id": "r", "field": "weird_col", "op": "le", "limit": 1.0}]}
    )
    assert ids(rules.evaluate_flags({"weird_col": 5.0}, None, cfg)) == ["r"]


# ------------------------------------------------------------------------- derived quantities


def test_betan_over_li_is_derived_from_the_two_columns(tmp_path):
    cfg = cfg_from(
        tmp_path, {"rules": [{"id": "b", "field": "betan_over_li", "op": "le", "limit": 4.0}]}
    )
    (flag,) = rules.evaluate_flags({"betan_mean": 4.5, "li_mean": 0.9}, None, cfg)
    assert flag.value == pytest.approx(5.0)
    assert rules.evaluate_flags({"betan_mean": 2.0, "li_mean": 1.0}, None, cfg) == []


def test_a_derived_quantity_that_is_missing_an_input_says_which(tmp_path):
    cfg = cfg_from(
        tmp_path, {"rules": [{"id": "b", "field": "betan_over_li", "op": "le", "limit": 4.0}]}
    )
    (flag,) = rules.evaluate_flags({"betan_mean": 2.0}, None, cfg)
    assert flag.severity == "info"
    assert "needs li_mean" in flag.message


def test_greenwald_fraction_from_the_v2_chord():
    """Shot 161172's flat top, values read off the built database: ne_line 9.82e13 m/cm3,
    Ip 985 kA, a 0.587 m, kappa 1.811, R0 1.778 m -> the chord at R=1.94 m crosses 1.97 m of
    plasma, n_e-bar = 4.98e19 m^-3, n_G = 0.910e20 m^-3, fraction 0.528 (computed 2026-09-05)."""
    row = {
        "ne_line_mean": 9.82e13,
        "ip_mean": 9.85e5,
        "aminor_mean": 0.587,
        "kappa_mean": 1.811,
        "r0_mean": 1.778,
    }
    assert rules._greenwald_frac(row) == pytest.approx(0.528, abs=0.002)
    assert (
        rules._greenwald_frac({**row, "ne_line_mean": -2.0e15}) is None
    )  # baseline drift, not physics
    assert rules._greenwald_frac({**row, "r0_mean": 1.2}) is None  # chord outside the ellipse
    assert rules._greenwald_frac({**row, "kappa_mean": None}) is None
    assert "greenwald_frac" in rules.DERIVED and rules.DERIVED["greenwald_frac"].blocked is None


# -------------------------------------------------------------------------------- when / load


def test_when_gates_on_a_category_the_caller_supplied(tmp_path):
    cfg = cfg_from(
        tmp_path,
        {"rules": [{"id": "r", "field": "x", "op": "le", "limit": 1.0, "when": {"regime": "H"}}]},
    )
    assert ids(rules.evaluate_flags({"x": 5.0}, {"regime": "H"}, cfg)) == ["r"]
    assert rules.evaluate_flags({"x": 5.0}, {"regime": "L"}, cfg) == []
    # Not knowing the regime is not a reason to stop checking a limit.
    assert ids(rules.evaluate_flags({"x": 5.0}, None, cfg)) == ["r"]


def test_later_config_files_override_a_rule_by_id(tmp_path):
    base = write_rules(tmp_path, {"rules": [{"id": "r", "field": "x", "op": "le", "limit": 1.0}]})
    over = write_rules(
        tmp_path, {"rules": [{"id": "r", "field": "x", "op": "le", "limit": 9.0}]}, "over.yaml"
    )
    cfg = rules.load_rules([base, over])
    assert [r["limit"] for r in cfg["rules"]] == [9.0]
    assert rules.evaluate_flags({"x": 5.0}, None, cfg) == []


def test_a_malformed_rule_is_refused_at_load_time_by_name(tmp_path):
    """A mistyped severity used to surface as a pydantic ValidationError out of evaluate_flags on
    the first shot checked, and a mistyped op as a bare KeyError -- neither naming the rule."""
    bad_severity = {"id": "bad", "field": "x", "op": "le", "limit": 1.0, "severity": "warning"}
    with pytest.raises(ValueError, match=r"rule 'bad': severity 'warning'"):
        cfg_from(tmp_path, {"rules": [bad_severity]})
    with pytest.raises(ValueError, match=r"rule 'bad': op 'gte'"):
        cfg_from(tmp_path, {"rules": [{"id": "bad", "field": "x", "op": "gte", "limit": 1.0}]})
    with pytest.raises(ValueError, match=r"rule 'bad': limit 'two'"):
        cfg_from(tmp_path, {"rules": [{"id": "bad", "field": "x", "op": "le", "limit": "two"}]})
    with pytest.raises(ValueError, match="has no id"):
        cfg_from(tmp_path, {"rules": [{"field": "x", "op": "le", "limit": 1.0}]})
    with pytest.raises(ValueError, match=r"rule 'bad' has no field"):
        cfg_from(tmp_path, {"rules": [{"id": "bad", "op": "le", "limit": 1.0}]})


def test_one_malformed_rule_does_not_lose_the_other_rules_flags():
    """A hand-built cfg bypasses load_rules. Even then one bad rule is one info flag, never a
    raise that drops every flag for the shot -- which is what rank._enrich's catch-all did."""
    cfg = {
        "aliases": {},
        "actuator_keys": {},
        "rules": [
            {"id": "good", "field": "x", "op": "le", "limit": 1.0, "severity": "error"},
            {"id": "bad", "field": "x", "op": "le", "limit": 1.0, "severity": "warning"},
            {"id": "worse", "field": "x", "op": "gte", "limit": 1.0},
        ],
    }
    flags = rules.evaluate_flags({"x": 5.0}, None, cfg)
    assert ids(flags) == ["good", "bad", "worse"]
    assert flags[0].severity == "error" and flags[0].value == 5.0
    assert all(f.severity == "info" and "malformed rule" in f.message for f in flags[1:])
    assert "severity 'warning'" in flags[1].message and "op 'gte'" in flags[2].message


def test_member_caps_expand_from_the_actuator_registry(tmp_path, monkeypatch):
    """`max:` is `[?]`/null for every system today, so nothing expands -- an undeclared cap must
    not become an invented one. Declare one and the rule appears with no edit to flags.yaml."""
    assert not [r for r in rules.load_rules()["rules"] if r["id"].endswith("_member_cap")]

    real = config.load_yaml

    def fake(name):
        doc = real(name)
        if name == "actuators.yaml":
            doc = {**doc, "systems": {**doc["systems"], "nbi": {**doc["systems"]["nbi"]}}}
            doc["systems"]["nbi"]["max"] = 2.5e6
        return doc

    monkeypatch.setattr(config, "load_yaml", fake)
    cfg = cfg_from(tmp_path, {"rules": []})
    (cap,) = [r for r in cfg["rules"] if r["id"] == "nbi_member_cap"]
    assert cap["field"] == "pnbi_*_peak" and cap["exclude"] == ["pnbi_total_*"]
    assert cap["source"] == "configs/shot_design/actuators.yaml"
    assert "configs/shot_design/actuators.yaml" in cap["message"]
    flags = rules.evaluate_flags({"pnbi_15L_peak": 2.6e6, "pnbi_total_peak": 1.1e7}, None, cfg)
    assert [f.value for f in flags] == [2.6e6]
    assert flags[0].source == "configs/shot_design/actuators.yaml"
    assert "provisional" not in flags[0].message  # a declared cap is measured, not guessed


# --------------------------------------------------------------------------------- envelopes


def test_an_envelope_flag_is_about_the_database_not_the_machine(tmp_path):
    cfg = cfg_from(tmp_path, {"rules": []})
    env = {"ip_mean": {"lo": 7.8e5, "hi": 1.5e6, "n": 105}}
    (flag,) = rules.evaluate_flags({"ip_mean": 1.9e6}, None, cfg, envelopes=env)
    assert flag.severity == "info" and flag.source == "envelope"
    assert flag.rule_id == "envelope:ip_mean"
    assert (
        "105 shots" in flag.message and "not necessarily outside what is possible" in flag.message
    )
    assert rules.evaluate_flags({"ip_mean": 1.0e6}, None, cfg, envelopes=env) == []


def test_envelopes_may_be_keyed_by_a_category_value(tmp_path):
    cfg = cfg_from(tmp_path, {"rules": []})
    env = {"2014_2015": {"ip_mean": {"lo": 5.0e5, "hi": 1.2e6, "n": 71}}}
    assert rules.evaluate_flags({"ip_mean": 1.4e6}, {"campaign": "2014_2015"}, cfg, env)
    # No matching category: the flat mapping is used, and this one has no bands at the top level.
    assert rules.evaluate_flags({"ip_mean": 1.4e6}, {"campaign": "2022_2025"}, cfg, env) == []


def test_envelopes_from_frame_drops_thin_columns():
    frame = pd.DataFrame(
        {"a": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [1.0, 2, None, None, None] + [None] * 5}
    )
    env = rules.envelopes_from_frame(frame)
    assert env["a"]["n"] == 10 and env["a"]["lo"] < env["a"]["hi"]
    assert "b" not in env  # 2 recorded values is not an envelope


# ------------------------------------------------------------------- the shipped config, real data


def test_shipped_rules_are_well_formed():
    cfg = rules.load_rules()
    assert cfg["rules"], "configs/shot_design/flags.yaml has no rules"
    for r in cfg["rules"]:
        assert r["op"] in rules.OPS, r
        assert r["severity"] in ("info", "warn", "error"), r
        assert isinstance(r["limit"], int | float), r
        assert r.get("message"), r
        # Nathan has supplied no operating limits, so every shipped number is a chosen one and
        # must say so on screen. Delete this only when a measured limit replaces a guessed one.
        assert r["provisional"] is True, f"{r['id']} claims a limit that was not measured"


@pytest.mark.real_data
@pytest.mark.skipif(not DB_DIR.exists(), reason=f"{DB_DIR} not mounted")
def test_shipped_limits_do_not_fire_on_most_good_shots():
    """The calibration claim configs/shot_design/flags.yaml makes, as a test.

    A limit that fires on a large fraction of real, successful shots is evidence the number is
    wrong, not evidence the shots were. The two coil rules are the interesting case: real good
    shots DO reach 7 kA on an I-coil and 5.7 kA on a C-coil, which is why they ship as `warn`.
    """
    cfg = rules.load_rules()
    seg = pd.read_parquet(DB_DIR / "segments.parquet")
    ft = seg[seg["segment"] == "flat_top"]
    fired: dict[str, int] = {r["id"]: 0 for r in cfg["rules"]}
    for _, row in ft.iterrows():
        hit = {
            f.rule_id
            for f in rules.evaluate_flags(row.to_dict(), None, cfg)
            if f.severity != "info"
        }
        for rid in hit:
            fired[rid] += 1
    assert len(ft) >= 100, "the reference database shrank; re-measure the limits in flags.yaml"
    for rid, n in fired.items():
        assert n <= 0.2 * len(ft), f"{rid} fires on {n}/{len(ft)} real flat tops -- recalibrate it"
    # The coil rules are `info` (no documented limit exists), so they never count here; the
    # Greenwald rules do, measured 2026-09-05 on the 201-shot database.
    assert fired["icoil_current"] == 0 and fired["ccoil_current"] == 0
    assert fired["greenwald_warn"] == 11 and fired["greenwald_error"] == 5
    assert fired["q95_floor"] == 0


@pytest.mark.real_data
@pytest.mark.skipif(not DB_DIR.exists(), reason=f"{DB_DIR} not mounted")
def test_greenwald_computes_on_nearly_every_real_flat_top():
    """Measured 2026-09-05 on the 201-shot database: 197/201 flat tops have the five inputs, the
    median fraction is 0.56 and the 95th percentile 0.91 -- ordinary DIII-D operation."""
    seg = pd.read_parquet(DB_DIR / "segments.parquet")
    ft = seg[seg["segment"] == "flat_top"]
    vals = [rules._greenwald_frac(r) for r in ft.to_dict("records")]
    got = [v for v in vals if v is not None]
    assert len(got) >= 0.95 * len(ft)
    assert 0.3 <= float(np.median(got)) <= 0.8
