"""Operating-limit rules: "can DIII-D actually do this?"

Two entry points. `load_rules` reads configs/ideate/flags.yaml (plus the per-member caps that
configs/ideate/actuators.yaml declares) into a plain dict; `evaluate_flags` runs that dict against
a flat mapping of numbers and returns `schema.Flag`s. The same call answers both questions the
demo asks: hand it a database row's flat-top values and it says whether that shot was near a
limit, hand it a user's proposed `QueryState.actuators` and it says whether the proposal is
possible at all.

Three things this module refuses to do, all of them deliberate:

* **No expression evaluator.** A rule is `{id, field, op, limit}`. A physicist adding a limit
  edits YAML, never Python, and a malformed rule cannot do anything more exciting than fail to
  match. Derived quantities that a limit needs (`betan_over_li`) are ordinary named features
  computed here, not formulas in the config.
* **No silence.** A rule whose field is missing produces an `info` flag saying which field was
  missing and, when known, why it could not be computed. "No flags" then means "checked and
  clean", never "checked nothing" -- which is the difference between a useful screen and a
  dangerous one.
* **No unearned authority.** Every limit in configs/ideate/flags.yaml today was chosen from
  published DIII-D practice rather than measured or supplied by the machine's operators, and is
  marked `provisional: true`. Messages built from such a rule carry a trailing
  "[provisional limit]".


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import fnmatch
import math
import operator
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from .. import config
from ..schema import Flag

# "op limit" names the ALLOWED side, so `{op: ge, limit: 2.0}` reads "must be >= 2.0". The second
# element is how a *violation* of that rule renders, which is the comparison the message wants:
# a value that fails `>= 2.0` is reported as "1.82 < 2.0".
OPS: dict[str, tuple[Callable[[float, float], bool], str]] = {
    "ge": (operator.ge, "<"),
    "gt": (operator.gt, "<="),
    "le": (operator.le, ">"),
    "lt": (operator.lt, ">="),
}

SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}
_GLOB_CHARS = set("*?[")


class _Derived(NamedTuple):
    """A quantity computed from other values, with the reason it might not be computable.

    `blocked` is not "this shot is missing an input" -- that is what `needs` covers. It is
    "this quantity cannot be computed for ANY shot in this database, and here is why", which is
    the honest answer for the Greenwald fraction as long as the density unit is unconfirmed.
    """

    fn: Callable[[Mapping[str, Any]], float | None]
    needs: tuple[str, ...]
    blocked: str | None = None


def _finite(x: Any) -> float | None:
    """A usable number, or None. NaN counts as missing: a pandas row hands unrecorded scalars
    over as NaN rather than None, and a NaN silently satisfies no comparison and violates none,
    which would make a missing field look like a clean one."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _betan_over_li(v: Mapping[str, Any]) -> float | None:
    b, li = _finite(v.get("betan_mean")), _finite(v.get("li_mean"))
    return None if b is None or not li else b / li


# The BCI V2 chord is vertical at this major radius (DIII-D CO2 interferometer geometry; V1 is
# at 1.48 m, V3 at 2.10 m, R0 is the radial chord). Only V2 is a registry signal today.
BCI_V2_RADIUS_M = 1.94


def _greenwald_frac(v: Mapping[str, Any]) -> float | None:
    """n_e / n_G from the V2 chord.

    ne_line is the MDSplus node \\ELECTRONS::TOP.BCI.MAIN:DENV2, whose units attribute reads
    "m/cm3" (confirmed on shot 205911, 2026-09-05): a path length in METRES times a density in
    cm^-3, DIII-D's mixed convention. Dividing by the chord's path length through the plasma gives
    a line-averaged density in cm^-3; x1e6 for m^-3; /1e20 for Greenwald units. The path length
    is taken from EFIT's shape as the vertical extent of an ellipse of minor radius a, elongation
    kappa and geometric axis R0 at the chord's major radius -- an approximation that ignores
    triangularity and the exact boundary, so treat the result as good to ~10-20 %. n_G = Ip[MA] /
    (pi a^2). A negative or zero density (baseline drift on some shots) is reported as None rather
    than a negative fraction.
    """
    ne, ip, a, kappa, r0 = (
        _finite(v.get(k))
        for k in ("ne_line_mean", "ip_mean", "aminor_mean", "kappa_mean", "r0_mean")
    )
    if None in (ne, ip, a, kappa, r0) or ne <= 0 or ip <= 0 or a <= 0 or kappa <= 0:
        return None
    half_chord = a * a - (BCI_V2_RADIUS_M - r0) ** 2
    if half_chord <= 0:
        return None  # the chord misses the plasma the ellipse describes
    path_m = 2.0 * kappa * math.sqrt(half_chord)
    nbar_1e20 = ne / path_m * 1e6 / 1e20
    n_g = (ip / 1e6) / (math.pi * a * a)
    return nbar_1e20 / n_g


DERIVED: dict[str, _Derived] = {
    "betan_over_li": _Derived(_betan_over_li, ("betan_mean", "li_mean")),
    # Greenwald needs a line-AVERAGED electron density in 10^20 m^-3; see _greenwald_frac.
    "greenwald_frac": _Derived(
        _greenwald_frac, ("ne_line_mean", "ip_mean", "aminor_mean", "kappa_mean", "r0_mean")
    ),
}


# --------------------------------------------------------------------------------- loading


def _paths(paths: Any) -> list[Path]:
    if paths is None:
        return [config.CONFIG_DIR / "flags.yaml"]
    if isinstance(paths, str | Path):
        return [Path(paths)]
    return [Path(p) for p in paths]


def _limit_stat(sysdef: Mapping[str, Any]) -> str:
    """The stat a single proposed value for an actuator system refers to.

    `peak` when the system records one -- the 95th percentile of the nonzero in-window samples
    (CLAUDE.md), which is the level a physicist asks the machine for and what an installed-power
    cap is written against -- otherwise the system's first declared stat.
    """
    stats = list(sysdef.get("stats", ["mean", "peak"]))
    return "peak" if "peak" in stats else stats[0]


def actuator_columns() -> dict[str, str]:
    """`nbi.total` / `ech.LUKE` (what a person types) -> the ONE database column it means.

    Built from configs/ideate/actuators.yaml so it cannot drift from the registry. `.total` is
    included explicitly because a user proposing "20 MW of NBI" is proposing a system total, not
    a member.

    This is the only resolver: `retrieval.channels._actuator_column` reads it too, so the scalar
    channel and the operating-limit rules agree on which column `--actuator nbi.total=5e6` names.
    They did not -- channels preferred `pnbi_total_mean` and this module `pnbi_total_peak`, so
    one flag was two quantities in one query.
    """
    out: dict[str, str] = {}
    for name, sysdef in config.load_yaml("actuators.yaml")["systems"].items():
        stat, prefix = _limit_stat(sysdef), sysdef["prefix"]
        out[f"{name}.total"] = f"{prefix}_total_{stat}"
        for m in sysdef["members"]:
            out[f"{name}.{m['id']}"] = f"{prefix}_{m['id']}_{stat}"
    return out


def _member_cap_rules() -> list[dict[str, Any]]:
    """One rule per actuator system that declares a `max:` in configs/ideate/actuators.yaml.

    Expanded rather than written out in flags.yaml so that adding a gyrotron stays a one-line
    registry edit, the way the rest of this codebase treats actuator members. Every `max:` is
    `null` today (they are `[?]` in the registry), so this returns nothing -- which is the
    correct behaviour, not a stub: an undeclared cap must not become an invented one.
    """
    out: list[dict[str, Any]] = []
    for name, sysdef in config.load_yaml("actuators.yaml")["systems"].items():
        cap = sysdef.get("max")
        if cap is None:
            continue
        stat = _limit_stat(sysdef)
        prefix = sysdef["prefix"]
        out.append(
            {
                "id": f"{name}_member_cap",
                "field": f"{prefix}_*_{stat}",
                # The system total is written into the same namespace as its members
                # (pnbi_total_peak next to pnbi_15L_peak) and is not a member, so a per-member
                # cap must not be applied to it.
                "exclude": [f"{prefix}_total_*"],
                "op": "le",
                "limit": float(cap),
                "severity": "error",
                "provisional": False,
                "source": "configs/ideate/actuators.yaml",
                "message": (
                    f"above the per-member {name} cap declared in configs/ideate/actuators.yaml"
                ),
            }
        )
    return out


def _actuator_key_map() -> dict[str, list[str]]:
    """`actuator_columns` in the alias table's shape (a key may stand for several columns)."""
    return {key: [col] for key, col in actuator_columns().items()}


def _check_rule(r: Mapping[str, Any], where: str) -> None:
    """Refuse a rule that cannot run, at load time and by name.

    Without this a mistyped `severity: warning` surfaced as a pydantic ValidationError out of
    `evaluate_flags` on the first shot checked, and a mistyped `op: gte` as a bare KeyError --
    neither naming the rule, both long after the config was read.
    """
    rid = r.get("id")
    if not rid:
        raise ValueError(f"{where}: a rule has no id: {dict(r)}")
    head = f"{where}: rule {str(rid)!r}"
    if not r.get("field"):
        raise ValueError(f"{head} has no field")
    if str(r.get("op")) not in OPS:
        raise ValueError(f"{head}: op {r.get('op')!r} is not one of {', '.join(OPS)}")
    if str(r.get("severity")) not in SEVERITY_ORDER:
        raise ValueError(
            f"{head}: severity {r.get('severity')!r} is not one of {', '.join(SEVERITY_ORDER)}"
        )
    if r.get("limit") is not None and _finite(r["limit"]) is None:
        raise ValueError(f"{head}: limit {r['limit']!r} is not a number")
    if r.get("exclude") is not None and not isinstance(r["exclude"], list):
        raise ValueError(f"{head}: exclude must be a list of globs, got {r['exclude']!r}")
    if r.get("when") is not None and not isinstance(r["when"], Mapping):
        raise ValueError(f"{head}: when must be a mapping, got {r['when']!r}")


def _alias_targets(name: str, v: Any, where: str) -> list[str]:
    """`q95: [q95_mean, q95_min]` or the scalar `q95: q95_min` -- a YAML author writes either.
    `list()` on the scalar form gave ['q', '9', '5', '_', 'm', 'i', 'n'], and the alias silently
    resolved to seven nonexistent columns."""
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(t, str) for t in v):
        return list(v)
    raise ValueError(f"{where}: alias {name!r} must be a column name or a list of them, got {v!r}")


def load_rules(paths: Any = None) -> dict[str, Any]:
    """The rule set, ready for `evaluate_flags`.

    `paths` is None (configs/ideate/flags.yaml), one path, or several. Several are merged in order:
    later files replace an earlier rule with the same `id` and extend the alias table, which is
    how a campaign- or user-specific overlay is meant to be applied without editing the shipped
    config. A rule that cannot run -- no id, unknown op or severity, non-numeric limit -- is
    refused here with a ValueError naming the file, the rule and the field.

    The returned dict is plain data -- `rules`, `aliases`, `systems`, `derived` -- so a caller
    can print it, diff it, or hand it to `evaluate_flags` unchanged.
    """
    merged: dict[str, Any] = {"version": None, "aliases": {}, "rules": []}
    by_id: dict[str, dict[str, Any]] = {}
    for p in _paths(paths):
        where = str(Path(p).name)
        doc = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        merged["version"] = doc.get("version", merged["version"])
        for name, v in (doc.get("aliases") or {}).items():
            merged["aliases"][str(name)] = _alias_targets(str(name), v, where)
        for rule in doc.get("rules") or []:
            r = dict(rule)
            r.setdefault("severity", "warn")
            r.setdefault("provisional", False)
            r.setdefault("source", where)
            _check_rule(r, where)
            by_id[str(r["id"])] = r
    for r in _member_cap_rules():
        _check_rule(r, "configs/ideate/actuators.yaml")
        by_id.setdefault(r["id"], r)
    merged["rules"] = list(by_id.values())
    merged["actuator_keys"] = _actuator_key_map()
    merged["derived"] = sorted(DERIVED)
    return merged


# ------------------------------------------------------------------------------ evaluation


def _canonical(
    values: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, str]]:
    """Whatever the caller passed -> database column names, plus why anything was dropped.

    Three key shapes arrive here and all three are accepted, because the same function has to
    read a Parquet row and a person's typed proposal: a column name already
    (`pnbi_total_peak`), an actuator key (`nbi.total`), and a bare quantity (`q95`). A bare name
    may stand for more than one column -- someone who types "q95 3.5" means it of the whole flat
    top, so the value is checked against both the mean rule and the minimum rule.
    """
    aliases: dict[str, list[str]] = dict(cfg.get("aliases") or {})
    actuator_keys: dict[str, list[str]] = dict(cfg.get("actuator_keys") or {})
    out: dict[str, float] = {}
    for key, raw in values.items():
        v = _finite(raw)
        if v is None:
            continue
        targets = actuator_keys.get(key) or aliases.get(key)
        if targets is None:
            # Already a column name (or something we have no mapping for -- carried through
            # unchanged so a rule written against a column this table has never heard of still
            # works, rather than being silently dropped here).
            targets = [key]
        for t in targets:
            out[t] = v
    reasons: dict[str, str] = {}
    for name, d in DERIVED.items():
        if d.blocked:
            reasons[name] = d.blocked
            continue
        got = d.fn(out)
        if got is None:
            missing = [n for n in d.needs if n not in out]
            reasons[name] = "needs " + ", ".join(missing or d.needs)
        else:
            out[name] = got
    return out, reasons


def _matches(rule: Mapping[str, Any], vals: Mapping[str, float]) -> list[tuple[str, float]]:
    field = str(rule["field"])
    if not _GLOB_CHARS & set(field):
        return [(field, vals[field])] if field in vals else []
    excl = [str(e) for e in rule.get("exclude") or []]
    hits = [
        (k, v)
        for k, v in vals.items()
        if fnmatch.fnmatchcase(k, field) and not any(fnmatch.fnmatchcase(k, e) for e in excl)
    ]
    return sorted(hits)


def _when_ok(rule: Mapping[str, Any], category: Mapping[str, str] | None) -> bool:
    """A `when:` guard applies only to category keys the caller actually supplied.

    An unknown category never suppresses a rule: not knowing whether a proposal is an H-mode is
    not a reason to stop checking whether it is inside the machine's envelope.
    """
    for key, want in (rule.get("when") or {}).items():
        have = (category or {}).get(key)
        if have is None:
            continue
        allowed = want if isinstance(want, list) else [want]
        if have not in [str(a) for a in allowed]:
            return False
    return True


def _num(v: float) -> str:
    return f"{v:.4g}"


def _is_band(x: Any) -> bool:
    return isinstance(x, Mapping) and ("lo" in x or "hi" in x)


def _select_envelope(
    envelopes: Mapping[str, Any] | None, category: Mapping[str, str] | None
) -> dict[str, Mapping[str, Any]]:
    """Flat `{field: band}`, or `{category_value: {field: band}}` keyed by any category value
    (a campaign id, a regime). The first category value that keys into the mapping wins."""
    if not envelopes:
        return {}
    for cv in (category or {}).values():
        sub = envelopes.get(cv)
        if isinstance(sub, Mapping):
            return {k: b for k, b in sub.items() if _is_band(b)}
    return {k: b for k, b in envelopes.items() if _is_band(b)}


def _envelope_flags(
    vals: Mapping[str, float],
    category: Mapping[str, str] | None,
    envelopes: Mapping[str, Any] | None,
) -> list[Flag]:
    """ "Nothing in the database has run there" -- a different claim from "the machine cannot".

    Always `info`: an observed range is a statement about this 105-shot slice, not about DIII-D.
    """
    out: list[Flag] = []
    for field, band in sorted(_select_envelope(envelopes, category).items()):
        v = vals.get(field)
        if v is None:
            continue
        lo, hi = _finite(band.get("lo")), _finite(band.get("hi"))
        side = (
            "below" if lo is not None and v < lo else "above" if hi is not None and v > hi else ""
        )
        if not side:
            continue
        n = band.get("n")
        span = f"{_num(lo) if lo is not None else '?'}-{_num(hi) if hi is not None else '?'}"
        out.append(
            Flag(
                rule_id=f"envelope:{field}",
                severity="info",
                message=(
                    f"{field} = {_num(v)} is {side} the observed {span} range"
                    + (f" of {n} shots in the database" if n else " in the database")
                    + " -- outside what has been run, not necessarily outside what is possible"
                ),
                value=v,
                limit=hi if side == "above" else lo,
                source="envelope",
            )
        )
    return out


def evaluate_flags(
    values: dict[str, float | None],
    category: dict[str, str] | None,
    cfg: dict,
    envelopes: Mapping[str, Any] | None = None,
) -> list[Flag]:
    """Run `cfg`'s rules over `values`; errors first, then warnings, then info.

    `values` is a flat mapping in any of the three key shapes `_canonical` accepts -- a database
    row's flat-top scalars, or a `QueryState.actuators` proposal. `category` labels the thing
    being checked (`{"campaign": "2014_2015", "regime": "H", "segment": "flat_top"}`); it gates
    `when:` rules and picks the envelope band. `envelopes` is optional observed ranges, see
    `envelopes_from_frame`.

    Every rule produces exactly one flag per outcome that is worth a line on screen: a violation
    (one per matching field, so a glob rule names each coil that is over), or a single `info`
    saying the rule could not run. A rule that ran and passed says nothing -- otherwise the
    clean case is unreadable.
    """
    vals, reasons = _canonical(values, cfg)
    flags: list[Flag] = []
    for rule in cfg.get("rules") or []:
        if not _when_ok(rule, category):
            continue
        try:
            flags.extend(_rule_flags(rule, vals, reasons))
        except (KeyError, TypeError, ValueError) as e:
            # `load_rules` refuses a rule like this, so only a hand-built cfg gets here. Even then
            # one bad rule must not take the other rules' flags down with it -- the caller used to
            # catch the exception and drop EVERY flag for the shot, which read as "checked and
            # clean". pydantic's ValidationError is a ValueError.
            rid = str(rule.get("id", "?")) if isinstance(rule, Mapping) else "?"
            flags.append(
                Flag(
                    rule_id=rid,
                    severity="info",
                    message=f"rule {rid} skipped: malformed rule ({str(e).splitlines()[0]})",
                    source=str(rule.get("source", "config")) if isinstance(rule, Mapping) else "?",
                )
            )
    flags.extend(_envelope_flags(vals, category, envelopes))
    return sorted(flags, key=lambda f: SEVERITY_ORDER.get(f.severity, 3))


def _rule_flags(
    rule: Mapping[str, Any], vals: Mapping[str, float], reasons: Mapping[str, str]
) -> list[Flag]:
    """One rule over the canonical values: its violations, or the single info flag saying why it
    could not run. Raises on a malformed rule; `evaluate_flags` turns that into an info flag."""
    rid, field = str(rule["id"]), str(rule["field"])
    limit = _finite(rule.get("limit"))
    if str(rule["op"]) not in OPS:
        raise ValueError(f"unknown op {rule['op']!r}")
    op, sym = OPS[str(rule["op"])]
    severity = str(rule.get("severity", "warn"))
    if severity not in SEVERITY_ORDER:
        raise ValueError(f"unknown severity {severity!r}")
    provisional = " [provisional limit]" if rule.get("provisional") else ""
    source = str(rule.get("source", "config"))
    hits = _matches(rule, vals)
    if not hits or limit is None:
        why = reasons.get(field) or ("no limit configured" if limit is None else None)
        return [
            Flag(
                rule_id=rid,
                severity="info",
                message=f"rule {rid} skipped: missing {field}" + (f" ({why})" if why else ""),
                limit=limit,
                source=source,
            )
        ]
    return [
        Flag(
            rule_id=rid,
            severity=severity,  # type: ignore[arg-type]  # checked against SEVERITY_ORDER above
            message=(
                f"{name} = {_num(v)} {sym} {_num(limit)}: "
                f"{rule.get('message', 'outside the configured limit')}{provisional}"
            ),
            value=v,
            limit=limit,
            source=source,
        )
        for name, v in hits
        if not op(v, limit)
    ]


# ------------------------------------------------------------------------------- envelopes


def envelopes_from_frame(
    frame: Any, fields: Iterable[str] | None = None, q: tuple[float, float] = (5.0, 95.0)
) -> dict[str, dict[str, float]]:
    """Observed `{field: {lo, hi, n}}` from a pandas frame of segment rows.

    Percentiles rather than min/max on purpose: one mis-fit EFIT slice should not widen the
    "this has been done" range to include a value nothing really ran at. Fields with fewer than
    five recorded values are dropped -- a band from three shots is not an envelope.
    """
    import numpy as np

    names = (
        list(fields)
        if fields is not None
        else [c for c in frame.columns if frame[c].dtype.kind == "f"]
    )
    out: dict[str, dict[str, float]] = {}
    for name in names:
        if name not in frame.columns:
            continue
        v = frame[name].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 5:
            continue
        lo, hi = (float(x) for x in np.percentile(v, q))
        out[name] = {"lo": lo, "hi": hi, "n": int(v.size)}
    return out
