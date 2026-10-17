#!/usr/bin/env python
"""Fetch the inputs of the detachment default view (plan D.B) on the login node.

    flock $G/raw_cache.lock pixi run --frozen --no-install -e labeler fdp run \\
        python scripts/labeler/detach_default_fetch.py [--shots ...] [--parts ...] [--pace 1]

Run under the fdp wrapper, never in a SLURM job. The shots default to the round-4
roster, the TangTV inversion shots, then every other detachment review store.
`--parts` picks from `equilibrium`, `prad`, `irtv`, `dts_blessed` and `irtv_temp` (the
default) or the retired `dts` (not in the default since 2026-10-17). Per shot:

* `equilibrium`: `$LABELER_ROOT/equilibria/<shot>.npz`, what `review/slices.py` draws
  in the dock: `t_ms` (the G-EQDSK clock), `tree` (EFIT02 or EFIT01) and `source` (the
  rule of `detachment.signals.tangtv_geometry`: EFIT02 when it has two slices or more,
  else `EFIT01_fallback_EFIT02_sparse`, or `EFIT01_fallback` without EFIT02); the
  boundary `rbbbs`/`zbbbs` (time, point; NaN past `nbbbs`), the wall `lim` (point,
  [r, z]); the X-points `rxpt1`/`zxpt1` (lower) and `rxpt2`/`zxpt2` (upper) and the
  strike points `rvsod`/`zvsod`, `rvsid`/`zvsid`, `rvsou`/`zvsou`, `rvsiu`/`zvsiu`
  from the A-EQDSK, matched to `t_ms` within 0.5 ms (NaN otherwise, EFIT sentinels
  kept); and, when the grid is at most 65 x 65, `r`, `z`, `psirz` (time, z, r),
  `ssimag` and `ssibry`. Float32 throughout, compressed.
* `prad`: `$LABELER_ROOT/detach_default/<shot>.npz`: `\\BOLOM::PRAD_DIVL` and
  `PRAD_DIVU` as `prad_divl__t`/`__y` and `prad_divu__t`/`__y` (ms, W) with a JSON
  `status`, only for a shot whose producer cache (`round4/detach/cache`) lacks them.
  The node declares "MW" but holds W, as the producer reads it (3.7e6 at peak on
  189061).
* `dts_blessed`: `$LABELER_ROOT/detach_default/dts_blessed/<shot>.npz`: the blessed
  divertor Thomson Te, `\\ELECTRONS::TOP.TS.BLESSED.DIVERTOR:TEMP` (eV, 0 is no fit):
  `temp` (chord, time) float32, `t_ms` (`:TIME`), and each chord's `r_m`/`z_m`/`phi_deg`
  (`:R`, `:Z`, `:PHI`); 8-82 chords, a new profile about every 20 ms. Absent on the
  shots the system did not read (logged, not an error).
* `dts` (retired from the default 2026-10-17, files kept): `$LABELER_ROOT/detach_default/
  dts/<shot>.npz`: the plasma control system's real-time divertor Thomson Te, PTDATA
  `tssdivte00`..`tssdivte05`, as `<name>__t` (ms) and `<name>__y` (eV, though PTDATA
  labels them "vo"; 0 is no measurement), and the blessed divertor chords'
  `chord_r_m`/`chord_z_m`/`chord_phi_m`: channel k follows chord k. The pointnames are
  absent before about shot 176000 (PTDATA then reports the `PTSERVER` service lookup),
  which is logged, not an error.
* `irtv`: `$LABELER_ROOT/detach_default/irtv/<shot>.npz`: the lower outer divertor
  infrared heat flux, `\\IRTVQ2DR2` (`\\IRTV::TOP.IRTV:LODIV_60RP2:DIGITAL_CAM:
  HEAT_FLUX2D`, W/cm^2), else `\\IRTVQ2DR201` (DIG_CAM_01): `heat_flux` (block,
  pixel) float16, the 1 ms block means of every pixel (`IRTV_BLOCK_MS`), `t_ms` the
  block centres, `r_cm`/`z_cm` each pixel's position (dimensions 0 and 2 of the
  record), and `node`/`system`/`units`.
* `irtv_temp`: `$LABELER_ROOT/detach_default/irtv_temp/<shot>.npz`: the twin map, the
  absolute surface temperature `\\IRTVAT2DR2` (`...:DIGITAL_CAM:ABS_T_2D`, deg C), else
  `\\IRTVAT2DR201`; the same arrays with `temperature` in place of `heat_flux`.

Resumable: a part logged as done for a shot is skipped, and so is a file already
written. One JSONL line per shot in `$LABELER_ROOT/runs/detach_default/fetch_log.jsonl`
(`done_parts` lists the parts it finished). An authentication failure stops the run
(exit 3) and is not retried.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detach_fetch_efit import orient  # the sys.path insert above makes this work

#: Words of an authentication failure, matched as whole words of the error text
#: (a shot or byte count holding "401" is not one); `auth...` and `unauth...` match
#: as prefixes (authentication, unauthorized).
AUTH_WORDS = ("token", "login", "credential", "credentials", "401", "403", "permission")
AUTH_PREFIXES = ("auth", "unauth")
#: The largest flux grid kept (r points x z points); a larger one keeps the geometry.
MAX_GRID = 65 * 65
#: A-EQDSK slices further than this from a G-EQDSK slice are left NaN.
MATCH_MS = 0.5
GEQDSK = ("rbbbs", "zbbbs", "nbbbs", "lim")
AEQDSK = (
    "rxpt1", "zxpt1", "rxpt2", "zxpt2", "rvsod", "zvsod", "rvsid", "zvsid",
    "rvsou", "zvsou", "rvsiu", "zvsiu",
)
FLUX = ("psirz", "ssimag", "ssibry", "rmaxis", "zmaxis")
PRAD = (("prad_divl", r"\BOLOM::PRAD_DIVL"), ("prad_divu", r"\BOLOM::PRAD_DIVU"))
PARTS = ("equilibrium", "prad", "dts", "irtv", "dts_blessed", "irtv_temp")
#: What `--parts` runs when unnamed: the retired tssdivte `dts` part is left out.
DEFAULT_PARTS = ("equilibrium", "prad", "irtv", "dts_blessed", "irtv_temp")
DTS_POINTNAMES = tuple(f"tssdivte{k:02d}" for k in range(6))
DTS_CHORDS = r"\ELECTRONS::TOP.TS.BLESSED.DIVERTOR"
#: The lower outer divertor heat flux, (system, tag) in order of preference: the
#: DIGITAL_CAM analysis (re-run 2025-11-28), else DIG_CAM_01.
IRTV_HEAT_FLUX = (("DIGITAL_CAM", r"\IRTVQ2DR2"), ("DIG_CAM_01", r"\IRTVQ2DR201"))
#: Its twin, the absolute surface temperature map (ABS_T_2D, deg C), same shape.
IRTV_TEMPERATURE = (("DIGITAL_CAM", r"\IRTVAT2DR2"), ("DIG_CAM_01", r"\IRTVAT2DR201"))
IRTV_BLOCK_MS = 1.0
#: float16 holds the 1 ms means; a value beyond this is clipped (and counted).
FLOAT16_MAX = 65000.0


#: The MDSplus tree `node` reads from, kept open across one shot's reads.
_OPEN: dict = {}


class AuthError(RuntimeError):
    """The fdp login has lapsed: stop, never retry."""


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"])


def checked(error: Exception) -> str:
    """The error as text; an authentication failure raises `AuthError` instead."""
    text = f"{type(error).__name__}: {str(error)[:200]}"
    words = re.findall(r"[a-z0-9]+", text.lower())
    if any(w in AUTH_WORDS or w.startswith(AUTH_PREFIXES) for w in words):
        raise AuthError(text)
    return text


def node(name: str, tree: str, shot: int) -> np.ndarray:
    """One EFIT node, read through an MDSplus tree kept open for the shot's reads
    (a toksearch signal reopens the tree per node: about 45 s a shot instead of 10)."""
    import MDSplus

    if name in GEQDSK:
        path = rf"\{tree}::top.results.geqdsk:{name}"
    elif name in AEQDSK or name == "atime":
        path = rf"\{tree}::top.results.aeqdsk:{name}"
    else:
        path = rf"\{tree}::{name}"
    if _OPEN.get("key") != (tree, shot):
        _OPEN.update(key=(tree, shot), tree=MDSplus.Tree(tree, shot, "readonly"))
    return np.asarray(_OPEN["tree"].getNode(path).data())


def choose_tree(shot: int) -> tuple[str, str, np.ndarray]:
    """EFIT02 with two slices or more, else EFIT01 flagged as the fallback."""
    try:
        gtime = node("gtime", "efit02", shot)
    except Exception as error:  # noqa: BLE001  a missing EFIT02 is the fallback case
        checked(error)
        return "EFIT01", "EFIT01_fallback", node("gtime", "efit01", shot)
    if gtime.size < 2:
        return "EFIT01", "EFIT01_fallback_EFIT02_sparse", node("gtime", "efit01", shot)
    return "EFIT02", "EFIT02", gtime


def on_clock(values: np.ndarray, atime: np.ndarray, gtime: np.ndarray) -> np.ndarray:
    """A-EQDSK `values` at each G-EQDSK time, NaN where no slice is within MATCH_MS."""
    out = np.full(len(gtime), np.nan, dtype=np.float32)
    if values.size != atime.size or not atime.size:
        return out
    order = np.argsort(atime)
    atime, values = atime[order], values[order]
    right = np.clip(np.searchsorted(atime, gtime), 0, len(atime) - 1)
    left = np.clip(right - 1, 0, len(atime) - 1)
    j = np.where(np.abs(atime[left] - gtime) <= np.abs(atime[right] - gtime), left, right)
    near = np.abs(atime[j] - gtime) <= MATCH_MS
    out[near] = values[j[near]]
    return out


def optional(name: str, tree: str, shot: int, missing: list) -> np.ndarray | None:
    """A node an older EFIT may lack: None (and its name in `missing`) when absent."""
    try:
        return node(name, tree, shot)
    except Exception as error:  # noqa: BLE001  an absent node is data
        checked(error)
        missing.append(name)
        return None


def equilibrium(shot: int) -> tuple[dict, dict]:
    """The arrays of `equilibria/<shot>.npz` and what the log says about them."""
    tree, source, gtime = choose_tree(shot)
    gtime = gtime.astype(np.float64).ravel()
    t = tree.lower()
    arrays = {"t_ms": gtime.astype(np.float32), "tree": tree, "source": source}
    missing: list[str] = []
    rb = np.atleast_2d(node("rbbbs", t, shot))
    zb = np.atleast_2d(node("zbbbs", t, shot))
    if rb.shape[0] != len(gtime):
        rb, zb = rb.T, zb.T
    rb, zb = rb.astype(np.float32), zb.astype(np.float32)
    count = optional("nbbbs", t, shot, missing)
    if count is not None and count.size == len(gtime):
        count = count.astype(np.int64).ravel()
        past = np.arange(rb.shape[1])[None, :] >= count[:, None]
        rb[past] = zb[past] = np.nan
        arrays["nbbbs"] = count.astype(np.int16)
    arrays.update(rbbbs=rb, zbbbs=zb)
    lim = node("lim", t, shot).astype(np.float32)
    arrays["lim"] = lim if lim.ndim == 2 and lim.shape[1] == 2 else lim.T
    atime = node("atime", t, shot).astype(np.float64).ravel()
    for name in AEQDSK:
        values = optional(name, t, shot, missing)
        arrays[name] = (
            np.full(len(gtime), np.nan, dtype=np.float32) if values is None
            else on_clock(values.astype(np.float64).ravel(), atime, gtime)
        )
    r = node("r", t, shot).astype(np.float64).ravel()
    z = node("z", t, shot).astype(np.float64).ravel()
    info = {
        "tree": tree, "source": source, "n_t": len(gtime), "grid": [len(r), len(z)],
        **({"missing": missing} if missing else {}),
    }
    if len(r) * len(z) <= MAX_GRID:
        flux = {name: node(name, t, shot) for name in FLUX}
        try:
            psirz = orient(
                flux["psirz"], r, z, flux["ssimag"], flux["rmaxis"], flux["zmaxis"]
            )
        except ValueError as error:
            # A square grid whose axis flux matches neither order is not guessed:
            # the boundary, X-points and wall are kept, the contours left out.
            info["psirz"] = f"left out: {error}"
        else:
            arrays.update(
                r=r.astype(np.float32),
                z=z.astype(np.float32),
                psirz=psirz.astype(np.float32),
                ssimag=flux["ssimag"].astype(np.float32).ravel(),
                ssibry=flux["ssibry"].astype(np.float32).ravel(),
            )
    else:
        info["psirz"] = f"left out: grid {len(r)} x {len(z)} is over {MAX_GRID} points"
    info["xpt2_finite"] = int(np.isfinite(arrays["rxpt2"]).sum())
    return arrays, info


def prad(shot: int) -> tuple[dict | None, dict]:
    """PRAD_DIVL and PRAD_DIVU, unless the producer cache already holds them."""
    cache = root() / "round4/detach/cache" / f"{shot}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            if {f"{n}__y" for n, _ in PRAD} <= set(data.files):
                return None, {"source": str(cache)}
    from labeler.features import resolve_fdp

    arrays, status = {}, {}
    for name, expr in PRAD:
        try:
            record = resolve_fdp._fetch_mds(expr, "BOLOM", shot)
            y = np.asarray(record["data"], dtype=np.float64).ravel()
            x = np.asarray(record.get("times", record.get("dim0")), dtype=np.float64)
        except Exception as error:  # noqa: BLE001  a node absent on a shot is data
            status[name] = "error " + checked(error)
            continue
        arrays[f"{name}__t"] = x.ravel().astype(np.float32)
        arrays[f"{name}__y"] = y.astype(np.float32)
        units = record.get("units", {})
        units = units.get("data", "") if isinstance(units, dict) else units
        status[name] = f"ok {y.size} {units}"
        status[f"{name}_max"] = float(np.nanmax(y)) if y.size else None
    return (arrays or None), {"source": "fdp", "status": status}


def dts(shot: int) -> tuple[dict | None, dict]:
    """tssdivte00-05 and the blessed divertor Thomson chords' positions."""
    from labeler.features import resolve_fdp

    arrays, status = {}, {}
    for name in DTS_POINTNAMES:
        try:
            record = resolve_fdp._fetch_ptdata(name, shot)
            y = np.asarray(record["data"], dtype=np.float32).ravel()
            x = np.asarray(record["times"], dtype=np.float64).ravel()
        except Exception as error:  # noqa: BLE001  a pointname absent on a shot is data
            status[name] = "error " + checked(error)
            continue
        if x.size != y.size or not x.size:
            status[name] = f"error {x.size} times for {y.size} values"
            continue
        arrays[f"{name}__t"] = x.astype(np.float32)
        arrays[f"{name}__y"] = y
        units = record.get("units", {})
        units = units.get("data", "") if isinstance(units, dict) else units
        live = int((np.isfinite(y) & (y > 0)).sum())
        status[name] = f"ok {y.size} {units} live {live}"
    if not arrays:
        return None, {"status": status}
    import MDSplus

    try:
        tree = MDSplus.Tree("electrons", shot, "readonly")
        chords = {
            f"chord_{leaf.lower()}_m": np.asarray(
                tree.getNode(f"{DTS_CHORDS}:{leaf}").data(), dtype=np.float32
            ).ravel()
            for leaf in ("R", "Z", "PHI")
        }
    except Exception as error:  # noqa: BLE001  a shot without blessed chords is data
        status["chords"] = "error " + checked(error)
    else:
        arrays.update(chords)
        status["chords"] = f"ok {chords['chord_z_m'].size}"
    return arrays, {"status": status}


def dts_blessed(shot: int) -> tuple[dict | None, dict]:
    """The blessed divertor Thomson Te, one profile per laser pulse: (chord, time)."""
    import MDSplus

    try:
        tree = MDSplus.Tree("electrons", shot, "readonly")
        temp = tree.getNode(f"{DTS_CHORDS}:TEMP")
        if int(temp.getLength()) == 0:
            return None, {"error": "empty"}
        leaves = {
            name: np.asarray(tree.getNode(f"{DTS_CHORDS}:{leaf}").data())
            for name, leaf in (
                ("t_ms", "TIME"), ("r_m", "R"), ("z_m", "Z"), ("phi_deg", "PHI")
            )
        }
        data = np.asarray(temp.data(), dtype=np.float32)
        units = str(tree.tdiExecute(f"units_of({DTS_CHORDS}:TEMP)").data()).strip()
    except Exception as error:  # noqa: BLE001  a shot without the blessed fit is data
        return None, {"error": checked(error)}
    t, z = leaves["t_ms"].ravel().astype(np.float64), leaves["z_m"].ravel()
    if data.shape == (t.size, z.size) and t.size != z.size:
        data = data.T
    if data.shape != (z.size, t.size) or t.size < 2:
        return None, {"error": f"shape {data.shape}, {t.size} times, {z.size} chords"}
    if leaves["r_m"].size != z.size or leaves["phi_deg"].size != z.size:
        return None, {"error": "R, Z and PHI differ in length"}
    arrays = {
        "temp": data,
        "t_ms": t.astype(np.float32),
        "r_m": leaves["r_m"].ravel().astype(np.float32),
        "z_m": z.astype(np.float32),
        "phi_deg": leaves["phi_deg"].ravel().astype(np.float32),
        "units": units,
    }
    live = int((np.isfinite(data) & (data > 0)).sum())
    return arrays, {"chords": int(z.size), "profiles": int(t.size), "units": units,
                    "live": live}


def block_means(t: np.ndarray, data: np.ndarray, block_ms: float):
    """Means of the finite samples in each `block_ms` block of `t` (time, pixel)."""
    order = np.argsort(t, kind="stable")
    t, data = t[order], data[order]
    index = np.floor((t - np.floor(t[0] / block_ms) * block_ms) / block_ms)
    starts = np.flatnonzero(np.r_[True, np.diff(index) > 0])
    finite = np.isfinite(data)
    sums = np.add.reduceat(np.where(finite, data, 0).astype(np.float32), starts, axis=0)
    counts = np.add.reduceat(finite.astype(np.int32), starts, axis=0)
    means = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    centres = np.add.reduceat(t, starts) / np.diff(np.r_[starts, len(t)])
    return centres, means


def irtv(
    shot: int, key: str = "heat_flux", candidates: tuple = IRTV_HEAT_FLUX
) -> tuple[dict | None, dict]:
    """The lower outer divertor infrared map named by `candidates` ((system, tag) in
    order of preference; the heat flux or, with `IRTV_TEMPERATURE`, the surface
    temperature), 1 ms block means of every pixel stored as `key`."""
    import MDSplus

    try:
        tree = MDSplus.Tree("irtv", shot, "readonly")
    except Exception as error:  # noqa: BLE001  a shot without the tree is data
        return None, {"error": checked(error)}
    tried = {}
    for system, tag in candidates:
        try:
            node = tree.getNode(tag)
            if int(node.getLength()) == 0:
                tried[tag] = "empty"
                continue
            data = np.asarray(node.data(), dtype=np.float32)
            r = np.asarray(node.dim_of(0).data(), dtype=np.float32).ravel()
            t = np.asarray(node.dim_of(1).data(), dtype=np.float64).ravel()
            z = np.asarray(node.dim_of(2).data(), dtype=np.float32).ravel()
            units = str(tree.tdiExecute(f"units_of({tag})").data())
        except Exception as error:  # noqa: BLE001  an unreadable record is data
            tried[tag] = "error " + checked(error)
            continue
        if data.shape == (r.size, t.size) and r.size != t.size:
            data = data.T
        if data.shape != (t.size, r.size) or z.size != r.size or t.size < 2:
            tried[tag] = f"error shape {data.shape}, {t.size} times, {r.size} pixels"
            continue
        centres, means = block_means(t, data, IRTV_BLOCK_MS)
        clipped = int((np.abs(means) > FLOAT16_MAX).sum())
        means = np.clip(means, -FLOAT16_MAX, FLOAT16_MAX)
        arrays = {
            key: means.astype(np.float16),
            "t_ms": centres.astype(np.float32),
            "r_cm": r,
            "z_cm": z,
            "node": tag,
            "system": system,
            "units": units,
        }
        return arrays, {
            "node": tag,
            "system": system,
            "units": units,
            "native": [int(t.size), round(float(np.median(np.diff(np.sort(t)))), 4)],
            "blocks": int(centres.size),
            "pixels": int(r.size),
            **({"clipped": clipped} if clipped else {}),
            **({"tried": tried} if tried else {}),
        }
    return None, {"tried": tried}


def write(path: Path, arrays: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path.stat().st_size


def default_shots() -> list[int]:
    detach = root() / "round4/detach"
    roster = [int(s) for s in (detach / "shots_roster.txt").read_text().split()]
    inverted = sorted(int(p.stem) for p in (detach / "inversions").glob("*.npz"))
    stores = sorted(
        int(p.stem) for p in (root() / "spectrograms/detachment").glob("*.h5")
        if p.stem.isdigit()
    )
    return list(dict.fromkeys(roster + inverted + stores))


def logged_parts(log_path: Path) -> dict[int, set[str]]:
    """The parts each shot finished; a line before `--parts` existed (`done` only)
    finished the equilibrium and Prad,div."""
    done: dict[int, set[str]] = {}
    if not log_path.is_file():
        return done
    for line in log_path.read_text().splitlines():
        entry = json.loads(line)
        parts = entry.get("done_parts")
        if parts is None:
            parts = ["equilibrium", "prad"] if entry.get("done") else []
        done.setdefault(int(entry["shot"]), set()).update(parts)
    return done


def fetch_part(part: str, shot: int, redo: bool) -> tuple[dict, bool]:
    """Run one part for a shot: (what the log says, whether it finished)."""
    if part == "equilibrium":
        path = root() / "equilibria" / f"{shot}.npz"
        if path.is_file() and not redo:
            return {"kept": str(path)}, True
        try:
            arrays, info = equilibrium(shot)
            info["bytes"] = write(path, arrays)
        except AuthError:
            raise
        except Exception as error:  # noqa: BLE001  record it, keep going
            return {"error": checked(error)}, False
        return info, True
    folder = {"prad": "", "dts": "dts", "irtv": "irtv", "dts_blessed": "dts_blessed",
              "irtv_temp": "irtv_temp"}[part]
    path = root() / "detach_default" / folder / f"{shot}.npz"
    if path.is_file() and not redo:
        return {"kept": str(path)}, True
    fetch = {
        "prad": prad,
        "dts": dts,
        "irtv": irtv,
        "dts_blessed": dts_blessed,
        "irtv_temp": lambda shot: irtv(shot, "temperature", IRTV_TEMPERATURE),
    }[part]
    arrays, info = fetch(shot)
    if arrays:
        if "status" in info:
            arrays = {**arrays, "status": json.dumps(info["status"])}
        info["bytes"] = write(path, arrays)
    return info, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", type=int, nargs="+")
    parser.add_argument(
        "--parts", nargs="+", choices=PARTS, default=list(DEFAULT_PARTS)
    )
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    log_path = root() / "runs/detach_default/fetch_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = {} if args.redo else logged_parts(log_path)
    todo = {
        shot: [p for p in args.parts if p not in done.get(shot, set())]
        for shot in (args.shots or default_shots())
    }
    todo = {shot: parts for shot, parts in todo.items() if parts}
    print(f"{len(todo)} shots to fetch ({', '.join(args.parts)})", flush=True)
    for shot, parts in todo.items():
        started = time.monotonic()
        entry = {"shot": shot, "at": datetime.now(UTC).isoformat(timespec="seconds")}
        finished = []
        try:
            for part in parts:
                entry[part], ok = fetch_part(part, shot, args.redo)
                if ok:
                    finished.append(part)
        except AuthError as error:
            entry.update(auth=str(error), done=False, done_parts=finished)
            with log_path.open("a") as log:
                log.write(json.dumps(entry) + "\n")
            print("AUTH ERROR: stopping, the login has lapsed:", error, file=sys.stderr)
            return 3
        entry["seconds"] = round(time.monotonic() - started, 1)
        entry["done_parts"] = finished
        entry["done"] = finished == parts
        with log_path.open("a") as log:
            log.write(json.dumps(entry) + "\n")
        short = {
            part: "kept" if "kept" in entry[part] else entry[part].get(
                "error", entry[part].get("bytes", entry[part].get("source", "none"))
            )
            for part in parts
        }
        print(shot, entry["seconds"], "s", json.dumps(short)[:300], flush=True)
        time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
