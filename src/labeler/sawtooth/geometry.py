"""Local ECE metadata and explicitly uncalibrated per-shot core proxies.

RF frequencies, channel order and the reference radius of Bt must be supplied
by local metadata. EFIT ``rmaxis`` is the magnetic axis and cannot substitute
for the radius at which the scalar toroidal field is defined. The analytic
second-harmonic resonance neglects relativistic and optical-depth corrections;
it is nominal R geometry, never a calibrated flux coordinate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np

from labeler.events import equilibrium
from labeler.events.panels.ece_geometry import align_q, q1_surface
from labeler.events.verify import NoDataError

ELECTRON_CYCLOTRON_HZ_PER_T = 27.992e9
FALLBACK_PHYSICAL_STOP = 40


@dataclass(frozen=True)
class RadiusGeometry:
    time_s: np.ndarray
    R_m: np.ndarray
    axis_R_m: np.ndarray
    q1_low_R_m: np.ndarray | None = None
    q1_high_R_m: np.ndarray | None = None
    lcfs_outer_R_m: np.ndarray | None = None
    nominal_rho: np.ndarray | None = None


def audit_frequency_grid(archive_root, *, minimum_shots=25, tolerance_ghz=0.001):
    """Verify the modal RF setup on independently identified archived shots.

    At least 95% of shots must agree on each channel 0..39. Deviations are
    recorded and same-shot metadata retains precedence over the modal grid.
    The verified scope is the archive's shot range, not all DIII-D history.
    """
    records, invalid = [], []
    for path in sorted((Path(archive_root) / "raw").glob("*.h5")):
        try:
            with h5py.File(path, "r", locking=False) as file:
                shot = int(path.stem)
                if int(file.attrs.get("shot", -1)) != shot:
                    raise ValueError("shot identifier mismatch")
                group = file["ecegeom"]
                if group.attrs.get("source") != "ELECTRONS":
                    raise ValueError("unverified RF source")
                count = len(group["FREQ"])
                expected = {f"ECEVS{i + 1:02d}" for i in range(count)}
                if not expected <= set(file["ece"]):
                    raise ValueError("unverified ECE channel join")
                frequency = _frequency_vector(group["FREQ"][...], "GHz", count)
                if frequency is None or count < FALLBACK_PHYSICAL_STOP:
                    raise ValueError("invalid RF vector")
                records.append((shot, frequency / 1e9))
        except (OSError, KeyError, ValueError) as error:
            invalid.append({"path": str(path), "error": str(error)})
    result = {
        "verified": False,
        "root": str(archive_root),
        "shots_audited": len(records),
        "shots": [shot for shot, _ in records],
        "minimum_shots": minimum_shots,
        "tolerance_ghz": tolerance_ghz,
        "terminal_channels_excluded": list(range(40, 48)),
        "invalid_files": invalid,
        "exceptions": [],
        "frequency_hz": None,
    }
    if not records:
        return result
    grid = np.stack([frequency[:40] for _, frequency in records])
    consensus = np.median(grid, axis=0)
    agrees = np.abs(grid - consensus) <= tolerance_ghz
    result.update(
        frequency_hz=(consensus * 1e9).tolist(),
        channel_agreement_fraction=agrees.mean(axis=0).tolist(),
        verified=bool(
            len(records) >= minimum_shots
            and (agrees.mean(axis=0) >= 0.95).all()
            and (np.diff(consensus) > 0).all()
        ),
    )
    for index, (shot, frequency) in enumerate(records):
        channels = np.flatnonzero(~agrees[index])
        if len(channels):
            result["exceptions"].append({
                "shot": shot, "channels": channels.tolist(),
                "frequency_ghz": frequency[channels].tolist(),
                "consensus_ghz": consensus[channels].tolist(),
                "maximum_difference_ghz": float(
                    np.max(np.abs(frequency[:40] - consensus))
                ),
            })
    result["limitations"] = (
        "modal archive grid transferred to other shots; historical RF changes "
        "outside the audited shot range are not excluded; same-shot metadata "
        "overrides it; channel 0 joins TECEF01/ECEVS01"
    )
    return result


@lru_cache(maxsize=8)
def _verified_archive_grid(root):
    return audit_frequency_grid(root)


@dataclass(frozen=True)
class ShotCore:
    core_channels: np.ndarray
    outer_channels: np.ndarray
    physical_channels: np.ndarray
    info: dict


def _frequency_vector(value, unit, channels):
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    try:
        values = np.asarray(value, dtype=float).ravel()
    except (ValueError, TypeError):
        return None
    scales = {"hz": 1, "ghz": 1e9, "mhz": 1e6}
    if str(unit).lower() not in scales:
        return None
    values = values * scales[str(unit).lower()]
    if (
        len(values) != channels
        or not np.isfinite(values).all()
        or (values < 10e9).any()
        or (values > 1e12).any()
    ):
        return None
    return values


def nominal_frequencies(file, channels):
    """Read explicit RF channel metadata, with no frequency-band guessing.

    An array in the ECE group shares its channel axis by construction. Generic
    frequency keys also require explicit units; waveform sampling rate and
    spectrogram frequencies cannot become a resonance calibration.
    """
    if "ece" not in file:
        return None, None
    group = file["ece"]
    names = (
        "frequency_hz",
        "frequencies_hz",
        "frequency_ghz",
        "frequencies_ghz",
        "channel_frequency_hz",
        "channel_frequencies_hz",
        "channel_frequency_ghz",
        "channel_frequencies_ghz",
        "frequency",
        "frequencies",
        "channel_frequency",
        "channel_frequencies",
    )
    for name in names:
        unit = name.rsplit("_", 1)[-1]
        if unit not in ("hz", "ghz"):
            unit = group.attrs.get("frequency_units", "")
        if name in group.attrs:
            values = _frequency_vector(group.attrs[name], unit, channels)
            if values is not None:
                return values, f"{file.filename}:/ece@{name}"
        if name in group and isinstance(group[name], h5py.Dataset):
            dataset = group[name]
            if dataset.size > channels:
                continue
            units = dataset.attrs.get("units", unit)
            values = _frequency_vector(dataset[...], units, channels)
            if values is not None:
                return values, f"{file.filename}:/ece/{name}"
    return None, None


def second_harmonic_R(t_s, frequency_hz, bt, reference_radius_m):
    """R=2*(e/2pi/me)*abs(Bt)*Rref/f; no temporal extrapolation across gaps."""
    frequency = np.asarray(frequency_hz, dtype=float)
    reference = float(reference_radius_m)
    if (
        frequency.ndim != 1
        or not np.isfinite(frequency).all()
        or (frequency <= 0).any()
        or not np.isfinite(reference)
        or reference <= 0
    ):
        raise ValueError("positive RF frequencies and field reference radius required")
    field = align_q(t_s, bt[0], np.atleast_2d(bt[1]))[0]
    return (
        2
        * ELECTRON_CYCLOTRON_HZ_PER_T
        * np.abs(field)[None]
        * reference
        / frequency[:, None]
    )


def _q1_major_radii(r, z, psirz, axis, axis_z, psi_axis, psi_edge, qpsi, height):
    """Axis-connected EFIT q=1 intersections with the measured ECE sightline."""
    surfaces = q1_surface(np.linspace(0, 1, qpsi.shape[1]), qpsi)
    low, high = np.full(len(axis), np.nan), np.full(len(axis), np.nan)
    for k, surface in enumerate(surfaces):
        if not np.isfinite(surface) or not np.isfinite(axis[k]):
            continue
        scale = psi_edge[k] - psi_axis[k]
        if not np.isfinite(scale) or scale == 0:
            continue
        # Native MDSplus square PSIRZ grids do not identify dimension order.
        # Resolve it from the independently stored flux at the magnetic axis.
        choices = []
        for candidate in (psirz[k], psirz[k].T):
            if candidate.shape != (len(z), len(r)):
                continue
            at_z = np.array(
                [np.interp(axis_z[k], z, candidate[:, i]) for i in range(len(r))]
            )
            at_axis = np.interp(axis[k], r, at_z)
            choices.append((abs(at_axis - psi_axis[k]), candidate))
        if not choices:
            continue
        error, oriented = min(choices, key=lambda pair: pair[0])
        if not np.isfinite(error) or error > 0.05 * abs(scale):
            continue
        line = np.array(
            [
                np.interp(height, z, oriented[:, i], left=np.nan, right=np.nan)
                for i in range(len(r))
            ]
        )
        psin = (line - psi_axis[k]) / scale
        for outward, target in ((r < axis[k], low), (r >= axis[k], high)):
            indices = np.flatnonzero(outward)
            indices = indices[np.argsort(np.abs(r[indices] - axis[k]))]
            if not len(indices) or not np.isfinite(psin[indices[0]]):
                continue
            if psin[indices[0]] >= surface:
                continue
            for before, after in pairwise(indices):
                if not np.isfinite(psin[[before, after]]).all():
                    break
                if psin[after] >= surface:
                    fraction = (surface - psin[before]) / (psin[after] - psin[before])
                    target[k] = r[before] + fraction * (r[after] - r[before])
                    break
    return low, high


def _equilibrium_radius(shot, t_s, channels, path, fixed_grid=None):
    """Read archive-layout per-shot EFIT, with independently audited RF fallback."""
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        with h5py.File(path, "r", locking=False) as file:
            if int(file.attrs.get("shot", -1)) != shot:
                raise ValueError("archive shot metadata does not agree")
            setup, eq = file.get("ecegeom"), file["eq"]
            frequency, frequency_source = None, None
            scope = "same_shot"
            if (
                setup is not None
                and setup.attrs.get("source") == "ELECTRONS"
                and "FREQ" in setup
            ):
                native = np.asarray(setup["FREQ"]).ravel()
                native = _frequency_vector(native, "GHz", len(native))
                if native is not None and len(native) >= min(channels, 40):
                    frequency = np.full(channels, np.nan)
                    stop = min(channels, 40)
                    frequency[:stop] = native[:stop]
                    frequency_source = f"{path}:/ecegeom/FREQ"
            if frequency is None and fixed_grid and fixed_grid["verified"]:
                frequency = np.full(channels, np.nan)
                stop = min(channels, 40)
                frequency[:stop] = np.asarray(fixed_grid["frequency_hz"])[:stop]
                frequency_source = f"{fixed_grid['root']}:/raw verified_modal_grid"
                scope = "audited_archive_modal_channels_0_to_39"
            if frequency is None:
                raise ValueError("RF metadata or verified fixed-grid audit unavailable")
            if "ece" in file:
                expected = {f"ECEVS{i + 1:02d}" for i in range(min(channels, 40))}
                if not expected <= set(file["ece"]):
                    raise ValueError("archive ECE identifiers do not establish order")
            clock = np.asarray(eq["gtime"], dtype=float) / 1000
            product = np.asarray(eq["fpol"], dtype=float)[:, -1]
            axis = np.asarray(eq["rmaxis"], dtype=float)
            height = (
                float(setup["ECEZH"][()])
                if setup is not None and "ECEZH" in setup else np.nan
            )
            measured_height = bool(np.isfinite(height))
            if not measured_height:
                # Muscatello describes the verified first-40 ECE array as
                # midplane. This is an explicit nominal sightline assumption.
                height = 0.0
            if (
                eq.attrs.get("source") != "efit01"
                or product.shape != clock.shape
                or axis.shape != clock.shape
            ):
                raise ValueError("archive EFIT array axes or source do not agree")
            q1_checked = {
                "zmaxis", "r", "z", "psirz", "qpsi", "ssimag", "ssibry"
            } <= set(eq)
            low, high, no_surface = None, None, None
            if q1_checked:
                axis_z = np.asarray(eq["zmaxis"], dtype=float)
                r, z = np.asarray(eq["r"]), np.asarray(eq["z"])
                psirz, qpsi = np.asarray(eq["psirz"]), np.asarray(eq["qpsi"])
                if (
                    psirz.shape
                    not in ((len(clock), len(z), len(r)), (len(clock), len(r), len(z)))
                    or qpsi.ndim != 2
                    or qpsi.shape[0] != len(clock)
                    or not (np.diff(r) > 0).all()
                    or not (np.diff(z) > 0).all()
                ):
                    raise ValueError("archive EFIT q/flux array axes do not agree")
                low, high = _q1_major_radii(
                    r, z, psirz, axis, axis_z,
                    np.asarray(eq["ssimag"]), np.asarray(eq["ssibry"]),
                    qpsi, height,
                )
                no_surface = int(
                    np.isnan(q1_surface(np.linspace(0, 1, qpsi.shape[1]), qpsi)).sum()
                )
            lcfs = np.full(len(clock), np.nan)
            if "bdry" in eq and "nbdry" in eq:
                boundary, counts = np.asarray(eq["bdry"]), np.asarray(eq["nbdry"])
                if (
                    boundary.ndim == 3
                    and boundary.shape[0] == len(clock)
                    and boundary.shape[-1] == 2
                    and counts.shape == clock.shape
                ):
                    for k, count in enumerate(counts):
                        if (
                            not np.isfinite(count) or count < 3
                            or count > boundary.shape[1]
                        ):
                            continue
                        points = boundary[k, :int(count), 0]
                        if len(points) > 2 and np.isfinite(points).all():
                            lcfs[k] = float(np.max(points))
        # EFIT's F(psi=1)=R*Bphi is the boundary/vacuum B0R0 product.
        # It avoids silently treating rmaxis as the Bt reference radius.
        radius = np.full((channels, len(t_s)), np.nan)
        valid_frequency = np.isfinite(frequency)
        radius[valid_frequency] = second_harmonic_R(
            t_s, frequency[valid_frequency], (clock, product), 1.0
        )
        axis_R = align_q(t_s, clock, axis[None])[0]
        lcfs_R = align_q(t_s, clock, lcfs[None])[0]
        minor = lcfs_R - axis_R
        minor[minor <= 0] = np.nan
        rho = np.abs(radius - axis_R[None]) / minor[None]
        mapped = RadiusGeometry(
            np.asarray(t_s),
            radius,
            axis_R,
            align_q(t_s, clock, low[None])[0] if q1_checked else None,
            align_q(t_s, clock, high[None])[0] if q1_checked else None,
            lcfs_R,
            rho,
        )
        steps = np.diff(frequency[:min(channels, 40)])
        info = {
            "status": "nominal_second_harmonic_R_from_archived_EFIT_F",
            "frequency_source": frequency_source,
            "frequency_hz": [float(v) if np.isfinite(v) else None for v in frequency],
            "frequency_scope": scope,
            "frequency_order_supported": bool((steps >= 0).all()),
            "frequency_order_break_channels": np.flatnonzero(steps < 0).tolist(),
            "near_duplicate_frequency_pairs": np.flatnonzero(
                np.abs(steps) <= 1e6
            ).tolist(),
            "frequency_units": (
                "GHz: documented omnimode ece_fwd reader contract; "
                "stored FREQ units are blank"
            ),
            "channel_order_source": "corpus row i joins TECEF(i+1)/ECEVS(i+1)",
            "field_product_source": f"{path}:/eq/fpol[:, -1] (F=R*Bphi)",
            "axis_source": f"{path}:/eq/rmaxis",
            "q1_source": (
                f"{path}:/eq/qpsi and /eq/psirz at ECEZH={height}m"
                if q1_checked else None
            ),
            "q1_checked": q1_checked,
            "q1_status": (
                "EFIT_psirz_and_qpsi_unavailable" if not q1_checked
                else "checked_EFIT01_sightline_intersections"
                if measured_height else "checked_EFIT01_nominal_midplane_intersections"
            ),
            "sightline_height_source": (
                f"{path}:/ecegeom/ECEZH" if measured_height
                else "Muscatello 2012 first-40 midplane ECE array; assumed z=0m"
            ),
            "q1_no_axis_connected_surface_slices": no_surface,
            "q1_low_supported_samples": (
                int(np.isfinite(mapped.q1_low_R_m).sum()) if q1_checked else 0
            ),
            "q1_high_supported_samples": (
                int(np.isfinite(mapped.q1_high_R_m).sum()) if q1_checked else 0
            ),
            "lcfs_outer_source": f"{path}:/eq/bdry[:nbdry,0] maximum R",
            "lcfs_outer_supported_samples": int(np.isfinite(lcfs_R).sum()),
            "nominal_rho_definition": "abs(R-axis_R)/(LCFS_outer_R-axis_R)",
            "formula": "R_m=2*27.992e9*abs(EFIT_F_boundary_Tm)/f_Hz",
            "calibrated_flux": False,
            "limitations": (
                "same-shot nominal second-harmonic vacuum-field mapping; "
                "no relativistic or optical-depth correction; geometric nominal "
                "rho is not normalized flux or the paper's measured rho"
            ),
        }
        return mapped, info
    except (OSError, KeyError, ValueError, IndexError) as error:
        return None, {
            "status": "same_shot_archive_geometry_invalid",
            "archive_source": str(path),
            "error": str(error),
        }


def load_radius_geometry(
    shot, t_s, channels, paths, *, archive_root=None, metadata_root=None
):
    """Read local frequency/Bt/EFIT-axis metadata; never invoke a resolver."""
    archive_info, fixed_grid = None, None
    if archive_root is not None:
        fixed_grid = _verified_archive_grid(str(archive_root))
        archived, archive_info = _equilibrium_radius(
            shot, t_s, channels, Path(archive_root) / "raw" / f"{int(shot)}.h5",
            fixed_grid,
        )
        if archived is not None:
            return archived, archive_info
    if metadata_root is not None:
        mapped, metadata_info = _equilibrium_radius(
            shot, t_s, channels, Path(metadata_root) / f"{int(shot)}.h5", fixed_grid
        )
        if mapped is not None:
            return mapped, metadata_info
        if metadata_info is not None:
            archive_info = metadata_info
    frequency, source = None, None
    stores = (
        paths.features_file(shot),
        paths.corpus_file(shot),
        paths.raw_cache / f"{int(shot)}_processed.h5",
    )
    for path in stores:
        try:
            with h5py.File(path, "r", locking=False) as file:
                frequency, source = nominal_frequencies(file, channels)
        except OSError:
            continue
        if frequency is not None:
            break
    if frequency is None and fixed_grid and fixed_grid["verified"]:
        frequency = np.full(channels, np.nan)
        stop = min(channels, 40)
        frequency[:stop] = np.asarray(fixed_grid["frequency_hz"])[:stop]
        source = f"{archive_root}:/raw verified_modal_grid"
    info = {
        "status": "frequency_metadata_unavailable",
        "frequency_source": source,
        "field_reference_source": None,
        "axis_source": None,
        "calibrated_flux": False,
        "formula": "R_m=2*27.992e9*abs(Bt_T)*Rref_m/f_Hz",
        "limitations": "nominal, no relativistic or optical-depth correction",
        "q1_checked": False,
        "q1_status": "EFIT_psirz_and_qpsi_unavailable",
        "frequency_grid_audit_verified": bool(fixed_grid and fixed_grid["verified"]),
    }
    if archive_info is not None:
        info["same_shot_archive"] = archive_info
    if frequency is None:
        return None, info
    try:
        field = equilibrium.signal(shot, "bt", paths, fetch=False)
        axis = equilibrium.signal(shot, "r0", paths, fetch=False)
    except (NoDataError, ValueError):
        info["status"] = "bt_or_efit_axis_unavailable"
        return None, info
    info["axis_source"] = {
        "store": axis.attrs.get("store"),
        "locator": axis.attrs.get("locator"),
    }
    reference = None
    for key in ("reference_radius_m", "bt_reference_radius_m", "rcentr_m"):
        try:
            candidate = float(field.attrs.get(key, "nan"))
        except (ValueError, TypeError):
            continue
        if np.isfinite(candidate) and candidate > 0:
            reference = candidate
            info["field_reference_source"] = f"{field.attrs['store']}:/bt@{key}"
            break
    if reference is None:
        info["status"] = "bt_reference_radius_unavailable"
        return None, info
    radius = np.full((channels, len(t_s)), np.nan)
    valid_frequency = np.isfinite(frequency)
    valid_frequency[40:] = False
    radius[valid_frequency] = second_harmonic_R(
        t_s, frequency[valid_frequency], (field.x, field.y[0]), reference
    )
    axis_R = align_q(t_s, axis.x, axis.y)[0]
    info.update(
        status="nominal_second_harmonic_R",
        reference_radius_m=reference,
        frequency_hz=[float(v) if np.isfinite(v) else None for v in frequency],
        field_source={
            "store": field.attrs.get("store"),
            "locator": field.attrs.get("locator"),
        },
        supported_samples=int(
            (np.isfinite(radius[:min(channels, 40)]).all(axis=0)
             & np.isfinite(axis_R)).sum()
        ),
    )
    return RadiusGeometry(np.asarray(t_s), radius, axis_R), info


def radius_evidence(radius, time_s, inversion_channel, flux=None):
    """Nominal inversion R and q=1 R on the same measured radial branch.

    The q=1 comparison also requires calibrated ECE psi and an EFIT surface.
    Unknown RF geometry, missing slices and missing bracketing channels produce
    nulls. A normalized flux radius is never relabeled as a major radius.
    """
    result = {
        "inversion_R_m": None,
        "q1_R_m": None,
        "q1_radius_difference_m": None,
        "inversion_nominal_rho": None,
        "q1_nominal_rho": None,
        "nominal_rho_difference": None,
    }
    if radius is None:
        return result
    channel = float(inversion_channel)
    lo, hi = int(np.floor(channel)), int(np.ceil(channel))
    if lo < 0 or hi >= len(radius.R_m):
        return result
    R = align_q([time_s], radius.time_s, radius.R_m)[:, 0]
    axis = align_q([time_s], radius.time_s, radius.axis_R_m[None])[0, 0]
    if not np.isfinite(R[[lo, hi]]).all():
        return result
    inversion = float(np.interp(channel, np.arange(len(R)), R))
    result["inversion_R_m"] = inversion
    minor = np.nan
    if radius.lcfs_outer_R_m is not None:
        lcfs = align_q(
            [time_s], radius.time_s, radius.lcfs_outer_R_m[None]
        )[0, 0]
        minor = lcfs - axis
        if np.isfinite(minor) and minor > 0:
            result["inversion_nominal_rho"] = abs(inversion - axis) / minor
    if radius.q1_low_R_m is not None and np.isfinite(axis):
        surface = radius.q1_high_R_m if inversion >= axis else radius.q1_low_R_m
        q1 = align_q([time_s], radius.time_s, surface[None])[0, 0]
        if np.isfinite(q1):
            result.update(
                q1_R_m=float(q1), q1_radius_difference_m=inversion - float(q1)
            )
            if np.isfinite(minor) and minor > 0:
                result.update(
                    q1_nominal_rho=float(abs(q1 - axis) / minor),
                    nominal_rho_difference=float((inversion - q1) / minor),
                )
        return result
    if flux is None or flux.psi is None or not np.isfinite(axis):
        return result
    psi = align_q([time_s * 1000], flux.time_ms, flux.psi)[:, 0]
    surface = align_q([time_s * 1000], flux.time_ms, flux.q1_psi[None])[0, 0]
    if not np.isfinite(surface) or not 0 < surface <= 1:
        return result
    branch = (R >= axis) if inversion >= axis else (R <= axis)
    selected = np.flatnonzero(
        branch & np.isfinite(R) & np.isfinite(psi) & (psi >= 0) & (psi <= 1)
    )
    if len(selected) < 2:
        return result
    selected = selected[np.argsort(psi[selected])]
    positions = psi[selected]
    if (np.diff(positions) <= 0).any():
        return result
    right = int(np.searchsorted(positions, surface))
    if right < len(selected) and positions[right] == surface:
        q1 = float(R[selected[right]])
    elif 0 < right < len(selected) and abs(selected[right] - selected[right - 1]) == 1:
        q1 = float(
            np.interp(
                surface,
                positions[right - 1 : right + 1],
                R[selected[right - 1 : right + 1]],
            )
        )
    else:
        return result
    result.update(q1_R_m=q1, q1_radius_difference_m=inversion - q1)
    return result


def _row_median(values):
    return np.array(
        [
            float(np.median(row[np.isfinite(row)]))
            if np.isfinite(row).any()
            else np.nan
            for row in values
        ]
    )


def select_core(
    y,
    *,
    radius=None,
    minimum_channels=2,
    width=7,
    maximum_to_core=1.5,
):
    """Use the EFIT axis and mask unverified/overlapping ECE spatial support.

    Channels 40..47 never contribute to core, outer or redistribution tests.
    With boundary geometry, channels whose median R2 is inside 2/3 R_LCFS,out
    are also excluded. Nominal rho is a geometric distance proxy, not flux.
    Missing geometry retains an explicitly unvalidated temperature proxy.
    """
    values = np.asarray(y, dtype=float)
    if values.ndim != 2 or len(values) < minimum_channels:
        raise ValueError("ECE requires multiple channel traces")
    # The proxy is chosen once per analysis window, without crash labels.
    median = _row_median(values)
    finite = np.isfinite(median) & (median > 0)
    stop = min(FALLBACK_PHYSICAL_STOP, len(values))
    eligible = finite.copy()
    eligible[stop:] = False
    smooth = np.full(len(values), np.nan)
    for channel in range(stop):
        neighbors = median[max(0, channel - 2) : min(stop, channel + 3)]
        neighbors = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
        if len(neighbors) >= minimum_channels:
            smooth[channel] = np.median(neighbors)
    if not np.isfinite(smooth).any():
        raise ValueError("insufficient coherent physical ECE core channels")
    coherent = int(np.nanargmax(smooth))
    peak = float(smooth[coherent])
    physical = finite & (median <= maximum_to_core * peak)
    physical[stop:] = False
    overlap = np.zeros(len(values), dtype=bool)
    nominal_rho = np.full(len(values), np.nan)
    if radius is not None:
        if radius.R_m.shape[0] != len(values):
            raise ValueError("radius and ECE channel axes disagree")
        if radius.lcfs_outer_R_m is not None:
            separation = _row_median(
                radius.R_m - 2 / 3 * radius.lcfs_outer_R_m[None]
            )
            overlap = np.isfinite(separation) & (separation < 0)
            physical &= ~overlap
        if radius.nominal_rho is not None:
            nominal_rho = _row_median(radius.nominal_rho)
    eligible &= physical
    candidates = np.flatnonzero(eligible)
    if len(candidates) < minimum_channels:
        raise ValueError("insufficient physically plausible ECE core channels")
    nearby = candidates[np.abs(candidates - coherent) <= 3]
    center = (
        int(nearby[np.argmax(median[nearby])]) if len(nearby)
        else int(candidates[np.argmin(np.abs(candidates - coherent))])
    )
    status = "shot_hottest_physical_channel_proxy"
    if radius is not None:
        distance = _row_median(np.abs(radius.R_m - radius.axis_R_m[None]))
        available = candidates[np.isfinite(distance[candidates])]
        if len(available) >= minimum_channels:
            center = int(available[np.argmin(distance[available])])
            core = np.sort(available[np.argsort(distance[available])[:width]])
            status = "nominal_second_harmonic_axis_proxy"
        else:
            core = candidates[np.argsort(np.abs(candidates - center))[:width]]
    else:
        core = candidates[np.argsort(np.abs(candidates - center))[:width]]
    core = np.sort(core)
    outer_status = "adjacent_channel_proxy_unvalidated"
    if radius is not None and np.isfinite(nominal_rho).any():
        side = _row_median(radius.R_m - radius.axis_R_m[None])
        outside = candidates[
            (side[candidates] > 0)
            & (nominal_rho[candidates] >= 0.4)
            & (nominal_rho[candidates] <= 0.65)
            & ~np.isin(candidates, core)
        ]
        outer = np.sort(
            outside[np.argsort(np.abs(nominal_rho[outside] - 0.525))[:width]]
        )
        outer_status = "nominal_low_field_side_rho_0.4_to_0.65"
        if len(outer) < minimum_channels:
            outer_status = "nominal_outer_band_insufficient_support"
    else:
        outside = candidates[candidates < core[0]]
        if len(outside) < minimum_channels:
            outside = candidates[candidates > core[-1]]
        outer = np.sort(outside[np.argsort(np.abs(outside - center))[:width]])
    info = {
        "status": status,
        "radius_status": "unavailable" if radius is None else "nominal_R",
        "central_channel": center,
        "core_channels": core.tolist(),
        "outer_channels": outer.tolist(),
        "outer_status": outer_status,
        "nominal_rho_median": [
            float(v) if np.isfinite(v) else None for v in nominal_rho
        ],
        "core_nominal_rho_median": (
            float(np.median(nominal_rho[core]))
            if len(core) and np.isfinite(nominal_rho[core]).all() else None
        ),
        "outer_nominal_rho_median": (
            float(np.median(nominal_rho[outer]))
            if len(outer) and np.isfinite(nominal_rho[outer]).all() else None
        ),
        "nominal_rho_definition": "abs(R-axis_R)/(LCFS_outer_R-axis_R)",
        "nominal_rho_limitations": (
            "vacuum second-harmonic geometric distance, not normalized flux; "
            "no relativistic or optical-depth correction"
        ),
        "harmonic_overlap_excluded_channels": np.flatnonzero(overlap).tolist(),
        "harmonic_overlap_mask": "median R2 < (2/3)*R_LCFS_outer",
        "terminal_channels_excluded": list(range(stop, len(values))),
        "shot_median_te_kev": [float(v) if np.isfinite(v) else None for v in median],
        "coherent_core_peak_kev": peak,
        "maximum_channel_to_core": maximum_to_core,
        "implausible_channels": np.flatnonzero(finite & ~physical).tolist(),
        "fallback_excluded_from_core": list(range(stop, len(values))),
        "calibrated": False,
    }
    return ShotCore(core, outer, physical, info)
