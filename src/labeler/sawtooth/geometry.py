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
from itertools import pairwise

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


def _archived_radius(shot, t_s, channels, archive_root):
    """Same-shot local FREQ/EFIT cache from the documented FDP acquisition."""
    path = archive_root / "raw" / f"{int(shot)}.h5"
    if not path.exists():
        return None, None
    try:
        with h5py.File(path, "r", locking=False) as file:
            if int(file.attrs.get("shot", -1)) != shot:
                raise ValueError("archive shot metadata does not agree")
            setup, eq = file["ecegeom"], file["eq"]
            expected = {f"ECEVS{i + 1:02d}" for i in range(channels)}
            if not expected <= set(file["ece"]):
                raise ValueError(
                    "archive ECE identifiers do not establish channel order"
                )
            frequency = _frequency_vector(setup["FREQ"][...], "GHz", channels)
            if frequency is None or setup.attrs.get("source") != "ELECTRONS":
                raise ValueError("malformed or unverified ELECTRONS setup FREQ")
            clock = np.asarray(eq["gtime"], dtype=float) / 1000
            product = np.asarray(eq["fpol"], dtype=float)[:, -1]
            axis = np.asarray(eq["rmaxis"], dtype=float)
            axis_z = np.asarray(eq["zmaxis"], dtype=float)
            r, z = np.asarray(eq["r"]), np.asarray(eq["z"])
            psirz, qpsi = np.asarray(eq["psirz"]), np.asarray(eq["qpsi"])
            height = float(setup["ECEZH"][()])
            if (
                eq.attrs.get("source") != "efit01"
                or product.shape != clock.shape
                or axis.shape != clock.shape
                or psirz.shape
                not in ((len(clock), len(z), len(r)), (len(clock), len(r), len(z)))
                or qpsi.ndim != 2
                or qpsi.shape[0] != len(clock)
                or not (np.diff(r) > 0).all()
                or not (np.diff(z) > 0).all()
            ):
                raise ValueError("archive EFIT array axes or source do not agree")
            low, high = _q1_major_radii(
                r,
                z,
                psirz,
                axis,
                axis_z,
                np.asarray(eq["ssimag"]),
                np.asarray(eq["ssibry"]),
                qpsi,
                height,
            )
        # EFIT's F(psi=1)=R*Bphi is the boundary/vacuum B0R0 product.
        # It avoids silently treating rmaxis as the Bt reference radius.
        radius = second_harmonic_R(t_s, frequency, (clock, product), 1.0)
        mapped = RadiusGeometry(
            np.asarray(t_s),
            radius,
            align_q(t_s, clock, axis[None])[0],
            align_q(t_s, clock, low[None])[0],
            align_q(t_s, clock, high[None])[0],
        )
        info = {
            "status": "nominal_second_harmonic_R_from_archived_EFIT_F",
            "frequency_source": f"{path}:/ecegeom/FREQ",
            "frequency_hz": frequency.tolist(),
            "frequency_scope": "same_shot",
            "frequency_units": (
                "GHz: documented omnimode ece_fwd reader contract; "
                "stored FREQ units are blank"
            ),
            "channel_order_source": f"{path}:/ece/ECEVS01..{channels:02d}",
            "field_product_source": f"{path}:/eq/fpol[:, -1] (F=R*Bphi)",
            "axis_source": f"{path}:/eq/rmaxis",
            "q1_source": f"{path}:/eq/qpsi and /eq/psirz at ECEZH={height}m",
            "q1_low_supported_samples": int(np.isfinite(mapped.q1_low_R_m).sum()),
            "q1_high_supported_samples": int(np.isfinite(mapped.q1_high_R_m).sum()),
            "formula": "R_m=2*27.992e9*abs(EFIT_F_boundary_Tm)/f_Hz",
            "calibrated_flux": False,
            "limitations": (
                "same-shot nominal second-harmonic vacuum-field mapping; "
                "no relativistic, optical-depth or harmonic-overlap calibration"
            ),
        }
        return mapped, info
    except (OSError, KeyError, ValueError) as error:
        return None, {
            "status": "same_shot_archive_geometry_invalid",
            "archive_source": str(path),
            "error": str(error),
        }


def load_radius_geometry(shot, t_s, channels, paths, *, archive_root=None):
    """Read local frequency/Bt/EFIT-axis metadata; never invoke a resolver."""
    archive_info = None
    if archive_root is not None:
        archived, archive_info = _archived_radius(shot, t_s, channels, archive_root)
        if archived is not None:
            return archived, archive_info
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
    info = {
        "status": "frequency_metadata_unavailable",
        "frequency_source": source,
        "field_reference_source": None,
        "axis_source": None,
        "calibrated_flux": False,
        "formula": "R_m=2*27.992e9*abs(Bt_T)*Rref_m/f_Hz",
        "limitations": "nominal, no relativistic or optical-depth correction",
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
    radius = second_harmonic_R(t_s, frequency, (field.x, field.y[0]), reference)
    axis_R = align_q(t_s, axis.x, axis.y)[0]
    info.update(
        status="nominal_second_harmonic_R",
        reference_radius_m=reference,
        frequency_hz=frequency.tolist(),
        field_source={
            "store": field.attrs.get("store"),
            "locator": field.attrs.get("locator"),
        },
        supported_samples=int(
            (np.isfinite(radius).all(axis=0) & np.isfinite(axis_R)).sum()
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
    if radius.q1_low_R_m is not None and np.isfinite(axis):
        surface = radius.q1_high_R_m if inversion >= axis else radius.q1_low_R_m
        q1 = align_q([time_s], radius.time_s, surface[None])[0, 0]
        if np.isfinite(q1):
            result.update(
                q1_R_m=float(q1), q1_radius_difference_m=inversion - float(q1)
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
    """Choose a shot proxy from a coherent hot profile, or nominal axis geometry.

    The fallback excludes channels 40-47 from *core selection*, as their
    spatial interpretation is unverified and the reviewed shots show harmonic
    overlap. It does not assert that all terminal channels are nonphysical.
    All channels with median Te above 1.5x the coherent core peak are masked.
    This conservative prior and every selected channel are reported per shot.
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
    eligible &= physical
    candidates = np.flatnonzero(eligible)
    if len(candidates) < minimum_channels:
        raise ValueError("insufficient physically plausible ECE core channels")
    nearby = candidates[np.abs(candidates - coherent) <= 3]
    center = int(nearby[np.argmax(median[nearby])])
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
    # A contiguous adjacent band on the lower-index side is the display/ML
    # outer proxy; absent RF metadata this remains an ordered-channel claim.
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
        "outer_status": "adjacent_channel_proxy_unvalidated",
        "shot_median_te_kev": [float(v) if np.isfinite(v) else None for v in median],
        "coherent_core_peak_kev": peak,
        "maximum_channel_to_core": maximum_to_core,
        "implausible_channels": np.flatnonzero(finite & ~physical).tolist(),
        "fallback_excluded_from_core": list(range(stop, len(values))),
        "calibrated": False,
    }
    return ShotCore(core, outer, physical, info)
