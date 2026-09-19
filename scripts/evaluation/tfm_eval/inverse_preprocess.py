"""Invert ``TokamakH5Dataset`` preprocessing so eval plots use physical units.

The dataset transforms every modality in
``TokamakH5Dataset._apply_preprocessing``
(``src/tokamak_foundation_model/data/data_loader.py``); there is no inverse
anywhere in the training code.  This module derives the exact forward
formulas from that method and provides their inverses.

Forward transforms (as implemented in the dataset, not its docstrings)
----------------------------------------------------------------------
With per-channel stats ``mean/std/min_val/max_val`` indexed by
``channels_to_use`` (stats arrays are stored for the FULL channel count and
sliced at apply time), ``eps = PreprocessConfig.eps = 1e-8``:

``standardize``      ``z = (x - mean) / clamp(std, min=1e-3)``
``normalize``        ``z = (x - min_val) / (max_val - min_val + eps)``
``log``              ``z = log10(clip(x, -0.99, None) + 1)``
``log_standardize``  ``u = log10(clip(x, -0.99, None) + 1)``;
                     ``z = (u - mean) / clamp(std, min=1e-3)``
``log_normalize``    ``u = log10(clip(x, -0.99, None) + 1)``;
                     ``z = (u - min_val) / (max_val - min_val + eps)``
``none``             identity

Note that ``standardize`` divides by ``std.clamp(min=1e-3)`` — NOT
``std + eps`` as the ``PreprocessConfig`` docstring claims.

Stats file layout (``_update_preprocessing_stats``): ``{signal: {"raw":
{...}, "log": {...}}}``; ``log_standardize`` / ``log_normalize`` read the
``"log"`` bucket, every other method reads ``"raw"`` (a legacy flat dict
without buckets is also accepted).  NaNs in the stats are replaced exactly
as the dataset does: ``mean -> 0``, ``std -> 1``, ``min_val -> 0``,
``max_val -> 1`` (infinities are left untouched, again mirroring the
dataset).

Inverse caveats
---------------
* The log-family clip means values below ``-0.99`` in the raw data are
  destroyed by the dataset itself; the inverse is exact only for
  ``x_raw >= -0.99``.  STFT magnitudes, densities, temperatures, and pixel
  values satisfy this; raw negative excursions of ``sxr`` / ``vib`` /
  ``bolo_raw`` (method ``log``) do not round-trip through the *dataset*.
* For spectrogram signals (``ece``, ``co2``, ``bes``, ``mhr``, ...) the
  "physical" quantity recovered here is the STFT magnitude spectrogram
  *before* preprocessing; the STFT itself is not inverted.
* Video (``tangtv``): the dataset-level transform is ``method='none'``,
  which is all this class inverts.  The Stage-1/2 trainer additionally
  applies a SECOND runtime per-(B, C) z-score over (T, H, W)
  (``_video_standardize_per_bc`` in ``scripts/training/train_e2e_stage1.py``,
  ``sd.clamp(min=1.0)``) whose statistics are computed on the fly and never
  persisted — undoing that one requires the ``(mu, sd)`` the trainer
  returned at runtime.

Validation (run ``python scripts/evaluation/tfm_eval/inverse_preprocess.py``)
------------------------------------------------------------------------------
1. Synthetic: fake per-channel stats (including NaN entries) for all five
   methods, plus ``channels_to_use`` as slices (``mhr`` 2:8, ``bes`` 48:64,
   ``filterscopes`` 0:8) and as an int list (``tangtv`` ``[4, 6]`` via
   ``dataset_kwargs``); asserts ``inverse(apply(x)) == x`` to rtol 1e-4.
2. Real data: one chunk of shot ``190000_processed.h5`` loaded through
   ``TokamakH5Dataset`` twice — once with the production
   ``preprocessing_stats.pt`` applied, once with preprocessing disabled by
   setting every per-instance ``config.preprocess.method = 'none'`` after
   construction (the dataset has no built-in disable flag; instance configs
   are deep copies so this is safe).  Asserts ``inverse(processed)`` matches
   the unprocessed tensors and ``apply(unprocessed)`` matches the processed
   tensors (rtol 1e-3, element-mask-valid positions only) for
   ``ts_core_density`` (log_standardize), ``filterscopes`` / ``pin``
   (standardize), ``gas_flow`` (none), and ``ece`` (STFT + log_standardize).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":  # script bootstrap: repo src + scripts/evaluation
    _HERE = Path(__file__).resolve()
    sys.path.insert(0, str(_HERE.parents[3] / "src"))
    sys.path.insert(0, str(_HERE.parents[1]))

import copy
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

__all__ = ["InversePreprocessor", "DEFAULT_STATS_PATH"]

DEFAULT_STATS_PATH = (
    "/lustre/orion/fus187/proj-shared/foundation_model_meta/"
    "preprocessing_stats.pt"
)
DEFAULT_SHOT_PATH = (
    "/lustre/orion/fus187/proj-shared/foundation_model/190000_processed.h5"
)

# Mirrors of constants hard-coded inside TokamakH5Dataset._apply_preprocessing
_LOG_METHODS = {"log_standardize", "log_normalize"}
_STD_CLAMP_MIN = 1e-3  # std.clamp(min=1e-3) in the (log_)standardize paths
_LOG_CLIP_MIN = -0.99  # np.clip(arr, a_min=-.99, ...) before log10(x + 1)

ChannelSelection = Union[slice, Sequence[int], None]


def _nan_filled(values, fill: float) -> np.ndarray:
    """Copy *values* to float64 and replace NaNs, as the dataset does."""
    arr = np.array(values, dtype=np.float64)
    arr[np.isnan(arr)] = fill
    return arr


def _selection_length(ch: ChannelSelection, n_full: int) -> int:
    """Number of channels after applying *ch* to *n_full* raw channels."""
    if ch is None:
        return n_full
    if isinstance(ch, slice):
        return len(range(*ch.indices(n_full)))
    return len(list(ch))


def _select_channels(arr: np.ndarray, ch: ChannelSelection) -> np.ndarray:
    """Index a full-channel stats array exactly like the dataset does."""
    if ch is None:
        return arr
    if isinstance(ch, slice):
        return arr[ch]
    return arr[list(ch)]


@dataclass
class _SignalSpec:
    """Resolved preprocessing recipe for one modality.

    Parameters
    ----------
    name : str
        Modality name (key in the dataset batch dict).
    kind : str
        ``'timeseries'`` (C, T), ``'spectrogram'`` (C, F, T) or
        ``'video'`` (C, T, H, W).
    configured_method : str
        Method from ``SIGNAL_CONFIGS`` / ``MOVIE_CONFIGS`` (possibly
        overridden via ``dataset_kwargs``).
    method : str
        Effective method after stats-availability degradation, mirroring
        the dataset (missing stats: ``standardize``/``normalize`` fall back
        to identity, ``log_standardize``/``log_normalize`` to plain
        ``log`` — because the dataset applies the log in-place before
        checking stats).
    n_trailing : int
        Number of tensor dims after the channel axis (1 / 2 / 3).
    n_channels : int
        Channel count AFTER ``channels_to_use`` selection.
    channels_to_use : slice or sequence of int or None
        Selection applied to the full-channel stats arrays.
    eps : float
        ``PreprocessConfig.eps`` (denominator guard for normalize).
    mean, std, min_val, max_val : numpy.ndarray or None
        Post-selection per-channel stats, float64, NaN-filled.
    error : str or None
        Deferred stats problem; raised when the signal is actually used.
    """

    name: str
    kind: str
    configured_method: str
    method: str
    n_trailing: int
    n_channels: int
    channels_to_use: ChannelSelection
    eps: float
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    error: Optional[str] = None


class InversePreprocessor:
    """Map tensors between the model's z-space and physical units.

    Builds a per-signal ``(method, channels_to_use, stats)`` table from
    ``TokamakH5Dataset.SIGNAL_CONFIGS`` / ``MOVIE_CONFIGS`` plus a
    ``preprocessing_stats.pt`` file, replicating the dataset's stats
    injection (``_update_preprocessing_stats``) and transform
    (``_apply_preprocessing``) semantics exactly — see the module
    docstring for the formulas.

    Parameters
    ----------
    stats_path : str
        Path to the ``preprocessing_stats.pt`` produced by
        ``compute_preprocessing_stats`` (loaded with
        ``torch.load(..., weights_only=False)``).
    dataset_kwargs : dict or None, optional
        Optional per-signal overrides matching how the eval datasets were
        built.  Recognised keys (all others are ignored so a dataset kwargs
        dict can be passed through unchanged):

        ``'channels_to_use'``
            ``{signal_name: slice | sequence of int | None}`` — e.g.
            ``{'tangtv': [4, 6]}``.
        ``'preprocess_method'``
            ``{signal_name: method_str}`` — e.g. if an eval dataset
            enabled video standardization.

    Attributes
    ----------
    specs : dict[str, _SignalSpec]
        Resolved per-signal table (all 30 signals + 2 movies).
    """

    def __init__(self, stats_path: str, dataset_kwargs: dict | None = None):
        dataset_kwargs = dict(dataset_kwargs or {})
        ch_overrides: dict = dataset_kwargs.get("channels_to_use") or {}
        method_overrides: dict = dataset_kwargs.get("preprocess_method") or {}

        self.stats_path = str(stats_path)
        self._stats: dict = torch.load(self.stats_path, weights_only=False)

        self.specs: dict[str, _SignalSpec] = {}
        for config in copy.deepcopy(TokamakH5Dataset.SIGNAL_CONFIGS):
            self.specs[config.name] = self._build_spec(
                name=config.name,
                kind="spectrogram" if config.apply_stft else "timeseries",
                method=method_overrides.get(
                    config.name, config.preprocess.method
                ),
                n_trailing=2 if config.apply_stft else 1,
                channels_to_use=ch_overrides.get(
                    config.name, config.channels_to_use
                ),
                n_channels_full=config.num_channels,
                eps=config.preprocess.eps,
            )
        for config in copy.deepcopy(TokamakH5Dataset.MOVIE_CONFIGS):
            self.specs[config.name] = self._build_spec(
                name=config.name,
                kind="video",
                method=method_overrides.get(
                    config.name, config.preprocess.method
                ),
                n_trailing=3,
                channels_to_use=ch_overrides.get(
                    config.name, config.channels_to_use
                ),
                n_channels_full=config.channels,
                eps=config.preprocess.eps,
            )

    # ── table construction ────────────────────────────────────────────

    def _build_spec(
        self,
        *,
        name: str,
        kind: str,
        method: str,
        n_trailing: int,
        channels_to_use: ChannelSelection,
        n_channels_full: int,
        eps: float,
    ) -> _SignalSpec:
        """Resolve one signal: select stats bucket, index channels, degrade."""
        n_channels = _selection_length(channels_to_use, n_channels_full)
        spec = _SignalSpec(
            name=name,
            kind=kind,
            configured_method=method,
            method=method,
            n_trailing=n_trailing,
            n_channels=n_channels,
            channels_to_use=channels_to_use,
            eps=eps,
        )
        if method in ("none", "log"):
            return spec  # no stats consumed

        entry = self._stats.get(name, {})
        if "raw" in entry or "log" in entry:
            bucket_key = "log" if method in _LOG_METHODS else "raw"
            bucket = entry.get(bucket_key, {})
        else:
            bucket = entry  # legacy flat format

        def load(key: str, fill: float) -> Optional[np.ndarray]:
            if key not in bucket:
                return None
            try:
                full = _nan_filled(bucket[key], fill)
                sel = np.atleast_1d(_select_channels(full, channels_to_use))
            except (IndexError, TypeError, ValueError) as exc:
                spec.error = (
                    f"stats[{name!r}].{key} (shape "
                    f"{np.shape(bucket[key])}) cannot be indexed by "
                    f"channels_to_use={channels_to_use!r}: {exc}"
                )
                return None
            if sel.shape != (n_channels,):
                spec.error = (
                    f"stats[{name!r}].{key}: got {sel.shape[0]} entries "
                    f"after channels_to_use={channels_to_use!r}, expected "
                    f"{n_channels}"
                )
                return None
            return sel

        spec.mean = load("mean", 0.0)
        spec.std = load("std", 1.0)
        spec.min_val = load("min_val", 0.0)
        spec.max_val = load("max_val", 1.0)
        if spec.error is not None:
            return spec

        # Mirror _apply_preprocessing's behaviour when stats are missing:
        # the affine part is skipped (with a warning), but for the log
        # family the in-place log10 has already been applied.
        missing = (
            spec.mean is None or spec.std is None
            if method in ("standardize", "log_standardize")
            else spec.min_val is None or spec.max_val is None
        )
        if missing:
            spec.method = "log" if method in _LOG_METHODS else "none"
            warnings.warn(
                f"{name}: no usable stats for {method!r} in "
                f"{self.stats_path}; treating as {spec.method!r} "
                "(matches the dataset's fallback)",
                stacklevel=3,
            )
        return spec

    # ── helpers ───────────────────────────────────────────────────────

    def _get_spec(self, name: str) -> _SignalSpec:
        try:
            spec = self.specs[name]
        except KeyError:
            raise KeyError(
                f"unknown signal {name!r}; known signals: "
                f"{sorted(self.specs)}"
            ) from None
        if spec.error is not None:
            raise ValueError(f"stats for {name!r} unusable: {spec.error}")
        return spec

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        t = torch.as_tensor(x)
        if not torch.is_floating_point(t):
            t = t.to(torch.float32)
        return t

    def _validate_channels(self, spec: _SignalSpec, x: torch.Tensor) -> None:
        layouts = {1: "(..., C, T)", 2: "(..., C, F, T)", 3: "(..., C, T, H, W)"}
        need = spec.n_trailing + 1
        if x.ndim < need:
            raise ValueError(
                f"{spec.name}: expected {layouts[spec.n_trailing]} with "
                f"C={spec.n_channels}, got shape {tuple(x.shape)}"
            )
        ax = x.ndim - need
        if x.shape[ax] != spec.n_channels:
            raise ValueError(
                f"{spec.name}: channel axis (dim {ax - x.ndim} of "
                f"{layouts[spec.n_trailing]}) has size {x.shape[ax]}, "
                f"expected {spec.n_channels} (post-channels_to_use="
                f"{spec.channels_to_use!r} count)"
            )

    def _stat(
        self, spec: _SignalSpec, arr: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Per-channel stats reshaped to (C, 1, ...) for broadcasting."""
        t = torch.as_tensor(arr, dtype=torch.float64, device=device)
        return t.reshape(spec.n_channels, *((1,) * spec.n_trailing))

    # ── public API ────────────────────────────────────────────────────

    def inverse(
        self, name: str, x: "torch.Tensor | np.ndarray"
    ) -> torch.Tensor:
        """Map a processed (z-space) tensor back to physical units.

        Parameters
        ----------
        name : str
            Signal name (a ``SIGNAL_CONFIGS`` / ``MOVIE_CONFIGS`` entry).
        x : torch.Tensor or numpy.ndarray
            Processed data shaped ``(..., C, T)`` for time series,
            ``(..., C, F, T)`` for spectrograms, or ``(..., C, T, H, W)``
            for video, where ``C`` is the post-``channels_to_use`` channel
            count.  Leading batch dims broadcast freely.  Never mutated.

        Returns
        -------
        torch.Tensor
            Physical-unit tensor, same shape, same floating dtype as *x*
            (integer inputs are promoted to float32).  For spectrogram
            signals this is the STFT magnitude, not the raw waveform.
            Exact for raw values ``>= -0.99`` (the dataset's log-path
            clip destroys anything below).
        """
        spec = self._get_spec(name)
        t = self._to_tensor(x)
        if spec.method == "none":
            return t.clone()
        self._validate_channels(spec, t)
        out_dtype = t.dtype
        z = t.to(torch.float64)

        if spec.method == "standardize":
            std = self._stat(spec, spec.std, z.device)
            mean = self._stat(spec, spec.mean, z.device)
            phys = z * std.clamp(min=_STD_CLAMP_MIN) + mean
        elif spec.method == "normalize":
            min_val = self._stat(spec, spec.min_val, z.device)
            max_val = self._stat(spec, spec.max_val, z.device)
            phys = z * (max_val - min_val + spec.eps) + min_val
        elif spec.method == "log":
            phys = torch.pow(10.0, z) - 1.0
        elif spec.method == "log_standardize":
            std = self._stat(spec, spec.std, z.device)
            mean = self._stat(spec, spec.mean, z.device)
            u = z * std.clamp(min=_STD_CLAMP_MIN) + mean
            phys = torch.pow(10.0, u) - 1.0
        elif spec.method == "log_normalize":
            min_val = self._stat(spec, spec.min_val, z.device)
            max_val = self._stat(spec, spec.max_val, z.device)
            u = z * (max_val - min_val + spec.eps) + min_val
            phys = torch.pow(10.0, u) - 1.0
        else:  # pragma: no cover - unreachable with known configs
            raise ValueError(f"{name}: unknown method {spec.method!r}")
        return phys.to(out_dtype)

    def apply(self, name: str, x: "torch.Tensor | np.ndarray") -> torch.Tensor:
        """Physical units -> processed z-space (the dataset's transform).

        Out-of-place mirror of ``TokamakH5Dataset._apply_preprocessing``
        (which operates in-place), including the ``clip(x, -0.99, None)``
        before the log.  Needed for round-trip tests and for actuator
        scans that perturb in physical units and re-normalize.

        Parameters
        ----------
        name : str
            Signal name.
        x : torch.Tensor or numpy.ndarray
            Physical-unit data with the same layout rules as
            :meth:`inverse`.  Never mutated.

        Returns
        -------
        torch.Tensor
            Processed tensor, same shape, same floating dtype as *x*.
        """
        spec = self._get_spec(name)
        t = self._to_tensor(x)
        if spec.method == "none":
            return t.clone()
        self._validate_channels(spec, t)
        out_dtype = t.dtype
        v = t.to(torch.float64)

        if spec.method in ("log", "log_standardize", "log_normalize"):
            v = torch.log10(v.clamp(min=_LOG_CLIP_MIN) + 1.0)
        if spec.method == "log":
            z = v
        elif spec.method in ("standardize", "log_standardize"):
            std = self._stat(spec, spec.std, v.device)
            mean = self._stat(spec, spec.mean, v.device)
            z = (v - mean) / std.clamp(min=_STD_CLAMP_MIN)
        else:  # normalize / log_normalize
            min_val = self._stat(spec, spec.min_val, v.device)
            max_val = self._stat(spec, spec.max_val, v.device)
            z = (v - min_val) / (max_val - min_val + spec.eps)
        return z.to(out_dtype)

    def physical_scale(self, name: str) -> torch.Tensor:
        """Per-channel scale mapping 1 z-unit to physical units.

        Parameters
        ----------
        name : str
            Signal name.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(C,)`` (post-``channels_to_use``):

            - ``standardize``: ``clamp(std, 1e-3)`` — physical units per
              z-unit.
            - ``normalize``: ``max_val - min_val + eps``.
            - ``log_standardize`` / ``log_normalize``: the same quantities
              of the LOG-space stats, i.e. log10-decades of ``(x + 1)``
              per z-unit.  The physical map is nonlinear there: a
              perturbation ``dz`` multiplies ``(x + 1)`` by
              ``10 ** (dz * scale)``.
            - ``log``: ones (1 z-unit = 1 decade of ``x + 1``).
            - ``none``: ones.
        """
        spec = self._get_spec(name)
        if spec.method in ("standardize", "log_standardize"):
            scale = torch.as_tensor(spec.std, dtype=torch.float64).clamp(
                min=_STD_CLAMP_MIN
            )
        elif spec.method in ("normalize", "log_normalize"):
            scale = torch.as_tensor(
                spec.max_val - spec.min_val + spec.eps, dtype=torch.float64
            )
        else:  # 'none' and 'log'
            scale = torch.ones(spec.n_channels, dtype=torch.float64)
        return scale.to(torch.float32)


# ── self-test ─────────────────────────────────────────────────────────


def _max_rel(
    a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> tuple[float, float, int]:
    """Return (max relative error, atol used, n compared) of a vs b."""
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    if mask is None:
        mask = torch.ones_like(b64, dtype=torch.bool)
    mask = mask & torch.isfinite(a64) & torch.isfinite(b64)
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0, 0
    ref_max = float(b64[mask].abs().max())
    atol = 1e-6 * ref_max + 1e-30
    rel = ((a64 - b64).abs() / (b64.abs() + atol))[mask]
    return float(rel.max()), atol, n


def _run_synthetic() -> bool:
    """Round-trip fake stats through every method / channel-selection mode."""
    import tempfile

    rng = np.random.default_rng(0)

    def entry(c: int) -> dict:
        return {
            "raw": {
                "mean": rng.normal(0.0, 2.0, c),
                "std": rng.uniform(0.5, 3.0, c),
                "min_val": rng.uniform(-1.0, 0.0, c),
                "max_val": rng.uniform(1.0, 3.0, c),
            },
            "log": {
                "mean": rng.normal(0.5, 0.3, c),
                "std": rng.uniform(0.1, 1.0, c),
                "min_val": rng.uniform(-0.5, 0.0, c),
                "max_val": rng.uniform(1.0, 2.0, c),
            },
        }

    fake = {
        name: entry(c)
        for name, c in [
            ("pin", 8),
            ("ts_core_density", 44),
            ("gas_flow", 11),
            ("mse", 69),
            ("mhr", 8),
            ("filterscopes", 104),
            ("bes", 64),
            ("tangtv", 7),
        ]
    }
    # NaN stats must be filled like the dataset does (mean->0, std->1).
    fake["pin"]["raw"]["mean"][3] = np.nan
    fake["pin"]["raw"]["std"][5] = np.nan

    with tempfile.TemporaryDirectory() as tmp:
        stats_path = str(Path(tmp) / "fake_stats.pt")
        torch.save(fake, stats_path)
        with warnings.catch_warnings():
            # The fake file intentionally covers only 8 signals; the
            # missing-stats fallback warnings for the rest are expected.
            warnings.simplefilter("ignore", UserWarning)
            ip = InversePreprocessor(
                stats_path,
                dataset_kwargs={
                    "preprocess_method": {
                        "gas_flow": "normalize",
                        "mse": "log_normalize",
                        "tangtv": "standardize",
                    },
                    "channels_to_use": {"tangtv": [4, 6]},
                },
            )

    # (name, shape, dtype) — shapes follow each signal's canonical layout;
    # x_raw > 0 so the log-path clip is inactive.
    cases = [
        ("pin", (2, 8, 50), torch.float64),  # standardize, NaN stats
        ("pin", (8, 50), torch.float32),  # float32 path, unbatched
        ("ts_core_density", (44, 30), torch.float64),  # log_standardize
        ("sxr", (3, 320, 20), torch.float64),  # log (no stats)
        ("gas_flow", (11, 40), torch.float64),  # normalize (override)
        ("mse", (2, 69, 25), torch.float64),  # log_normalize (override)
        ("mhr", (6, 12, 10), torch.float64),  # slice(2, 8) -> C=6, (C,F,T)
        ("filterscopes", (5, 8, 7), torch.float64),  # slice(0, 8)
        ("bes", (2, 16, 8, 9), torch.float64),  # slice(48, 64), (B,C,F,T)
        ("tangtv", (2, 2, 3, 4, 5), torch.float64),  # list [4,6], video
        ("gas_raw", (11, 12), torch.float64),  # method 'none' identity
    ]
    ok = True
    for name, shape, dtype in cases:
        gen = torch.Generator().manual_seed(hash(name) % (2**31))
        x = (torch.rand(*shape, generator=gen, dtype=torch.float64) * 99 + 0.5)
        x = x.to(dtype)
        x_orig = x.clone()
        z = ip.apply(name, x)
        x2 = ip.inverse(name, z)
        rel, _, _ = _max_rel(x2, x)
        passed = bool(
            torch.allclose(x2, x, rtol=1e-4, atol=1e-6 * float(x.abs().max()))
        )
        passed &= bool(torch.equal(x, x_orig))  # no in-place mutation
        ok &= passed
        spec = ip.specs[name]
        print(
            f"[synthetic] {name:16s} {spec.method:16s} shape={str(shape):20s}"
            f" {str(dtype).replace('torch.', ''):9s}"
            f" max_rel={rel:.3e}  {'PASS' if passed else 'FAIL'}"
        )

    # apply(inverse(z)) for the affine methods (z-space round trip).
    for name in ("pin", "gas_flow"):
        z = torch.randn(4, ip.specs[name].n_channels, 17, dtype=torch.float64)
        z2 = ip.apply(name, ip.inverse(name, z))
        rel, _, _ = _max_rel(z2, z)
        passed = bool(torch.allclose(z2, z, rtol=1e-4, atol=1e-9))
        ok &= passed
        print(
            f"[synthetic] {name:16s} z-space round trip       "
            f" max_rel={rel:.3e}  {'PASS' if passed else 'FAIL'}"
        )

    # NaN handling: pin std[5] was NaN -> filled with 1.0.
    scale = ip.physical_scale("pin")
    passed = scale.shape == (8,) and float(scale[5]) == 1.0
    ok &= passed
    print(
        f"[synthetic] physical_scale NaN std -> 1.0"
        f"                          {'PASS' if passed else 'FAIL'}"
    )

    # Wrong channel count must raise.
    try:
        ip.inverse("mhr", torch.zeros(8, 12, 10, dtype=torch.float64))
        passed = False
    except ValueError:
        passed = True
    ok &= passed
    print(
        f"[synthetic] wrong channel count raises ValueError"
        f"                  {'PASS' if passed else 'FAIL'}"
    )
    return ok


def _run_real(stats_path: str, shot_path: str) -> bool:
    """Round-trip one real chunk: dataset-with-stats vs dataset-monkeypatched.

    The reference ("physical") tensors come from a second
    ``TokamakH5Dataset`` on the same shot whose per-instance configs are
    all forced to ``preprocess.method = 'none'`` after construction, i.e.
    the exact pre-preprocessing tensors (post channel-select, resample,
    NaN->0, and STFT for spectrogram signals).  Comparison is restricted
    to the dataset's own ``{name}_mask`` positions — masked elements are
    zero-filled *after* preprocessing and are not invertible by design.
    """
    signals = ["ts_core_density", "filterscopes", "pin", "gas_flow", "ece"]
    stats = torch.load(stats_path, weights_only=False)
    common = dict(
        chunk_duration_s=0.5, prediction_mode=False, input_signals=signals
    )
    ds_proc = TokamakH5Dataset(shot_path, preprocessing_stats=stats, **common)
    ds_ref = TokamakH5Dataset(shot_path, preprocessing_stats=None, **common)
    for cfg in ds_ref.signal_configs + ds_ref.movie_configs:
        cfg.preprocess.method = "none"

    ip = InversePreprocessor(stats_path)

    # Pick a mid-shot chunk where every signal has valid data.
    sample_p = sample_r = None
    for idx in (4, 3, 5, 2, 1, 0):
        if idx >= len(ds_proc):
            continue
        cand = ds_proc[idx]
        if all(
            cand[f"{s}_valid"] > 0 and bool(cand[f"{s}_mask"].any())
            for s in signals
        ):
            sample_p, sample_r = cand, ds_ref[idx]
            print(f"[real] shot {Path(shot_path).name}, chunk idx={idx}")
            break
    if sample_p is None:
        print("[real] FAIL: no chunk with valid data for all signals")
        return False

    ok = True
    for s in signals:
        proc = sample_p[s]
        ref = sample_r[s]
        mask = sample_p[f"{s}_mask"] & sample_r[f"{s}_mask"]
        inv = ip.inverse(s, proc)
        rel_inv, atol, n = _max_rel(inv, ref, mask)
        pass_inv = bool(
            ((inv - ref).abs() <= 1e-3 * ref.abs() + atol)[mask].all()
        )
        fwd = ip.apply(s, ref)
        rel_fwd, _, _ = _max_rel(fwd, proc, mask)
        pass_fwd = bool(
            ((fwd - proc).abs() <= 1e-3 * proc.abs() + 1e-5)[mask].all()
        )
        ok &= pass_inv and pass_fwd
        frac = n / mask.numel()
        print(
            f"[real] {s:16s} {ip.specs[s].method:16s} n={n:>8d} "
            f"({frac:5.1%} valid)  inverse max_rel={rel_inv:.3e} "
            f"{'PASS' if pass_inv else 'FAIL'}  "
            f"apply max_rel={rel_fwd:.3e} {'PASS' if pass_fwd else 'FAIL'}"
        )
    return ok


def main() -> int:
    """Run the synthetic and real-data self-tests; return exit code."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH)
    parser.add_argument("--shot", default=DEFAULT_SHOT_PATH)
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="only run the synthetic round-trip (no HDF5 access)",
    )
    args = parser.parse_args()

    ok = _run_synthetic()
    if not args.skip_real:
        if Path(args.stats).exists() and Path(args.shot).exists():
            ok &= _run_real(args.stats, args.shot)
        else:
            print(
                f"[real] SKIPPED: {args.stats} or {args.shot} not found "
                "(pass --skip-real to silence)"
            )
    print("ALL PASS" if ok else "FAILURES — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
