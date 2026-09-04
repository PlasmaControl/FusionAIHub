"""The canonical feature namespace.

One name per physical quantity. A model never names an MDSplus node, a corpus
group or an archive column; it names a canonical feature, so every
substitution is written down once, here, and shows up in the model card.

Most quantities are available from more than one place with different shot
coverage, so `sources` is an ordered preference list and the group attribute
`resolver` records which source actually produced a feature for a given
shot. Provenance is therefore per shot, which a static name could not be.

`locators` is parallel to `sources`: the archive column name, the corpus
group name, the MDSplus node. Registry order is not meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Radial grid every profile feature is evaluated on: the
#: `spatial_coordinates` dataset of
#: /projects/EKOLEMEN/simple_ae_predictor/test/test_shots.h5.
RHO_GRID = np.linspace(0.0, 1.0, 33)

#: The 25 ms output grid, in seconds. Identical to that file's `times`
#: dataset (0..5975 ms) and to the archive store's 240 rows.
STEP_S = 0.025
GRID_S = STEP_S * np.arange(240, dtype=np.float64)

KINDS = ("scalar", "profile")
SOURCES = ("archive", "corpus", "fdp")


@dataclass(frozen=True)
class FeatureSpec:
    """What a canonical feature is and where it can come from."""

    name: str
    kind: str
    units: str
    sources: tuple[str, ...]
    locators: tuple[str, ...]
    step: float = 0.0  # storage step in seconds; 0.0 keeps the native rate
    notes: str = ""

    def locator_for(self, source: str) -> str:
        """This feature's name in the given source, or KeyError."""
        for src, loc in zip(self.sources, self.locators):
            if src == source:
                return loc
        raise KeyError(f"{self.name} has no {source} source")


# The `archive` locators are columns of
# /projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/example_191450_183224.h5
# (5,000 shots, 183224..191450, 240 rows at 25 ms), which is the store the
# Phase 1 model's training features were built from. `fdp` locators are
# verified against that store in Task 11 Step 1 before any bulk fetch.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="ip", kind="scalar", units="A",
        sources=("archive", "fdp"),
        locators=("ip", "ip"),
        step=0.001,
        notes="archive column is bit-identical to the model's training input",
    ),
    FeatureSpec(
        name="bt", kind="scalar", units="T",
        sources=("archive", "fdp"),
        locators=("bt", "bt"),
        step=0.001,
        notes="archive column is bit-identical to the model's training input",
    ),
    FeatureSpec(
        name="pinj_total", kind="scalar", units="kW",
        sources=("archive", "corpus"),
        locators=("pinj", "pinj"),
        step=0.001,
        notes="corpus holds 8 beams in W; sum and divide by 1000",
    ),
    FeatureSpec(
        name="tinj_total", kind="scalar", units="N m",
        sources=("archive", "corpus"),
        locators=("tinj", "tinj"),
        step=0.001,
        notes="corpus holds 8 beams already in N m; sum only",
    ),
    FeatureSpec(
        name="ech_power_total", kind="scalar", units="W",
        sources=("archive", "corpus"),
        locators=("ech_pwr", "ech_power"),
        step=0.001,
        notes="corpus holds 12 gyrotrons and can contain NaN channels; "
              "nansum, then NaN or negative -> 0 (upstream rule). Both "
              "sources are in W, MEASURED: the corpus channel sum over "
              "the archive column `EC.PECH`, time-aligned onto the 25 ms "
              "grid, has per-shot median ratios of 0.64 to 1.10 (global "
              "median 1.001 over 1,738 samples on 20 overlap shots, "
              "185950/186114/186508/186985/189295/189549 among them), so "
              "the corpus needs no scaling - the ratio is order 1, not "
              "1e-3 or 1e3. "
              "OPEN, MEASURED: this archive locator is the wrong column. "
              "`ech_pwr` is stored (1, 240) and is ONE gyrotron - "
              "`ech_names` is length 1 ('LEIA', 'LUKE') - 0.544 MW on "
              "shot 186504 at t=3.05 s, where the three gyrotrons that "
              "are on sum to 2.11 MW and `ech_pwr` equals corpus channel "
              "5 to 0.4%. The corpus/`ech_pwr` ratio is one stable value "
              "per shot in 1.88..4.01 (IQR 0.06 within a shot, and the "
              "same under nearest-sample, forward-window and centred-"
              "window alignment, so it is not a duty-cycle artefact). "
              "`EC.PECH` is the archive's total: it matches the model's "
              "own training column x0[:, 9] in magnitude (2.027e6 vs "
              "2.042e6 on 186504, 9.69e5 vs 9.71e5 on 185950) where "
              "`ech_pwr` is 4x smaller, and it is present on 174 of 200 "
              "sampled archive shots against `ech_pwr`'s 52. Switching "
              "the locator also invalidates the negative-baseline "
              "statistics measured on `ech_pwr` in "
              "d3d_tearing_onset_cnn1d/spec.py (18.9% negative readings, "
              "min -4,086 W: `EC.PECH` is NaN off-window instead), so it "
              "is left for Task 15 to settle with those together",
    ),
    FeatureSpec(
        name="r0", kind="scalar", units="m",
        sources=("archive", "fdp"),
        locators=("rmaxis_EFIT01", r"\efit01::top.results.geqdsk:rmaxis"),
        notes="stands in for R0_EFITRT1; measured 8.8e-3 median relative "
              "difference on shot 185945",
    ),
    FeatureSpec(
        name="kappa", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("kappa_EFIT01", r"\efit01::top.results.aeqdsk:kappa"),
        notes="stands in for kappa_EFITRT1; measured 3.1e-3",
    ),
    FeatureSpec(
        name="tritop", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("tritop_EFIT01", r"\efit01::top.results.aeqdsk:tritop"),
        notes="bit-identical to the training input",
    ),
    FeatureSpec(
        name="tribot", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("tribot_EFIT01", r"\efit01::top.results.aeqdsk:tribot"),
        notes="bit-identical to the training input",
    ),
    FeatureSpec(
        name="gapin", kind="scalar", units="m",
        sources=("archive", "fdp"),
        locators=("gapin_EFIT01", r"\efit01::top.results.aeqdsk:gapin"),
        notes="bit-identical to the training input",
    ),
    FeatureSpec(
        name="betan", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("betan_EFIT01", r"\efit01::top.results.aeqdsk:betan"),
        notes="not a model input; the model predicts it, and validation "
              "compares against it",
    ),
    FeatureSpec(
        name="qpsi", kind="profile", units="",
        sources=("archive", "fdp"),
        locators=("qpsi_EFIT01", r"\efit01::top.results.geqdsk:qpsi"),
        notes="stands in for qpsi_EFITRT1; measured 6.4e-2. The model "
              "consumes 1/qpsi, applied by the adapter, not here",
    ),
    FeatureSpec(
        name="pres", kind="profile", units="Pa",
        sources=("archive", "fdp"),
        locators=("pres_EFIT01", r"\efit01::top.results.geqdsk:pres"),
        notes="bit-identical to the training input",
    ),
    FeatureSpec(
        name="ne_zipfit", kind="profile", units="1e19 m^-3",
        sources=("archive", "fdp"),
        locators=("zipfit_edensfit_rho", r"\ZIPFIT01::TOP.PROFILES.EDENSFIT"),
        notes="stands in for thomson_density_mtanh_1d; measured 2.0e-1 "
              "median relative difference, correlation 0.984. Our own mtanh "
              "fit to raw Thomson is Phase 2",
    ),
    FeatureSpec(
        name="te_zipfit", kind="profile", units="keV",
        sources=("archive", "fdp"),
        locators=("zipfit_etempfit_rho", r"\ZIPFIT01::TOP.PROFILES.ETEMPFIT"),
        notes="stands in for thomson_temp_mtanh_1d; measured 1.8e-1, 0.991",
    ),
    FeatureSpec(
        name="rot_zipfit", kind="profile", units="krad/s",
        sources=("archive", "fdp"),
        locators=("zipfit_trotfit_rho", r"\ZIPFIT01::TOP.PROFILES.TROTFIT"),
        notes="stands in for cer_rot_csaps_1d; measured 1.7e-1, 0.980",
    ),
    FeatureSpec(
        name="ech_rho", kind="scalar", units="",
        sources=("archive",),
        locators=("EC.RHO_ECH",),
        notes="TORBEAM deposition location; 0 where unavailable, which the "
              "upstream training filter admitted (x0[:, 10] >= 0)",
    ),
)

_BY_NAME = {f.name: f for f in FEATURES}


def by_name(name: str) -> FeatureSpec:
    """The spec for a canonical name, or KeyError."""
    return _BY_NAME[name]


def by_source(source: str) -> tuple[FeatureSpec, ...]:
    """Every feature a single resolver can produce."""
    return tuple(f for f in FEATURES if source in f.sources)
