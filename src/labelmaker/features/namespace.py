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
        locators=("EC.PECH", "ech_power"),
        step=0.001,
        notes="corpus holds 12 gyrotrons and can contain NaN channels; "
              "nansum, then NaN or negative -> 0 (upstream rule). "
              "The archive locator is `EC.PECH`, the machine total, and NOT "
              "`ech_pwr`, which holds ONE gyrotron: it is stored (1, 240) "
              "and `ech_names` is length 1 ('LEIA', 'LUKE' or 'TINMAN'). "
              "ESTABLISHED by exact row alignment to the model's own "
              "training array: matching x0.npy rows to archive time indices "
              "on the five columns that are bit-identical (bt, ip, tritop, "
              "tribot, gapin) resolves 9,805 of 9,805 rows to a unique "
              "index, so this store is the row-level provenance of the "
              "training set. At those aligned rows, powered only "
              "(x0[:, 9] > 0.2 MW, n=1,361), `EC.PECH` has median relative "
              "error 0.097% with 91.2% of rows inside 1%, while `ech_pwr` "
              "has median relative error 61% with 0.0% inside 1% and is "
              "~2.6x smaller. Upstream's own name for the input is "
              "`ech_pwr_total` (inputs_0d[9], "
              "simple_ae_predictor/models/rt_multi_io/train.py:31), which is "
              "not a column of this store - hence the 0.1% residue rather "
              "than bit-equality: it was fetched and resampled separately. "
              "Coverage: `EC.PECH` on 2,409 of 5,000 archive shots against "
              "`ech_pwr`'s 1,636, and 1,299 of the 1,497 PoC-pool shots "
              "against 384. That reverses on the early shot range, where "
              "the 49 pool shots carrying only `ech_pwr` live: below 187000 "
              "it is `ech_pwr` 1,488 against `EC.PECH` 1,000. "
              "Both sources are in W, so no scaling - the ratio is order 1, "
              "not 1e-3 or 1e3. But they are NOT interchangeable, and this "
              "is OPEN for Task 15: the corpus channel sum over `EC.PECH`, "
              "time-aligned onto the 25 ms grid over a RANDOM 60 overlap "
              "shots (n=5,186), has global median 0.779, per-shot medians "
              "spanning 0.444 to 1.406, with only 2 of 60 inside 0.3% of "
              "unity and 39 of 60 below 0.95. The corpus under-reports by "
              "~22% at the median because its 12-channel `ech_power` is "
              "missing gyrotrons that `EC.PECH` counts. An earlier note "
              "here claimed median 1.0051 with 16 of 20 shots inside 0.3%; "
              "that sample was the FIRST 20 overlap shots by number, and "
              "the agreement is real only at the low end of the range",
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
