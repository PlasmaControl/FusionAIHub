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

#: `waveform` is a raw, native-rate record kept whole - millions of samples,
#: several channels - for a model that transforms the signal itself rather
#: than reading a value off the 25 ms grid. `InputSpec.build` hands one
#: straight to the adapter instead of sampling it (see `models.base`).
KINDS = ("scalar", "profile", "waveform")
SOURCES = ("archive", "corpus", "fdp")

#: How a stored array from each source must be sampled at model-input time
#: An archive-served field is already the archive's own 50 ms boxcar
#: (see `models.base.ARCHIVE_WINDOW_S`) and must be read nearest-sample at
#: its own stamp - windowing it again would double-average it. A corpus- or
#: fdp-served field is a true-time, high-rate record and must be turned into
#: the SAME 50 ms mean to be comparable - MEASURED in Task 15 to beat every
#: other convention by one to two orders of magnitude on both paths (see
#: `pinj_total`'s notes below and the Task 15 report's four-candidate scan).
#: `models.base.sample_by_resolver` and `validate`'s ECH diagnostic both read
#: this dict rather than each hand-coding the rule, and
#: `test_every_source_has_a_declared_sampling_convention` asserts every
#: entry of `SOURCES` has one - the same shape commit 38b137b gave the
#: correction/fill split, so a fourth source added to `SOURCES` cannot
#: silently fall through to the wrong convention (or to no convention) the
#: way an untotaled `resolver in (...)` check could.
SAMPLING_BY_SOURCE: dict[str, str] = {
    "archive": "nearest",
    "corpus": "window",
    "fdp": "window",
}


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
# Phase 1 model's training features were built from. Every `fdp` locator
# below was fetched live and compared against that store before any bulk
# fetch; the per-feature figures are in `resolve_fdp`'s docstring, which also
# records the ~25 ms lag the archive's rows carry relative to the raw
# records, since the comparison had to correct for it.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="ip", kind="scalar", units="A",
        sources=("archive", "fdp"),
        locators=("ip", "ip"),
        step=0.001,
        notes="archive column is bit-identical to the model's training input. "
              "The fdp locator is a PTDATA point in AMPS and needs no scale "
              "factor: median ratio to the archive column 1.000034 over a "
              "random 120 overlap shots, median relative difference 1.1e-3. "
              "It is also the wider source - the archive omits `ip` on 2,192 "
              "of its 5,000 shots (present on 2,808, 56.2%) while fdp served "
              "it on all 120",
    ),
    FeatureSpec(
        name="bt", kind="scalar", units="T",
        sources=("archive", "fdp"),
        locators=("bt", "bt"),
        step=0.001,
        notes="archive column is bit-identical to the model's training input. "
              "The fdp locator is a PTDATA point in TESLA and needs no scale "
              "factor: median ratio 0.999992, median relative difference "
              "1.0e-3, same 120-shot sample. Same archive coverage gap as "
              "`ip`: the two columns are present on exactly the same 2,808 "
              "shots",
    ),
    FeatureSpec(
        name="pinj_total", kind="scalar", units="kW",
        sources=("archive", "corpus"),
        locators=("pinj", "pinj"),
        step=0.001,
        notes="corpus holds 8 beams in W; sum and divide by 1000. SAMPLING, "
              "Task 15: at t=1.025s on shot 185945 the corpus sum reads "
              "10.02 MW where the archive reads 10.996 MW, ~10% - sampling, "
              "not units. Scanned nearest/25ms-window/25ms-window-back/the "
              "archive's own 50ms-window-ending-at-t against the archive "
              "column over 8 RANDOM overlap shots (seed 42, both `pinj_total` "
              "and `tinj_total`): the archive convention's per-shot median "
              "relative error ranges 5.9e-4 to 1.4e-2, against 1.5e-2 to "
              "1.17 for the best of the other three candidates on the same "
              "row - 11x to 232x better (tightest at shot 190512 `tinj`: "
              "0.001311 vs 0.014514; loosest at shot 190000 `pinj`), and "
              "98x to 233x better than nearest-sample alone. That is ONE TO "
              "TWO orders of magnitude, never three or four - an earlier "
              "version of this note overstated the margin and also "
              "misattributed nearest-sample's own worst value (1.532485, "
              "shot 190308 `tinj`) to 'the best of the other three', whose "
              "actual value on that row is window-back's 1.170076. "
              "`models.base.build` (via `sample_by_resolver`) now samples "
              "any corpus- or fdp-resolved field with the archive's "
              "convention (`ARCHIVE_WINDOW_S`, per-resolver via "
              "`SAMPLING_BY_SOURCE`, not per-feature) rather than "
              "nearest-sample; an archive-resolved field is untouched. See "
              "`validate`'s module docstring and the Task 15 report for the "
              "full table",
    ),
    FeatureSpec(
        name="tinj_total", kind="scalar", units="N m",
        sources=("archive", "corpus"),
        locators=("tinj", "tinj"),
        step=0.001,
        notes="corpus holds 8 beams already in N m; sum only. Same sampling "
              "finding and fix as `pinj_total` - see its notes",
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
        notes="stands in for R0_EFITRT1. fdp against the archive: median "
              "relative difference 7.9e-4 over a RANDOM 120 overlap shots "
              "(seed 12), median ratio 1.000000. An earlier note here said "
              "8.8e-3 on shot 185945 alone; that comparison had not "
              "corrected for the archive's 25 ms row lag - see "
              "`resolve_fdp`'s docstring for the lag and the full table",
    ),
    FeatureSpec(
        name="kappa", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("kappa_EFIT01", r"\efit01::top.results.aeqdsk:kappa"),
        notes="stands in for kappa_EFITRT1; fdp against the archive 1.2e-3 "
              "median relative difference, ratio 1.000000, 120 shots",
    ),
    FeatureSpec(
        name="tritop", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("tritop_EFIT01", r"\efit01::top.results.aeqdsk:tritop"),
        notes="bit-identical to the training input; fdp against the archive "
              "3.2e-3 median relative difference, ratio 1.000000, 120 shots",
    ),
    FeatureSpec(
        name="tribot", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("tribot_EFIT01", r"\efit01::top.results.aeqdsk:tribot"),
        notes="bit-identical to the training input; fdp against the archive "
              "2.8e-3 median relative difference, ratio 1.000000, 120 shots",
    ),
    FeatureSpec(
        name="gapin", kind="scalar", units="m",
        sources=("archive", "fdp"),
        locators=("gapin_EFIT01", r"\efit01::top.results.aeqdsk:gapin"),
        notes="bit-identical to the training input; fdp against the archive "
              "7.3e-3 median relative difference, ratio 1.000000, 120 shots",
    ),
    FeatureSpec(
        name="betan", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("betan_EFIT01", r"\efit01::top.results.aeqdsk:betan"),
        notes="not a model input; the model predicts it, and validation "
              "compares against it. fdp against the archive 1.3e-2 median "
              "relative difference, ratio 1.000000, 120 shots",
    ),
    FeatureSpec(
        name="qpsi", kind="profile", units="",
        sources=("archive", "fdp"),
        locators=("qpsi_EFIT01", r"\efit01::top.results.geqdsk:qpsi"),
        notes="stands in for qpsi_EFITRT1. The model consumes 1/qpsi, "
              "applied by the adapter, not here. fdp against the archive "
              "6.9e-3 median relative difference over 120 shots. NOTE the "
              "radial axis: the geqdsk node arrives on 65 points of "
              "NORMALIZED PSI, and the archive column is those points' even "
              "indices - i.e. `RHO_GRID` here is a uniform psi grid for the "
              "two EFIT profiles, not rho, despite its name. Converting psi "
              "to rho with geqdsk `rhovn` was tried and is 7x WORSE "
              "(6.9e-2); see `resolve_fdp`'s docstring for the table",
    ),
    FeatureSpec(
        name="pres", kind="profile", units="Pa",
        sources=("archive", "fdp"),
        locators=("pres_EFIT01", r"\efit01::top.results.geqdsk:pres"),
        notes="bit-identical to the training input. fdp against the archive "
              "3.0e-2 median relative difference over 120 shots - the "
              "loosest of the thirteen, and it barely improves on "
              "exactly-coincident rows (2.9e-2), so it is the offline "
              "EFIT01 tree having been rerun since the store was built, not "
              "a sampling artefact. Same normalized-psi radial axis as "
              "`qpsi`, where the rhovn conversion is 10x worse on the "
              "shot-185945 exact-time comparison",
    ),
    FeatureSpec(
        name="ne_zipfit", kind="profile", units="1e19 m^-3",
        sources=("archive", "fdp"),
        locators=("zipfit_edensfit_rho", r"\ZIPFIT01::TOP.PROFILES.EDENSFIT"),
        notes="stands in for thomson_density_mtanh_1d. Our own mtanh fit to "
              "raw Thomson is Phase 2. fdp is in 1e19 m^-3 (the node's own "
              "units field) and needs no scale factor: median ratio to the "
              "archive column 0.999352, median relative difference 1.3e-2 "
              "over 120 shots. An earlier note here said 2.0e-1 with "
              "correlation 0.984, which is not reproducible - that "
              "comparison had not corrected for the archive's 25 ms row "
              "lag. Its x axis IS rho (121 points), so this path "
              "interpolates only, with no coordinate conversion. Absent on "
              "10 of the 120 sampled shots and single-sliced on 1",
    ),
    FeatureSpec(
        name="te_zipfit", kind="profile", units="keV",
        sources=("archive", "fdp"),
        locators=("zipfit_etempfit_rho", r"\ZIPFIT01::TOP.PROFILES.ETEMPFIT"),
        notes="stands in for thomson_temp_mtanh_1d; fdp in keV, no scale "
              "factor (ratio 0.999438), 1.2e-2 median relative difference "
              "over 120 shots. Absent on the same 10 shots as the density",
    ),
    FeatureSpec(
        name="rot_zipfit", kind="profile", units="kHz",
        sources=("archive", "fdp"),
        locators=("zipfit_trotfit_rho", r"\ZIPFIT01::TOP.PROFILES.TROTFIT"),
        notes="stands in for cer_rot_csaps_1d. Units are kHz, not krad/s "
              "and not km/s: MEASURED from the node's own units field "
              "(the TROTFIT node reports `kHz`), which "
              "settles the question carried from Task 8 - magnitude alone "
              "could not, since v = omega*R with R ~ 1.75 m puts all three "
              "readings in the same range. The model's domain rule "
              "`absmax < 150` therefore reads as 150 kHz. No scale factor "
              "against the archive (ratio 1.000000), 1.1e-2 median relative "
              "difference over 120 shots. The THINNEST of the thirteen: "
              "absent on 26 of the 120 sampled shots and single-sliced on "
              "1, so ~22% of shots have no rotation profile at all",
    ),
    # Added for d3d_tearing_time_to_event_dsm (2026-09-05). None is in the
    # tearing CNN's archive, so each is fdp-only; every node was fetched live
    # on shot 186563 before being listed (aeqdsk scalars: 390 slices, all
    # finite; itempfit (304, 121) in keV; pcbcoil 162,497 samples, 'raw').
    FeatureSpec(
        name="qmin", kind="scalar", units="",
        sources=("fdp",),
        locators=(r"\efit01::top.results.aeqdsk:qmin",),
        notes="stands in for qmin_EFITRT2 (offline EFIT01 for real-time EFIT)",
    ),
    FeatureSpec(
        name="li", kind="scalar", units="",
        sources=("fdp",),
        locators=(r"\efit01::top.results.aeqdsk:li",),
        notes="stands in for li_EFITRT2",
    ),
    FeatureSpec(
        name="aminor", kind="scalar", units="m",
        sources=("fdp",),
        locators=(r"\efit01::top.results.aeqdsk:aminor",),
        notes="stands in for aminor_EFITRT2",
    ),
    FeatureSpec(
        name="volume", kind="scalar", units="m^3",
        sources=("fdp",),
        locators=(r"\efit01::top.results.aeqdsk:volume",),
        notes="stands in for volume_EFITRT2",
    ),
    FeatureSpec(
        name="pcbcoil", kind="scalar", units="A",
        sources=("fdp",),
        locators=("pcbcoil",),
        step=0.001,
        notes="B-coil current, PTDATA. The survival model multiplies it by "
              "1.69861e-5 (get_survival_from_shot.py, the line commented "
              "'fix BT') before normalising, i.e. it is used as a toroidal "
              "field in tesla; kept as the raw current here so the adapter "
              "reproduces that step verbatim",
    ),
    FeatureSpec(
        name="ti_zipfit", kind="profile", units="keV",
        sources=("fdp",),
        locators=(r"\ZIPFIT01::TOP.PROFILES.ITEMPFIT",),
        notes="stands in for cer_temp_csaps_1d, the CER ion temperature fit. "
              "Same tree and layout as te_zipfit; not priced against any "
              "archive (the survival model's training data are not on disk "
              "in a form labelmaker reads)",
    ),
    # Added for d3d_ae_activity_seldnet (2026-09-06). The first `waveform`
    # feature: 4 chords at 500 kHz over the whole record, kept at the native
    # rate because the model's own STFT is what turns it into an input.
    FeatureSpec(
        name="co2", kind="waveform", units="cm^-2",
        sources=("corpus",),
        locators=("co2",),
        step=0.0,
        notes="CO2 interferometer line-integrated density, chords r0/v1/v2/v3 "
              "in that channel order. MEASURED on the corpus: `(4, 4.0-5.5e6)` "
              "float32 spanning -1.45 s to 6.55-9.55 s at 500,000.0 Hz "
              "(shot 198279: 4,000,001 samples over 8.000 s, 499,999.98 Hz "
              "from the span). Take the rate from the SPAN, never from a "
              "median `diff`: `xdata` is float32, whose spacing at t ~ 3 s is "
              "2.4e-7 s, so a 2 us step quantises and a median diff reads "
              "524,288 Hz. The rate matters - it fixes the frequency axis of "
              "the AE band, whose 250 kHz ceiling is this record's Nyquist. "
              "Availability is the real limit: filled on 12 of the 24 corpus "
              "shots sampled with `sample_shots(corpus, 24, seed=0)` - the "
              "other 12 carry the `(4, 1)` absent-signal sentinel - and the "
              "12 filled ones are all above shot 198279. Never scaled or "
              "decimated: `SCALE_TO_CANONICAL` does not list it and "
              "`resolve_corpus` skips both steps for a waveform",
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
