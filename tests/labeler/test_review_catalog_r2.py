"""The grown signal catalog (R2): the `description` and `avail` keys, a fetched
source's `free` flag and channels, `Entry.points`, the repo catalog's size and
integrity, and that a typed name still validates and fetches whether or not the
catalog lists it. Nothing here fetches."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from labeler.events.review import catalog as cat
from labeler.events.review import profile_fetch

REPO_CATALOG = Path(__file__).parents[2] / "configs" / "labeler" / cat.CATALOG_FILE
# What the catalog said before this round, and the groups the page shows.
GROUPS = {"Plasma", "Equilibrium", "Heating", "Fuelling", "Density", "Radiation",
          "Magnetics", "Edge", "Divertor", "Core", "Ions", "Fluctuations", "Control",
          "Neutrons", "Profiles", "Disruption"}


@pytest.fixture(scope="module")
def repo():
    return cat.load(REPO_CATALOG)


def _one(**entry):
    base = {"id": "x", "label": "X", "group": "G", "units": "", "kind": "trace",
            "sources": [{"ptdata": "IP"}]}
    return {"signals": [{**base, **entry}]}


# -- the new keys ------------------------------------------------------------------


def test_description_is_one_line_of_text_and_avail_a_known_word():
    got = cat.parse(_one(description="  Plasma  current\n from PTDATA.  ",
                         avail="early")).entries[0]
    assert got.description == "Plasma current from PTDATA." and got.avail == "early"
    plain = cat.parse(_one()).entries[0]
    assert plain.description == "" and plain.avail is None
    for avail in cat.AVAIL:
        assert cat.parse(_one(avail=avail)).entries[0].avail == avail


@pytest.mark.parametrize("raw, problem", [
    (_one(description="x" * (cat.DESCRIPTION_MAX + 1)), "description"),
    (_one(description=3), "description"),
    (_one(avail="some"), "avail"),
    (_one(avail=True), "avail"),
])
def test_a_bad_description_or_avail_is_refused(raw, problem):
    with pytest.raises(cat.CatalogError, match=problem):
        cat.parse(raw)


def test_the_longest_allowed_description_loads():
    entry = cat.parse(_one(description="y" * cat.DESCRIPTION_MAX)).entries[0]
    assert len(entry.description) == cat.DESCRIPTION_MAX


def test_free_belongs_to_an_mds_node_and_gives_it_the_long_limit():
    node = {"mds": "\\D3D::TOP.X:Y", "tree": "D3D"}
    plain = cat.parse(_one(sources=[node])).entries[0].sources[0]
    free = cat.parse(_one(sources=[{**node, "free": True}])).entries[0].sources[0]
    assert (plain.free, plain.long_limit) == (False, False)
    assert (free.free, free.long_limit) == (True, True)
    assert plain.key() == free.key()  # the cached values do not depend on the limit
    for bad in ({"ptdata": "IP", "free": True}, {**node, "free": "yes"},
                {"corpus": "ece", "rows": 0, "free": True}):
        with pytest.raises(cat.CatalogError, match="free"):
            cat.parse(_one(sources=[bad]))


def test_a_fetched_source_has_the_channels_the_entry_says():
    wide = cat.parse(_one(channels=10, draw="map")).entries[0]
    assert wide.channels == 10
    assert cat.parse(_one()).entries[0].channels == 1
    with pytest.raises(cat.CatalogError, match="legend"):
        cat.parse(_one(channels=3, legend=["a", "b"]))
    # a fetched source cannot make a corpus source's three rows one channel
    with pytest.raises(cat.CatalogError, match="channel"):
        cat.parse(_one(sources=[{"corpus": "ece", "rows": "0-2"}, {"ptdata": "ip"}]))


def test_points_name_the_points_an_entry_fetches():
    entry = cat.parse(_one(sources=[
        {"corpus": "co2", "rows": 1}, {"ptdata": "denv1uf"},
        {"mds": "\\D3D::TOP.ELECTRONS.BCI.DPD.V1:DENUF", "tree": "D3D"},
        {"feature": "co2_v1"}])).entries[0]
    assert entry.points == ("DENV1UF", "\\D3D::TOP.ELECTRONS.BCI.DPD.V1:DENUF")
    assert cat.parse(_one(sources=[{"corpus": "ece", "rows": 0}])).entries[0].points \
        == ()


# -- the repo catalog --------------------------------------------------------------


def test_the_repo_catalog_is_a_few_hundred_described_entries(repo):
    assert len(repo.entries) >= 200
    assert {e.group for e in repo.entries} <= GROUPS
    for entry in repo.entries:
        assert 20 <= len(entry.description) <= cat.DESCRIPTION_MAX, entry.id
        assert entry.avail in (*cat.AVAIL, None), entry.id


def test_an_entry_that_fetches_says_where_it_was_seen(repo):
    # a wide corpus-backed entry may carry `avail` from the archive points of a few
    # of its channels; an entry that fetches always does
    for entry in repo.entries:
        if entry.points or any(s.fetches for s in entry.sources):
            assert entry.avail is not None, entry.id


def test_no_new_entry_is_listed_without_data_on_either_shot(repo):
    # `none` is for an entry that was in the catalog before the probe (kept so the
    # page can say so); a name nobody has seen data for is left out.
    original = {"fs01", "fs02", "fs03", "fs04", "fs05", "fs06", "fs07", "fs08",
                "wmhd", "betan", "betap", "li", "q95", "qmin", "kappa", "tritop",
                "tribot", "n1rms", "n2rms", "mpi66m322d", "pcphd03", "prad_divl",
                "prad_divu", "ip", "bt", "density"}
    assert {e.id for e in repo.entries if e.avail == "none"} <= original


def test_every_group_the_owner_asked_for_has_entries(repo):
    wanted = {"Plasma", "Equilibrium", "Heating", "Fuelling", "Radiation",
              "Magnetics", "Edge", "Divertor", "Core", "Ions", "Fluctuations",
              "Control", "Neutrons", "Profiles"}
    assert wanted <= {e.group for e in repo.entries}


@pytest.mark.parametrize("ident", [
    "ip", "bt", "density", "betan", "betap", "li", "q95", "qmin", "wmhd", "kappa",
    "gapout", "rmaxis", "zmaxis", "pinj_total", "pinj_30l", "ech_total",
    "gas_gasa", "gas_total", "prad_divl", "fs01", "tplang01", "n1rms", "n3rms",
    "mpi66m322d", "co2_v1", "ts_core_te_ch", "tecef_20", "cer_ti_24", "mse_pitch_35",
    "besfu24", "neutronsrate", "pressure_efit", "ne_zipfit", "te_zipfit",
    "ti_zipfit", "pprime_efit", "ffprime_efit", "fpol_efit", "q_efit02",
])
def test_the_entries_the_brief_names_exist(repo, ident):
    assert ident in repo.by_id


def test_the_original_entries_keep_their_sources(repo):
    ip = repo.by_id["ip"]
    assert [(s.kind, s.name, s.scale) for s in ip.sources] == [
        ("feature", "ip", 1e-6), ("ptdata", "ip", 1e-6)]
    assert [s.kind for s in repo.by_id["qmin"].sources] == ["park", "feature", "mds"]
    assert repo.by_id["ece_map"].channels == 40
    assert repo.by_id["rotation"].sources[0].name == "zipfit_rotation"


def test_every_recipe_the_catalog_names_is_a_reader(repo):
    from labeler.events.review import profiles

    local = set(profiles.READERS) | {"efit01_q"}
    fetched = set(profile_fetch.SPECS)
    for entry in repo.entries:
        for source in entry.sources:
            if source.kind == "recipe":
                assert source.name in local | fetched, (entry.id, source.name)
                assert source.name in cat.RECIPES


def test_the_new_profiles_are_profiles_with_a_radial_axis(repo):
    for ident in ("pressure_efit", "pprime_efit", "ffprime_efit", "fpol_efit",
                  "q_efit02", "ne_zipfit", "te_zipfit", "ti_zipfit"):
        entry = repo.by_id[ident]
        assert entry.kind == "profile" and entry.group == "Profiles"
        assert entry.axis in ("psi_n", "rho") and entry.description
        assert all(s.fetches for s in entry.sources), ident


def test_a_name_typed_into_the_page_resolves_listed_or_not(repo):
    # a listed point typed by name is the free entry, not the catalog's
    typed = repo.resolve("custom:denv1uf")
    assert typed.id == "custom:denv1uf" and typed.sources[0].name == "DENV1UF"
    assert typed.group == "PTDATA" and typed.sources[0].kind == "ptdata"
    unlisted = repo.resolve("custom:ZZQQ99")
    assert unlisted.sources[0].name == "ZZQQ99" and unlisted.sources[0].fetches
    free = repo.resolve("mds:EFIT01:\\EFIT01::TOP.RESULTS.AEQDSK:ZXPT2")
    assert free.sources[0].kind == "mds" and free.sources[0].long_limit


def test_every_catalog_node_also_resolves_when_typed(repo):
    # the catalog's trees stay on the allow-list a typed `mds:` id is held to, so a
    # listed node can always be typed, and the page's autocomplete never offers a
    # name its own text box refuses
    for entry in repo.entries:
        for source in entry.sources:
            if source.kind == "mds":
                typed = repo.resolve(f"mds:{source.tree}:{source.name}")
                assert typed.sources[0].name.upper() == source.name.upper()
            elif source.kind == "ptdata":
                assert repo.resolve(f"custom:{source.name}").sources[0].kind == \
                    "ptdata"


def test_a_typed_name_is_validated_before_it_can_reach_a_fetch(repo):
    for ident in ("custom:ip()", "custom:ip;bt", "mds:CER:\\CER::X",
                  "mds:D3D:\\D3D::TOP.X:DIM_OF(Y)", "mds:D3D:\\D3D::TOP.X;Y"):
        with pytest.raises(cat.Invalid):
            repo.resolve(ident)


def test_the_catalog_loads_quickly_and_lists_by_id(repo):
    start = time.perf_counter()
    again = cat.load(REPO_CATALOG)
    assert time.perf_counter() - start < 2.0
    assert len(again.entries) == len(repo.entries) == len(repo.by_id)
