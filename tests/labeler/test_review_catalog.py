"""The signal catalog (configs/labeler/signals.yaml) and the free ids a client may
send: what validates, and every injection the server must refuse before a fetch."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from labeler.events.review import catalog as cat

REPO_CATALOG = Path(__file__).parents[2] / "configs" / "labeler" / cat.CATALOG_FILE


@pytest.fixture(scope="module")
def repo():
    return cat.load(REPO_CATALOG)


def test_the_repo_catalog_loads_with_the_designs_ids(repo):
    ids = set(repo.by_id)
    design = {"ip", "betan", "q95", "wmhd", "ece_map", "q_profile", "ts_core_te",
              "ts_core_ne", "rotation", "ece_te_r"}
    detachment = {"bolo", "fs_all", "prad_divl", "prad_divu",
                  *(f"fs0{i}" for i in range(1, 9))}
    assert design <= ids and detachment <= ids
    ece = repo.by_id["ece_map"]
    assert (ece.channels, ece.draw, ece.sources[0].kind) == (40, "map", "corpus")
    assert repo.by_id["bolo"].channels == 48 and repo.by_id["bolo"].draw == "map"
    assert repo.by_id["fs_all"].channels == 8
    assert repo.by_id["fs_all"].legend_for(8)[0] == "FS01"
    axes = {i: repo.by_id[i].axis for i in ("q_profile", "ts_core_te", "rotation",
                                            "ece_te_r")}
    assert axes == {"q_profile": "psi_n", "ts_core_te": "R_m", "rotation": "rho",
                    "ece_te_r": "R_m"}
    assert repo.by_id["prad_divl"].sources[0].name == "\\BOLOM::PRAD_DIVL"


def test_the_repo_catalogs_events_are_events(repo):
    events = yaml.safe_load((REPO_CATALOG.parent / "events.yaml").read_text())
    known = set(events["taxonomy"]) | {
        "alfven_eigenmode", "confinement", "high_confinement_mode", "detachment",
        "edge_localized_mode", "locked_mode", "minimum_safety_factor",
        "neoclassical_tearing_mode", "poloidal_beta", "resistive_wall_mode",
        "sawtooth_oscillation",
    }
    named = {e for entry in repo.entries for e in entry.events}
    assert named and named <= known


def test_suggested_follows_the_store_and_the_parent(repo):
    betan = repo.by_id["betan"]
    assert betan.suggested("mhd_n1")
    assert betan.suggested("mhd_n2")  # shows mhd_n1's store
    assert betan.suggested("mhd_n1.m1")  # a subcategory of mhd_n1
    assert not betan.suggested(None)
    assert not repo.by_id["bolo"].suggested("mhd_n1")


@pytest.mark.parametrize("name", ["DUSBRADIAL", "ip", "MPI66M322D", "a", "A_1"])
def test_a_ptdata_name_resolves(repo, name):
    entry = repo.resolve(f"custom:{name}")
    assert entry.sources[0].kind == "ptdata" and entry.sources[0].name == name.upper()
    assert not entry.sources[0].long_limit and entry.label == name


def test_free_ids_differing_in_case_share_one_source(repo):
    def key(ident):
        return repo.resolve(ident).sources[0].key()

    assert key("custom:denr0uf") == key("custom:DENR0UF")
    assert key("mds:mhd:\\mhd::n1rms") == key("mds:MHD:\\MHD::N1RMS")


@pytest.mark.parametrize("name", [
    "", "DUSB-RADIAL", "ip()", "ip'", 'ip"', "ip,bt", "ip bt", "ip$", "$ip",
    "1IP", "_IP", "ABCDEFGHIJKLM", "dim_of(ip)", "ip;bt", "ip\n", "IP\\X",
])
def test_a_bad_ptdata_name_is_invalid(repo, name):
    with pytest.raises(cat.Invalid):
        repo.resolve(f"custom:{name}")


@pytest.mark.parametrize("ident", [
    "mds:EFIT01:\\EFIT01::TOP.RESULTS.AEQDSK:WMHD",
    "mds:efit01:\\efit01::top.results.aeqdsk:q95",
    "mds:MHD:\\MHD::N1RMS",
    "mds:BOLOM:BOLOM::PRAD_DIVL",
])
def test_an_allowed_node_resolves_to_a_free_fetch(repo, ident):
    source = repo.resolve(ident).sources[0]
    assert source.kind == "mds" and source.free and source.long_limit
    assert source.tree == source.tree.upper() and source.name.startswith("\\")


@pytest.mark.parametrize("ident", [
    "mds:EFIT01",  # no node
    "mds:EFIT01:",  # empty node
    "mds:MHD:\\MHD::N1RMS()",  # parentheses
    "mds:MHD:dim_of(\\MHD::N1RMS)",
    "mds:MHD:\\MHD::DIM_OF",  # never even a bare dim_of segment
    "mds:MHD:\\MHD::PTDATA2",
    "mds:MHD:\\MHD::PTDATA2X",
    "mds:MHD:\\MHD::N1RMS'",  # quotes
    'mds:MHD:\\MHD::"N1RMS"',
    "mds:MHD:\\MHD::N1RMS,1",  # commas
    "mds:MHD:\\MHD::N1 RMS",  # spaces
    "mds:MHD:\\MHD::$N1RMS",  # $
    "mds:MHD:\\MHD::N1RMS$",
    "mds:MHD:\\MHD::" + "A" * 92,  # overlong (97 characters)
    "mds:CER:\\CER::CERQROTC",  # a tree off the list
    "mds:PTDATA2:\\PTDATA2::X",
    "mds:EFIT01:\\EFIT02::TOP.RESULTS.AEQDSK:Q95",  # a node outside its tree
    "mds:EFIT01:\\EFIT01::_TOP",  # a segment starting with _
    "mds:EFIT01:\\EFIT01::TOP._X",
    "mds:EFIT01:\\EFIT01::TOP.1X",  # a segment starting with a digit
    "mds:EFIT01:\\EFIT01::TOP..X",  # an empty segment
    "mds:EFIT01:\\EFIT01::TOP.",
    "mds:EFIT01:\\EFIT01:::X",
    "mds:EFIT01:TOP.RESULTS",  # no tree prefix
    "mds:EFIT01:\\\\EFIT01::Q95",
    "mds:EFIT01:\\EFIT01::Q95\n",
    "mds:EFIT01:\\EFIT01::Q95/2",
    "mds:EFIT01:\\EFIT01::Q95*2",
    "mds:EFIT01:\\EFIT01::Q95[0]",
])
def test_a_bad_node_is_invalid_before_any_fetch(repo, ident):
    with pytest.raises(cat.Invalid):
        repo.resolve(ident)


def test_the_node_max_is_96(repo):
    node = "\\MHD::" + "A" * (cat.NODE_MAX - 6)
    assert len(node) == cat.NODE_MAX
    assert repo.resolve(f"mds:MHD:{node}").sources[0].name == node
    with pytest.raises(cat.Invalid):
        repo.resolve(f"mds:MHD:{node}A")


@pytest.mark.parametrize("ident", [None, "", "IP", "nope", "../ip", "ip?x=1"])
def test_an_unknown_id_is_invalid(repo, ident):
    with pytest.raises(cat.Invalid):
        repo.resolve(ident)


def test_the_page_pattern_agrees_with_the_server():
    for node in ("\\EFIT01::TOP.RESULTS.AEQDSK:WMHD", "MHD::N1RMS"):
        assert cat.NODE_STRICT.match(node)
    for node in ("\\MHD::N1RMS()", "\\MHD::_X", "MHD", "\\MHD::A..B"):
        assert not cat.NODE_STRICT.match(node)


def _one(**entry):
    base = {"id": "x", "label": "X", "group": "G", "units": "", "kind": "trace",
            "sources": [{"corpus": "ece", "rows": 0}]}
    return {"signals": [{**base, **entry}]}


@pytest.mark.parametrize("raw, problem", [
    ({"signals": "no"}, "signals"),
    (_one(id="Bad Id"), "id"),
    (_one(kind="image"), "kind"),
    (_one(colour="red"), "unknown keys"),
    (_one(sources=[]), "sources"),
    (_one(sources=[{"corpus": "ece"}]), "rows"),
    (_one(sources=[{"corpus": "ece", "rows": 0, "ptdata": "ip"}]), "exactly one"),
    (_one(sources=[{"ptdata": "ip()"}]), "PTDATA"),
    (_one(sources=[{"mds": "\\EFIT02::Q95", "tree": "EFIT01"}]), "not EFIT01"),
    (_one(sources=[{"mds": "\\CER::X", "tree": "CER"}]), "allowed MDS tree"),
    (_one(sources=[{"ptdata": "ip", "scale": 0}]), "scale"),
    (_one(sources=[{"ptdata": "ip", "tree": "MHD"}]), "tree is for"),
    (_one(sources=[{"recipe": "park_q"}]), "recipe"),
    (_one(kind="profile", axis="psi_n"), "recipes"),
    (_one(kind="profile", axis="time", sources=[{"recipe": "park_q"}]), "axis"),
    (_one(sources=[{"corpus": "ece", "rows": "0-2"}, {"ptdata": "ip"}]), "channel"),
    (_one(legend=["a", "b"]), "legend"),
    (_one(draw="bars"), "draw"),
    (_one(sources=[{"park": "a_beta"}]), "park"),
])
def test_a_malformed_catalog_is_refused(raw, problem):
    with pytest.raises(cat.CatalogError, match=problem):
        cat.parse(raw)


def test_a_duplicate_id_is_refused():
    raw = _one()
    raw["signals"].append(dict(raw["signals"][0]))
    with pytest.raises(cat.CatalogError, match="twice"):
        cat.parse(raw)


def test_rows_spans_totals_and_keys():
    parsed = cat.parse({"signals": [
        {"id": "a", "label": "A", "group": "G", "kind": "trace",
         "sources": [{"corpus": "pinj", "rows": "0-7", "total": True, "scale": 1e-6},
                     {"feature": "pinj_total", "scale": 1e-3}]},
        {"id": "b", "label": "B", "group": "G", "kind": "trace", "legend": ["p", "q"],
         "sources": [{"raw": "co2", "rows": [0, 2]}]},
    ]})
    a, b = parsed.entries
    assert a.channels == 1 and a.sources[0].rows == tuple(range(8))
    assert b.channels == 2 and b.legend_for(2) == ["p", "q"]
    # The cache key holds what the values depend on, not their scale or tag.
    same = cat.Source("corpus", "pinj", rows=tuple(range(8)), total=True, scale=5.0,
                      tag="t")
    assert a.sources[0].key() == same.key()
    assert a.sources[0].where() == "corpus pinj rows 0-7, summed"
    assert b.sources[0].where() == "raw cache co2 rows 0,2"


def test_an_absent_catalog_is_empty(tmp_path):
    assert cat.load(tmp_path / "none.yaml").entries == ()
