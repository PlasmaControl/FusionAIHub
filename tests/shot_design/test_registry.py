"""Ported from shot-recommender-system (shotrec) @565d548."""

# tests/test_registry.py
import pytest
from pydantic import ValidationError

from shot_design import config


def test_campaign_for_shot():
    assert config.campaign_for_shot(161172) == "2014_2015"
    assert config.campaign_for_shot(200443) == "2022_2025"
    assert config.campaign_for_shot(100) == "unknown"


def test_expand_registry_members_and_templates():
    specs = {s.name: s for s in config.expand_registry(161172)}
    assert specs["pnbi_15L"].group == "p_inj" and specs["pnbi_15L"].col == "pinjf_15l"
    # Task 11 measured all three of these against the staged columns of 161172; the values they
    # replace were guesses that did not fetch at all (see configs/shot_design/actuators.yaml for the
    # evidence).
    assert specs["pnbi_15L"].fetch == {
        "kind": "mds",
        "tree": "nb",
        "node": "\\NB::TOP.NB15L:PINJF_15L",
    }
    assert specs["pech_LEIA"].col == "ecleifpwrc"
    assert specs["pech_LEIA"].fetch["node"] == "\\RF::TOP.ECH.LEIA:ECLEIFPWRC"
    assert specs["pech_HAN"].fetch["node"] == "\\RF::TOP.ECH.HAN:ECHANDLFPWRC"  # code_upper
    assert specs["irmp_IU30"].fetch["node"] == "iu30"  # no trailing F: iu30f does not exist here
    assert specs["irmp_C19"].fetch["node"] == "c19"  # c19f returns C79F's array
    assert (
        specs["ip"].abs is True and specs["ip"].tier == "raw" and specs["betan"].tier == "derived"
    )
    assert specs["ne0"].fetch is None
    assert all(s.installed for s in specs.values())


def test_installation_windows(monkeypatch):
    cfg = config.load_yaml("actuators.yaml")
    cfg["systems"]["ech"]["members"][0]["until"] = 170000  # LEIA retired after 170000 (test only)
    monkeypatch.setattr(
        config,
        "load_yaml",
        lambda name, _c=cfg: _c if name == "actuators.yaml" else config._load_yaml_file(name),
    )
    names = {s.name for s in config.expand_registry(180000)}
    assert "pech_LEIA" not in names
    full = {s.name: s for s in config.expand_registry(180000, include_not_installed=True)}
    assert full["pech_LEIA"].installed is False


def test_actuator_systems():
    systems = config.actuator_systems(161172)
    assert systems["nbi"].prefix == "pnbi" and systems["nbi"].members == [
        "15L",
        "15R",
        "21L",
        "21R",
        "30L",
        "30R",
        "33L",
        "33R",
    ]


def test_campaign_for_shot_boundaries_are_inclusive():
    # The brief's own test only checks shots deep inside a range (161172, 200443) or deep
    # outside all of them (100), so it would still pass even if the comparison were off by
    # one (e.g. "<" instead of "<="). These check every campaign edge explicitly.
    assert config.campaign_for_shot(158000) == "2014_2015"  # first shot of the first campaign
    assert config.campaign_for_shot(164999) == "2014_2015"  # last shot before the next campaign
    assert config.campaign_for_shot(165000) == "2016_2018"  # first shot of the next campaign
    assert config.campaign_for_shot(176999) == "2016_2018"
    assert config.campaign_for_shot(177000) == "2019_2021"
    assert config.campaign_for_shot(189999) == "2019_2021"
    assert config.campaign_for_shot(190000) == "2022_2025"
    assert config.campaign_for_shot(209999) == "2022_2025"  # last shot of the last campaign
    assert config.campaign_for_shot(210000) == "unknown"  # one past the last campaign


def test_signalspec_rejects_misspelled_key():
    # "unit" (for "units") is exactly the authoring typo extra="forbid" exists to catch: under
    # the pydantic default (extra="ignore") this would silently construct with units=None
    # instead of failing, and a signals.yaml typo would go undetected.
    bad = {"group": "ip", "col": "ipsip", "unit": "A"}
    with pytest.raises(ValidationError):
        config.SignalSpec(name="ip", **bad)


def test_signalspec_accepts_scale_and_null_value_but_still_rejects_unknown_key():
    # scale and null_value are the two fields Task 7's unit fix adds; both must be legal on the
    # model. extra="forbid" must still catch an unrelated typo (mirrors
    # test_signalspec_rejects_misspelled_key's "unit" case, just with a different bad key) so this
    # doesn't silently loosen the model while adding the new fields.
    spec = config.SignalSpec(name="ip", group="ip", col="ipsip", scale=1.0e6, null_value=-9.99)
    assert spec.scale == 1.0e6
    assert spec.null_value == -9.99
    with pytest.raises(ValidationError):
        config.SignalSpec(name="ip", group="ip", col="ipsip", scael=1.0)


def test_installed_boundary_edges_are_inclusive():
    # No member in configs/shot_design/actuators.yaml has a concrete numeric since/until: every window
    # is either fully open (key absent entirely, e.g. the base NBI/coil_rmp members) or an explicit
    # `since: null` [?] placeholder awaiting a real installation shot (the later ECH gyrotrons,
    # LOB1/LOB2/PFX1-3/UOB gas valves) -- so there is no real registry entry to pin the edges
    # against. Exercise the window predicate directly with a synthetic member instead.
    member = {"id": "SYNTH", "since": 170000, "until": 180000}
    assert config._installed(member, 170000) is True  # since edge: installed
    assert config._installed(member, 169999) is False  # since - 1: not installed
    assert config._installed(member, 180000) is True  # until edge: installed
    assert config._installed(member, 180001) is False  # until + 1: not installed
