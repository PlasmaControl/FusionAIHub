"""The detachment review source: vote tiers, divertor Thomson Te bins, the merges."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.events import suggestions
from labeler.events.detachment import source

VOTE_CSV = (
    "shot,category,t_start,t_end,confidence,attrs\n"
    '7,1,0,100,,"{""tier"": ""certain""}"\n'
    '7,4,150,200,,"{""tier"": ""candidate_marfe""}"\n'
    '7,2,200,250,,"{""tier"": ""conflict""}"\n'
)


def test_vote_tiers_map_to_page_categories():
    assert source.vote_state(1, "certain") == 1
    assert source.vote_state(2, "tangtv_only") == 2
    assert source.vote_state(4, "candidate_marfe") == 3
    assert source.vote_state(2, "conflict") == 4
    assert source.vote_state(1, "tangtv_only_lmode") == 4
    with pytest.raises(ValueError):
        source.vote_state(4, "certain")


def test_vote_rows_become_bins_and_rows_with_unmarked_gaps(tmp_path):
    path = tmp_path / "vote.csv"
    path.write_text(VOTE_CSV)
    bins = source.vote_bins(path)
    assert bins.start.tolist() == [0, 50, 150, 200]
    assert bins.category.tolist() == [1, 1, 3, 4]
    assert source.bin_rows(bins) == [
        [7, 1, 0, 100, ""],
        [7, 0, 100, 150, ""],
        [7, 3, 150, 200, ""],
        [7, 4, 200, 250, ""],
    ]
    path.write_text(VOTE_CSV + '7,1,50,100,,"{""tier"": ""certain""}"\n')
    with pytest.raises(ValueError, match="overlapping"):
        source.vote_bins(path)
    path.write_text(VOTE_CSV + '7,1,300,320,,"{""tier"": ""certain""}"\n')
    with pytest.raises(ValueError, match="whole 50 ms bins"):
        source.vote_bins(path)  # its one bin would end past the row's t_end


def test_te_bands():
    got = source.te_state([0.5, 5.0, 5.1, 9.9, 10.0, 40.0, np.nan])
    assert got.tolist() == [2, 2, 4, 4, 1, 1, 0]


def flat_map(times, values):
    """A flux map whose normalised flux is `values[k]` everywhere at `times[k]`."""
    r, z = np.linspace(1.3, 1.7, 5), np.linspace(-1.4, -1.0, 5)
    psirz = np.stack([np.full((len(z), len(r)), v) for v in values])
    zeros = np.zeros(len(times))
    return {
        "gtime_ms": np.asarray(times, float),
        "psirz": psirz,
        "r": r,
        "z": z,
        "ssimag": zeros,
        "ssibry": zeros + 1,
    }


def test_bin_te_keeps_shelf_chords_in_the_near_sol_and_falls_back_to_efit01():
    # Chords 0.4, 2.7, 4.5 and 25 cm above the shelf: only the middle two count.
    dts = {
        "r": np.full(4, 1.485),
        "z": np.array([-1.246, -1.223, -1.205, -1.0]),
        "t_ms": np.array([10.0, 30.0, 110.0, 130.0, 210.0]),
        "te": np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [3.0, 4.0, 2.0, 2.0, 12.0],
                [6.0, 60.0, 2.0, 2.0, 14.0],  # 60 eV is outside 0.1-50 eV
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    }
    maps = [
        ("EFIT02", flat_map([0.0, 20.0], [1.02, 1.02])),  # covers t <= 60 ms only
        ("EFIT01", flat_map([100.0, 200.0], [0.9, 1.03])),  # core at 100, SOL at 200
    ]
    value, count, used = source.bin_te(dts, maps, np.arange(0.0, 250.0, 50.0))
    np.testing.assert_allclose(value, [4.0, np.nan, np.nan, np.nan, 13.0])
    assert count.tolist() == [3, 0, 0, 0, 2]
    assert used.tolist() == [0, -1, -1, -1, 1]
    assert source.te_state(value).tolist() == [2, 0, 0, 0, 1]


def park(path, t_ms, te, status=None):
    """A parked divertor Thomson record: two chords 2.7 and 4.5 cm above the shelf."""
    if status is not None:
        np.savez(path, status=np.array(json.dumps(status)))
        return
    z = np.array([-1.223, -1.205])
    np.savez(
        path,
        t_ms=np.asarray(t_ms, float),
        te=np.asarray(te, float),
        r=np.full(2, 1.485),
        z=z,
        status=np.array("{}"),
    )


def test_te_bins_keep_the_plasma_window_and_two_samples_and_skip_with_reasons(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        source,
        "load_maps",
        lambda shot: [("EFIT02", flat_map(range(0, 300, 20), [1.02] * 15))],
    )
    nan = np.nan
    # Bins 0 and 200 lie outside the (50, 200) ms window; bin 50 has one sample;
    # bin 100 has two at one laser time and bin 150 one at each of two times.
    t = [10.0, 30.0, 60.0, 110.0, 160.0, 170.0, 210.0]
    te = [[2.0, 2.0, 3.0, 2.0, 20.0, nan, 2.0], [2.0, 2.0, nan, 4.0, nan, 30.0, 2.0]]
    park(tmp_path / "1.npz", t, te)
    park(tmp_path / "2.npz", [nan, nan], [[1.0, 1.0], [1.0, 1.0]])  # no finite time
    nodata = "error TreeNODATA: %TREE-E-NODATA, No data available for this node"
    park(tmp_path / "3.npz", None, None, status={"te": nodata, "t_ms": nodata})
    park(tmp_path / "6.npz", t, te)  # no window
    got, skipped, dropped = source.te_bins(
        [1, 2, 3, 4, 5, 6],
        [tmp_path],
        windows={1: (50, 200), 2: (0, 300), 3: (0, 300)},
        min_samples=2,
        selection={4: "selected", 5: "off_shelf"},
    )
    assert got.start.tolist() == [100.0, 150.0]
    assert got.category.tolist() == [2, 1]
    assert got.n.tolist() == [2, 2]
    assert dropped == {"bins_with_te": 5, "outside_window": 2, "too_few_samples": 1}
    assert skipped == {
        "2": "no_dts_time",
        "3": "no_dts_node",
        "4": "not_fetched",
        "5": "not_selected:off_shelf",
        "6": "no_window",
    }
    ungated, _, _ = source.te_bins([1], [tmp_path])  # detach_te v1's rule
    assert ungated.start.tolist() == [0.0, 50.0, 100.0, 150.0, 200.0]


def bins(rows, tier=False):
    columns = ["shot", "start", "category"] + (["tier"] if tier else [])
    return pd.DataFrame(rows, columns=columns)


def test_merge_puts_the_vote_first_and_agreement_counts_shared_state_bins():
    vote = bins(
        [
            (1, 0.0, 1, "certain"),
            (1, 50.0, 2, "tangtv_only"),
            (1, 100.0, 4, "conflict"),
        ],
        tier=True,
    )
    te = bins([(1, 0.0, 1), (1, 50.0, 1), (1, 100.0, 2), (1, 150.0, 2), (2, 0.0, 4)])
    merged = source.merge(vote, te)
    assert merged.category.tolist() == [1, 2, 4, 2, 4]
    assert merged["from"].tolist() == ["detach_vote"] * 3 + ["detach_te"] * 2
    record = source.agreement(vote, te)
    assert (record["bins"], record["shots"]) == (3, 1)
    assert record["vote_by_te"] == {
        "attached": {"attached": 1},
        "detached": {"attached": 1},
        "uncertain": {"detached": 1},
    }
    states = record["states"]
    assert (states["bins"], states["agree"], states["agreement"]) == (2, 1, 0.5)
    assert states["kappa"] == pytest.approx(0.0)
    uncertain = record["vote_uncertain_with_te"]
    assert (uncertain["bins"], uncertain["bins_with_te_state"]) == (1, 1)
    assert uncertain["by_tier"] == {"conflict": {"detached": 1}}


def test_v3_merge_lets_te_fill_only_the_vote_abstentions():
    vote = bins(
        [
            (1, 0.0, 4, "no_vote"),
            (1, 50.0, 4, "insufficient_support"),
            (1, 100.0, 4, "conflict"),
            (1, 150.0, 4, "tangtv_only_lmode"),
            (1, 200.0, 4, "lower_shelf_window"),
            (1, 250.0, 1, "certain"),
            (1, 300.0, 4, "no_vote"),  # no Te here: stays as voted
        ],
        tier=True,
    )
    te = bins([(1, s, c) for s, c in zip(range(0, 300, 50), (2, 1, 2, 2, 1, 2))])
    te = pd.concat([te, bins([(1, 350.0, 4)])], ignore_index=True)
    merged = source.merge(vote, te, source.ABSTAIN_TIERS)
    assert merged.start.tolist() == [0, 50, 100, 150, 200, 250, 300, 350]
    assert merged.category.tolist() == [2, 1, 4, 4, 4, 1, 4, 4]
    assert merged["from"].tolist() == ["detach_te"] * 2 + ["detach_vote"] * 5 + [
        "detach_te"
    ]
    assert source.filled(merged) == {
        "insufficient_support": {"attached": 1},
        "no_vote": {"detached": 1},
    }
    assert (source.merge(vote, te)["from"] == "detach_vote").sum() == 7  # v2


def test_write_never_overwrites_and_moves_the_pointer(tmp_path, monkeypatch):
    event_dir = tmp_path / "events" / "detachment"
    event_dir.mkdir(parents=True)
    monkeypatch.setattr(source, "EVENT_DIR", event_dir)
    paths = Paths(root=tmp_path / "root")
    rows = bins([(7, 0.0, 1), (7, 100.0, 2)])
    out = source.write(paths, "detach_vote", "v9", rows, {"rule": "test"}, pointer=True)
    table = suggestions.table_path(paths, "detachment", "detach_vote", "v9")
    assert out["table"] == str(table) and out["rows"] == 3
    meta = json.loads(table.with_suffix(".meta.json").read_text())
    assert meta["coverage"]["detached"] == {"bins": 1, "shots": 1}
    pointer = json.loads((event_dir / "review" / "source.json").read_text())
    assert (pointer["method"], pointer["version"]) == ("detach_vote", "v9")
    with pytest.raises(FileExistsError):
        source.write(paths, "detach_vote", "v9", rows, {}, pointer=False)
