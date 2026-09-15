"""Which shots exist, which are usable, and which 100 the PoC runs on."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from labelmaker import catalog
from labelmaker.config import Paths

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
TM = catalog.TM_ARCHIVE


def _fake_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for shot, n_ech in ((190000, 24), (190001, 1), (190002, 24)):
        with h5py.File(corpus / f"{shot}_processed.h5", "w") as f:
            g = f.create_group("pinj")
            g.create_dataset("xdata", data=np.arange(10, dtype=np.float32))
            g.create_dataset("ydata", data=np.zeros((8, 10), dtype=np.float32))
            g = f.create_group("ech_power")
            g.create_dataset("xdata", data=np.arange(n_ech, dtype=np.float32))
            g.create_dataset("ydata", data=np.zeros((12, n_ech), dtype=np.float32))
    (corpus / "not_a_shot.h5").touch()
    return Paths(root=tmp_path / "out", corpus=corpus)


def test_corpus_shots_are_sorted_ints_from_filenames(tmp_path):
    paths = _fake_corpus(tmp_path)
    assert catalog.corpus_shots(paths) == [190000, 190001, 190002]


def test_corpus_groups_reports_lengths_and_absence(tmp_path):
    paths = _fake_corpus(tmp_path)
    groups = catalog.corpus_groups(paths.corpus_file(190001))
    assert groups == {"pinj": 10, "ech_power": 1}
    assert catalog.available_groups(paths.corpus_file(190001)) == {"pinj"}
    assert catalog.available_groups(paths.corpus_file(190000)) == {"pinj", "ech_power"}


def test_sample_shots_is_deterministic_and_bounded():
    shots = list(range(1000, 1100))
    a = catalog.sample_shots(shots, 10, seed=0)
    b = catalog.sample_shots(shots, 10, seed=0)
    c = catalog.sample_shots(shots, 10, seed=1)
    assert a == b and a != c
    assert len(a) == 10 and set(a) <= set(shots) and a == sorted(a)
    assert catalog.sample_shots(shots, 500, seed=0) == shots  # n > len is all


def test_shot_file_round_trip(tmp_path):
    p = tmp_path / "shots.txt"
    catalog.write_shot_file(p, [190002, 190000])
    assert catalog.read_shot_file(p) == [190000, 190002]
    p.write_text("# a comment\n190005\n\n190004\n")
    assert catalog.read_shot_file(p) == [190004, 190005]


@pytest.mark.skipif(not CORPUS.exists(), reason=f"corpus not available: {CORPUS}")
def test_real_corpus_is_enumerated():
    shots = catalog.corpus_shots(Paths())
    assert len(shots) > 16_000
    assert all(100_000 < s < 300_000 for s in shots)


@pytest.mark.skipif(
    not (CORPUS.exists() and TM.exists()), reason="corpus or tm archive missing"
)
def test_overlap_with_the_tearing_archive_is_the_poc_pool():
    overlap = catalog.overlap_shots(Paths())
    assert len(overlap) > 1_000       # measured 1,503 on 2026-09-03
    assert min(overlap) >= 185_000


def test_sample_shots_deduplicates_before_sampling():
    """A shot file may list a shot twice; the sample must not."""
    shots = [5, 5, 5, 1, 2, 2]
    assert catalog.sample_shots(shots, 10, seed=0) == [1, 2, 5]
    # 300 entries but three distinct shots: the sample is those three, not
    # three draws from a list full of repeats.
    assert catalog.sample_shots(shots * 50, 3, seed=0) == [1, 2, 5]
