"""The dock's 2-D slices (plan D.B): EFIT equilibria and TangTV inversions."""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
from PIL import Image

from labeler.config import Paths
from labeler.events.review import slices

SHOT = 190001
WALL = [[1.0, -1.4], [2.4, -1.4], [2.4, 1.4], [1.0, 1.4]]


def write_equilibrium(paths, shot=SHOT, *, times=(100.0, 200.0, 300.0),
                      tree="EFIT02", grid=True, wall=WALL, rule=None, counts=True):
    """Elliptic flux surfaces: psi_N = ((R - 1.7) / 0.5)^2 + (Z / 0.9)^2."""
    path = paths.root / "equilibria" / f"{shot}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(times)
    theta = np.linspace(0, 2 * np.pi, 41)
    rb = np.full((n, 50), np.nan, dtype=np.float32)
    zb = np.full((n, 50), np.nan, dtype=np.float32)
    rb[:, :41] = 1.7 + 0.5 * np.cos(theta)
    zb[:, :41] = 0.9 * np.sin(theta)
    arrays = {
        "t_ms": np.asarray(times, dtype=np.float32),
        "tree": np.asarray(tree),
        "source": np.asarray(rule or (tree if tree == "EFIT02" else f"{tree}_fallback")),
        "nbbbs": np.full(n, 41, dtype=np.int16),
        "rbbbs": rb,
        "zbbbs": zb,
        "lim": np.asarray(wall, dtype=np.float32).reshape(-1, 2),
        "rxpt1": np.full(n, 1.4, np.float32),
        "zxpt1": np.full(n, -1.1, np.float32),
        "rxpt2": np.full(n, -9.99, np.float32),
        "zxpt2": np.full(n, -9.99, np.float32),
        "rvsod": np.full(n, 1.5, np.float32),
        "zvsod": np.full(n, -1.25, np.float32),
        "rvsid": np.full(n, 0.0, np.float32),
        "zvsid": np.full(n, 0.0, np.float32),
        "rvsou": np.full(n, -0.89, np.float32),
        "zvsou": np.full(n, -0.89, np.float32),
        "rvsiu": np.full(n, -0.89, np.float32),
        "zvsiu": np.full(n, -0.89, np.float32),
    }
    if not counts:
        del arrays["nbbbs"]  # an EFIT without NBBBS; the fetch then writes none
    if grid:
        r = np.linspace(0.84, 2.54, 69, dtype=np.float32)
        z = np.linspace(-1.6, 1.6, 65, dtype=np.float32)
        psin = ((r[None, :] - 1.7) / 0.5) ** 2 + (z[:, None] / 0.9) ** 2
        arrays |= {
            "r": r,
            "z": z,
            "psirz": np.broadcast_to(2.0 * psin - 0.5, (n, *psin.shape)).astype(
                np.float32
            ),
            "ssimag": np.full(n, -0.5, np.float32),
            "ssibry": np.full(n, 1.5, np.float32),
        }
    np.savez_compressed(path, **arrays)
    return path


def write_inversion(paths, shot=SHOT, *, times=(1000.0, 1016.7), radii=(1.0, 2.0)):
    """frames[t, zi, ri] = 10 zi + ri on `radii` (increasing by default) and
    increasing elevation."""
    path = paths.root / "round4/detach/inversions" / f"{shot}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    zi, ri = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    frames = np.broadcast_to(10 * zi + ri, (len(times), 3, 4)).astype(np.float16)
    np.savez(
        path,
        frames=frames,
        times_ms=np.asarray(times, dtype=np.float32),
        radii=np.linspace(*radii, 4, dtype=np.float32),
        elevation=np.asarray([-1.4, -0.9, -0.4], dtype=np.float32),
        source=np.asarray("/somewhere/emission_structure_190001.sav"),
    )
    return path


@pytest.fixture
def paths(tmp_path):
    return Paths(root=tmp_path / "root")


def test_available_lists_what_is_on_disk(paths):
    assert slices.available(SHOT, paths) == []
    write_equilibrium(paths, tree="EFIT01")
    write_inversion(paths)
    assert slices.available(SHOT, paths) == [
        {"kind": "equilibrium", "label": "EFIT equilibrium (EFIT01)", "n_t": 3,
         "t_range_ms": [100.0, 300.0]},
        {"kind": "inversion", "label": "TangTV inversion", "n_t": 2,
         "t_range_ms": [1000.0, pytest.approx(1016.7)]},
    ]


def test_equilibrium_json_shape_and_nearest_slice(paths):
    write_equilibrium(paths)
    view = slices.equilibrium(SHOT, 230, paths)
    assert set(view) == {
        "t_ms", "source", "source_rule", "r_range", "z_range", "boundary",
        "xpoints", "strikes", "wall", "levels",
    }
    assert view["t_ms"] == 200.0 and view["source"] == "EFIT02"
    assert view["r_range"] == pytest.approx([0.84, 2.54])
    assert view["z_range"] == pytest.approx([-1.6, 1.6])
    assert len(view["boundary"]) == 41 and len(view["boundary"][0]) == 2
    assert view["xpoints"] == [[1.4, -1.1]]  # the -9.99 sentinel is dropped
    assert view["strikes"] == [[1.5, -1.25]]  # 0.0 and -0.89 are dropped
    assert view["wall"] == WALL
    assert [level["psi_n"] for level in view["levels"]] == list(slices.LEVELS)
    for level in view["levels"]:
        assert level["paths"], level["psi_n"]
        for line in level["paths"]:
            rz = np.asarray(line)
            psin = ((rz[:, 0] - 1.7) / 0.5) ** 2 + (rz[:, 1] / 0.9) ** 2
            np.testing.assert_allclose(psin, level["psi_n"], atol=0.01)


def test_a_gap_beyond_40_ms_is_missing(paths):
    write_equilibrium(paths)
    assert slices.equilibrium(SHOT, 340, paths)["t_ms"] == 300.0
    with pytest.raises(slices.SliceMissing, match="within 40 ms"):
        slices.equilibrium(SHOT, 341, paths)
    with pytest.raises(slices.SliceMissing, match="no inversion"):
        slices.inversion(SHOT, 1000, paths)


def test_levels_are_empty_without_a_psi_grid(paths):
    write_equilibrium(paths, grid=False)
    view = slices.equilibrium(SHOT, 100, paths)
    assert view["levels"] == []
    assert view["r_range"] == pytest.approx([1.0, 2.4])  # the wall's extent


def test_contours_are_cut_at_the_wall(paths):
    narrow = [[1.0, -1.4], [2.15, -1.4], [2.15, 1.4], [1.0, 1.4]]
    write_equilibrium(paths, wall=narrow)
    view = slices.equilibrium(SHOT, 100, paths)
    outer = next(level for level in view["levels"] if level["psi_n"] == 1.05)
    assert len(outer["paths"]) == 1  # the ellipse, open where it leaves the wall
    assert max(r for line in outer["paths"] for r, _ in line) <= 2.15


def test_inversion_png_row_zero_is_z_max_and_x_slice_record(paths):
    write_inversion(paths)
    data, record = slices.inversion(SHOT, 1010, paths)
    pixels = np.asarray(Image.open(io.BytesIO(data)))
    assert pixels.dtype == np.uint8 and pixels.shape == (3, 4)
    assert record["t_ms"] == pytest.approx(1016.7)
    assert record["r_range"] == [1.0, 2.0]
    assert record["z_range"] == pytest.approx([-1.4, -0.4])
    assert record["units"] == "a.u."
    assert record["source"] == "emission_structure_190001.sav"
    frame = np.arange(3)[:, None] * 10 + np.arange(4)
    shot = np.stack([frame, frame])  # the range is the shot's, both frames
    assert (record["vmin"], record["vmax"]) == pytest.approx(
        tuple(np.percentile(shot, slices.PERCENTILES))
    )
    # Row 0 holds the top elevation (z index 2), column 0 the inner radius.
    assert pixels[0, 0] > pixels[1, 0] > pixels[2, 0]
    assert pixels[0, 0] < pixels[0, 3]
    assert pixels[0, 3] == 255 and pixels[2, 0] == 0


def test_a_rewritten_file_is_read_again(paths):
    path = write_equilibrium(paths)
    assert slices.available(SHOT, paths)[0]["n_t"] == 3
    write_equilibrium(paths, times=(100.0, 200.0))
    os.utime(path, ns=(1, 1))
    assert slices.available(SHOT, paths)[0]["n_t"] == 2


def test_decreasing_radii_put_the_inner_radius_in_column_zero(paths):
    write_inversion(paths, radii=(2.0, 1.0))  # ri = 3 is now the inner radius
    data, record = slices.inversion(SHOT, 1000, paths)
    pixels = np.asarray(Image.open(io.BytesIO(data)))
    assert record["r_range"] == [1.0, 2.0]
    assert pixels[0, 0] > pixels[0, 3]  # column 0 holds ri = 3 (the larger value)
    assert pixels[0, 0] == 255 and pixels[2, 3] == 0


@pytest.mark.parametrize(
    "rule", ["EFIT01_fallback", "EFIT01_fallback_EFIT02_sparse"]
)
def test_an_efit01_fallback_keeps_its_rule(paths, rule):
    write_equilibrium(paths, tree="EFIT01", rule=rule)
    view = slices.equilibrium(SHOT, 100, paths)
    assert (view["source"], view["source_rule"]) == ("EFIT01", rule)
    assert slices.available(SHOT, paths)[0]["label"] == "EFIT equilibrium (EFIT01)"


def test_no_wall_and_no_grid_falls_back_to_the_boundary(paths):
    write_equilibrium(paths, grid=False, wall=np.zeros((0, 2)))
    view = slices.equilibrium(SHOT, 100, paths)
    assert view["wall"] == [] and view["levels"] == []
    assert view["r_range"] == pytest.approx([1.2, 2.2], abs=1e-4)
    assert view["z_range"] == pytest.approx([-0.9, 0.9], abs=1e-3)


def test_a_boundary_without_nbbbs_keeps_its_finite_points(paths):
    write_equilibrium(paths, counts=False)
    assert len(slices.equilibrium(SHOT, 100, paths)["boundary"]) == 41


def test_a_broken_file_leaves_the_listing_and_answers_missing(paths):
    write_equilibrium(paths)
    broken = paths.root / "round4/detach/inversions" / f"{SHOT}.npz"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"not an npz")
    assert [k["kind"] for k in slices.available(SHOT, paths)] == ["equilibrium"]
    with pytest.raises(slices.SliceMissing, match="unreadable"):
        slices.inversion(SHOT, 1000, paths)


def test_the_listing_reads_clocks_only(paths):
    path = write_inversion(paths)
    with np.load(path) as source:
        arrays = dict(source)
    arrays["radii"] = arrays["radii"][:3]  # frames no longer match the axes
    np.savez(path, **arrays)
    assert [k["kind"] for k in slices.available(SHOT, paths)] == ["inversion"]
    with pytest.raises(slices.SliceMissing, match="disagree with the axes"):
        slices.inversion(SHOT, 1000, paths)
