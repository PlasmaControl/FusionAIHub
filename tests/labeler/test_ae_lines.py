"""labeler.ae.lines: a fixed pickup line is notched, a drifting mode is not, and the torch path
(ae_fdp_label.py infer --notch) gives the numpy path's masks and output."""
import numpy as np
import pytest

from labeler.ae import lines


def _shot(seed=0, frames=6000, chords=4, bins=348):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (frames, chords, bins)).astype(np.float32)
    x[:, :2, 200] += 2.0  # a pickup line over the whole shot in chords 0 and 1
    x[3000:, 2, 200] += 2.0  # the same line in chord 2 for the second half only
    k = (60 + np.arange(frames) * 40 // frames).astype(int)  # a mode drifting 40 bins over the shot
    x[np.arange(frames), 3, k] += 3.0
    return x


def test_line_notched_mode_kept():
    x = _shot()
    out, masks = lines.notch_shot(x)
    assert all(m[:, 200].all() for m in masks[:2])  # the line, in every block of chords 0 and 1
    assert masks[2][-1, 200]  # its second half in chord 2, through the known frequency
    assert not masks[3][:, 60:100].any()  # the drifting mode is never flagged
    np.testing.assert_array_equal(out[:, 3, 60:100], x[:, 3, 60:100])
    assert out[:, 0, 200].mean() < 0.5 < x[:, 0, 200].mean()


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    x = _shot(seed=1)
    ref, ref_masks = lines.notch_shot(x)
    got, got_masks = lines.notch_shot_torch(torch.from_numpy(x))
    for a, b in zip(ref_masks, got_masks, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(got.numpy(), ref, atol=1e-5)
