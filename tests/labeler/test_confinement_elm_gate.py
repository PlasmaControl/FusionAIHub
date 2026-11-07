import numpy as np

from labeler.confinement import elm_gate as g


def _filterscope(onsets_ms, height, n_ms=4000, dt=0.1):
    """A flat filterscope (1.0) with a 1 ms spike of ``height`` at each onset."""
    fs = np.ones(int(n_ms / dt), np.float32)
    for t in onsets_ms:
        i = int(np.rint(t / dt))
        fs[i:i + int(1 / dt)] = height
    return fs[None], 0.0, dt


def test_train_gates_qh_inside_only_and_breakthrough_elms_leave_wpqh():
    qh_train = np.arange(1000, 1401, 20.0)  # 50 Hz for 400 ms: 21 onsets
    wp_train = np.arange(2000, 3001, 10.0)  # 100 Hz for 1 s
    lone = np.array([3500.0])  # one breakthrough ELM
    on = np.r_[qh_train, wp_train, lone]
    sizes = g.onset_sizes(on, *_filterscope(on, 3.0))
    assert np.allclose(sizes, 3.0)
    qh, wp = g.masks(on, sizes, 4000)
    assert qh[1200] and not qh[900] and not qh[1450]  # inside the train, not 50 ms after it
    assert not wp[1200] and wp[2500]  # 21 onsets in 500 ms are too few for WPQH
    assert not qh[3500] and not wp[3500]
    small = g.onset_sizes(on, *_filterscope(on, 1.2))  # below the size cut: no train
    assert not g.masks(on, small, 4000)[0].any()


def test_flip_moves_qh_and_wpqh_to_h_only_where_called():
    prob = np.array([[0.1, 0.6, 0.1, 0.1],  # L
                     [0.1, 0.1, 0.1, 0.2],  # H
                     [0.7, 0.2, 0.1, 0.3],  # QH
                     [0.1, 0.1, 0.7, 0.4]])  # WPQH
    qh = np.array([True, True, True, False])
    wp = np.array([False, True, False, True])
    flip = g.flips(prob, qh, wp)
    assert flip.tolist() == [True, False, False, True]  # QH in qh; L untouched; WPQH needs wp
    out = g.move(prob, flip)
    assert out.argmax(0).tolist() == [1, 0, 3, 1]
    assert np.allclose(out.sum(0), prob.sum(0)) and np.allclose(out[:, ~flip], prob[:, ~flip])
