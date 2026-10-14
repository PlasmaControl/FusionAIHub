"""elm-inst: instance targets, focal loss, peak decoding and the network's shapes."""

import numpy as np
import pandas as pd
import torch

from labeler.elm import inputs, inst, net


def test_smith_target_marks_the_window_known_and_one_peak_cell():
    n = 2000  # 200 ms from -50 ms
    target = inst.smith_target([10.03], [5.0], [20.0], n)
    c = int(inst.cell_of(10.03))
    assert target.heat[c] == 1.0 and target.heat.argmax() == c
    assert target.known[inst.cell_of(5.0)] and not target.known[inst.cell_of(4.0)]
    assert not target.known[inst.cell_of(25.0)]
    assert np.isclose(inst.time_of(c), 10.05)


def test_review_target_guards_absent_edges_and_keeps_pseudo_onsets_in_present_time():
    spans = pd.DataFrame(
        {
            "t_start": [0.0, 100.0],
            "t_end": [100.0, 200.0],
            "kind": ["absent", "crowd"],
        }
    )
    n = 3000
    target = inst.review_target(spans, [50.0, 150.0], n)
    assert target.known[inst.cell_of(50.0)] and target.heat[inst.cell_of(50.0)] == 0
    assert not target.known[inst.cell_of(2.0)] and not target.known[inst.cell_of(97.0)]
    c = int(inst.cell_of(150.0))
    assert target.heat[c] == 1.0 and target.weight[c] == inst.PSEUDO_WEIGHT
    assert not target.known[inst.cell_of(160.0)]  # crowd time: no negative asserted


def test_focal_loss_prefers_the_right_peak_and_ignores_unknown_cells():
    heat = torch.zeros(1, 100)
    heat[0, 50] = 1.0
    known = torch.ones(1, 100, dtype=torch.bool)
    weight = torch.ones(1, 100)
    good = torch.full((1, 100), -8.0)
    good[0, 50] = 8.0
    bad = torch.full((1, 100), -8.0)
    bad[0, 20] = 8.0
    assert inst.focal_loss(good, heat, known, weight) < inst.focal_loss(
        bad, heat, known, weight
    )
    known[0, 20] = False
    assert torch.isfinite(inst.focal_loss(bad, heat, known, weight))


def test_peaks_keep_the_higher_of_close_maxima_and_onsets_gate_by_occupancy():
    heat = np.zeros(1000)
    heat[100], heat[110], heat[300] = 0.9, 0.8, 0.6  # 110 is 1 ms from 100
    assert inst.peaks(heat, 0.5).tolist() == [100, 300]
    assert inst.peaks(heat, 0.7).tolist() == [100]
    gate = np.zeros(200)
    gate[:20] = 1.0  # occupancy only near the first peak (t = -40 ms)
    t = inst.onsets(heat, 0.5, gate, 0.5)
    assert np.allclose(t, [inst.time_of(100)])


def test_match_is_one_to_one_nearest_first():
    tp, fp, fn, off = inst.match([10.0, 10.5, 30.0], [10.4, 50.0], 2.0)
    assert (tp, fp, fn) == (1, 2, 1) and np.isclose(off[0], 0.1)


def test_instnet_returns_ms_occupancy_and_cell_instance_logits():
    model = inst.InstNet()
    x = torch.zeros(2, inputs.N_CHANNELS, net.UNIT)
    occ, ins = model(x)
    assert occ.shape == (2, net.UNIT // inputs.CELLS_PER_MS)
    assert ins.shape == (2, net.UNIT)
