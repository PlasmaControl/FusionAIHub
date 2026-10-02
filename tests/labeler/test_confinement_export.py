import numpy as np
import pandas as pd

from labeler.confinement.export import export
from labeler.events.interval_tables import read_label_grid


def test_unified_h_l_exports_are_complementary_and_keep_unknown(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(
        {
            "shot": [1, 1, 1],
            "t_start": [0.0, 50.0, 100.0],
            "t_end": [50.0, 100.0, 150.0],
            "label": [0, 1, -1],
        }
    ).to_csv(run / "targets.csv", index=False)
    export(run)
    h = read_label_grid(run / "format/high_confinement_mode/shots/1.npz")["label"]
    l = read_label_grid(run / "format/low_confinement_mode/shots/1.npz")["label"]
    assert np.all(h[0] == 0) and np.all(l[0] == 1)
    assert np.all(h[1] == 1) and np.all(l[1] == 0)
    assert np.isnan(h[2]).all() and np.isnan(l[2]).all()
    assert np.isnan(h[3:]).all() and np.isnan(l[3:]).all()
