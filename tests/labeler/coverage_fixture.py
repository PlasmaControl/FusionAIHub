"""The critic's controlled D-alpha dropout, entirely inside a test's tmp_path."""

import h5py
import numpy as np


def gapped_filterscopes(corpus, shot=198658):
    corpus.mkdir(parents=True, exist_ok=True)
    t = np.arange(2**16 + 1, dtype=float) / 10000 - 0.05
    # Millisecond ELM pulses, the same Gaussian construction as the clock fixtures.
    y = np.ones(t.size)
    for peak in np.arange(0.2, 6.4, 0.15):
        y += np.exp(-0.5 * ((t - peak) / 0.0005) ** 2)
    y[(t >= 1) & (t <= 2)] = np.nan
    y[0] = y[-1] = np.nan
    with h5py.File(corpus / f"{shot}_processed.h5", "w") as f:
        group = f.create_group("filterscopes")
        group["xdata"] = t
        group["ydata"] = np.tile(y, (8, 1))
    return ((float(t[1]), float(t[t < 1][-1])),
            (float(t[t > 2][0]), float(t[-2])))
