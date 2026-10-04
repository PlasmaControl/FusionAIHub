"""Which merged-interval sources a window's interval carries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeler.confinement import bes_windows as bw


def test_source_mask_marks_windows_of_intervals_with_the_source():
    intervals = pd.DataFrame(
        {
            "shot": [5, 5, 6],
            "interval": [0, 1, 0],
            "sources": ["kevin_bes", "jalal|kevin_workbook", "jalal|kevin_bes"],
        }
    )
    got = bw.source_mask(
        np.array([5, 5, 6, 6]), np.array([0, 1, 0, 3]), intervals, "kevin_bes"
    )
    # the last window names an interval the table does not hold: not marked
    assert got.tolist() == [True, False, True, False]
    assert bw.source_mask(
        np.array([5]), np.array([1]), intervals, "kevin_workbook"
    ).tolist() == [True]
