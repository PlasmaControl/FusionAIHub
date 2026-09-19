"""The frame layout read from a pinned bundle's codec MANIFEST.json.

The static FROZEN_MODALITIES table describes one codec generation. A pinned bundle carries its
own manifest, and that -- not the table -- is what a checkpoint trained against it agrees with,
so the layout has to be constructible from the manifest alone.
"""

import json

from tokamak_foundation_model.ignite import dynamics_config as dc


def test_modalities_from_manifest_preserves_order_and_totals(tmp_path):
    m = {
        "modalities": {
            "ece": {"family": "spectro", "n_tok": 192, "codebook_size": 1000},
            "mirnov": {"family": "spectro", "n_tok": 192, "codebook_size": 1000},
            "filterscopes": {"family": "fastts", "n_tok": 5, "codebook_size": 1000},
        }
    }
    p = tmp_path / "MANIFEST.json"
    p.write_text(json.dumps(m))
    mods = dc.modalities_from_manifest(p)
    assert [x.name for x in mods] == ["ece", "mirnov", "filterscopes"]
    assert sum(x.n_tok for x in mods) == 389
    assert mods[1] == dc.ModalitySpec("mirnov", "spectro", 192, 1000)
