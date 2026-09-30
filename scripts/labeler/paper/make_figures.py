"""Draw the paper's figures and tables into `$LABELER_ROOT/paper/`.

    PYTHONPATH=src pixi run --frozen -e labelmaker \\
        python scripts/labeler/paper/make_figures.py \\
        [--version v1] --copy-to dev/label_paper/figures

`--version v2` reads v2's frame model once it exists, and skips its products
until then; `--seg-version` (default v1) names the segmentation's apart, and
`--interpreter-seg-version` (default v3) fig_interpreter's; `--shot` names the
interpreter's shot, a roster candidate.
See `labeler.paper.build` for what it reads, draws and skips.
"""

from labeler.paper.build import main

if __name__ == "__main__":
    raise SystemExit(main())
