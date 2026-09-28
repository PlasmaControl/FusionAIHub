"""Draw the paper's figures and tables into `$LABELER_ROOT/paper/`.

    PYTHONPATH=src pixi run --frozen -e labelmaker \\
        python scripts/labeler/paper/make_figures.py \\
        [--version v1] --copy-to dev/label_paper/figures

`--version v2` reads v2's models once they exist, and skips the AE products
until then. See `labeler.paper.build` for what it reads, draws and skips.
"""

from labeler.paper.build import main

if __name__ == "__main__":
    raise SystemExit(main())
