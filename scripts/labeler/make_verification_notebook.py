"""Scaffold a category's verification.ipynb.

The generic notebook plots `ip`, `betan` and `pinj_total` from the feature store
against the saved labels. That is enough to confirm a shot exists and that its
labels sit inside the discharge, and nowhere near enough to verify a
phenomenon - which is the point. Whoever takes a category on replaces the
panel cell with the traces that actually settle it, the way
`minimum_safety_factor`, `sawtooth_oscillation`, `alfven_eigenmode` and
`fishbone` already have.

    pixi run -e labelmaker python scripts/labeler/make_verification_notebook.py \
        --event detachment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

from labeler.config import Paths

HEADER = """# {title} - verification

Use the **Python (FAITH labelmaker)** kernel. Set `shot` below, drag a time
range on any panel, then press *Mark present* / *Mark absent*, *Verify* and
*Save*.

**These are generic panels.** `ip`, `betan` and `pinj_total` show that the shot
exists and that its labels sit inside the discharge. They do not show whether
{title} actually happened. Replace the panel cell below with the traces that
settle this phenomenon, then delete this paragraph.

*Save* writes two files. The corrected intervals go to a NEW file,
`review/<shot>__<reviewer>__<stamp>.csv`, in the same five-column schema as
`format/`; corrections are append-only, so every press of *Save* writes its
own file and nothing under `review/` is ever overwritten or deleted by this
tooling - a second reviewer cannot destroy the first's work. Merging rows
into `format/` is a manual step the repository author does by hand. Your
review also goes into `shots.csv`, which is the one file edited in place and
only ever gains a reviewer and today's date. `tier` and `holdout` are
curation calls set by hand in that file, not derived from who has reviewed a
shot.
"""

SETUP = """%load_ext autoreload
%autoreload 2"""

PANELS = '''from labeler.config import Paths
from labeler.events.verify import Panel, review
from labeler.features.store import read_feature

event = "{event}"
shot = 1  # replace with a shot from shots.csv
source = "format/shots"

features = Paths.from_env().features_file(shot)

def trace(name, ylabel):
    """One scalar feature as a panel; seconds on disk, milliseconds on the plot."""
    array = read_feature(features, name)
    return Panel(title=name, x=array.x * 1000.0, y=array.y, ylabel=ylabel)

panels = [
    trace("ip", "A"),
    trace("betan", ""),
    trace("pinj_total", "kW"),
]'''

REVIEW = """session = review(event, shot, panels, source=source)
session"""

FOOTER = """After pressing *Save*, check what was written. Every save writes its
own file, so list what this shot now has and read back the most recent one:

```python
from labeler.events.verify import corrections_for, read_latest_corrections

for path in corrections_for(event, shot):
    print(path.name)
read_latest_corrections(event, shot)
```

These files are never overwritten; the rows are merged into `format/` by hand.
"""


def build(event: str) -> nbformat.NotebookNode:
    """The generic verification notebook for one category."""
    title = event.replace("_", " ")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(HEADER.format(title=title)),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_code_cell(PANELS.format(event=event)),
        nbformat.v4.new_code_cell(REVIEW),
        nbformat.v4.new_markdown_cell(FOOTER),
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python (FAITH labelmaker)",
        "language": "python",
        "name": "faith-labelmaker",
    }
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = Paths.from_env().label_tables if args.root is None else args.root
    path = root / args.event / "verification.ipynb"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build(args.event), path)
    print("wrote", path)


if __name__ == "__main__":
    main()
