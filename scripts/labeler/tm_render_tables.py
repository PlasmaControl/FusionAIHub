#!/usr/bin/env python
"""Render the final committed TM TeX tables at a 6.75-inch text width.

The preview is generated from the exact TeX, never transcribed numbers. Each
PDF/150-dpi PNG has hashes linking the source TeX, label table and benchmark JSON.
Temporary TeX products are confined to TMPDIR; rendered artifacts live in round4/tm.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TM = (
    Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
    / "round4/tm"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tmp = Path(os.environ["TMPDIR"]) / "table_render_fix4"
    tmp.mkdir(parents=True, exist_ok=True)
    out = TM / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "table_tm_benchmark",
        "table_tm_benchmark_appendix",
        "table_tm_paired",
        "table_tm_shot_sets",
    ):
        source = REPO / "docs/labeler" / f"{name}.tex"
        tex = tmp / f"{name}.tex"
        tex.write_text(
            r"\documentclass{article}"
            "\n"
            r"\usepackage{booktabs,array,geometry}"
            "\n"
            r"\geometry{paperwidth=7.25in,paperheight=5.5in,margin=0.25in}"
            "\n"
            r"\pagestyle{empty}\begin{document}"
            "\n" + source.read_text().replace("[t]", "[!ht]") + "\n" + r"\end{document}"
            "\n"
        )
        run = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={tmp}",
                str(tex),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (tmp / f"{name}.stdout").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f"TeX rendering failed: {tmp / (name + '.stdout')}")
        log = (tmp / f"{name}.log").read_text()
        if "Overfull \\hbox" in log:
            raise RuntimeError(f"table exceeds its 6.75-inch text width: {name}")
        preview = out / f"{name}_preview"
        pdf = preview.with_suffix(".pdf")
        pdf.write_bytes((tmp / f"{name}.pdf").read_bytes())
        subprocess.run(
            ["pdftoppm", "-png", "-r", "150", "-singlefile", str(pdf), str(preview)],
            check=True,
            capture_output=True,
        )
        record = {
            "made_by": "scripts/labeler/tm_render_tables.py",
            "tex_source": str(source.relative_to(REPO)),
            "tex_sha256": sha(source),
            "benchmark_sha256": sha(
                REPO
                / "data/events/neoclassical_tearing_mode/benchmark/tm_benchmark.json"
            ),
            "labels_sha256": sha(
                REPO
                / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
            ),
            "png_sha256": sha(preview.with_suffix(".png")),
            "pdf_sha256": sha(pdf),
            "text_width_inches": 6.75,
            "dpi": 150,
            "overfull_boxes": False,
        }
        preview.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        print(preview)


if __name__ == "__main__":
    main()
