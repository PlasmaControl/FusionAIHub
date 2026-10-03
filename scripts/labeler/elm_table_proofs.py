#!/usr/bin/env python
"""Compile every ELM table standalone and record hashes of all rendered pages.

Run through the labelmaker pixi environment with TMPDIR set to the stream temp.
After inspecting every recorded PNG, --record-viewed marks the unchanged audit.
Large PDF/PNG proofs stay under $LABELER_ROOT/round4/elm/table-proofs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs" / "labeler" / "elm"
AUDIT = OUTPUTS / "table_render_verification.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mark_viewed(audit):
    record = json.loads(audit.read_text())
    for entry in record["artifacts"]:
        for artifact in (entry["source"], entry["pdf"], *entry["pngs"]):
            path = Path(artifact["path"])
            if sha256(path) != artifact["sha256"]:
                raise ValueError(f"Rendered artifact changed since inspection: {path}")
        entry["viewed"] = True
        for png in entry["pngs"]:
            png["viewed"] = True
    record["script"]["sha256"] = sha256(Path(__file__))
    record["viewed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["all_pages_viewed"] = True
    audit.write_text(json.dumps(record, indent=1) + "\n")
    print("Recorded inspection of", record["png_page_count"], "unchanged PNG pages")


def render(out, audit):
    tmp = Path(os.environ["TMPDIR"]) / "table-proofs"
    tmp.mkdir(exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted(OUTPUTS.rglob("*.tex"))
    current = {source.stem for source in sources}
    removed = (
        json.loads(audit.read_text()).get("removed_stale_proofs", [])
        if audit.exists()
        else []
    )
    # Old individual table proofs must not outlive their consolidated sources.
    for path in sorted(out.iterdir()):
        if path.suffix not in (".pdf", ".png"):
            continue
        stem = path.stem
        if path.suffix == ".png":
            stem = stem.rsplit("-", 1)[0]
        if stem.startswith(("table_elm_", "elm_table_")) and stem not in current:
            removed.append({"path": str(path), "sha256": sha256(path)})
            path.unlink()
    figures = Path(os.environ["LABELER_ROOT"]) / "round4/elm/figures"
    for path in sorted(figures.glob("tableproof*")):
        if path.is_file() and path.suffix in (".png", ".pdf", ".aux", ".log"):
            removed.append({"path": str(path), "sha256": sha256(path)})
            path.unlink()
    artifacts = []
    for source in sources:
        name = source.stem
        build = tmp / name
        build.mkdir(exist_ok=True)
        wrapper = build / (name + ".tex")
        wrapper.write_text(
            r"\documentclass[10pt]{article}" + "\n"
            r"\usepackage[paperwidth=8.5in,paperheight=11in,margin=0.65in]{geometry}"
            + "\n"
            r"\usepackage[T1]{fontenc}" + "\n"
            r"\usepackage{booktabs,amsmath}" + "\n"
            r"\pagestyle{empty}" + "\n"
            r"\begin{document}" + "\n"
            r"\input{" + str(source) + "}\n"
            r"\clearpage\end{document}" + "\n"
        )
        for _ in range(2):
            proc = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(build),
                    str(wrapper),
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode:
                raise RuntimeError(f"Compile failed: {source}\n{proc.stdout[-7000:]}")
        log = (build / (name + ".log")).read_text()
        warnings = [
            line
            for line in log.splitlines()
            if any(
                term in line
                for term in (
                    "Overfull",
                    "Underfull",
                    "Float too large",
                    "undefined",
                    "LaTeX Warning",
                )
            )
        ]
        pdf = out / (name + ".pdf")
        shutil.copy2(build / (name + ".pdf"), pdf)
        for previous in out.glob(name + "-[0-9]*.png"):
            previous.unlink()
        subprocess.run(
            ["pdftoppm", "-r", "150", "-png", str(pdf), str(out / name)],
            check=True,
            capture_output=True,
        )
        images = sorted(out.glob(name + "-[0-9]*.png"))
        for image in images:
            subprocess.run(
                ["convert", str(image), "-fuzz", "2%", "-trim", "+repage", str(image)],
                check=True,
                capture_output=True,
            )
        artifacts.append(
            {
                "source": {"path": str(source), "sha256": sha256(source)},
                "pdf": {"path": str(pdf), "sha256": sha256(pdf)},
                "pngs": [
                    {"path": str(image), "sha256": sha256(image), "viewed": False}
                    for image in images
                ],
                "pages": len(images),
                "compile_passes": 2,
                "compile_exit_code": 0,
                "warning_count": len(warnings),
                "warnings": warnings,
                "viewed": False,
            }
        )
        print(name, "pages", len(images), "warnings", warnings, flush=True)
    record = {
        "script": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "proof_directory": str(out),
        "compile_protocol": "article 10 pt; 8.5x11 in; 0.65 in margins; booktabs/amsmath; two pdflatex passes",
        "raster_dpi": 150,
        "table_and_note_files": len(artifacts),
        "png_page_count": sum(a["pages"] for a in artifacts),
        "compile_failures": 0,
        "warning_count": sum(a["warning_count"] for a in artifacts),
        "all_pages_viewed": False,
        "removed_stale_proofs": removed,
        "artifacts": artifacts,
    }
    audit.write_text(json.dumps(record, indent=1) + "\n")
    print("Verified", len(artifacts), "files;", record["png_page_count"], "pages")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--audit", type=Path, default=AUDIT)
    ap.add_argument("--record-viewed", action="store_true")
    args = ap.parse_args()
    if args.record_viewed:
        mark_viewed(args.audit)
    else:
        out = (
            args.out_dir or Path(os.environ["LABELER_ROOT"]) / "round4/elm/table-proofs"
        )
        render(out, args.audit)


if __name__ == "__main__":
    main()
