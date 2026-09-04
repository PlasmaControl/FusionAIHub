"""Discovering model folders and reading their cards.

A model folder is `models/<slug>/` with a `README.md` whose YAML front
matter follows the HuggingFace model-card standard plus one custom
`labelmaker:` block, and a `spec.py` exposing `ADAPTER`. The card is the
single place a physicist reads to learn what a model is, where its weights
came from, what it consumes, and how its labels scored; `card_discrepancies`
keeps the card honest about the first two.
"""
from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path

import yaml

from .base import ModelAdapter

MODELS_DIR = Path(__file__).resolve().parent

_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def model_slugs() -> list[str]:
    """Model folder names, sorted."""
    return sorted(
        p.name
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and (p / "README.md").exists() and (p / "spec.py").exists()
    )


def card_path(slug: str) -> Path:
    return MODELS_DIR / slug / "README.md"


def parse_card(text: str) -> dict:
    """The YAML front matter of a model card."""
    m = _FRONT_MATTER.match(text)
    if not m:
        raise ValueError("card has no YAML front matter (expected a leading --- block)")
    data = yaml.safe_load(m.group(1))
    if not isinstance(data, dict):
        # ValueError, not TypeError, despite ruff's TRY004: `text` is always a
        # str, so this is a malformed *data file*, not a caller passing the
        # wrong argument type - and the sibling branch above raises ValueError
        # for the same category. A caller wanting to catch "bad card" should
        # need one except clause, not two.
        raise ValueError(  # noqa: TRY004
            f"card front matter is not a mapping: {type(data).__name__}"
        )
    return data


def read_card(slug: str) -> dict:
    return parse_card(card_path(slug).read_text())


def status(slug: str) -> str:
    return read_card(slug)["labelmaker"]["status"]


def implemented() -> list[str]:
    return [s for s in model_slugs() if status(s) == "implemented"]


def scaffolds() -> list[str]:
    return [s for s in model_slugs() if status(s) == "scaffold"]


def load_adapter(slug: str) -> ModelAdapter:
    """The `ADAPTER` of a model's `spec.py`.

    A scaffold's `spec.py` raises `NotImplementedError` at import, which is
    the intended behaviour: the folder documents a model that cannot run yet.
    """
    module = importlib.import_module(f"labelmaker.models.{slug}.spec")
    adapter = getattr(module, "ADAPTER", None)
    if adapter is None:
        raise AttributeError(f"{slug}/spec.py defines no ADAPTER")
    return adapter


def card_discrepancies(slug: str) -> list[str]:
    """Where a card disagrees with its own `spec.py`.

    Checked: the input mapping (`"<model name> <- <canonical>"`), the output
    names with their task and activation, the framework, the artifact list
    and the ensemble size. A card that drifts from the code is worse than no
    card, so a test fails on any entry here.
    """
    card = read_card(slug)["labelmaker"]
    adapter = load_adapter(slug)
    out: list[str] = []
    want_in = [f"{f.model_name} <- {f.canonical}" for f in adapter.input_spec.fields]
    got_in = list(card.get("inputs") or [])
    if got_in != want_in:
        out.append(f"inputs: card {got_in} != spec {want_in}")
    want_out = [
        {"name": f.name, "task": f.task, "activation": f.activation}
        for f in adapter.output_spec.fields
    ]
    got_out = [
        {"name": d.get("name"), "task": d.get("task"), "activation": d.get("activation")}
        for d in (card.get("outputs") or [])
    ]
    if got_out != want_out:
        out.append(f"outputs: card {got_out} != spec {want_out}")
    if card.get("framework") != adapter.framework:
        out.append(f"framework: card {card.get('framework')} != {adapter.framework}")
    if card.get("card_id") != adapter.card_id:
        out.append(f"card_id: card {card.get('card_id')} != {adapter.card_id}")
    artifacts = list((card.get("upstream") or {}).get("artifacts") or [])
    if artifacts != list(adapter.artifacts):
        out.append(f"artifacts: card {artifacts} != spec {list(adapter.artifacts)}")
    if int(card.get("ensemble_n") or 1) != adapter.ensemble_n:
        out.append(f"ensemble_n: card {card.get('ensemble_n')} != {adapter.ensemble_n}")
    return out


def sha256_of(path) -> str:
    """Hex digest of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_artifacts(slug: str, model_dir) -> None:
    """Raise unless every artifact the spec loads matches the card's sha256.

    Labels are only worth what the weights behind them are, so `run.py`'s
    `_predictor` calls this before loading a model rather than silently
    producing a label file whose provenance is wrong. Absence and corruption
    are reported separately: they are different failures.
    """
    card = read_card(slug)["labelmaker"]
    expected = (card.get("upstream") or {}).get("sha256") or {}
    if not expected:
        raise ValueError(f"{slug}: card records no sha256 for its artifacts")
    # Verify the artifacts the SPEC loads, not merely the ones the card
    # happens to list. Iterating `expected` alone fails open: a card with ten
    # entries under `artifacts` but three under `sha256` - a hand edit, a
    # partial regeneration, a bad merge - would pass while seven unverified
    # weight files got loaded. This is the only guard between the pipeline and
    # unverified weights, so it has to fail closed.
    unlisted = [a for a in load_adapter(slug).artifacts if a not in expected]
    if unlisted:
        raise ValueError(
            f"{slug}: card records no sha256 for {unlisted}; every artifact the "
            "spec loads must be verifiable"
        )
    model_dir = Path(model_dir)
    absent, mismatched = [], []
    for name, want in sorted(expected.items()):
        path = model_dir / name
        if not path.exists():
            absent.append(name)
            continue
        got = sha256_of(path)
        if got != want:
            mismatched.append(f"{name}: {got[:12]} != card {want[:12]}")
    problems = []
    if absent:
        problems.append(f"missing from {model_dir}: {absent}")
    if mismatched:
        problems.append(f"sha256 mismatch: {mismatched}")
    if problems:
        raise ValueError(f"{slug}: " + "; ".join(problems))
