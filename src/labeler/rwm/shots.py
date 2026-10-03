"""Which shots the RWM baseline uses, and why.

Jeremy Hanson's onset tables give 56 onsets on 33 shots from two campaigns, none
in the corpus. The comparison shots must come from the same campaigns, because a
model fitted to one era against another learns the era, not the physics. Two
rules choose them, both from the operator logbook's run record (`run`, the run
day, and `run_title`):

``same_day``
    every other plasma shot of a run day that holds a Hanson shot;
``same_experiment``
    every plasma shot of a run day of the same calendar year whose title is one
    of the high-beta or RWM experiments the Hanson days belong to
    (`EXPERIMENT_TITLES`).

A comparison shot is *unlabelled*, not negative: the onset tables list onsets and
say nothing about shots they omit, and these are RWM experiment days.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

#: The shot ranges read from the logbook. They cover the two Hanson campaigns.
LOG_RANGES = ((156000, 160000), (175500, 177000))

RUN_COLUMNS = (
    "shot",
    "run",
    "shot_entered",
    "run_title",
    "mpid",
    "configuration",
    "shot_type",
)

#: Title fragments (lower case, whitespace collapsed) of the experiments the
#: Hanson run days belong to: the 2014 beta_N ~ 5 and kinetic-RWM-stability days
#: and the 2018 RWM-control and high-beta_p scenario days.
EXPERIMENT_TITLES = (
    "explore access to bn~5",
    "explore access to beta_n~5",
    "testing kinetic rwm stabilization",
    "rwm control development",
    "fully non-inductive high beta-p scenario",
    "control of divertor radiation with impurity seeding in high betap",
)

ROLES = ("hanson", "same_day", "same_experiment")


def squash(title) -> str:
    """A run title in lower case with its whitespace collapsed."""
    return re.sub(r"\s+", " ", str(title)).strip().lower()


def read_runs(logs_jsonl, ranges=LOG_RANGES) -> pd.DataFrame:
    """The logbook's one-row-per-shot run record for shots inside `ranges`.

    The dump is 616 MB with every line starting `{"shot": <digits>,`, so a line is
    decoded only when its shot number is inside a range.
    """
    rows = {}
    with Path(logs_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                shot = int(line[8 : line.index(",")])
            except ValueError:
                continue
            if not any(lo <= shot <= hi for lo, hi in ranges):
                continue
            record = json.loads(line)
            rows.setdefault(shot, {key: record.get(key) for key in RUN_COLUMNS})
    return pd.DataFrame(list(rows.values()), columns=list(RUN_COLUMNS)).sort_values(
        "shot", ignore_index=True
    )


def choose(runs: pd.DataFrame, hanson_shots) -> pd.DataFrame:
    """Hanson shots plus their same-day and same-experiment comparison shots.

    Returns `shot, role, run, run_title, mpid`, one row per shot, `role` being
    the first of `ROLES` that applies. Only plasma shots are kept. A Hanson shot
    missing from the logbook keeps an empty run.
    """
    hanson = sorted({int(s) for s in hanson_shots})
    runs = runs.drop_duplicates("shot").set_index("shot", drop=False)
    runs = runs.assign(year=runs.shot_entered.astype(str).str[:4])
    own = runs.loc[runs.index.intersection(hanson)]
    days, years = set(own.run), set(own.year)
    titled = runs.run_title.map(squash).map(
        lambda title: any(fragment in title for fragment in EXPERIMENT_TITLES)
    )
    plasma = runs.shot_type == "plasma"
    same_day = runs.run.isin(days) & plasma
    same_experiment = titled & runs.year.isin(years) & plasma
    rows = []
    for shot in sorted(set(hanson) | set(runs.index[same_day | same_experiment])):
        if shot in hanson:
            role = "hanson"
        elif same_day.get(shot, False):
            role = "same_day"
        else:
            role = "same_experiment"
        record = runs.loc[shot] if shot in runs.index else None
        rows.append(
            {
                "shot": shot,
                "role": role,
                "run": "" if record is None else record.run,
                "run_title": "" if record is None else squash(record.run_title),
                "mpid": "" if record is None else record.mpid,
            }
        )
    return pd.DataFrame(rows, columns=["shot", "role", "run", "run_title", "mpid"])
