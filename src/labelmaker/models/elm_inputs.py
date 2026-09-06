"""The 124 input columns the upstream ELM survival model was fitted on.

The order is `current_diagnostic_order` from cell 43 of
`/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/data_processing.ipynb`, which is
the order the columns sit in inside
`/projects/EKOLEMEN/wpqh_elm_hiro/data/train_test_split_model10.pkl`. It is
recorded here so a column set can be named in code and in a card rather than
written out as a slice, and so the trainer can assert the order it loaded.

**There is a second order and it is a trap.** `new_diagnostic_order` puts the 76
non-ECE columns first and `ece_slow_channel_1..48` last; it is what the PCS
wants and it is materialised in `data/reordered_model10.pkl`. Feeding it to a
model fitted on this order produces silent garbage, so every consumer names the
set through `COLUMN_SETS` instead of slicing by position.

Two sets matter for labelmaker:

* `all124` - everything, what upstream fitted;
* `no_bes` - slots 0-59, everything except `bes_slow_channel_1..64`. BES fills
  only 2 of 24 sampled corpus shots, so a model that needs it cannot be served
  at corpus scale without an fdp/toksearch BES fetch. `no_bes` is the set that
  can be served today.
"""
from __future__ import annotations

SUFFIX = "_downsampled"

CURRENT_DIAGNOSTIC_ORDER: tuple[str, ...] = (
    "ip" + SUFFIX,
    "bt" + SUFFIX,
    "gas" + SUFFIX,
    "pinj" + SUFFIX,
    "tinj" + SUFFIX,
    "ech" + SUFFIX,
    *(f"ece_slow_channel_{i}{SUFFIX}" for i in range(1, 49)),
    "pcphd02" + SUFFIX,
    "pcphd03" + SUFFIX,
    *(f"co2_density_slow_{c}{SUFFIX}" for c in ("r0", "v1", "v2", "v3")),
    *(f"bes_slow_channel_{i}{SUFFIX}" for i in range(1, 65)),
)

BES_COLUMNS: tuple[str, ...] = tuple(
    name for name in CURRENT_DIAGNOSTIC_ORDER if name.startswith("bes_slow_channel_")
)

COLUMN_SETS: dict[str, tuple[str, ...]] = {
    "all124": CURRENT_DIAGNOSTIC_ORDER,
    "no_bes": tuple(
        name for name in CURRENT_DIAGNOSTIC_ORDER if name not in set(BES_COLUMNS)
    ),
}


class UnknownColumnSet(KeyError):
    """The caller named a column set this module does not define."""


def column_indices(set_name: str) -> tuple[int, ...]:
    """Positions of a named set's columns within `CURRENT_DIAGNOSTIC_ORDER`.

    Selection is by name, never by a hard-coded slice: an upstream reorder
    would then be a wrong answer here rather than a silently wrong model.
    """
    try:
        wanted = COLUMN_SETS[set_name]
    except KeyError:
        raise UnknownColumnSet(
            f"{set_name!r}; known sets are {sorted(COLUMN_SETS)}"
        ) from None
    position = {name: i for i, name in enumerate(CURRENT_DIAGNOSTIC_ORDER)}
    return tuple(position[name] for name in wanted)
