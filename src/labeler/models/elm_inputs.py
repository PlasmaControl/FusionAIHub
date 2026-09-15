"""The 124 input columns the upstream ELM survival split pickle carries.

**Two orders exist and the difference is 118 of 124 columns.** Both are written
out verbatim in cell 43 of
`/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/data_processing.ipynb`:

* `current_diagnostic_order` - `CURRENT_DIAGNOSTIC_ORDER` below - puts the 48
  ECE channels at slots 6-53 and BES last. It is the order of
  `data/compiled_model10.pkl` and of the saved Keras graphs
  (`wpqh1_embedding_with_norm.keras`, whose baked normalisation means at slots
  56-59 are 8.6e13-1.3e14, i.e. CO2 line density);
* `new_diagnostic_order` - `SPLIT_COLUMN_ORDER` below - puts all 76 non-ECE
  columns first and `ece_slow_channel_1..48` last. Cell 43 materialises it as
  `data/reordered_model10.pkl`, and cell 47 builds
  `data/train_test_split_model10.pkl` **from that reordered file**.

So the split pickle - the rows labeler fits on - is in `SPLIT_COLUMN_ORDER`.
MEASURED 2026-09-06, not assumed: the pickle carries both `*_final_x` (raw) and
`*_final_x_normalized`, so the per-column normalisation upstream applied is
recoverable exactly as `s = std(raw)/std(norm)`, `m = mean(raw) - s*mean(norm)`,
and can be matched against the named `normalizations` dict inside
`data/testing_model.pkl`. All 124 columns of both the train and the test side
match `new_diagnostic_order` to a worst relative error of 1.2e-12; only 6 of 124
match `current_diagnostic_order` (slots 0-5, which the two orders share).

Under `SPLIT_COLUMN_ORDER` the blocks are:

    0-5     ip, bt, gas, pinj, tinj, ech
    6-7     pcphd02, pcphd03            (the two D-alpha photodiodes)
    8-11    co2_density_slow r0/v1/v2/v3
    12-75   bes_slow_channel_1..64
    76-123  ece_slow_channel_1..48

Two sets matter for labeler:

* `all124` - everything, what upstream fitted;
* `no_bes` - the 60 columns that are not BES, i.e. slots 0-11 and 76-123. BES
  fills only 2 of 24 sampled corpus shots, so a model that needs it cannot be
  served at corpus scale; `no_bes` is the set that can be served today.

Selection is always by NAME (`column_indices`), never by a hard-coded slice.
The first fit of `no_bes` (Task 8a, commit 54d201d) took slots 0-59 of the
pickle believing them to be the non-BES columns; under the real order those
slots are the 12 non-ECE, non-BES columns plus `bes_slow_channel_1..48`, so
that model dropped every ECE channel and required 48 BES ones. Naming the set
is what makes that a wrong index rather than a silently wrong model.
"""
from __future__ import annotations

SUFFIX = "_downsampled"

#: `current_diagnostic_order`, cell 43. The Keras graphs' order. NOT the order
#: of `train_test_split_model10.pkl` - see the module docstring.
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

#: `new_diagnostic_order`, cell 43: the column order of
#: `train_test_split_model10.pkl`, measured to 1.2e-12 (module docstring).
SPLIT_COLUMN_ORDER: tuple[str, ...] = (
    "ip" + SUFFIX,
    "bt" + SUFFIX,
    "gas" + SUFFIX,
    "pinj" + SUFFIX,
    "tinj" + SUFFIX,
    "ech" + SUFFIX,
    "pcphd02" + SUFFIX,
    "pcphd03" + SUFFIX,
    *(f"co2_density_slow_{c}{SUFFIX}" for c in ("r0", "v1", "v2", "v3")),
    *(f"bes_slow_channel_{i}{SUFFIX}" for i in range(1, 65)),
    *(f"ece_slow_channel_{i}{SUFFIX}" for i in range(1, 49)),
)

BES_COLUMNS: tuple[str, ...] = tuple(
    name for name in SPLIT_COLUMN_ORDER if name.startswith("bes_slow_channel_")
)

COLUMN_SETS: dict[str, tuple[str, ...]] = {
    "all124": SPLIT_COLUMN_ORDER,
    "no_bes": tuple(
        name for name in SPLIT_COLUMN_ORDER if name not in set(BES_COLUMNS)
    ),
}


class UnknownColumnSet(KeyError):
    """The caller named a column set this module does not define."""


def column_indices(set_name: str) -> tuple[int, ...]:
    """Positions of a named set's columns within `SPLIT_COLUMN_ORDER`.

    Selection is by name, never by a hard-coded slice: an upstream reorder
    would then be a wrong answer here rather than a silently wrong model.
    """
    try:
        wanted = COLUMN_SETS[set_name]
    except KeyError:
        raise UnknownColumnSet(
            f"{set_name!r}; known sets are {sorted(COLUMN_SETS)}"
        ) from None
    position = {name: i for i, name in enumerate(SPLIT_COLUMN_ORDER)}
    return tuple(position[name] for name in wanted)
