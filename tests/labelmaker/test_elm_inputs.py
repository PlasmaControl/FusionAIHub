"""The ELM model's 124 input columns and the two named column sets.

**Two orders exist and they differ in 118 of 124 columns.**
`CURRENT_DIAGNOSTIC_ORDER` (ECE at slots 6-53, BES last) is the Keras graphs'
order; `SPLIT_COLUMN_ORDER` (BES at 12-75, ECE last) is the order of
`train_test_split_model10.pkl`, the rows labelmaker fitted on - measured, not
read, to a worst relative error of 1.2e-12 (see the module docstring and
`scripts/labelmaker/elm_write_normalization.py --verify-split`). Feeding one to
weights fitted on the other produces silent garbage, so column sets are selected
by NAME and these tests pin both the names and the positions they resolve to.
"""
import pytest

from labelmaker.models import elm_inputs


def test_the_keras_order_is_the_124_columns_with_ece_early():
    order = elm_inputs.CURRENT_DIAGNOSTIC_ORDER
    assert len(order) == 124
    assert len(set(order)) == 124
    assert order[:6] == tuple(
        f"{n}_downsampled" for n in ("ip", "bt", "gas", "pinj", "tinj", "ech")
    )
    assert order[6] == "ece_slow_channel_1_downsampled"
    assert order[53] == "ece_slow_channel_48_downsampled"
    assert order[54:56] == ("pcphd02_downsampled", "pcphd03_downsampled")
    assert order[56:60] == tuple(
        f"co2_density_slow_{c}_downsampled" for c in ("r0", "v1", "v2", "v3")
    )
    assert order[60] == "bes_slow_channel_1_downsampled"
    assert order[123] == "bes_slow_channel_64_downsampled"


def test_the_split_order_puts_bes_at_12_to_75_and_ece_last():
    order = elm_inputs.SPLIT_COLUMN_ORDER
    assert len(order) == 124
    assert len(set(order)) == 124
    assert order[:6] == elm_inputs.CURRENT_DIAGNOSTIC_ORDER[:6]
    assert order[6:8] == ("pcphd02_downsampled", "pcphd03_downsampled")
    assert order[8:12] == tuple(
        f"co2_density_slow_{c}_downsampled" for c in ("r0", "v1", "v2", "v3")
    )
    assert order[12] == "bes_slow_channel_1_downsampled"
    assert order[75] == "bes_slow_channel_64_downsampled"
    assert order[76] == "ece_slow_channel_1_downsampled"
    assert order[123] == "ece_slow_channel_48_downsampled"


def test_the_two_orders_agree_on_exactly_the_first_six_columns():
    same = [
        i for i, (a, b) in enumerate(
            zip(elm_inputs.CURRENT_DIAGNOSTIC_ORDER, elm_inputs.SPLIT_COLUMN_ORDER)
        )
        if a == b
    ]
    assert same == list(range(6))
    assert set(elm_inputs.CURRENT_DIAGNOSTIC_ORDER) == set(elm_inputs.SPLIT_COLUMN_ORDER)


def test_bes_is_64_columns_at_slots_12_to_75_of_the_split_order():
    assert len(elm_inputs.BES_COLUMNS) == 64
    order = elm_inputs.SPLIT_COLUMN_ORDER
    assert [order.index(c) for c in elm_inputs.BES_COLUMNS] == list(range(12, 76))


def test_no_bes_is_the_60_non_bes_columns_and_keeps_every_ece_channel():
    cols = elm_inputs.COLUMN_SETS["no_bes"]
    assert len(cols) == 60
    assert not any(c.startswith("bes_") for c in cols)
    assert sum(c.startswith("ece_slow_channel_") for c in cols) == 48
    assert elm_inputs.column_indices("no_bes") == tuple(range(12)) + tuple(range(76, 124))
    assert elm_inputs.column_indices("all124") == tuple(range(124))
    assert elm_inputs.COLUMN_SETS["all124"] == elm_inputs.SPLIT_COLUMN_ORDER


def test_selection_follows_the_names_not_a_hard_coded_slice(monkeypatch):
    """If upstream ever reorders again, `no_bes` must follow the names it asked for.

    This is the failure the first fit actually hit: slots 0-59 of the split
    pickle are NOT the non-BES columns, and only a name-keyed selection notices.
    """
    swapped = (
        elm_inputs.SPLIT_COLUMN_ORDER[76:] + elm_inputs.SPLIT_COLUMN_ORDER[:76]
    )
    monkeypatch.setattr(elm_inputs, "SPLIT_COLUMN_ORDER", swapped)
    got = elm_inputs.column_indices("no_bes")
    assert [swapped[i] for i in got] == list(elm_inputs.COLUMN_SETS["no_bes"])
    assert not any(swapped[i].startswith("bes_") for i in got)


def test_an_unknown_set_names_the_sets_that_exist():
    with pytest.raises(elm_inputs.UnknownColumnSet, match="all124"):
        elm_inputs.column_indices("no_ece")
