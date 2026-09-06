"""The ELM model's 124 input columns and the two named column sets.

The order is upstream's `current_diagnostic_order`; a second order exists
(`new_diagnostic_order`, ECE last) and feeding it to weights fitted on this one
produces silent garbage, so the sets are selected by column name and these tests
pin both the names and the positions they resolve to.
"""
import pytest

from labelmaker.models import elm_inputs


def test_the_order_is_the_124_columns_upstream_fitted_on():
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


def test_bes_is_64_columns_and_occupies_the_last_64_slots():
    assert len(elm_inputs.BES_COLUMNS) == 64
    positions = elm_inputs.CURRENT_DIAGNOSTIC_ORDER
    assert [positions.index(c) for c in elm_inputs.BES_COLUMNS] == list(range(60, 124))


def test_no_bes_is_slots_0_to_59_and_names_no_bes_channel():
    assert elm_inputs.column_indices("no_bes") == tuple(range(60))
    assert elm_inputs.column_indices("all124") == tuple(range(124))
    assert not any(c.startswith("bes_") for c in elm_inputs.COLUMN_SETS["no_bes"])
    assert elm_inputs.COLUMN_SETS["all124"] == elm_inputs.CURRENT_DIAGNOSTIC_ORDER


def test_selection_follows_the_names_not_a_hard_coded_slice(monkeypatch):
    """If upstream ever reorders, `no_bes` must follow the names it asked for."""
    swapped = (
        elm_inputs.CURRENT_DIAGNOSTIC_ORDER[60:]
        + elm_inputs.CURRENT_DIAGNOSTIC_ORDER[:60]
    )
    monkeypatch.setattr(elm_inputs, "CURRENT_DIAGNOSTIC_ORDER", swapped)
    assert elm_inputs.column_indices("no_bes") == tuple(range(64, 124))


def test_an_unknown_set_names_the_sets_that_exist():
    with pytest.raises(elm_inputs.UnknownColumnSet, match="all124"):
        elm_inputs.column_indices("no_ece")
