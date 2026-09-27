"""Names keep joiners but refuse other Unicode C characters and line separators."""

from __future__ import annotations

import json

import pytest

from labeler.events.review import labels, versions

from .test_review_versions import _label, client, event_dir  # noqa: F401


@pytest.mark.parametrize(
    "name",
    [
        "محمد‌رضا",  # Persian with ZWNJ, U+200C
        "क्‍ष",  # Devanagari with ZWJ, U+200D
    ],
)
def test_joiners_are_kept_byte_for_byte(name):
    assert versions.clean_name(name).encode("utf-8") == name.encode("utf-8")


@pytest.mark.parametrize(
    "character",
    [
        "\n",
        "\t",
        "\x00",
        "\x85",
        "\u2028",
        "\u2029",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
        "\ud800",
        "\u200b",  # zero width space
        "\u2060",  # word joiner
        "\ufeff",  # zero width no-break space
        "\u00ad",  # soft hyphen
        "\ue000",  # private use
        "\u0378",  # unassigned
    ],
)
def test_a_non_joiner_c_character_or_line_separator_in_a_name_is_refused(character):
    with pytest.raises(ValueError, match="a name cannot hold control characters"):
        versions.clean_name(f"Ada{character}Lovelace")


def test_the_server_saves_a_zwnj_name_exactly(client, event_dir):  # noqa: F811
    name = "محمد‌رضا"
    response = client.post("/api/label", json=_label(name=name))
    assert response.status_code == 200
    assert response.json()["last_save"]["name"] == name
    [line] = labels.history_path(event_dir).read_text().splitlines()
    assert json.loads(line)["name"].encode("utf-8") == name.encode("utf-8")
    [version] = client.get("/api/history?event=alfven_eigenmode&shot=170815").json()[
        "versions"
    ]
    assert version["name"] == name
