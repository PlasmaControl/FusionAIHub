"""Every model folder is discoverable and its card is machine-readable."""
import pytest

from labelmaker.models import registry

REQUIRED_TOP = ("pipeline_tag", "tags", "library_name", "labelmaker")
REQUIRED_LM = ("status", "slug", "card_id", "framework", "upstream", "inputs", "outputs")


def test_front_matter_is_parsed():
    card = registry.parse_card(
        "---\nlibrary_name: keras\nlabelmaker:\n  status: scaffold\n---\n\n# Title\n"
    )
    assert card["library_name"] == "keras"
    assert card["labelmaker"]["status"] == "scaffold"


def test_a_card_without_front_matter_is_an_error():
    with pytest.raises(ValueError, match="front matter"):
        registry.parse_card("# Just a heading\n")


def test_every_model_folder_has_a_well_formed_card():
    slugs = registry.model_slugs()
    assert slugs, "no model folders found"
    for slug in slugs:
        card = registry.read_card(slug)
        for key in REQUIRED_TOP:
            assert key in card, f"{slug}: card is missing {key}"
        lm = card["labelmaker"]
        for key in REQUIRED_LM:
            assert key in lm, f"{slug}: labelmaker block is missing {key}"
        assert lm["slug"] == slug
        assert lm["card_id"] == f"plasmacontrol/{slug.replace('_', '-')}"
        assert lm["status"] in ("implemented", "scaffold")


def test_the_roster_readme_lists_every_folder_and_the_exclusions():
    text = (registry.MODELS_DIR / "README.md").read_text()
    for slug in registry.model_slugs():
        assert slug in text, f"{slug} missing from models/README.md"
    for excluded in ("tokeye", "ae_tf_maskrcnn", "tokamind", "diag2diag"):
        assert excluded in text.lower(), f"{excluded} not recorded as excluded"


def test_scaffolds_declare_what_blocks_them_and_refuse_to_load():
    assert registry.scaffolds(), "expected scaffolded models"
    for slug in registry.scaffolds():
        card = registry.read_card(slug)
        assert card["labelmaker"]["blocked_on"], f"{slug}: no blocked_on entries"
        with pytest.raises(NotImplementedError, match=slug):
            registry.load_adapter(slug)


def test_implemented_models_load_and_match_their_card():
    for slug in registry.implemented():
        adapter = registry.load_adapter(slug)
        assert adapter.slug == slug
        assert registry.card_discrepancies(slug) == []
