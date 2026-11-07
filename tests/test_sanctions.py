from core.sanctions_matcher import SanctionsMatcher
from core.schemas import MatchStrength
from core.text import normalize_name


def test_normalize_strips_legal_suffixes_and_punctuation():
    assert normalize_name("Orient Global Trading F.Z.E.") == "ORIENT GLOBAL TRADING F Z E"
    assert normalize_name("Acme Infotech Pvt. Ltd.") == "ACME INFOTECH"
    assert normalize_name("Tata & Sons Private Limited") == "TATA AND SONS"


def test_loads_all_lists_and_excludes_individuals():
    m = SanctionsMatcher.from_directory()
    assert set(m.lists_loaded) == {"OFAC_SDN", "UN_CONSOLIDATED", "SYNTHETIC_TEST"}
    names = {e.name for e in m.entries}
    assert "DOE, John" not in names          # individual excluded
    assert "IGNORED" not in names            # UN individual excluded
    assert "ZZ TEST MARITIME" in names       # OFAC alias loaded
    assert "ZZ FIXTURE AID TRUST" in names   # UN alias loaded


def test_exact_synthetic_hit_is_strong():
    res = SanctionsMatcher.from_directory().screen(["ZZ-SYNTHETIC Sanctioned Trading LLC"])
    assert res.has_strong_hit
    assert res.hits[0].list_name == "SYNTHETIC_TEST"


def test_alias_and_suffix_variation_still_matches():
    res = SanctionsMatcher.from_directory().screen(["ZZ Test Maritime Pvt Ltd"])
    assert any(h.entry_id == "101" and h.strength == MatchStrength.STRONG for h in res.hits)


def test_word_order_variation_matches():
    res = SanctionsMatcher.from_directory().screen(["Global Orient Trading"])
    assert any(h.entry_id == "104" for h in res.hits)


def test_unrelated_company_has_no_hits():
    res = SanctionsMatcher.from_directory().screen(["Acme Infotech Limited", "Acme Infotech"])
    assert res.hits == []


def test_one_token_names_need_exact_match():
    m = SanctionsMatcher.from_directory()
    assert m.screen(["Novas"]).hits == []                 # fuzzy near-miss on a short name: no hit
    assert m.screen(["Nova Ltd"]).hits[0].entry_id == "103"  # exact after normalisation: hit


def test_un_entity_matches():
    res = SanctionsMatcher.from_directory().screen(["ZZ Fixture Relief Foundation"])
    assert res.hits and res.hits[0].list_name == "UN_CONSOLIDATED"
