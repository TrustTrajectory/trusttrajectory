from trusttrajectory.extraction import (
    CUISINE_SYNONYMS, contains, contains_cuisine, extract_number, find_first_json, strip_think_blocks,
)


def test_extract_number_prefers_digits_then_words():
    assert extract_number("a table for 4 people") == 4
    assert extract_number("party of twelve") == 12
    assert extract_number("twenty five guests") == 25
    assert extract_number("no numbers here") is None
    assert extract_number("room 101") is None  # only 1-99 counts as a party size


def test_contains_is_case_insensitive_and_rejects_empty_target():
    assert contains("Booked in New York", "new york")
    assert not contains("Booked in New York", "")


def test_contains_cuisine_uses_synonyms():
    assert contains_cuisine("we found a great sushi bar", "japanese")
    assert contains_cuisine("Italian it is", "italian")
    assert not contains_cuisine("burgers all round", "italian")
    assert not contains_cuisine("anything", "")


def test_cuisine_synonym_order_is_stable():
    # several callers stop at the first matching cuisine, so order is part of the contract
    assert list(CUISINE_SYNONYMS)[:4] == ["thai", "italian", "japanese", "korean"]
    assert "bibimbap" in CUISINE_SYNONYMS["korean"]


def test_find_first_json_returns_balanced_object():
    assert find_first_json('prefix {"a": {"b": 1}} suffix {"c": 2}') == '{"a": {"b": 1}}'
    assert find_first_json("no braces") is None


def test_strip_think_blocks():
    assert strip_think_blocks("<think>hmm\nthinking</think>  Answer") == "Answer"
