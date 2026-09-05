import re

import pytest

from trusttrajectory.scenarios import (
    SMOKE_IDS, TIERS, load_scenarios, smoke_scenarios, tier_counts, validate_scenario,
)


def test_suite_has_46_scenarios_in_the_published_split():
    scenarios = load_scenarios()
    assert len(scenarios) == 46
    assert tier_counts(scenarios) == {"easy": 10, "medium": 10, "hard": 10, "conflicting": 8, "adversarial": 8}
    by_domain = {(s["difficulty"], s["domain"]) for s in scenarios}
    assert len({s["id"] for s in scenarios}) == 46
    assert all((t, d) in by_domain for t in TIERS for d in ("restaurant", "flight"))


def test_domain_counts_match_paper_table():
    scenarios = load_scenarios()
    rest = sum(1 for s in scenarios if s["domain"] == "restaurant")
    assert rest == 26 and len(scenarios) - rest == 20


def test_every_scenario_is_well_formed():
    for s in load_scenarios():
        assert 8 <= s["complication_turn"] <= 12
        assert s["expected_pre"].keys() and s["expected_post"].keys()
        assert len(s["expected_pre"]) >= 3, s["id"]  # PARAM_DRIFT requires >= 3 slots
        for pattern in s["clarification_drip"]:
            re.compile(pattern)
        assert s["post_booking_probes"]


def test_filters_and_smoke_set():
    assert [s["id"] for s in smoke_scenarios()] == SMOKE_IDS
    assert all(s["domain"] == "flight" for s in load_scenarios(domains=["flight"]))
    assert [s["id"] for s in load_scenarios(ids=["easy_01"])] == ["easy_01"]
    assert len(load_scenarios(tiers=["conflicting", "adversarial"])) == 16


def test_validation_rejects_bad_scenarios():
    good = load_scenarios(ids=["easy_01"])[0]
    bad = dict(good)
    bad.pop("pivot")
    with pytest.raises(ValueError):
        validate_scenario(bad)
    bad = dict(good, difficulty="impossible")
    with pytest.raises(ValueError):
        validate_scenario(bad)
