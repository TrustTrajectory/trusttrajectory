import json

import pytest
from conftest import ScriptedClient

from trusttrajectory.config import PRESETS
from trusttrajectory.models import ModelConfig
from trusttrajectory.replay import ReplayClient, ReplayExhausted, config_for, replay_trajectories, replay_trajectory
from trusttrajectory.runner import required_slots, run_scenario
from trusttrajectory.scenarios import load_scenarios
from trusttrajectory.scoring import make_scorer, missing_params_in_context

IDS = ["easy_01", "flight_easy_01", "hard_02", "conflict_03", "adversarial_04"]


@pytest.fixture(scope="module")
def stored():
    cfg = PRESETS["v4or_t13"]
    scorer = make_scorer(cfg)
    return [run_scenario(ScriptedClient(str(i), confab=(i % 2 == 0)), ModelConfig("fake/m", "fake"), sc, 1, cfg, scorer,
                         log=lambda s: None, sleep=lambda s: None)
            for i, sc in enumerate(load_scenarios(ids=IDS))]


def strip(t):
    skip = {"latency_mean", "latency_total", "replayed_from", "replay_mismatch", "scorer"}
    return {k: v for k, v in t.items() if k not in skip and k != "turn_records"}, [
        {k: v for k, v in r.items() if k not in ("latency_s", "usage", "contradictions", "consistency_score")}
        for r in t["turn_records"]]


def test_replay_reproduces_the_stored_trajectory(stored):
    cfg = PRESETS["v4or_t13"]
    again = replay_trajectories(stored, make_scorer(cfg), cfg, scenarios=load_scenarios(ids=IDS))
    for a, b in zip(stored, again):
        assert not b["replay_mismatch"]
        assert strip(a) == strip(b)
        assert b["replayed_from"] == "regex" and b["scorer"] == "regex"


def test_replay_with_the_assertion_scorer_and_workers(stored):
    cfg = PRESETS["v4or_t13"]
    one = replay_trajectories(stored, make_scorer(cfg, "assertion"), cfg, workers=1)
    two = replay_trajectories(stored, make_scorer(cfg, "assertion"), cfg, workers=3)
    assert json.dumps(one, sort_keys=True, default=str) == json.dumps(two, sort_keys=True, default=str)
    assert all(t["scorer"] == "assertion" and not t["replay_mismatch"] for t in one)
    assert all("missing_params" in r for t in one for r in t["turn_records"])


def test_replay_marks_truncated_or_foreign_trajectories(stored):
    cfg = PRESETS["v4or_t13"]
    t = dict(stored[0]); t["turn_records"] = t["turn_records"][:3]
    out = replay_trajectory(t, load_scenarios(ids=[t["scenario_id"]])[0], make_scorer(cfg), cfg)
    assert out["replay_mismatch"] and out["error"]
    with pytest.raises(ReplayExhausted):
        ReplayClient([]).create(messages=[])


def test_config_for_applies_stored_controls():
    cfg = PRESETS["v4or_t13"]
    t = {"pivot_turn": 11, "config": {"gate_mode": "none", "state_card_at_pivot": True, "filler_tokens_per_turn": 40,
                                       "thresholds": {"constraint_ignore": 0.9}, "unknown_key": 1}}
    c = config_for(t, cfg)
    assert c.pivot_turn == 11 and c.gate_mode == "none" and c.state_card_at_pivot and c.filler_tokens_per_turn == 40
    assert c.thresholds == cfg.thresholds


def test_premature_commit_uses_mandatory_slots_only():
    expected = {"people": 4, "cuisine": "italian", "city": "boston", "day": "friday", "time": "7 PM", "occasion": "birthday"}
    msgs = [{"role": "user", "content": "Table for 4 people, italian, in boston on friday at 7 PM."}]
    assert missing_params_in_context(msgs, expected) == ["occasion"]
    assert missing_params_in_context(msgs, required_slots("restaurant", expected)) == []
    assert required_slots("restaurant", expected, "all") == expected
    flight = {"origin": "boston", "destination": "paris", "date_out": "may 1", "passengers": 2, "cabin": "business", "meal": "kosher"}
    assert set(required_slots("flight", flight)) == {"origin", "destination", "date_out", "passengers"}


def test_in_context_check_understands_time_spellings():
    expected = {"time": "12 PM"}
    for said in ("noon", "12:00 PM", "12pm", "12 p.m."):
        assert missing_params_in_context([{"role": "user", "content": f"Let's say {said}."}], expected) == []
    assert missing_params_in_context([{"role": "user", "content": "Let's say 1 PM."}], expected) == ["time"]
    assert missing_params_in_context([{"role": "user", "content": "around 7:00pm works"}], {"time": "7 PM"}) == []
