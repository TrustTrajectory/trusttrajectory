import random

import pytest

from trusttrajectory.config import PRESETS, RunConfig, get_preset


def test_presets_reproduce_original_settings():
    t13 = PRESETS["v4or_t13"]
    assert (t13.max_turns, t13.pivot_turn, t13.min_booking_turn) == (22, 13, 4)
    assert t13.effective_param_drift_floor == 8 and t13.context_window_messages == 20
    t15 = PRESETS["v4or_t15"]
    assert (t15.max_turns, t15.pivot_turn, t15.min_booking_turn) == (30, 15, 8)
    assert t15.effective_param_drift_floor == 12
    nudge = PRESETS["v4or_t13_nudge"]
    assert nudge.post_pivot_nudge and nudge.premature_commit_respects_gate


def test_randomised_pivot_is_seeded_and_in_range():
    cfg = RunConfig(pivot_turn=(11, 15))
    draws = {cfg.resolve_pivot_turn(random.Random(i)) for i in range(50)}
    assert draws <= {11, 12, 13, 14, 15} and len(draws) > 1
    assert cfg.resolve_pivot_turn(random.Random(3)) == cfg.resolve_pivot_turn(random.Random(3))


def test_gate_none_disables_execution_gate():
    assert RunConfig(gate_mode="none", min_booking_turn=4).effective_min_booking_turn == 1
    with pytest.raises(ValueError):
        RunConfig(gate_mode="soft")


def test_invalid_pivot_rejected():
    with pytest.raises(ValueError):
        RunConfig(max_turns=10, pivot_turn=12)
    with pytest.raises(ValueError):
        RunConfig(pivot_turn=(15, 11))


def test_overrides_and_unknown_preset():
    cfg = get_preset("v4or_t13").with_overrides(max_turns=30, pivot_turn=15)
    assert cfg.max_turns == 30 and cfg.pivot_turn == 15 and cfg.min_booking_turn == 4
    with pytest.raises(KeyError):
        get_preset("nope")
