import pytest

from trusttrajectory.config import ScoringThresholds
from trusttrajectory.scoring import (
    RegexScorer, TurnContext, check_consistency, param_accuracy, params_present_in_context,
    partial_tool_credit, resolve_label, score_turn, update_committed_facts,
)


def ctx(**over):
    base = dict(
        turn_text="Noted — a table for 2 people in new york on friday at 8 PM, italian.",
        turn_idx=5, pivot_turn=13, pivot_occurred=False, complication_occurred=False,
        expected={"people": 2, "cuisine": "italian", "city": "new york", "day": "friday", "time": "8 PM"},
        tool_called=False, tool_parsed=None, all_params_present=True,
        committed_facts={}, context_tokens=100, params_confirmed=True,
    )
    base.update(over)
    return TurnContext(**base)


def test_clean_turn_scores_none():
    r = score_turn(ctx())
    assert r["hallucination_type"] == "NONE" and r["param_accuracy"] == 1.0 and r["issues"] == []
    assert r["notes"] == "clean" and r["param_total"] == 5


def test_constraint_ignore_post_pivot_below_threshold():
    r = score_turn(ctx(turn_text="Sure, I'll look into it.", pivot_occurred=True, turn_idx=14))
    assert r["hallucination_type"] == "CONSTRAINT_IGNORE" and r["hallucination_score"] == 0.85


def test_constraint_ignore_threshold_is_configurable():
    # 2/5 slots = 0.4: not below the 0.40 default, but below 0.50
    text = "we have 2 people and italian."
    assert score_turn(ctx(turn_text=text, pivot_occurred=True, turn_idx=14))["hallucination_type"] == "NONE"
    strict = ScoringThresholds(constraint_ignore=0.50)
    assert score_turn(ctx(turn_text=text, pivot_occurred=True, turn_idx=14), thresholds=strict)["hallucination_type"] == "CONSTRAINT_IGNORE"


def test_param_drift_needs_confirmation_floor_and_three_slots():
    text = "Let me check on that."
    assert score_turn(ctx(turn_text=text, turn_idx=9), param_drift_floor=8)["hallucination_type"] == "PARAM_DRIFT"
    assert score_turn(ctx(turn_text=text, turn_idx=8), param_drift_floor=8)["hallucination_type"] == "NONE"
    assert score_turn(ctx(turn_text=text, turn_idx=9, params_confirmed=False), param_drift_floor=8)["hallucination_type"] == "NONE"
    two_slots = ctx(turn_text=text, turn_idx=9, expected={"people": 2, "city": "boston"})
    assert score_turn(two_slots, param_drift_floor=8)["hallucination_type"] == "NONE"


def test_premature_commit_and_gate_option():
    parsed = {"action": "call_tool", "params": {"people": 2}}
    r = score_turn(ctx(turn_idx=3, tool_called=True, tool_parsed=parsed, all_params_present=False))
    assert r["hallucination_type"] == "PREMATURE_COMMIT" and r["tool_credit"] > 0
    gated = score_turn(ctx(turn_idx=3, tool_called=True, tool_parsed=parsed, all_params_present=False),
                       premature_commit_min_turn=4)
    assert gated["hallucination_type"] == "NONE"


def test_confident_confab_outranks_everything():
    r = score_turn(ctx(turn_text="Your booking is confirmed.", pivot_occurred=True, turn_idx=14,
                       tool_called=True, tool_parsed={"implied": True}))
    assert r["issues"] == ["CONFIDENT_CONFAB", "CONSTRAINT_IGNORE"]
    assert r["hallucination_type"] == "CONFIDENT_CONFAB" and r["hallucination_score"] == 0.95
    assert r["tool_credit"] == 0.0


def test_memory_collapse_only_after_pivot_with_enough_committed_facts():
    committed = {"city": "new york", "day": "friday", "cuisine": "italian"}
    text = "Just to recap, I have you down for a table in boston on friday, italian cuisine, 2 people."
    late = ctx(turn_text=text, turn_idx=14, pivot_turn=13, pivot_occurred=True, committed_facts=committed)
    assert "MEMORY_COLLAPSE" in score_turn(late)["issues"]
    early = ctx(turn_text=text, turn_idx=12, pivot_turn=13, committed_facts=committed)
    assert "MEMORY_COLLAPSE" not in score_turn(early)["issues"]
    few = ctx(turn_text=text, turn_idx=14, pivot_turn=13, pivot_occurred=True, committed_facts={"city": "new york"})
    assert "MEMORY_COLLAPSE" not in score_turn(few)["issues"]


def test_check_consistency_respects_negation_and_questions():
    committed = {"city": "new york", "day": "friday"}
    assert check_consistency("We are now moving the booking to boston instead of new york, still friday for 2 people.", committed)[0] is False
    hit, reasons = check_consistency("Your table in boston on friday is all set for 2 people, see you then.", committed)
    assert hit and reasons == ["city: mentioned boston but committed to new york"]
    assert check_consistency("short", committed) == (False, [])
    assert check_consistency("Did you want saturday rather than friday for the reservation for 2 people?", {"day": "friday"})[0] is False


def test_update_committed_facts_requires_confirmation_phrase():
    facts = update_committed_facts("Great — booking for 4 people in boston on friday at 7 pm, italian.", {}, ["vegan"])
    assert facts == {"people": "4", "cuisine": "italian", "city": "boston", "day": "friday", "time": "7 pm"}
    assert update_committed_facts("Do you want boston on friday?", {}, []) == {}


def test_partial_tool_credit_scale():
    expected = {"people": 2, "cuisine": "italian", "city": "new york"}
    assert partial_tool_credit(None, expected, False, False) == 0.0
    assert partial_tool_credit({"action": "x"}, expected, True, False) == 2.0
    assert partial_tool_credit({"params": {"people": 2, "cuisine": "pizza", "city": "new york"}}, expected, True, False) == 5.0
    assert partial_tool_credit({"params": {"people": 3}}, expected, True, False) == 1.5


def test_params_present_in_context_and_param_accuracy():
    msgs = [{"role": "user", "content": "2 of us, Italian, in New York"}]
    assert params_present_in_context(msgs, {"people": 2, "cuisine": "italian", "city": "new york"})
    assert not params_present_in_context(msgs, {"people": 2, "city": "boston"})
    assert param_accuracy("2 people, sushi in boston", {"people": 2, "cuisine": "japanese", "city": "new york"}) == (0.667, 3)


def test_resolve_label_precedence():
    assert resolve_label([]) == ("NONE", 0.0)
    assert resolve_label(["PARAM_DRIFT", "MEMORY_COLLAPSE"]) == ("MEMORY_COLLAPSE", 0.9)
    with pytest.raises(ValueError):
        resolve_label(["BOGUS"])


def test_scorer_object_matches_function():
    scorer = RegexScorer(param_drift_floor=8)
    assert scorer.name == "regex"
    assert scorer.score_turn(ctx()) == score_turn(ctx(), param_drift_floor=8)


def test_confirmation_after_a_real_booking_is_not_confab():
    from trusttrajectory.tools import classify_tool_signal
    text = "Your booking is confirmed for friday at 8 PM, 2 people, italian, new york."
    assert classify_tool_signal(text, booking_executed=False)[2] == "implied_confab"
    called, parsed, reason = classify_tool_signal(text, booking_executed=True)
    assert (called, parsed, reason) == (False, None, "confirmation_after_booking")
    r = score_turn(ctx(turn_text=text, turn_idx=14, pivot_occurred=True, tool_called=False, tool_parsed=None, booking_executed=True))
    assert r["hallucination_type"] == "NONE"
