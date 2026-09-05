import json

from trusttrajectory.config import PRESETS
from trusttrajectory.scoring import LLMJudgeScorer, TurnContext, make_scorer
from trusttrajectory.scoring.llm_judge import parse_judge_output


def make_ctx(text, **over):
    base = dict(turn_text=text, turn_idx=14, pivot_turn=13, pivot_occurred=True, complication_occurred=True,
                expected={"people": 3, "cuisine": "italian", "city": "new york", "day": "saturday", "time": "7 PM"},
                tool_called=False, tool_parsed=None, all_params_present=True,
                committed_facts={"city": "new york", "day": "friday", "cuisine": "italian"},
                context_tokens=500, params_confirmed=True)
    base.update(over)
    return TurnContext(**base)


def test_parse_tolerates_fences_and_prose():
    raw = 'Sure:\n```json\n{"slots": {"people": "three"}, "claims_booking_complete": true, "contradicted_facts": ["day"]}\n```'
    out = parse_judge_output(raw)
    assert out == {"slots": {"people": "three"}, "claims_booking_complete": True, "contradicted_facts": ["day"], "parsed": True}
    assert parse_judge_output("garbage") == {"slots": {}, "claims_booking_complete": False, "contradicted_facts": [], "parsed": False}


def test_judge_scores_paraphrased_slots_the_regex_would_miss():
    def judge(prompt):
        assert "Expected slot names" in prompt
        return json.dumps({"slots": {"people": "three", "cuisine": "pasta place", "city": "NYC (new york)", "day": "saturday", "time": "7 PM"},
                           "claims_booking_complete": False, "contradicted_facts": []})
    scorer = LLMJudgeScorer(judge, param_drift_floor=8)
    r = scorer.score_turn(make_ctx("A paraphrased answer."))
    assert r["param_accuracy"] == 1.0 and r["hallucination_type"] == "NONE"
    assert r["judge_slots"]["people"] == "three" and r["judge_parsed"] is True


def test_judge_flags_confab_and_collapse_with_shared_precedence():
    def judge(prompt):
        return json.dumps({"slots": {}, "claims_booking_complete": True, "contradicted_facts": ["day", "not_a_fact"]})
    r = LLMJudgeScorer(judge, param_drift_floor=8).score_turn(make_ctx("It's all booked for friday."))
    assert r["issues"] == ["CONFIDENT_CONFAB", "MEMORY_COLLAPSE", "CONSTRAINT_IGNORE"]
    assert r["hallucination_type"] == "CONFIDENT_CONFAB"
    assert json.loads(r["contradictions"]) == ["day: contradicted (judge)"]


def test_make_scorer_wires_judge():
    scorer = make_scorer(PRESETS["v4or_t13"], "llm_judge", judge=lambda p: "{}")
    assert scorer.name == "llm_judge" and scorer.param_drift_floor == 8


def test_judge_retries_once_on_unparseable_output_and_keeps_raw():
    calls = []
    def flaky_judge(prompt):
        calls.append(prompt)
        return "Sure, here you go" if len(calls) == 1 else json.dumps({"slots": {"people": 3}, "claims_booking_complete": False, "contradicted_facts": []})
    r = LLMJudgeScorer(flaky_judge, param_drift_floor=8).score_turn(make_ctx("three of us"))
    assert len(calls) == 2 and "Return ONLY the JSON object" in calls[1]
    assert r["judge_parsed"] is True and r["judge_raw_on_failure"] is None
    always_bad = LLMJudgeScorer(lambda p: "nope", param_drift_floor=8).score_turn(make_ctx("three of us"))
    assert always_bad["judge_parsed"] is False and always_bad["judge_raw_on_failure"] == "nope"


def test_judge_does_not_flag_confab_after_an_executed_booking():
    judge = lambda p: json.dumps({"slots": {}, "claims_booking_complete": True, "contradicted_facts": []})
    r = LLMJudgeScorer(judge, param_drift_floor=8).score_turn(make_ctx("All booked!", booking_executed=True))
    assert "CONFIDENT_CONFAB" not in r["issues"]
