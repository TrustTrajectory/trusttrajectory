import json

import pytest
from conftest import ScriptedClient

from trusttrajectory.analysis import agreement_report, cohen_kappa, rescore_trajectories
from trusttrajectory.analysis.rescore import turn_context
from trusttrajectory.config import PRESETS
from trusttrajectory.models import ModelConfig
from trusttrajectory.runner import run_scenario
from trusttrajectory.scenarios import load_scenarios
from trusttrajectory.scoring import LLMJudgeScorer, make_scorer


@pytest.fixture(scope="module")
def trajectories():
    cfg = PRESETS["v4or_t13"]
    scorer = make_scorer(cfg)
    return [run_scenario(ScriptedClient(str(i), confab=(i % 2 == 0)), ModelConfig("fake/m", "fake"), sc, 1, cfg, scorer,
                         log=lambda s: None, sleep=lambda s: None)
            for i, sc in enumerate(load_scenarios(ids=["easy_01", "flight_easy_01", "hard_02", "conflict_03", "adversarial_04"]))]


def labels(trajs):
    return [(t["scenario_id"], r["turn_idx"], r["hallucination_type"]) for t in trajs for r in t["turn_records"]]


def test_rescoring_with_the_regex_scorer_reproduces_stored_labels(trajectories):
    again = rescore_trajectories(trajectories, make_scorer(PRESETS["v4or_t13"]))
    assert labels(again) == labels(trajectories)
    assert [t["first_hall_turn"] for t in again] == [t["first_hall_turn"] for t in trajectories]
    assert again[0]["rescored_from"] == "regex" and again[0]["turn_records"][0]["usage"] is None
    assert trajectories[0]["turn_records"][0] is not again[0]["turn_records"][0]  # inputs untouched


def test_turn_context_falls_back_to_scenario_for_older_records(trajectories):
    t = trajectories[0]
    older = {k: v for k, v in t["turn_records"][0].items() if k not in ("assistant_text_full", "expected_state", "committed_facts_before")}
    ctx = turn_context(t, older, load_scenarios(ids=[t["scenario_id"]])[0])
    assert ctx.expected == t["turn_records"][0]["expected_state"] and ctx.committed_facts == {}
    with pytest.raises(ValueError):
        turn_context(t, older, None)


def test_judge_rescoring_and_agreement(trajectories):
    def judge(prompt):  # a judge that reads no slots at all: everything post-pivot becomes CONSTRAINT_IGNORE
        return json.dumps({"slots": {}, "claims_booking_complete": False, "contradicted_facts": []})
    judged = rescore_trajectories(trajectories, LLMJudgeScorer(judge, param_drift_floor=8))
    assert all(r["hallucination_type"] in ("CONSTRAINT_IGNORE", "PREMATURE_COMMIT", "CONFIDENT_CONFAB")
               for t in judged for r in t["turn_records"] if r["pivot_occurred"])
    rep = agreement_report(trajectories, judged, names=("regex", "judge"))
    assert rep["n_turns"] == sum(len(t["turn_records"]) for t in trajectories)
    assert 0 <= rep["label_agreement"] <= 1 and -1 <= rep["label_kappa"] <= 1
    assert sum(sum(row.values()) for row in rep["confusion"].values()) == rep["n_turns"]
    assert rep["judge_rates"]["post_constraint_ignore_rate"] >= rep["regex_rates"]["post_constraint_ignore_rate"]


def test_cohen_kappa_values():
    assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == 1.0
    assert cohen_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"]) == pytest.approx(0.0)
    assert cohen_kappa(["A", "A"], ["A", "A"]) == 1.0
    assert cohen_kappa([], []) != cohen_kappa([], [])  # nan


def test_parallel_rescore_matches_sequential(trajectories):
    scorer = make_scorer(PRESETS["v4or_t13"], "assertion")
    seq = rescore_trajectories(trajectories, scorer, workers=1)
    par = rescore_trajectories(trajectories, scorer, workers=3)
    assert [t["scenario_id"] for t in seq] == [t["scenario_id"] for t in par]
    assert json.dumps(seq, sort_keys=True, default=str) == json.dumps(par, sort_keys=True, default=str)
    assert all(r["committed_facts_before"] is not None for t in par for r in t["turn_records"])


def test_rescore_checkpoint_resumes_and_skips_scored_trajectories(trajectories, tmp_path):
    scorer = make_scorer(PRESETS["v4or_t13"], "assertion")
    ck = str(tmp_path / "checkpoint.json")
    first = rescore_trajectories(trajectories[:2], scorer, checkpoint_path=ck, checkpoint_every=1)
    assert len(json.load(open(ck))) == 2

    calls = []

    class Counting:
        name = scorer.name
        harvest = staticmethod(scorer.harvest)

        def score_turn(self, ctx):
            calls.append(ctx.turn_idx)
            return scorer.score_turn(ctx)

    again = rescore_trajectories(trajectories, Counting(), checkpoint_path=ck, checkpoint_every=1, workers=2)
    assert len(again) == len(trajectories) and again[:2] == first
    n_new_turns = sum(len(t["turn_records"]) for t in trajectories[2:])
    assert len(calls) == n_new_turns                     # the two checkpointed trajectories were not re-scored
    assert len(json.load(open(ck))) == len(trajectories)
    # a checkpoint written by a different scorer is ignored
    other = rescore_trajectories(trajectories[:1], make_scorer(PRESETS["v4or_t13"], "regex"), checkpoint_path=ck)
    assert other[0]["scorer"] == "regex"
