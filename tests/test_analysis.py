import json

import pandas as pd
import pytest
from conftest import ScriptedClient

from trusttrajectory.analysis import (
    booked_by_tier, bootstrap_pre_post, build_decay_df, build_summary_df, build_turn_df,
    fht_by_tier, label_collisions, depth_mean_pre_post, load_trajectories,
    pre_post_by_tier, pre_post_rates, relabel, taxonomy_counts, threshold_sensitivity,
)
from trusttrajectory.config import PRESETS, ScoringThresholds
from trusttrajectory.models import ModelConfig
from trusttrajectory.runner import run_scenario
from trusttrajectory.scenarios import load_scenarios
from trusttrajectory.scoring import make_scorer


@pytest.fixture(scope="module")
def trajectories():
    cfg = PRESETS["v4or_t13"]
    scorer = make_scorer(cfg)
    out = []
    for i, sc in enumerate(load_scenarios(ids=["easy_01", "flight_easy_01", "medium_01", "conflict_01", "adversarial_01"])):
        for run_idx in (1, 2):
            out.append(run_scenario(ScriptedClient(f"{i}{run_idx}", confab=(run_idx == 2)), ModelConfig("fake/m", "fake"),
                                    sc, run_idx, cfg, scorer, log=lambda s: None, sleep=lambda s: None))
    return out


def test_frames(trajectories):
    turn_df, summary_df = build_turn_df(trajectories), build_summary_df(trajectories)
    assert len(summary_df) == 10 and len(turn_df) == sum(t["turns_used"] for t in trajectories)
    assert set(turn_df["hallucinated"].unique()) <= {0, 1}
    assert str(summary_df["difficulty"].dtype) == "category"
    decay = build_decay_df(turn_df)
    assert list(decay["turn_idx"]) == sorted(decay["turn_idx"]) and decay["n"].iloc[0] == 10
    pruned = build_decay_df(turn_df, min_n=10)
    assert (pruned["n"] >= 10).all() and len(pruned) <= len(decay)


def test_pre_post_rates_are_pooled_turn_proportions(trajectories):
    turn_df = build_turn_df(trajectories)
    r = pre_post_rates(turn_df)
    post = turn_df[turn_df["pivot_occurred"]]
    assert r["post_rate"] == pytest.approx(post["hallucinated"].mean()) and r["n_post"] == len(post)
    assert r["delta"] == pytest.approx(r["post_rate"] - r["pre_rate"])
    tiers = pre_post_by_tier(turn_df)
    assert set(tiers["difficulty"]) == {"easy", "medium", "conflicting", "adversarial"}


def test_bootstrap_is_trajectory_level_and_reproducible(trajectories):
    turn_df = build_turn_df(trajectories)
    a = bootstrap_pre_post(turn_df, n_boot=500, seed=1)
    b = bootstrap_pre_post(turn_df, n_boot=500, seed=1)
    assert a == b and a["n_trajectories"] == 10
    assert a["pre_ci"][0] <= a["pre_rate"] <= a["pre_ci"][1]
    assert a["post_ci"][0] <= a["post_rate"] <= a["post_ci"][1]
    assert a["delta_ci"][0] <= a["delta"] <= a["delta_ci"][1]


def test_tier_summaries_and_counts(trajectories):
    summary_df, turn_df = build_summary_df(trajectories), build_turn_df(trajectories)
    fht = fht_by_tier(summary_df)
    assert (fht["n_with_fht"] <= fht["n"]).all()
    booked = booked_by_tier(summary_df)
    assert (booked["booked"] <= booked["n"]).all() and booked.loc[booked["difficulty"] == "easy", "n"].iat[0] == 4
    tax = taxonomy_counts(turn_df)
    assert tax["total"].sum() == turn_df["hallucinated"].sum()
    assert list(tax["type"]) == ["PARAM_DRIFT", "PREMATURE_COMMIT", "CONSTRAINT_IGNORE", "CONFIDENT_CONFAB", "MEMORY_COLLAPSE"]


def test_relabel_reproduces_stored_labels_at_baseline(trajectories):
    turn_df = build_turn_df(trajectories)
    floor = PRESETS["v4or_t13"].effective_param_drift_floor
    assert relabel(turn_df, ScoringThresholds(), floor).tolist() == turn_df["hallucination_type"].tolist()
    older = turn_df.drop(columns=["issues", "param_total"])  # records without the newer fields
    assert relabel(older, ScoringThresholds(), floor).tolist() == turn_df["hallucination_type"].tolist()


def test_threshold_sensitivity_is_monotone_in_constraint_ignore(trajectories):
    turn_df = build_turn_df(trajectories)
    sens = threshold_sensitivity(turn_df, param_drift_floor=8)
    ci = sens[sens["threshold"] == "constraint_ignore"].sort_values("value")
    rates = ci["post_pivot_constraint_ignore_rate"].tolist()
    assert rates == sorted(rates)
    assert set(sens["threshold"]) == {"constraint_ignore", "param_drift"}


def test_label_collisions_and_depth_mean_metric(trajectories):
    turn_df = build_turn_df(trajectories)
    lc = label_collisions(turn_df)
    assert lc["n_multi_condition"] <= lc["n_hallucinated"] == turn_df["hallucinated"].sum()
    pre, post = depth_mean_pre_post(build_decay_df(turn_df), 13)
    assert 0 <= pre <= 1 and 0 <= post <= 1


def test_load_trajectories_concatenates(tmp_path, trajectories):
    for i, chunk in enumerate((trajectories[:3], trajectories[3:])):
        (tmp_path / f"raw_{i}.json").write_text(json.dumps(chunk, default=str))
    loaded = load_trajectories([str(tmp_path / "raw_0.json"), str(tmp_path / "raw_1.json")])
    assert len(loaded) == len(trajectories)
