"""Metrics: FHT, pre/post-pivot rates, trajectory-level bootstrap CIs,
taxonomy counts, threshold sensitivity and label-collision statistics."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import ScoringThresholds
from ..scoring.taxonomy import TAXONOMY_ORDER, HallucinationType, resolve_label
from .frames import DIFFICULTY_ORDER

_TRAJ_KEY = ["model", "scenario_id", "run_idx"]


# ── pre / post pivot ────────────────────────────────────────────────────────

def pre_post_rates(turn_df: pd.DataFrame) -> Dict[str, Any]:
    """Pooled (turn-level) hallucination rate before and after the pivot."""
    pre = turn_df[~turn_df["pivot_occurred"].astype(bool)]["hallucinated"]
    post = turn_df[turn_df["pivot_occurred"].astype(bool)]["hallucinated"]
    pre_rate = float(pre.mean()) if len(pre) else float("nan")
    post_rate = float(post.mean()) if len(post) else float("nan")
    return {
        "pre_rate": pre_rate, "post_rate": post_rate, "delta": post_rate - pre_rate,
        "n_pre": int(len(pre)), "n_post": int(len(post)),
    }


def pre_post_by_tier(turn_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier in DIFFICULTY_ORDER:
        sub = turn_df[turn_df["difficulty"] == tier]
        if sub.empty:
            continue
        rows.append({"difficulty": tier, **pre_post_rates(sub)})
    return pd.DataFrame(rows)


def _per_trajectory_counts(turn_df: pd.DataFrame) -> pd.DataFrame:
    df = turn_df.copy()
    df["post"] = df["pivot_occurred"].astype(int)
    df["pre_h"] = df["hallucinated"] * (1 - df["post"])
    df["post_h"] = df["hallucinated"] * df["post"]
    df["pre_n"] = 1 - df["post"]
    df["post_n"] = df["post"]
    return df.groupby(_TRAJ_KEY, observed=True)[["pre_h", "pre_n", "post_h", "post_n"]].sum().reset_index()


def bootstrap_pre_post(
    turn_df: pd.DataFrame, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05,
) -> Dict[str, Any]:
    """Percentile bootstrap CIs for pre, post and delta, resampling *trajectories*.

    Resampling whole trajectories (not turns) keeps within-trajectory
    dependence intact, as described in Section 3.4 of the paper.
    """
    per = _per_trajectory_counts(turn_df)
    k = len(per)
    if k == 0:
        return {"n_trajectories": 0}
    counts = per[["pre_h", "pre_n", "post_h", "post_n"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    sums = counts[idx].sum(axis=1)  # (n_boot, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        pre = sums[:, 0] / sums[:, 1]
        post = sums[:, 2] / sums[:, 3]
    delta = post - pre
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def ci(x: np.ndarray) -> Tuple[float, float]:
        x = x[np.isfinite(x)]
        return (float(np.percentile(x, lo)), float(np.percentile(x, hi))) if len(x) else (float("nan"),) * 2

    point = pre_post_rates(turn_df)
    return {
        "pre_rate": point["pre_rate"], "pre_ci": ci(pre),
        "post_rate": point["post_rate"], "post_ci": ci(post),
        "delta": point["delta"], "delta_ci": ci(delta),
        "n_trajectories": int(k), "n_boot": int(n_boot), "alpha": alpha,
    }


def depth_mean_pre_post(decay_df: pd.DataFrame, pivot_turn: int) -> Tuple[float, float]:
    """The mean of per-turn-depth
    means* before/after the pivot.  Kept for comparison only; it up-weights
    sparse late turns relative to the pooled rate reported in the paper."""
    pre = decay_df[decay_df["turn_idx"] < pivot_turn]["hallucination_rate"].mean()
    post = decay_df[decay_df["turn_idx"] >= pivot_turn]["hallucination_rate"].mean()
    return float(pre), float(post)


# ── per-tier summaries ──────────────────────────────────────────────────────

def fht_by_tier(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/min/max First Hallucination Turn per tier over trajectories that hallucinated."""
    rows = []
    for tier in DIFFICULTY_ORDER:
        sub = summary_df[summary_df["difficulty"] == tier]
        if sub.empty:
            continue
        fht = sub["first_hall_turn"].dropna()
        rows.append({
            "difficulty": tier, "n": int(len(sub)), "n_with_fht": int(len(fht)),
            "mean_fht": float(fht.mean()) if len(fht) else float("nan"),
            "min_fht": int(fht.min()) if len(fht) else None,
            "max_fht": int(fht.max()) if len(fht) else None,
        })
    return pd.DataFrame(rows)


def booked_by_tier(summary_df: pd.DataFrame) -> pd.DataFrame:
    """API-Booked Rate as successes / rollouts (binary per trajectory)."""
    rows = []
    for tier in DIFFICULTY_ORDER:
        sub = summary_df[summary_df["difficulty"] == tier]
        if sub.empty:
            continue
        booked = int(sub["booking_done"].astype(bool).sum())
        rows.append({"difficulty": tier, "booked": booked, "n": int(len(sub)), "rate": booked / len(sub)})
    return pd.DataFrame(rows)


def taxonomy_counts(turn_df: pd.DataFrame) -> pd.DataFrame:
    """Turn-level counts of each class, pre- and post-pivot."""
    post_mask = turn_df["pivot_occurred"].astype(bool)
    rows = []
    for t in TAXONOMY_ORDER:
        is_t = turn_df["hallucination_type"] == t
        pre, post = int((is_t & ~post_mask).sum()), int((is_t & post_mask).sum())
        rows.append({"type": t, "pre": pre, "post": post, "total": pre + post})
    return pd.DataFrame(rows)


# ── label collisions & re-labelling ─────────────────────────────────────────

def issues_of(row: pd.Series) -> List[str]:
    """Fired conditions for a turn (``issues`` column, or parsed from ``notes``)."""
    issues = row.get("issues") if hasattr(row, "get") else None
    if isinstance(issues, list):
        return list(issues)
    notes = row.get("notes", "") if hasattr(row, "get") else ""
    if not isinstance(notes, str) or notes in ("", "clean"):
        return []
    return [s.strip() for s in notes.split(";") if s.strip()]


def label_collisions(turn_df: pd.DataFrame) -> Dict[str, Any]:
    """How often more than one taxonomy condition fires on the same turn."""
    n_hall, n_multi, pairs = 0, 0, Counter()
    for _, row in turn_df.iterrows():
        issues = issues_of(row)
        if not issues:
            continue
        n_hall += 1
        if len(issues) > 1:
            n_multi += 1
            pairs["+".join(sorted(issues))] += 1
    return {
        "n_hallucinated": n_hall, "n_multi_condition": n_multi,
        "frac_multi_condition": (n_multi / n_hall) if n_hall else float("nan"),
        "combinations": dict(pairs),
    }


def relabel(
    turn_df: pd.DataFrame,
    thresholds: ScoringThresholds,
    param_drift_floor: int,
) -> pd.Series:
    """Re-derive labels from stored per-turn measurements under new thresholds.

    Threshold-free conditions (CONFIDENT_CONFAB, MEMORY_COLLAPSE,
    PREMATURE_COMMIT) are taken from the stored ``issues``; the two
    threshold-based ones are recomputed from ``param_accuracy``.
    """
    fixed = {HallucinationType.CONFIDENT_CONFAB.value,
             HallucinationType.MEMORY_COLLAPSE.value,
             HallucinationType.PREMATURE_COMMIT.value}
    labels = []
    has_total = "param_total" in turn_df.columns
    for _, row in turn_df.iterrows():
        issues = [i for i in issues_of(row) if i in fixed]
        acc = float(row["param_accuracy"])
        post = bool(row["pivot_occurred"])
        total = int(row["param_total"]) if has_total and pd.notna(row["param_total"]) else thresholds.min_params_for_drift
        if post and acc < thresholds.constraint_ignore:
            issues.append(HallucinationType.CONSTRAINT_IGNORE.value)
        if (not post and bool(row.get("params_confirmed", False))
                and acc < thresholds.param_drift
                and int(row["turn_idx"]) > param_drift_floor
                and total >= thresholds.min_params_for_drift):
            issues.append(HallucinationType.PARAM_DRIFT.value)
        labels.append(resolve_label(issues)[0])
    return pd.Series(labels, index=turn_df.index, name="hallucination_type")


def threshold_sensitivity(
    turn_df: pd.DataFrame,
    param_drift_floor: int,
    base: ScoringThresholds = ScoringThresholds(),
    constraint_ignore_values: Sequence[float] = (0.30, 0.40, 0.50),
    param_drift_values: Sequence[float] = (0.40, 0.50, 0.60),
) -> pd.DataFrame:
    """Post-pivot CONSTRAINT_IGNORE rate (and overall post-pivot hallucination
    rate) when each threshold is perturbed with the other held at baseline."""
    post_mask = turn_df["pivot_occurred"].astype(bool)
    rows = []

    def evaluate(which: str, value: float, thr: ScoringThresholds) -> None:
        labels = relabel(turn_df, thr, param_drift_floor)
        post_labels = labels[post_mask]
        rows.append({
            "threshold": which, "value": value,
            "post_pivot_constraint_ignore_rate": float((post_labels == "CONSTRAINT_IGNORE").mean()) if len(post_labels) else float("nan"),
            "post_pivot_hallucination_rate": float((post_labels != "NONE").mean()) if len(post_labels) else float("nan"),
            "dominant_post_pivot_class": post_labels[post_labels != "NONE"].mode().iat[0] if (post_labels != "NONE").any() else None,
        })

    for v in constraint_ignore_values:
        evaluate("constraint_ignore", v, ScoringThresholds(
            constraint_ignore=v, param_drift=base.param_drift,
            min_committed_for_collapse=base.min_committed_for_collapse,
            min_params_for_drift=base.min_params_for_drift))
    for v in param_drift_values:
        evaluate("param_drift", v, ScoringThresholds(
            constraint_ignore=base.constraint_ignore, param_drift=v,
            min_committed_for_collapse=base.min_committed_for_collapse,
            min_params_for_drift=base.min_params_for_drift))
    return pd.DataFrame(rows)


def probe_outcomes(turn_df: pd.DataFrame) -> pd.DataFrame:
    """Per probe kind (direct / presupposition): how the model answered.

    Columns: n, stale (old value affirmed or asserted), wrong (another value),
    denied (the model says there is no such booking), ok (current value stated:
    "answered" for direct probes, "corrected" for presupposition probes),
    unclear (no value committed to) and the rates.
    """
    if turn_df.empty or "probe_kind" not in turn_df or turn_df["probe_kind"].isna().all():
        return pd.DataFrame(columns=["probe_kind", "n", "stale", "wrong", "denied", "ok", "unclear", "stale_rate", "ok_rate"])
    probes = turn_df[turn_df["probe_kind"].notna()]
    rows = []
    for kind, g in probes.groupby("probe_kind", sort=True):
        out = g["probe_outcome"].fillna("unclear")
        n = int(len(g))
        stale = int((out == "stale").sum())
        wrong = int((out == "wrong").sum())
        denied = int((out == "denied").sum())
        ok = int(out.isin(["answered", "corrected"]).sum())
        rows.append({"probe_kind": kind, "n": n, "stale": stale, "wrong": wrong, "denied": denied, "ok": ok,
                     "unclear": n - stale - wrong - denied - ok,
                     "stale_rate": stale / n if n else float("nan"), "ok_rate": ok / n if n else float("nan")})
    return pd.DataFrame(rows)
