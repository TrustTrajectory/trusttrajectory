"""Dataframes over trajectory records."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ..scenarios import TIERS

DIFFICULTY_ORDER: List[str] = list(TIERS)


def load_trajectories(paths: Iterable[str]) -> List[Dict[str, Any]]:
    """Load and concatenate ``raw_*.json`` / ``checkpoint_*.json`` files."""
    out: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list of trajectories")
        out.extend(data)
    return out


def _with_tier_order(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "difficulty" in df.columns:
        df["difficulty"] = pd.Categorical(df["difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
    return df


_PROBE_FIELDS = ("kind", "variant", "slot", "idx")


def build_turn_df(trajectories: List[Dict[str, Any]], include_interference: bool = False) -> pd.DataFrame:
    """One row per scored turn, with trajectory metadata repeated on each row.

    Turns of the interference condition's distractor bookings (``segment ==
    "interference"``) are about another booking and are left out of the rates
    unless ``include_interference`` is set.  A probe turn's spec is flattened to
    ``probe_kind`` / ``probe_variant`` / ``probe_slot`` / ``probe_idx``.
    """
    rows = []
    for t in trajectories:
        base = {
            "scenario_id": t["scenario_id"],
            "model": t["model"],
            "run_idx": t["run_idx"],
            "difficulty": t["difficulty"],
            "domain": t.get("domain", "restaurant"),
            "pivot_turn": t.get("pivot_turn"),
            "pivot_injected": t["pivot_injected"],
            "complication_given": t["complication_given"],
        }
        for r in t["turn_records"]:
            if r.get("segment", "main") == "interference" and not include_interference:
                continue
            row = {**base, **{k: v for k, v in r.items() if k != "probe"}}
            probe = r.get("probe") or {}
            for k in _PROBE_FIELDS:
                row[f"probe_{k}"] = probe.get(k)
            row.setdefault("segment", "main")
            row.setdefault("probe_outcome", None)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["hallucinated"] = (df["hallucination_type"] != "NONE").astype(int)
    return _with_tier_order(df)


def build_summary_df(trajectories: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per trajectory."""
    rows = []
    for t in trajectories:
        types = [r["hallucination_type"] for r in t["turn_records"] if r["hallucination_type"] != "NONE"]
        rows.append({
            "scenario_id": t["scenario_id"],
            "model": t["model"],
            "run_idx": t["run_idx"],
            "difficulty": t["difficulty"],
            "domain": t.get("domain", "restaurant"),
            "pivot_turn": t.get("pivot_turn"),
            "turns_used": t["turns_used"],
            "booking_done": t["booking_done"],
            "early_booking_turn": t.get("early_booking_turn"),
            "first_hall_turn": t["first_hall_turn"],
            "total_hallucinations": t["total_hallucinations"],
            "hallucination_types": json.dumps(types),
            "latency_total_s": t.get("latency_total"),
        })
    return _with_tier_order(pd.DataFrame(rows))


def build_decay_df(turn_df: pd.DataFrame, min_n: Optional[int] = None) -> pd.DataFrame:
    """Per-turn-depth means: the Trust Decay Curve.

    ``min_n`` prunes turn depths supported by fewer than ``min_n`` trajectories
    (the paper uses 20) so the sparse tail does not dominate the curve.
    """
    if turn_df.empty:
        return pd.DataFrame()
    df = turn_df.copy()
    df["post_pivot"] = df["pivot_occurred"].astype(int)
    out = df.groupby("turn_idx").agg(
        hallucination_rate=("hallucinated", "mean"),
        param_accuracy_mean=("param_accuracy", "mean"),
        consistency_score_mean=("consistency_score", "mean"),
        tool_credit_mean=("tool_credit", "mean"),
        n=("hallucinated", "count"),
        post_pivot_frac=("post_pivot", "mean"),
        context_tokens_mean=("context_tokens", "mean"),
    ).reset_index()
    if min_n is not None:
        out = out[out["n"] >= min_n].reset_index(drop=True)
    return out
