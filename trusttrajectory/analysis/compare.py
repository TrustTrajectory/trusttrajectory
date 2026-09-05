"""Multi-model comparison: the paper's Tables 1--3 and cross-model figures
from one raw file per model (or per experimental condition)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import RunConfig
from .frames import build_decay_df, build_summary_df, build_turn_df
from .metrics import (
    booked_by_tier, bootstrap_pre_post, fht_by_tier, label_collisions, pre_post_by_tier,
    taxonomy_counts, threshold_sensitivity,
)


def condition_report(trajectories: List[Dict[str, Any]], label: str, param_drift_floor: int,
                     n_boot: int = 10_000, min_n: int = 20) -> Dict[str, Any]:
    turn_df, summary_df = build_turn_df(trajectories), build_summary_df(trajectories)
    boot = bootstrap_pre_post(turn_df, n_boot=n_boot)
    return {
        "label": label,
        "model": str(summary_df["model"].iloc[0]),
        "n_trajectories": int(len(summary_df)),
        "n_turns": int(len(turn_df)),
        "pivot_turns": sorted(set(int(p) for p in summary_df["pivot_turn"].dropna())),
        "gate_mode": trajectories[0].get("config", {}).get("gate_mode"),
        "scorer": trajectories[0].get("scorer"),
        "pre_post": boot,
        "pre_post_by_tier": pre_post_by_tier(turn_df).to_dict(orient="records"),
        "fht_by_tier": fht_by_tier(summary_df).to_dict(orient="records"),
        "booked_by_tier": booked_by_tier(summary_df).to_dict(orient="records"),
        "taxonomy_counts": taxonomy_counts(turn_df).to_dict(orient="records"),
        "label_collisions": label_collisions(turn_df),
        "threshold_sensitivity": threshold_sensitivity(turn_df, param_drift_floor).to_dict(orient="records"),
        "_frames": {"turn": turn_df, "summary": summary_df, "decay": build_decay_df(turn_df, min_n=min_n)},
    }


def compare_conditions(
    raw_by_label: Dict[str, List[Dict[str, Any]]],
    out_dir: str,
    cfg: Optional[RunConfig] = None,
    n_boot: int = 10_000,
    min_n: int = 20,
    figures: bool = True,
) -> Dict[str, Any]:
    """Write summary.json and figures for several conditions."""
    os.makedirs(out_dir, exist_ok=True)
    floor = cfg.effective_param_drift_floor if cfg else 8
    reports = {label: condition_report(traj, label, floor, n_boot, min_n) for label, traj in raw_by_label.items()}

    public = {label: {k: v for k, v in r.items() if k != "_frames"} for label, r in reports.items()}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(public, f, indent=2, default=str)

    for label, r in reports.items():
        safe = _safe(label)
        r["_frames"]["turn"].to_csv(os.path.join(out_dir, f"turn_level_{safe}.csv"), index=False)
        r["_frames"]["summary"].to_csv(os.path.join(out_dir, f"trajectory_summary_{safe}.csv"), index=False)

    if figures:
        from ..plotting import (
            fig_decay_by_condition, fig_hallucination_taxonomy, fig_trust_decay_by_difficulty,
            fig_trust_decay_curve,
        )
        fig_decay_by_condition({label: r["_frames"]["decay"] for label, r in reports.items()}, out_dir, "all")
        for label, r in reports.items():
            safe, fr = _safe(label), r["_frames"]
            pivot = r["pivot_turns"][0] if len(r["pivot_turns"]) == 1 else int(round(sum(r["pivot_turns"]) / len(r["pivot_turns"])))
            fig_trust_decay_curve(fr["decay"], out_dir, safe, pivot, title=f"Trust Decay Curve — {label}")
            fig_trust_decay_by_difficulty(fr["turn"], out_dir, safe, pivot)
            fig_hallucination_taxonomy(fr["turn"], out_dir, safe)
    return public


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in label).strip("_").lower()


def print_comparison(public: Dict[str, Any]) -> None:
    print("\n" + "═" * 78)
    print(f"  {'condition':28s} {'n_traj':>6s} {'pre':>7s} {'post':>7s} {'delta':>8s}   95% CI (delta)")
    print("═" * 78)
    for label, r in public.items():
        pp = r["pre_post"]
        print(f"  {label:28s} {r['n_trajectories']:6d} {pp['pre_rate']:7.3f} {pp['post_rate']:7.3f} "
              f"{pp['delta']:+8.3f}   [{pp['delta_ci'][0]:.3f}, {pp['delta_ci'][1]:.3f}]")
    print("═" * 78)
