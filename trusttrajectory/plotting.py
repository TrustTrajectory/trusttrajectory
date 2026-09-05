"""Figures.  Every function takes the dataframes plus ``out_dir``/``tag`` and
writes a PNG; nothing is shown interactively."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis.frames import DIFFICULTY_ORDER  # noqa: E402
from .scoring.taxonomy import TAXONOMY_ORDER  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")

DIFF_COLORS = {
    "easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c",
    "conflicting": "#9b59b6", "adversarial": "#2c3e50",
}
TAXONOMY_COLORS = ["#f39c12", "#e67e22", "#e74c3c", "#8e44ad", "#2c3e50"]
LABEL_KW = dict(fontsize=12, fontweight="bold")


def _save(fig, out_dir: str, name: str, tag: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_{tag}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_trust_decay_curve(decay_df: pd.DataFrame, out_dir: str, tag: str, pivot_turn: int,
                          title: Optional[str] = None) -> Optional[str]:
    if decay_df.empty:
        return None
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(decay_df["turn_idx"], decay_df["hallucination_rate"], color="#e74c3c", lw=2.5,
             marker="o", ms=5, label="Hallucination Rate")
    ax1.fill_between(decay_df["turn_idx"], decay_df["hallucination_rate"], alpha=0.12, color="#e74c3c")
    ax2.plot(decay_df["turn_idx"], decay_df["param_accuracy_mean"], color="#3498db", lw=2,
             linestyle="--", marker="s", ms=4, label="Param Accuracy")
    ax2.plot(decay_df["turn_idx"], 1 - decay_df["consistency_score_mean"], color="#27ae60", lw=1.8,
             linestyle=":", marker="^", ms=4, label="Consistency (1-score)")
    ax1.axvline(x=pivot_turn, color="orange", lw=2.5, linestyle="--", alpha=0.9, label=f"Pivot (T{pivot_turn})")
    ax1.axhspan(0, 0.1, alpha=0.06, color="green", label="Reliable zone (<10%)")
    ax1.set_xlabel("Trajectory Turn Depth", **LABEL_KW)
    ax1.set_ylabel("Hallucination Rate", color="#e74c3c", **LABEL_KW)
    ax2.set_ylabel("Accuracy / Consistency", color="#3498db", **LABEL_KW)
    ax1.set_ylim([-0.05, 1.05]); ax2.set_ylim([-0.05, 1.05])
    ax1.set_title(title or f"Trust Decay Curve — {tag}", **LABEL_KW)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, out_dir, "fig1_trust_decay", tag)


def fig_trust_decay_by_difficulty(turn_df: pd.DataFrame, out_dir: str, tag: str, pivot_turn: int) -> Optional[str]:
    if turn_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for diff in DIFFICULTY_ORDER:
        subset = turn_df[turn_df["difficulty"] == diff]
        if subset.empty:
            continue
        grp = subset.groupby("turn_idx").agg(hall_rate=("hallucinated", "mean"), n=("hallucinated", "count")).reset_index()
        se = np.sqrt(grp["hall_rate"] * (1 - grp["hall_rate"]) / grp["n"].clip(lower=1))
        ci = 1.96 * se
        color = DIFF_COLORS.get(diff, "#888888")
        ax.plot(grp["turn_idx"], grp["hall_rate"], color=color, lw=2.2, marker="o", ms=5, label=diff.capitalize())
        ax.fill_between(grp["turn_idx"], (grp["hall_rate"] - ci).clip(lower=0), (grp["hall_rate"] + ci).clip(upper=1),
                        alpha=0.12, color=color)
    ax.axvline(x=pivot_turn, color="orange", lw=2.5, linestyle="--", alpha=0.9, label=f"Pivot (T{pivot_turn})")
    ax.set_xlabel("Trajectory Turn Depth", **LABEL_KW); ax.set_ylabel("Hallucination Rate", **LABEL_KW)
    ax.set_ylim([-0.05, 1.05]); ax.set_title("Trust Decay by Difficulty Tier", **LABEL_KW)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_decay_by_difficulty", tag)


def fig_hallucination_taxonomy(turn_df: pd.DataFrame, out_dir: str, tag: str) -> Optional[str]:
    if turn_df.empty:
        return None
    post_mask = turn_df["pivot_occurred"].astype(bool)
    pre_counts = [int(((turn_df["hallucination_type"] == h) & ~post_mask).sum()) for h in TAXONOMY_ORDER]
    post_counts = [int(((turn_df["hallucination_type"] == h) & post_mask).sum()) for h in TAXONOMY_ORDER]
    x, w = np.arange(len(TAXONOMY_ORDER)), 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - w / 2, pre_counts, w, label="Pre-Pivot", alpha=0.85, color=TAXONOMY_COLORS, edgecolor="white")
    b2 = ax.bar(x + w / 2, post_counts, w, label="Post-Pivot", alpha=0.55, color=TAXONOMY_COLORS, edgecolor="white", hatch="//")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2, str(int(h)), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(TAXONOMY_ORDER, fontsize=10, fontweight="bold")
    ax.set_ylabel("Count (turn-level)", fontsize=11)
    ax.set_title("Hallucination Taxonomy: Pre-Pivot vs Post-Pivot", **LABEL_KW)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_taxonomy", tag)


def fig_fht_distribution(summary_df: pd.DataFrame, out_dir: str, tag: str, pivot_turn: int, max_turns: int) -> Optional[str]:
    df = summary_df.dropna(subset=["first_hall_turn"]) if not summary_df.empty else summary_df
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for diff in DIFFICULTY_ORDER:
        subset = df[df["difficulty"] == diff]["first_hall_turn"]
        if subset.empty:
            continue
        color = DIFF_COLORS.get(diff, "#888888")
        ax.hist(subset, bins=range(1, max_turns + 2), alpha=0.55, color=color, edgecolor="white",
                label=f"{diff.capitalize()} (n={len(subset)})")
        ax.axvline(x=subset.mean(), color=color, lw=2, linestyle="--", alpha=0.9, label=f"{diff.capitalize()} μ=T{subset.mean():.1f}")
    ax.axvline(x=pivot_turn, color="orange", lw=2.5, linestyle="-", alpha=0.8, label=f"Pivot (T{pivot_turn})")
    ax.set_xlabel("First Hallucination Turn (FHT)", **LABEL_KW); ax.set_ylabel("Frequency", **LABEL_KW)
    ax.set_title("FHT Distribution by Difficulty Tier", **LABEL_KW)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_fht_distribution", tag)


def fig_context_vs_hallucination(turn_df: pd.DataFrame, out_dir: str, tag: str) -> Optional[str]:
    if turn_df.empty or "context_tokens" not in turn_df.columns:
        return None
    df = turn_df.copy()
    df["ctx_bin"] = pd.cut(df["context_tokens"], bins=10)
    grp = df.groupby("ctx_bin", observed=True).agg(hall_rate=("hallucinated", "mean"), n=("hallucinated", "count")).reset_index()
    grp["ctx_mid"] = grp["ctx_bin"].apply(lambda x: x.mid).astype(float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sc = ax.scatter(grp["ctx_mid"], grp["hall_rate"], s=grp["n"] * 8, c=grp["hall_rate"], cmap="RdYlGn_r",
                    vmin=0, vmax=1, alpha=0.85, edgecolors="black", lw=0.5)
    valid = grp.dropna()
    if len(valid) >= 3:
        z = np.polyfit(valid["ctx_mid"], valid["hall_rate"], 1)
        xs = np.linspace(valid["ctx_mid"].min(), valid["ctx_mid"].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), "k--", lw=1.5, alpha=0.6, label="Trend")
    fig.colorbar(sc, ax=ax, label="Hallucination Rate")
    ax.set_xlabel("Approx. Context Window (tokens)", **LABEL_KW); ax.set_ylabel("Hallucination Rate", **LABEL_KW)
    ax.set_title("Context Load vs Hallucination Rate", **LABEL_KW)
    ax.set_ylim([-0.05, 1.05]); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_context_vs_hall", tag)


def fig_ablation_bar(summary_df: pd.DataFrame, out_dir: str, tag: str) -> Optional[str]:
    if summary_df.empty:
        return None
    grp = summary_df.groupby("difficulty", observed=True).agg(
        mean_hall=("total_hallucinations", "mean"),
        se=("total_hallucinations", lambda x: x.std() / (len(x) ** 0.5) if len(x) > 1 else 0.0),
    ).reset_index()
    grp["difficulty"] = pd.Categorical(grp["difficulty"].astype(str), categories=DIFFICULTY_ORDER, ordered=True)
    grp = grp.sort_values("difficulty")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [DIFF_COLORS.get(str(d), "#888888") for d in grp["difficulty"]]
    bars = ax.bar(grp["difficulty"].astype(str), grp["mean_hall"], color=colors, edgecolor="white", alpha=0.85,
                  yerr=grp["se"].fillna(0), capsize=5)
    for bar, val in zip(bars, grp["mean_hall"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xlabel("Difficulty Tier", **LABEL_KW); ax.set_ylabel("Mean Hallucinations / Trajectory", **LABEL_KW)
    ax.set_title("Hallucinations per Trajectory by Difficulty Tier", **LABEL_KW)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_ablation_bar", tag)


def fig_ablation_delta(turn_df: pd.DataFrame, out_dir: str, tag: str) -> Optional[str]:
    if turn_df.empty:
        return None
    rows = []
    post_mask = turn_df["pivot_occurred"].astype(bool)
    for diff in DIFFICULTY_ORDER:
        sub = turn_df["difficulty"] == diff
        if not sub.any():
            continue
        pre = turn_df[sub & ~post_mask]["hallucinated"].mean()
        post = turn_df[sub & post_mask]["hallucinated"].mean()
        rows.append({"difficulty": diff, "pre": pre, "post": post, "delta": post - pre})
    if not rows:
        return None
    rdf = pd.DataFrame(rows)
    x, w = np.arange(len(rdf)), 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [DIFF_COLORS.get(d, "#888888") for d in rdf["difficulty"]]
    ax.bar(x - w / 2, rdf["pre"], w, label="Pre-Pivot", alpha=0.85, color=colors, edgecolor="white")
    ax.bar(x + w / 2, rdf["post"], w, label="Post-Pivot", alpha=0.55, color=colors, edgecolor="white", hatch="//")
    for i, row in rdf.iterrows():
        ax.annotate(f"Δ{row['delta']:+.2f}", xy=(i, np.nanmax([row["pre"], row["post"]]) + 0.02), ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in rdf["difficulty"]], fontsize=11, fontweight="bold")
    ax.set_ylabel("Hallucination Rate", **LABEL_KW)
    ax.set_title("Pre/Post-Pivot Hallucination Rate by Difficulty Tier", **LABEL_KW)
    ax.set_ylim([0, min(1.05, float(np.nanmax(rdf["post"])) + 0.15)])
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "fig8_ablation_delta", tag)


def fig_decay_by_condition(decay_by_label: dict, out_dir: str, tag: str) -> Optional[str]:
    """Hallucination rate vs turn depth, one line per model / condition."""
    if not decay_by_label:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, decay_df) in enumerate(decay_by_label.items()):
        if decay_df.empty:
            continue
        ax.plot(decay_df["turn_idx"], decay_df["hallucination_rate"], lw=2.2, marker="o", ms=4,
                color=palette[i % len(palette)], label=label)
    ax.set_xlabel("Trajectory Turn Depth", **LABEL_KW); ax.set_ylabel("Hallucination Rate", **LABEL_KW)
    ax.set_ylim([-0.05, 1.05]); ax.set_title("Trust Decay by Model / Condition", **LABEL_KW)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_decay_by_condition", tag)


def make_all_figures(turn_df, summary_df, decay_df, out_dir: str, tag: str, pivot_turn: int, max_turns: int) -> list:
    paths = [
        fig_trust_decay_curve(decay_df, out_dir, tag, pivot_turn),
        fig_trust_decay_by_difficulty(turn_df, out_dir, tag, pivot_turn),
        fig_hallucination_taxonomy(turn_df, out_dir, tag),
        fig_fht_distribution(summary_df, out_dir, tag, pivot_turn, max_turns),
        fig_context_vs_hallucination(turn_df, out_dir, tag),
        fig_ablation_bar(summary_df, out_dir, tag),
        fig_ablation_delta(turn_df, out_dir, tag),
    ]
    return [p for p in paths if p]
