"""Command-line entry points: ``trusttrajectory run | analyze | list-scenarios``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .config import PRESETS, RunConfig, get_preset
from .scenarios import TIERS, load_scenarios, smoke_scenarios, tier_counts


def _build_config(args: argparse.Namespace) -> RunConfig:
    cfg = get_preset(args.preset)
    overrides = {}
    if args.max_turns is not None:
        overrides["max_turns"] = args.max_turns
    if args.pivot_range is not None:
        overrides["pivot_turn"] = (args.pivot_range[0], args.pivot_range[1])
    elif args.pivot_turn is not None:
        overrides["pivot_turn"] = args.pivot_turn
    if args.gate is not None:
        overrides["gate_mode"] = args.gate
    if args.min_booking_turn is not None:
        overrides["min_booking_turn"] = args.min_booking_turn
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.full_context:
        overrides["context_window_messages"] = None
    if getattr(args, "filler_tokens", None):
        overrides["filler_tokens_per_turn"] = args.filler_tokens
    if getattr(args, "state_card", False):
        overrides["state_card_at_pivot"] = True
    if getattr(args, "context_window", None) is not None:
        overrides["context_window_messages"] = args.context_window or None
    if getattr(args, "memory", None):
        overrides["memory_mode"] = args.memory
    if getattr(args, "memory_keep", None) is not None:
        overrides["memory_keep_messages"] = args.memory_keep
    if getattr(args, "memory_every", None) is not None:
        overrides["memory_summary_every"] = args.memory_every
    if getattr(args, "interference", None) is not None:
        overrides["interference_tasks"] = args.interference
    if getattr(args, "interference_max_turns", None) is not None:
        overrides["interference_max_turns"] = args.interference_max_turns
    if getattr(args, "probes", None):
        overrides["probe_style"] = args.probes
    return cfg.with_overrides(**overrides) if overrides else cfg


def _select_scenarios(args: argparse.Namespace):
    if args.smoke:
        return smoke_scenarios()
    tiers = args.tiers.split(",") if args.tiers else None
    domains = args.domains.split(",") if args.domains else None
    ids = args.ids.split(",") if args.ids else None
    return load_scenarios(tiers=tiers, domains=domains, ids=ids)


def cmd_run(args: argparse.Namespace) -> int:
    from .models import make_client, model_config_from_env
    from .runner import make_tag, run_benchmark
    from .scoring import make_openai_judge, make_scorer

    cfg = _build_config(args)
    scenarios = _select_scenarios(args)
    if not scenarios:
        print("No scenarios selected.", file=sys.stderr)
        return 2
    model_cfg = model_config_from_env(
        args.model, args.display_name or "",
        temperature=cfg.temperature, strip_think_tags=args.strip_think, reasoning=args.reasoning,
    )
    client = make_client(args.api_key)

    judge = None
    if args.scorer == "llm_judge":
        judge = make_openai_judge(client, args.judge_model or args.model)
    scorer = make_scorer(cfg, args.scorer, judge)

    mode = "smoke" if args.smoke else ("subset" if (args.tiers or args.domains or args.ids) else "full")
    tag = args.tag or make_tag(model_cfg, mode)

    counts = tier_counts(scenarios)
    print("═" * 65)
    print(f"  TrustTrajectory {args.preset}  | model={model_cfg.name} | scorer={scorer.name}")
    print("═" * 65)
    print(f"  Scenarios: {len(scenarios)}  " + "  ".join(f"{t}={c}" for t, c in counts.items()))
    print(f"  Runs: {args.runs}  Max turns: {cfg.max_turns}  Pivot: {cfg.pivot_turn}  Gate: {cfg.gate_mode}  "
          f"Reasoning: {model_cfg.reasoning or 'provider default'}  Workers: {args.workers}")
    print(f"  Output: {args.out}/raw_{tag}.json")

    trajectories = run_benchmark(
        client, model_cfg, scenarios, cfg, n_runs=args.runs, out_dir=args.out,
        tag=tag, scorer=scorer, resume=not args.no_resume, workers=args.workers,
    )
    if not args.no_analysis:
        _analyze(trajectories, os.path.join(args.out, f"analysis_{tag}"), tag, cfg, args.n_boot, args.min_n, True)
    return 0


def _pivot_for_figures(trajectories, cfg: Optional[RunConfig], override: Optional[int]) -> int:
    if override is not None:
        return override
    pivots = [t.get("pivot_turn") for t in trajectories if t.get("pivot_turn")]
    if pivots:
        return int(round(sum(pivots) / len(pivots)))
    if cfg is not None and isinstance(cfg.pivot_turn, int):
        return cfg.pivot_turn
    return 13


def _analyze(trajectories, out_dir: str, tag: str, cfg: Optional[RunConfig], n_boot: int, min_n: int,
             figures: bool, pivot_override: Optional[int] = None) -> None:
    from .analysis import (
        booked_by_tier, bootstrap_pre_post, build_decay_df, build_summary_df, build_turn_df, fht_by_tier, label_collisions, pre_post_by_tier, probe_outcomes,
        taxonomy_counts, threshold_sensitivity,
    )

    os.makedirs(out_dir, exist_ok=True)
    turn_df = build_turn_df(trajectories)
    summary_df = build_summary_df(trajectories)
    if turn_df.empty:
        print("  No turns to analyse.")
        return
    decay_df = build_decay_df(turn_df, min_n=min_n)
    pivot_turn = _pivot_for_figures(trajectories, cfg, pivot_override)
    max_turns = int(turn_df["turn_idx"].max())
    floor = cfg.effective_param_drift_floor if cfg else 8
    model = str(summary_df["model"].iloc[0])

    turn_df.to_csv(os.path.join(out_dir, "turn_level.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "trajectory_summary.csv"), index=False)
    decay_df.to_csv(os.path.join(out_dir, "decay_curve.csv"), index=False)

    boot = bootstrap_pre_post(turn_df, n_boot=n_boot)
    sens = threshold_sensitivity(turn_df, floor)
    report = {
        "tag": tag, "model": model, "n_trajectories": int(len(summary_df)), "n_turns": int(len(turn_df)),
        "pre_post": boot,
        "pre_post_by_tier": pre_post_by_tier(turn_df).to_dict(orient="records"),
        "fht_by_tier": fht_by_tier(summary_df).to_dict(orient="records"),
        "booked_by_tier": booked_by_tier(summary_df).to_dict(orient="records"),
        "taxonomy_counts": taxonomy_counts(turn_df).to_dict(orient="records"),
        "label_collisions": label_collisions(turn_df),
        "threshold_sensitivity": sens.to_dict(orient="records"),
        "probe_outcomes": probe_outcomes(turn_df).to_dict(orient="records"),
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "═" * 65)
    print(f"  RESULTS SUMMARY  ({model}, {len(summary_df)} trajectories, {len(turn_df)} turns)")
    print("═" * 65)
    print("\n  First Hallucination Turn by tier:")
    for r in report["fht_by_tier"]:
        if r["n_with_fht"]:
            print(f"    {r['difficulty']:12s}: mean T{r['mean_fht']:.1f}  min T{r['min_fht']}  max T{r['max_fht']}  (n={r['n_with_fht']}/{r['n']})")
    print("\n  Booked rate by tier:")
    for r in report["booked_by_tier"]:
        print(f"    {r['difficulty']:12s}: {r['booked']}/{r['n']} ({100 * r['rate']:.0f}%)")
    print(f"\n  Pre-pivot  hallucination rate : {boot['pre_rate']:.3f}  CI {tuple(round(v, 3) for v in boot['pre_ci'])}")
    print(f"  Post-pivot hallucination rate : {boot['post_rate']:.3f}  CI {tuple(round(v, 3) for v in boot['post_ci'])}")
    print(f"  Delta                         : {boot['delta']:+.3f}  CI {tuple(round(v, 3) for v in boot['delta_ci'])}")
    lc = report["label_collisions"]
    print(f"\n  Multi-condition turns: {lc['n_multi_condition']}/{lc['n_hallucinated']} hallucinated turns")
    if report["probe_outcomes"]:
        print("\n  Probe outcomes:")
        for r in report["probe_outcomes"]:
            print(f"    {r['probe_kind']:15s}: n={r['n']:3d}  stale={r['stale']:3d} ({100 * r['stale_rate']:.1f}%)  "
                  f"wrong={r['wrong']:3d}  denied={r['denied']:3d}  ok={r['ok']:3d} ({100 * r['ok_rate']:.1f}%)  unclear={r['unclear']}")
    print("═" * 65)

    if figures:
        from .plotting import make_all_figures
        for p in make_all_figures(turn_df, summary_df, decay_df, out_dir, tag, pivot_turn, max_turns):
            print(f"  Saved: {p}")
    print(f"  Analysis written to {out_dir}/")


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import load_trajectories

    trajectories = load_trajectories(args.raw)
    if not trajectories:
        print("No trajectories found.", file=sys.stderr)
        return 2
    cfg = get_preset(args.preset) if args.preset else None
    tag = args.tag or os.path.splitext(os.path.basename(args.raw[0]))[0].replace("raw_", "").replace("checkpoint_", "")
    _analyze(trajectories, args.out, tag, cfg, args.n_boot, args.min_n, not args.no_figures, args.pivot_turn)
    return 0


def cmd_rescore(args: argparse.Namespace) -> int:
    from .analysis import agreement_report, load_trajectories, rescore_trajectories
    from .analysis.rescore import save_json
    from .scoring import make_openai_judge, make_scorer

    trajectories = load_trajectories(args.raw)
    cfg = get_preset(args.preset)
    if args.judge_fields:
        from .analysis.rescore import merge_judge_fields
        trajectories = merge_judge_fields(trajectories, load_trajectories(args.judge_fields))
    judge = None
    if args.scorer == "llm_judge":
        from .models import make_client
        if not args.judge_model:
            print("--judge-model is required for --scorer llm_judge", file=sys.stderr)
            return 2
        judge = make_openai_judge(make_client(args.api_key), args.judge_model)
    scorer = make_scorer(cfg, args.scorer, judge)
    os.makedirs(args.out, exist_ok=True)

    n_turns = sum(len(t["turn_records"]) for t in trajectories)
    workers = args.workers if args.workers else (4 if args.scorer == "llm_judge" else 1)
    print(f"  Re-scoring {len(trajectories)} trajectories / {n_turns} turns with {scorer.name}"
          + (f" ({args.judge_model})" if judge else "") + (f", {workers} workers" if workers > 1 else ""), flush=True)
    counter = {"n": 0}

    def progress(t, rec):
        counter["n"] += 1
        if counter["n"] % 100 == 0:
            print(f"    {counter['n']}/{n_turns} turns", flush=True)

    suffix = args.scorer if args.scorer != "llm_judge" else "judge_" + _safe_name(args.judge_model)
    # Checkpoints only pay off for network-bound judges; for offline scorers a stale checkpoint
    # written by an older scorer version would silently mix labels, so they are opt-in.
    use_checkpoint = args.checkpoint or (args.scorer == "llm_judge" and not args.no_checkpoint)
    checkpoint = os.path.join(args.out, f"checkpoint_rescored_{suffix}.json") if use_checkpoint else None
    if args.replay:
        from .replay import replay_trajectories
        print("  Replaying stored assistant turns through the runner (harness state recomputed)", flush=True)
        rescored = replay_trajectories(trajectories, scorer, cfg, workers=max(1, args.workers or 1))
    else:
        rescored = rescore_trajectories(trajectories, scorer, progress=progress, workers=workers, checkpoint_path=checkpoint)
    raw_out = os.path.join(args.out, f"raw_rescored_{suffix}.json")
    save_json(rescored, raw_out)
    reference = load_trajectories(args.reference) if args.reference else trajectories
    ref_name = reference[0].get("scorer", "stored") if reference else "stored"
    report = agreement_report(reference, rescored, names=(ref_name, scorer.name))
    save_json(report, os.path.join(args.out, f"agreement_{suffix}.json"))
    print(f"  Saved: {raw_out}")
    if report.get("n_turns"):
        if report.get("judge_parse_rate") is not None:
            print(f"  Judge output parsed on {100 * report['judge_parse_rate']:.1f}% of turns")
        print(f"  Turn-level label agreement: {report['label_agreement']:.3f}  (Cohen's kappa {report['label_kappa']:.3f})")
        print(f"  Hallucinated-vs-clean agreement: {report['binary_agreement']:.3f}  (kappa {report['binary_kappa']:.3f})")
        for name in (k for k in report if k.endswith("_rates")):
            r = report[name]
            print(f"  {name:22s} pre={r['pre_rate']:.3f} post={r['post_rate']:.3f} post CONSTRAINT_IGNORE={r['post_constraint_ignore_rate']:.3f}")
    return 0


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in s)


def cmd_compare(args: argparse.Namespace) -> int:
    from .analysis import compare_conditions, load_trajectories, print_comparison

    labels = args.labels.split(",") if args.labels else None
    if labels and len(labels) != len(args.raw):
        print("--labels must have one entry per raw file", file=sys.stderr)
        return 2
    raw_by_label = {}
    for i, path in enumerate(args.raw):
        traj = load_trajectories([path])
        if not traj:
            print(f"{path}: no trajectories", file=sys.stderr)
            return 2
        label = labels[i] if labels else str(traj[0]["model"])
        raw_by_label[label] = traj
    cfg = get_preset(args.preset) if args.preset else None
    public = compare_conditions(raw_by_label, args.out, cfg, args.n_boot, args.min_n, not args.no_figures)
    print_comparison(public)
    print(f"  Summary: {os.path.join(args.out, 'summary.json')}")
    return 0


# OpenRouter list prices, USD per million tokens, fetched 2026-09-03.  Override with --price-in/--price-out.
PRICES = {
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "google/gemini-3.5-flash": (1.50, 9.00),
    "openai/gpt-5.4": (2.50, 15.00),
    "openai/gpt-5.4-mini": (0.75, 4.50),
}


def cmd_estimate(args: argparse.Namespace) -> int:
    n_scen = len(_select_scenarios(args))
    turns, tok_in, tok_out = args.turns_per_trajectory, args.tokens_in, args.tokens_out
    total_cost, total_calls = 0.0, 0
    print(f"  Assumptions: {n_scen} scenarios, {turns} turns/trajectory, {tok_in} in / {tok_out} out tokens per call")
    print(f"  {'model':40s} {'traj':>5s} {'calls':>7s} {'est. USD':>9s}")
    for model in args.models.split(","):
        price = PRICES.get(model)
        if price is None and (args.price_in is None or args.price_out is None):
            print(f"  {model:40s}  (unknown price; pass --price-in/--price-out)")
            continue
        pin, pout = price or (args.price_in, args.price_out)
        n_traj = n_scen * args.runs + (2 * n_scen if args.controls else 0)
        calls = n_traj * turns
        cost = calls * (tok_in * pin + tok_out * pout) / 1e6
        total_cost += cost; total_calls += calls
        print(f"  {model:40s} {n_traj:5d} {calls:7d} {cost:9.2f}")
    if args.judge_model:
        pin, pout = PRICES.get(args.judge_model, (args.price_in or 0, args.price_out or 0))
        judge_turns = n_scen * args.runs * turns * len(args.models.split(","))
        cost = judge_turns * (450 * pin + 120 * pout) / 1e6
        total_cost += cost; total_calls += judge_turns
        print(f"  {'judge: ' + args.judge_model:40s} {'':5s} {judge_turns:7d} {cost:9.2f}")
    print(f"  {'TOTAL':40s} {'':5s} {total_calls:7d} {total_cost:9.2f}")
    print(f"  Wall clock at ~4 s/call: {total_calls * 4 / 3600:.1f} h sequential, "
          f"~{total_calls * 4 / 3600 / args.workers:.1f} h with {args.workers} workers")
    return 0


def cmd_annotate_export(args: argparse.Namespace) -> int:
    from .analysis import export_sample, load_trajectories
    from .analysis.annotation import FIELDS, GUIDELINES, write_csv

    trajectories = load_trajectories(args.raw)
    blind, key = export_sample(trajectories, n=args.n, seed=args.seed, strata=tuple(args.strata.split(",")))
    os.makedirs(args.out, exist_ok=True)
    write_csv(blind, os.path.join(args.out, "annotation_sample.csv"), FIELDS)
    write_csv(key, os.path.join(args.out, "annotation_key.csv"))
    with open(os.path.join(args.out, "GUIDELINES.md"), "w", encoding="utf-8") as f:
        f.write(GUIDELINES)
    strata_counts = {}
    for k in key:
        strata_counts[(k["model"], k["difficulty"], k["phase"])] = strata_counts.get((k["model"], k["difficulty"], k["phase"]), 0) + 1
    print(f"  {len(blind)} turns -> {args.out}/annotation_sample.csv (blind), annotation_key.csv (automated labels), GUIDELINES.md")
    print(f"  strata covered: {len(strata_counts)}; per-stratum sizes {sorted(set(strata_counts.values()))}")
    return 0


def cmd_annotate_score(args: argparse.Namespace) -> int:
    from .analysis import annotation_report
    from .analysis.annotation import read_labels
    from .analysis.rescore import save_json

    auto = read_labels(args.key, column="auto_label")
    annotators = {os.path.splitext(os.path.basename(p))[0]: read_labels(p) for p in args.annotations}
    report = annotation_report(auto, annotators)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_json(report, args.out)
    print(f"  sample: {report['n_sample']} turns; annotators: {', '.join(report['annotators'])}")
    for name, r in report["auto_vs_annotator"].items():
        print(f"  automated vs {name:12s}: agreement {r['agreement']:.3f}  kappa {r['kappa']:.3f}  "
              f"(hallucinated-vs-clean kappa {r['binary_kappa']:.3f}, n={r['n']})")
    for pair, k in report.get("pairwise_kappa", {}).items():
        print(f"  {pair:24s}: Cohen's kappa {k:.3f}")
    if "fleiss_kappa" in report:
        print(f"  Fleiss' kappa across annotators: {report['fleiss_kappa']:.3f}  "
              f"(majority-vote reference on {report['n_majority_items']} items)")
    print(f"  Saved: {args.out}")
    return 0


def cmd_context_gap(args: argparse.Namespace) -> int:
    """Filler per pre-pivot user turn needed for tier A to match tier B's context at the pivot."""
    from .analysis import build_turn_df, load_trajectories

    turn_df = build_turn_df(load_trajectories(args.raw))
    at_pivot = turn_df[turn_df["pivot_occurred"].astype(bool)].groupby(["scenario_id", "run_idx", "difficulty"], observed=True).first().reset_index()
    means = at_pivot.groupby("difficulty", observed=True)["context_tokens"].mean()
    if args.source not in means.index or args.target not in means.index:
        print(f"need both tiers in the data; have {list(means.index)}", file=sys.stderr)
        return 2
    gap = float(means[args.target] - means[args.source])
    pre_turns = max(1, int(at_pivot["turn_idx"].median()) - 1)
    per_turn = max(0, int(round(gap / pre_turns)))
    print(f"  mean context at the pivot: {args.source}={means[args.source]:.0f} tokens, {args.target}={means[args.target]:.0f} tokens")
    print(f"  gap {gap:+.0f} tokens over ~{pre_turns} pre-pivot user turns -> --filler-tokens {per_turn}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    scenarios = _select_scenarios(args)
    for s in scenarios:
        print(f"{s['id']:24s} {s['difficulty']:12s} {s['domain']:10s} "
              f"pre={len(s['expected_pre'])} post={len(s['expected_post'])} complication@T{s['complication_turn']}")
    print(f"\n{len(scenarios)} scenarios  " + "  ".join(f"{t}={c}" for t, c in tier_counts(scenarios).items()))
    return 0


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--smoke", action="store_true", help="one flight scenario per tier")
    p.add_argument("--tiers", help=f"comma-separated subset of {','.join(TIERS)}")
    p.add_argument("--domains", help="comma-separated subset of restaurant,flight")
    p.add_argument("--ids", help="comma-separated scenario ids")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trusttrajectory", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the benchmark against a model")
    run.add_argument("--model", required=True, help="OpenRouter model id, e.g. google/gemini-3.5-flash")
    run.add_argument("--display-name", help="short label for filenames and figures")
    run.add_argument("--preset", default="v4or_t13", choices=sorted(PRESETS), help="harness preset (default: v4or_t13)")
    run.add_argument("--runs", type=int, default=1, help="rollouts per scenario (N)")
    run.add_argument("--out", default="results", help="output directory")
    run.add_argument("--tag", help="run tag (default: <model>_<mode>_<timestamp>)")
    _add_selection_args(run)
    run.add_argument("--max-turns", type=int)
    run.add_argument("--pivot-turn", type=int, help="fixed pivot turn")
    run.add_argument("--pivot-range", type=int, nargs=2, metavar=("LO", "HI"), help="randomise the pivot per trajectory")
    run.add_argument("--gate", choices=["hard", "none"], help="execution gate mode")
    run.add_argument("--min-booking-turn", type=int)
    run.add_argument("--temperature", type=float)
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--full-context", action="store_true", help="send the full history every turn (no message window)")
    run.add_argument("--filler-tokens", type=int, help="length control: neutral filler tokens appended to each pre-pivot user turn")
    run.add_argument("--state-card", action="store_true", help="intervention: restate the confirmed state right before the pivot")
    run.add_argument("--context-window", type=int, metavar="N",
                     help="send only the system prompt plus the last N-1 messages (0 = full history)")
    run.add_argument("--memory", choices=["full", "summary"],
                     help="condition: 'summary' keeps a rolling summary the model writes itself plus the last few messages")
    run.add_argument("--memory-keep", type=int, metavar="N", help="summary memory: messages kept verbatim (default 6)")
    run.add_argument("--memory-every", type=int, metavar="N", help="summary memory: re-summarise after N further messages (default 4)")
    run.add_argument("--interference", type=int, metavar="K",
                     help="condition: K unrelated bookings between the post-pivot booking and the probes")
    run.add_argument("--interference-max-turns", type=int, metavar="N", help="turn cap per distractor booking (default 8)")
    run.add_argument("--probes", choices=["direct", "presupposition", "both"],
                     help="condition: post-pivot probes that assert the pre-pivot value ('we're still at 7 PM, right?')")
    run.add_argument("--strip-think", action="store_true", help="strip <think> blocks from model output")
    run.add_argument("--reasoning", choices=["off", "low", "medium", "high"],
                     help="OpenRouter reasoning setting applied to every model (matched inference); default: provider default")
    run.add_argument("--workers", type=int, default=1, help="concurrent trajectories")
    run.add_argument("--scorer", default="regex", choices=["regex", "assertion", "llm_judge", "judge_assertion"])
    run.add_argument("--judge-model", help="judge model id for --scorer llm_judge (default: same as --model)")
    run.add_argument("--api-key", help="API key (default: $OPENROUTER_API_KEY)")
    run.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint with this tag")
    run.add_argument("--no-analysis", action="store_true")
    run.add_argument("--n-boot", type=int, default=10_000)
    run.add_argument("--min-n", type=int, default=20, help="prune decay-curve depths with fewer trajectories")
    run.set_defaults(func=cmd_run)

    an = sub.add_parser("analyze", help="analyse raw_*.json / checkpoint_*.json files")
    an.add_argument("raw", nargs="+", help="trajectory JSON files (concatenated)")
    an.add_argument("--out", required=True, help="analysis output directory")
    an.add_argument("--tag", help="tag used in figure filenames")
    an.add_argument("--preset", choices=sorted(PRESETS), help="preset the run used (sets scoring floors)")
    an.add_argument("--pivot-turn", type=int, help="pivot line for figures (default: from the records)")
    an.add_argument("--n-boot", type=int, default=10_000)
    an.add_argument("--min-n", type=int, default=20)
    an.add_argument("--no-figures", action="store_true")
    an.set_defaults(func=cmd_analyze)

    rs = sub.add_parser("rescore", help="re-score stored trajectories with another scorer and report agreement")
    rs.add_argument("raw", nargs="+", help="trajectory JSON files")
    rs.add_argument("--out", required=True)
    rs.add_argument("--preset", default="v4or_t13", choices=sorted(PRESETS))
    rs.add_argument("--scorer", default="llm_judge", choices=["regex", "assertion", "llm_judge", "judge_assertion"])
    rs.add_argument("--judge-model", help="judge model id (required for llm_judge)")
    rs.add_argument("--reference", nargs="+", help="report agreement against the labels in these files instead of the stored labels")
    rs.add_argument("--workers", type=int, help="concurrent trajectories (default: 4 for llm_judge, else 1)")
    rs.add_argument("--replay", action="store_true",
                    help="rebuild each conversation from the stored assistant turns (recomputes in-context slot checks, "
                         "committed facts and phase bookkeeping with current code) instead of re-scoring stored records")
    rs.add_argument("--judge-fields", nargs="+",
                    help="copy per-turn judge_* fields from these judge-rescored files before scoring (for judge_assertion)")
    rs.add_argument("--checkpoint", action="store_true",
                    help="write/resume checkpoint_rescored_<scorer>.json in --out (default for llm_judge only)")
    rs.add_argument("--no-checkpoint", action="store_true", help="disable checkpointing even for llm_judge")
    rs.add_argument("--api-key")
    rs.set_defaults(func=cmd_rescore)

    cp = sub.add_parser("compare", help="multi-model / multi-condition tables and figures")
    cp.add_argument("raw", nargs="+", help="one raw JSON file per model or condition")
    cp.add_argument("--out", required=True)
    cp.add_argument("--labels", help="comma-separated labels, one per raw file (default: model label)")
    cp.add_argument("--preset", choices=sorted(PRESETS))
    cp.add_argument("--n-boot", type=int, default=10_000)
    cp.add_argument("--min-n", type=int, default=20)
    cp.add_argument("--no-figures", action="store_true")
    cp.set_defaults(func=cmd_compare)

    es = sub.add_parser("estimate", help="estimate API calls and cost for a run")
    es.add_argument("--models", required=True, help="comma-separated model ids")
    es.add_argument("--runs", type=int, default=3)
    es.add_argument("--controls", action="store_true", help="add the randomised-pivot and no-gate runs (N=1 each)")
    es.add_argument("--judge-model", help="also estimate judge rescoring of the main runs")
    es.add_argument("--turns-per-trajectory", type=int, default=18)
    es.add_argument("--tokens-in", type=int, default=1800)
    es.add_argument("--tokens-out", type=int, default=200)
    es.add_argument("--price-in", type=float, help="USD per M input tokens for unknown models")
    es.add_argument("--price-out", type=float, help="USD per M output tokens for unknown models")
    es.add_argument("--workers", type=int, default=4)
    _add_selection_args(es)
    es.set_defaults(func=cmd_estimate)

    ae = sub.add_parser("annotate-export", help="export a blind stratified sample of turns for human annotation")
    ae.add_argument("raw", nargs="+")
    ae.add_argument("--out", required=True)
    ae.add_argument("--n", type=int, default=120)
    ae.add_argument("--seed", type=int, default=0)
    ae.add_argument("--strata", default="model,difficulty,phase")
    ae.set_defaults(func=cmd_annotate_export)

    asc = sub.add_parser("annotate-score", help="agreement between annotators and the automated scorer")
    asc.add_argument("--key", required=True, help="annotation_key.csv from annotate-export")
    asc.add_argument("annotations", nargs="+", help="one filled-in annotation_sample.csv per annotator")
    asc.add_argument("--out", default="annotation_report.json")
    asc.set_defaults(func=cmd_annotate_score)

    cg = sub.add_parser("context-gap", help="filler needed so one tier's context matches another's at the pivot")
    cg.add_argument("raw", nargs="+")
    cg.add_argument("--source", default="medium")
    cg.add_argument("--target", default="hard")
    cg.set_defaults(func=cmd_context_gap)

    ls = sub.add_parser("list-scenarios", help="print the scenario suite")
    _add_selection_args(ls)
    ls.set_defaults(func=cmd_list)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
