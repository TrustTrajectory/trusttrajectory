"""Re-score stored trajectories with another scorer and measure agreement.

Turn records written by the package carry the full assistant text, the
committed-fact snapshot and the expected state, so any scorer can be re-run
offline: rescoring with the regex scorer reproduces the stored labels
exactly, and rescoring with the LLM-judge scorer gives the regex-versus-judge
comparison at the cost of one judge call per turn.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..scenarios import load_scenarios
from ..scoring import Scorer, TurnContext
from ..scoring.taxonomy import TAXONOMY_ORDER
from ..tools import classify_tool_signal

LABELS: List[str] = ["NONE"] + TAXONOMY_ORDER
_CARRY = ("latency_s", "assistant_text", "assistant_text_full", "committed_facts_before",
          "expected_state", "booking_executed", "detect_reason", "usage",
          "segment", "distractor_id", "probe", "memory_summary", "summary_usage", "memory_folded_messages",
          "judge_slots", "judge_claims_booking_complete", "judge_contradicted_facts", "judge_parsed")
_JUDGE_FIELDS = ("judge_slots", "judge_claims_booking_complete", "judge_contradicted_facts", "judge_parsed")


def _pivot_turn_of(t: Dict[str, Any]) -> int:
    if t.get("pivot_turn"):
        return int(t["pivot_turn"])
    for r in t["turn_records"]:
        if r.get("pivot_occurred"):
            return int(r["turn_idx"])
    return int(t.get("config", {}).get("pivot_turn", 13))


def turn_context(t: Dict[str, Any], r: Dict[str, Any], scenario: Optional[Dict[str, Any]],
                 committed_facts: Optional[Dict[str, str]] = None) -> TurnContext:
    """Rebuild the scorer input for one stored turn.

    ``committed_facts`` overrides the stored snapshot (used when a scorer
    recomputes committed facts with its own harvester).
    """
    text = r.get("assistant_text_full")
    if text is None:
        text = r.get("assistant_text", "")
    expected = r.get("expected_state")
    if expected is None:
        if scenario is None:
            raise ValueError(f"turn {t['scenario_id']}/T{r['turn_idx']} has no expected_state and no scenario given")
        expected = scenario["expected_post"] if r["pivot_occurred"] else scenario["expected_pre"]
    booking_executed = bool(r.get("booking_executed", False))
    tool_called, tool_parsed, _ = classify_tool_signal(text, booking_executed)
    extra = {k: r[k] for k in _JUDGE_FIELDS if k in r}
    segment = r.get("segment", "main")
    extra["segment"] = segment
    if r.get("probe"):
        extra["probe"] = r["probe"]
    if scenario is not None and scenario.get("accepted") and segment == "main":
        extra["accepted"] = scenario["accepted"]
    if segment == "interference":
        # a distractor booking: its own expected state is the ground truth and it has no committed facts
        return TurnContext(
            turn_text=text, turn_idx=int(r["turn_idx"]), pivot_turn=_pivot_turn_of(t),
            pivot_occurred=True, complication_occurred=False,
            expected=expected, tool_called=tool_called, tool_parsed=tool_parsed,
            all_params_present=bool(r.get("all_params_present", False)), committed_facts={},
            context_tokens=int(r.get("context_tokens", 0)), params_confirmed=False,
            booking_executed=booking_executed, expected_pre=expected, expected_post=expected, extra=extra,
        )
    return TurnContext(
        turn_text=text, turn_idx=int(r["turn_idx"]), pivot_turn=_pivot_turn_of(t),
        pivot_occurred=bool(r["pivot_occurred"]), complication_occurred=bool(r.get("complication_occurred", False)),
        expected=expected, tool_called=tool_called, tool_parsed=tool_parsed,
        all_params_present=bool(r.get("all_params_present", False)),
        committed_facts=dict(committed_facts if committed_facts is not None else (r.get("committed_facts_before") or {})),
        context_tokens=int(r.get("context_tokens", 0)), params_confirmed=bool(r.get("params_confirmed", False)),
        booking_executed=booking_executed,
        expected_pre=scenario["expected_pre"] if scenario else None,
        expected_post=scenario["expected_post"] if scenario else None,
        extra=extra,
    )


def _rescore_one(t: Dict[str, Any], scorer: Scorer, scenario: Optional[Dict[str, Any]],
                 progress: Optional[Any]) -> Tuple[Dict[str, Any], int]:
    """Re-score one trajectory; returns the new record and the number of truncated turns."""
    harvest = getattr(scorer, "harvest", None)
    truncated = 0
    new_records = []
    committed: Dict[str, str] = {}
    for r in t["turn_records"]:
        if r.get("assistant_text_full") is None:
            truncated += 1
        ctx = turn_context(t, r, scenario, dict(committed) if harvest else None)
        rec = scorer.score_turn(ctx)
        for key in _CARRY:
            if key in r:
                rec[key] = r[key]
        if harvest:
            rec["committed_facts_before"] = dict(committed) if r.get("segment", "main") == "main" else {}
            if r.get("segment", "main") == "main":
                committed = harvest(ctx.turn_text, committed)
        new_records.append(rec)
        if progress:
            progress(t, rec)
    hall = [r for r in new_records if r["hallucination_type"] != "NONE" and r.get("segment", "main") == "main"]
    new_t = {k: v for k, v in t.items() if k != "turn_records"}
    new_t.update({
        "turn_records": new_records,
        "first_hall_turn": hall[0]["turn_idx"] if hall else None,
        "total_hallucinations": len(hall),
        "scorer": getattr(scorer, "name", "unknown"),
        "rescored_from": t.get("scorer"),
    })
    return new_t, truncated


def _key(t: Dict[str, Any]) -> Tuple[str, int]:
    return (str(t["scenario_id"]), int(t.get("run_idx", 0)))


def rescore_trajectories(
    trajectories: List[Dict[str, Any]],
    scorer: Scorer,
    scenarios: Optional[Iterable[Dict[str, Any]]] = None,
    progress: Optional[Any] = None,
    workers: int = 1,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 5,
) -> List[Dict[str, Any]]:
    """Return new trajectory records scored by ``scorer`` (inputs are not mutated).

    ``workers`` > 1 scores trajectories concurrently (useful for network-bound
    LLM judges); turns within a trajectory stay sequential because committed
    facts accumulate along it.  Output order matches the input order.

    With ``checkpoint_path`` the finished trajectories are written every
    ``checkpoint_every`` completions, and trajectories already present in that
    file (same scenario and run, same scorer name) are not scored again, so an
    interrupted judge pass resumes where it stopped.
    """
    by_id = {s["id"]: s for s in (scenarios if scenarios is not None else load_scenarios())}
    lock = threading.Lock()
    done: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            for t in json.load(f):
                if t.get("scorer") == getattr(scorer, "name", None):
                    done[_key(t)] = t
        if done:
            print(f"  Resuming: {len(done)} trajectories already scored in {checkpoint_path}", flush=True)
    todo = [t for t in trajectories if _key(t) not in done]
    finished_since_save = [0]

    def safe_progress(t, rec):
        if progress:
            with lock:
                progress(t, rec)

    def save() -> None:
        if not checkpoint_path:
            return
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(list(done.values()), f)
        os.replace(tmp, checkpoint_path)

    def work(t):
        new_t, truncated = _rescore_one(t, scorer, by_id.get(t["scenario_id"]), safe_progress)
        with lock:
            done[_key(t)] = new_t
            finished_since_save[0] += 1
            if checkpoint_path and finished_since_save[0] >= checkpoint_every:
                save()
                finished_since_save[0] = 0
        return truncated

    if workers <= 1 or len(todo) <= 1:
        truncations = [work(t) for t in todo]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            truncations = list(pool.map(work, todo))
    if checkpoint_path and todo:
        save()
    truncated = sum(truncations)
    if truncated:
        print(f"  [warning] {truncated} turns had only truncated text (older records); "
              "labels for those turns are approximate", flush=True)
    return [done[_key(t)] for t in trajectories]


def merge_judge_fields(base: List[Dict[str, Any]], judge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy the per-turn ``judge_*`` fields of ``judge`` onto ``base`` (matched by scenario, run and turn)."""
    index = {(t["scenario_id"], t.get("run_idx", 0), r["turn_idx"]): r for t in judge for r in t["turn_records"]}
    out = []
    missing = 0
    for t in base:
        new_t = dict(t)
        new_records = []
        for r in t["turn_records"]:
            rec = dict(r)
            src = index.get((t["scenario_id"], t.get("run_idx", 0), r["turn_idx"]))
            if src is None:
                missing += 1
            else:
                for key in _JUDGE_FIELDS:
                    if key in src:
                        rec[key] = src[key]
            new_records.append(rec)
        new_t["turn_records"] = new_records
        out.append(new_t)
    if missing:
        print(f"  [warning] {missing} turns had no judge record to merge", flush=True)
    return out


# ── agreement ────────────────────────────────────────────────────────────

def cohen_kappa(a: List[str], b: List[str]) -> float:
    if not a or len(a) != len(b):
        return float("nan")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def _aligned_labels(a: List[Dict[str, Any]], b: List[Dict[str, Any]], parsed_only: bool = False) -> List[Tuple[str, str, bool]]:
    def index(trajs):
        return {(t["scenario_id"], t["run_idx"], r["turn_idx"]): (r["hallucination_type"], bool(r["pivot_occurred"]), r.get("judge_parsed", True))
                for t in trajs for r in t["turn_records"]}
    ia, ib = index(a), index(b)
    keys = sorted(set(ia) & set(ib))
    if parsed_only:
        keys = [k for k in keys if ib[k][2] is not False]
    return [(ia[k][0], ib[k][0], ia[k][1]) for k in keys]


def agreement_report(reference: List[Dict[str, Any]], other: List[Dict[str, Any]],
                     names: Tuple[str, str] = ("regex", "judge")) -> Dict[str, Any]:
    """Turn-level agreement between two scorings of the same trajectories."""
    pairs = _aligned_labels(reference, other)
    if not pairs:
        return {"n_turns": 0}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    a_bin = ["HALL" if x != "NONE" else "NONE" for x in a]
    b_bin = ["HALL" if x != "NONE" else "NONE" for x in b]
    confusion = {la: {lb: 0 for lb in LABELS} for la in LABELS}
    for x, y in zip(a, b):
        confusion[x][y] += 1
    per_class = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        fp = sum(confusion[o][lab] for o in LABELS if o != lab)
        fn = sum(confusion[lab][o] for o in LABELS if o != lab)
        per_class[lab] = {
            f"{names[0]}_count": sum(confusion[lab].values()),
            f"{names[1]}_count": sum(confusion[o][lab] for o in LABELS),
            "precision_of_" + names[0] + "_vs_" + names[1]: tp / (tp + fp) if tp + fp else None,
            "recall_of_" + names[0] + "_vs_" + names[1]: tp / (tp + fn) if tp + fn else None,
        }

    def rates(labels: List[str]) -> Dict[str, float]:
        pre = [l for l, p in zip(labels, [x[2] for x in pairs]) if not p]
        post = [l for l, p in zip(labels, [x[2] for x in pairs]) if p]
        f = lambda xs: (sum(1 for l in xs if l != "NONE") / len(xs)) if xs else float("nan")
        post_ci = (sum(1 for l in post if l == "CONSTRAINT_IGNORE") / len(post)) if post else float("nan")
        return {"pre_rate": f(pre), "post_rate": f(post), "post_constraint_ignore_rate": post_ci}

    judge_parsed = [r.get("judge_parsed") for t in other for r in t["turn_records"] if "judge_parsed" in r]
    parsed_pairs = _aligned_labels(reference, other, parsed_only=True)
    parsed_only = None
    if judge_parsed and len(parsed_pairs) != len(pairs):
        pa, pb = [p[0] for p in parsed_pairs], [p[1] for p in parsed_pairs]
        parsed_only = {"n_turns": len(parsed_pairs),
                       "label_agreement": sum(1 for x, y in zip(pa, pb) if x == y) / len(parsed_pairs) if parsed_pairs else None,
                       "label_kappa": cohen_kappa(pa, pb)}
    return {
        "n_turns": len(pairs),
        "judge_parse_rate": (sum(1 for x in judge_parsed if x) / len(judge_parsed)) if judge_parsed else None,
        "parsed_only": parsed_only,
        "label_agreement": sum(1 for x, y in zip(a, b) if x == y) / len(pairs),
        "label_kappa": cohen_kappa(a, b),
        "binary_agreement": sum(1 for x, y in zip(a_bin, b_bin) if x == y) / len(pairs),
        "binary_kappa": cohen_kappa(a_bin, b_bin),
        "confusion": confusion,           # rows: reference scorer, columns: other scorer
        "per_class": per_class,
        f"{names[0]}_rates": rates(a),
        f"{names[1]}_rates": rates(b),
    }


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
