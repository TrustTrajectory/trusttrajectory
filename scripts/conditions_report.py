"""Summarise the full condition runs across models.

    python scripts/conditions_report.py [results/conditions] [--main results/paper_v3/main]

Reads <cond>/<model>/rescored/raw_rescored_assertion.json when present (else the raw file).
For the four paper models the full-context column comes from the main suite (N=3).
"""
from __future__ import annotations

import glob
import json
import os
import sys

from trusttrajectory.analysis import build_turn_df, pre_post_rates, probe_outcomes, bootstrap_pre_post

root = next((a for a in sys.argv[1:] if not a.startswith("--")), "results/conditions")
main_dir = sys.argv[sys.argv.index("--main") + 1] if "--main" in sys.argv else "results/paper_v3/main"
MODELS = [("claude-sonnet-4.5", "Claude S. 4.5"), ("claude-sonnet-4.6", "Claude S. 4.6"),
          ("gemini-3.5-flash", "Gemini 3.5 F."), ("llama-3.3-70b-instruct", "Llama-3.3-70B"),
          ("llama-3.1-8b-instruct", "Llama-3.1-8B"), ("gemma-3-12b-it", "Gemma 3 12B")]
PAPER = {"claude-sonnet-4.5", "claude-sonnet-4.6", "gemini-3.5-flash", "llama-3.3-70b-instruct"}
WINDOWS = [("window6", 6), ("window10", 10), ("window20", 20), ("full", None)]


def load(model, cond):
    if cond == "full" and model in PAPER:
        f = os.path.join(main_dir, model, "raw_rescored_assertion.json")
        return json.load(open(f)) if os.path.exists(f) else None
    d = os.path.join(root, cond, model)
    f = os.path.join(d, "rescored", "raw_rescored_assertion.json")
    if not os.path.exists(f):
        fs = glob.glob(os.path.join(d, "raw_*.json"))
        if not fs:
            return None
        f = fs[0]
    return json.load(open(f))


def stats(trajs):
    df = build_turn_df(trajs)
    r = bootstrap_pre_post(df, n_boot=2000)
    n_post = int(df["pivot_occurred"].sum())
    booked = sum(1 for t in trajs if t["booking_done"])
    po = probe_outcomes(df)
    return {"n": len(trajs), "booked": booked, "post": r["post_rate"], "ci": r["post_ci"], "n_post": n_post,
            "pre": r["pre_rate"], "probes": po.set_index("probe_kind").to_dict(orient="index") if not po.empty else {}}


# The system prompt's example booking (prompts.py): values a model can only have taken from the prompt
# when they are not the scenario's ground truth.
EXAMPLE = {"restaurant": {"cuisine": "italian", "city": "new york", "day": "friday", "time": "8 pm"},
           "flight": {"origin": "new york", "destination": "london", "date_out": "march 15", "date_return": "march 22"}}
CALL_HITS = {"restaurant": 3, "flight": 2}       # distinctive example values a tool call must carry to count


from trusttrajectory.scenarios import load_scenarios
SCENARIOS = {sc["id"]: sc for sc in load_scenarios()}


def example_share(trajs):
    """Post-pivot flagged turns whose wrong values include a prompt-example value that is neither the
    scenario's pre- nor post-pivot value; and post-pivot bookings made with >= 2 such values."""
    from trusttrajectory.scoring.assertions import normalize, _param_values
    flagged = recited = booked_example = 0
    for t in trajs:
        ex = EXAMPLE.get(t.get("domain", "restaurant"), {})
        sc = SCENARIOS.get(t["scenario_id"], {})
        own = {(k, normalize(str(v)).strip()) for st in (sc.get("expected_pre", {}), sc.get("expected_post", {})) for k, v in st.items()}
        for r in t["turn_records"]:
            if not r["pivot_occurred"] or r.get("segment", "main") != "main":
                continue
            exp_norm = {k: v for k, v in ex.items() if (k, v) not in own}   # example values the scenario never uses
            if r["hallucination_type"] != "NONE":
                flagged += 1
                vals = {(k, normalize(str(v)).strip()) for k, vs in r.get("wrong_values", {}).items() for v in vs}
                if any(exp_norm.get(k) == v for k, v in vals):
                    recited += 1
            if r.get("tool_called") and r.get("detect_reason") == "json_block":
                from trusttrajectory.tools import detect_tool_call
                ok, parsed, _ = detect_tool_call(r.get("assistant_text_full") or "")
                params = _param_values(parsed) if ok else {}
                hits = sum(1 for k, v in params.items() if exp_norm.get(k) == v)
                if hits >= CALL_HITS.get(t.get("domain", "restaurant"), 2):
                    booked_example += 1
    return flagged, recited, booked_example


rows = {}
for model, label in MODELS:
    for cond, _ in WINDOWS + [("presup", None)]:
        trajs = load(model, cond)
        if trajs:
            rows[(model, cond)] = stats(trajs)
            rows[(model, cond)]["example"] = example_share(trajs)

# ── console table ──
print(f"{'model':16s} " + " ".join(f"{c:>14s}" for c, _ in WINDOWS) + f" {'presup stale/n':>15s} {'presup post':>11s}")
for model, label in MODELS:
    cells = []
    for cond, _ in WINDOWS:
        st = rows.get((model, cond))
        cells.append(f"{st['post']:.3f} ({st['booked']}/{st['n']})" if st else "-")
    st = rows.get((model, "presup"))
    pre = st["probes"].get("presupposition") if st else None
    cells.append(f"{int(pre['stale'])}/{int(pre['n'])}" if pre else "-")
    cells.append(f"{st['post']:.3f}" if st else "-")
    print(f"{label:16s} " + " ".join(f"{c:>14s}" for c in cells[:-2]) + f" {cells[-2]:>15s} {cells[-1]:>11s}")

print("\nprompt-example recitation (flagged post-pivot turns / of which carry a distinctive example value / post-pivot tool calls built from the example):")
for model, label in MODELS:
    parts = []
    for cond, _ in WINDOWS:
        st = rows.get((model, cond))
        if st:
            f, rcd, bk = st["example"]
            parts.append(f"{cond}: {rcd}/{f} flagged, {bk} calls")
    print(f"  {label:16s} " + " | ".join(parts))
