"""Summarise a conditions pilot: per model x condition rates, probe outcomes, and flagged probe turns.

    python scripts/pilot_report.py [results/pilot] [--dump N] [--rescored]

``--rescored`` reads ``<cond>/<model>/rescored/raw_rescored_assertion.json`` (written by
``trusttrajectory rescore --replay``) instead of the run-time labels.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from trusttrajectory.analysis import build_turn_df, pre_post_rates, probe_outcomes, taxonomy_counts

root = next((a for a in sys.argv[1:] if not a.startswith("--")), "results/pilot")
dump = int(sys.argv[sys.argv.index("--dump") + 1]) if "--dump" in sys.argv else 0
rescored = "--rescored" in sys.argv
ORDER = ["baseline", "full", "window6", "summary", "interference1", "interference2", "presup", "presup_window"]

rows = []
flagged = []
for f in sorted(glob.glob(os.path.join(root, "*", "*", "raw_*.json"))):
    cond, label = f.split(os.sep)[-3], f.split(os.sep)[-2]
    if rescored:
        rf = os.path.join(os.path.dirname(f), "rescored", "raw_rescored_assertion.json")
        if not os.path.exists(rf):
            print(f"  [skip] {rf} missing", file=sys.stderr)
            continue
        f = rf
    trajs = json.load(open(f))
    df = build_turn_df(trajs)
    if df.empty:
        continue
    r = pre_post_rates(df)
    tax = {t["type"]: t["post"] for t in taxonomy_counts(df).to_dict(orient="records")}
    po = probe_outcomes(df).set_index("probe_kind") if "probe_kind" in df else None
    probes = df[df["probe_kind"].notna()] if "probe_kind" in df else df.iloc[0:0]
    n_bad = int(probes["probe_outcome"].isin(["stale", "wrong", "denied"]).sum()) if len(probes) else 0
    rows.append({
        "model": label, "cond": cond, "n": len(trajs), "booked": sum(t["booking_done"] for t in trajs),
        "pre": r["pre_rate"], "post": r["post_rate"], "post_n": int(df["pivot_occurred"].sum()),
        "CI": tax.get("CONSTRAINT_IGNORE", 0), "MC": tax.get("MEMORY_COLLAPSE", 0), "PC": tax.get("PREMATURE_COMMIT", 0),
        "probes": len(probes), "bad_probes": n_bad,
        "presup_stale": int(po.loc["presupposition", "stale"]) if po is not None and "presupposition" in po.index else None,
        "presup_n": int(po.loc["presupposition", "n"]) if po is not None and "presupposition" in po.index else None,
        "summaries": sum(t.get("n_summaries", 0) for t in trajs),
        "ctx_tokens": int(df["context_tokens"].max()),
    })
    for t in trajs:
        for rec in t["turn_records"]:
            if rec.get("probe") and rec.get("probe_outcome") in ("stale", "wrong", "denied"):
                flagged.append((label, cond, t["scenario_id"], rec))

rows.sort(key=lambda r: (r["model"], ORDER.index(r["cond"]) if r["cond"] in ORDER else 99))
hdr = f"{'model':24s} {'condition':14s} {'n':>3s} {'bk':>3s} {'pre':>6s} {'post':>6s} {'#post':>5s} {'CI':>3s} {'MC':>3s} {'PC':>3s} {'probes':>6s} {'bad':>4s} {'presup':>8s} {'summ':>4s} {'ctx':>6s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    presup = f"{r['presup_stale']}/{r['presup_n']}" if r["presup_n"] is not None else "-"
    print(f"{r['model']:24s} {r['cond']:14s} {r['n']:3d} {r['booked']:3d} {r['pre']:6.3f} {r['post']:6.3f} {r['post_n']:5d} "
          f"{r['CI']:3d} {r['MC']:3d} {r['PC']:3d} {r['probes']:6d} {r['bad_probes']:4d} {presup:>8s} {r['summaries']:4d} {r['ctx_tokens']:6d}")

if dump:
    print(f"\n=== flagged probe turns (first {dump}) ===")
    for label, cond, sid, rec in flagged[:dump]:
        print(f"\n[{label} / {cond} / {sid} T{rec['turn_idx']}] {rec['hallucination_type']} outcome={rec['probe_outcome']} "
              f"stale={rec['stale_slots']} wrong={ {k: sorted(set(v)) for k, v in rec['wrong_values'].items()} }")
        print("  USER:", rec["probe"]["text"])
        print("  ASSISTANT:", rec["assistant_text_full"][:380].replace("\n", " | "))
