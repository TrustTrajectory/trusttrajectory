# Results

Curated outputs of the runs reported in the paper. The repository keeps the small, derived files
(per-run summaries, bootstrap tables, judge agreement); the raw trajectory files
(`raw_*.json`, ~560 MB uncompressed) and the replay-rescored copies (`rescored/`) are distributed
as the release asset `trusttrajectory_raw_trajectories.tar.gz`. Unpack it at the repository root
to restore the full tree:

```bash
tar xzf trusttrajectory_raw_trajectories.tar.gz      # creates results/{paper,paper_v3,conditions,pilot}/...
```

| Directory | Contents |
|---|---|
| `paper/` | Matched re-run of the four paper models (N=3, full context, `--reasoning low`), the reviewer-requested controls, the length-matched Medium tier, the GPT-5.6 judge readings and the annotation export. `summary_v1/` is scored with the omission-based rules, `summary/` with the assertion-based rules. |
| `paper_v3/` | The same trajectories replayed and rescored with the final assertion-based scorer (`trusttrajectory rescore --replay --scorer assertion`). |
| `conditions/` | Harder conditions on all 46 scenarios (context budgets k=6/10/20/full, presupposition probes) for six models; `scripts/conditions_report.py` summarises them. |
| `pilot/` | Ten-scenario pilot of every condition (rolling summary, interference, presupposition, windows) with `summary/` tables from `scripts/pilot_report.py`. |

Regenerate the analysis for any raw file with

```bash
trusttrajectory analyze results/paper/main/<model>/raw_*.json --out results/paper/main/<model>/analysis --preset v4or_t13
trusttrajectory rescore results/paper/main/<model>/raw_*.json --replay --scorer assertion --out results/paper_v3/main/<model>
```

