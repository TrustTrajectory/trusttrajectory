# TrustTrajectory

Benchmark and evaluation harness for *TrustTrajectory: Measuring When LLM Agents Begin to
Hallucinate in Long-Trajectory Agentic Execution* (COLM 2026).

TrustTrajectory is a 46-scenario benchmark (26 restaurant, 20 flight bookings; five difficulty
tiers) in which a scripted user drips constraints one at a time, adds a mid-task complication,
and then fires a hard constraint pivot that forces the agent to overwrite slots it has already
confirmed. Every assistant turn is scored against the scripted ground truth with a five-class
taxonomy (`PARAM_DRIFT`, `PREMATURE_COMMIT`, `CONSTRAINT_IGNORE`, `CONFIDENT_CONFAB`,
`MEMORY_COLLAPSE`), which yields the First Hallucination Turn and the Trust Decay Curve. Two
scorers implement the taxonomy: omission-based rules (`--scorer regex`) and assertion-based rules
(`--scorer assertion`); an LLM judge and a replay mode re-score stored runs without new API calls.

## Install

```bash
git clone git@github.com:TrustTrajectory/trusttrajectory.git && cd trusttrajectory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # put your OpenRouter key in it, or export OPENROUTER_API_KEY
python -m pytest                # 147 tests, no network needed
```

All model traffic goes through OpenRouter, so any `provider/model` id it serves works.

## Reproducing the paper

Each step is a script; every knob it uses is an environment variable documented at the top of
the script. `trusttrajectory estimate --models <ids> --runs 3 --controls` prints the number of
API calls before you spend anything, and `SMOKE=1` runs five scenarios per model first.

1. **Main evaluation, controls and judge readings** (four models, N=3, matched settings):
   ```bash
   ./scripts/run_paper_suite.sh                    # -> results/paper
   ```
   Runs the main suite, the randomised-pivot / no-gate / state-card controls, the length-matched
   Medium tier, and the GPT-5.6 judge pass. Trajectories are labelled with the omission-based
   rules as they run.

2. **Assertion-based scoring by replay** (no API calls):
   ```bash
   IN=results/paper OUT=results/paper_v3 ./scripts/rescore_suite.sh
   ```
   Replays every stored trajectory through the scripted user, re-scores it with the
   assertion-based rules, applies the same rules to the judge's readings, and writes the
   per-model and cross-model summaries under `results/paper_v3/`.

3. **Harder conditions** (six models, N=1; context budgets and presupposition probes):
   ```bash
   ./scripts/run_conditions.sh                     # -> results/conditions, rescored by replay
   python scripts/conditions_report.py             # cross-model summary
   ```
   The rolling-summary and interference conditions were piloted on ten scenarios:
   ```bash
   ./scripts/pilot_conditions.sh && python scripts/pilot_report.py --rescored
   ```

Steps 1 and 3 sample new trajectories from the models, so they give an independent replication
with similar but not identical numbers (temperature 0.2, provider-side nondeterminism). The exact
tables in the paper are reproduced from the stored trajectories: download
`trusttrajectory_raw_trajectories.tar.gz` from the Releases page, unpack it at the repository
root, and run step 2 and `RESCORE=1 ./scripts/run_conditions.sh` followed by
`python scripts/conditions_report.py`. Scoring is deterministic given the trajectories.
`results/README.md` describes the layout of the result directories.

## Running your own model

```bash
trusttrajectory run --model <provider/model> --preset v4or_t13 --full-context --reasoning low --runs 3 --out results/<name>
trusttrajectory rescore results/<name>/raw_*.json --scorer assertion --replay --out results/<name>/assertion
trusttrajectory analyze results/<name>/assertion/raw_rescored_assertion.json --out results/<name>/analysis
```

`--context-window N`, `--memory summary`, `--interference K` and `--probes both` switch on the
harder conditions; `--pivot-range`, `--gate none`, `--state-card` and `--filler-tokens` are the
controls. `trusttrajectory list-scenarios` prints the suite.

## Layout

```
trusttrajectory/data/scenarios/   the 46 scenarios, one JSON file per tier
trusttrajectory/                  simulator, runner, scorers, replay, analysis, CLI
scripts/                          the reproduction scripts above
tests/                            pytest suite with a scripted fake model
results/                          curated outputs of the paper's runs (raw files in the release asset)
```

## Citation

```bibtex
@inproceedings{hegde2026trusttrajectory,
  title     = {TrustTrajectory: Measuring When LLM Agents Begin to Hallucinate in Long-Trajectory Agentic Execution},
  author    = {Hegde, Preetam and Li, Vincent and Dang, Jacob and Chatterjee, Aviv},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```
