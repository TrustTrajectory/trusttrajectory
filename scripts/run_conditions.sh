#!/usr/bin/env bash
# Full-suite runs of the harder conditions, matched to the paper suite (preset v4or_t13,
# reasoning low, temperature 0.2), N=1 per condition on all 46 scenarios.
#
#   ./scripts/run_conditions.sh                                   # six models, four conditions
#   MODELS="google/gemma-3-12b-it" CONDITIONS="full window6" ./scripts/run_conditions.sh
#   RESCORE=1 ./scripts/run_conditions.sh                         # replay-rescore + report only
#
# Conditions (assertion scorer):
#   full      full context, direct probes  (the paper models already have this at N=3 in results/paper/main)
#   window6   system prompt + last 5 messages     window10  + last 9     window20  + last 19 (the default preset's setting)
#   presup    full context, presupposition + direct probes (--probes both)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT="${OUT:-results/conditions}"
MODELS="${MODELS:-anthropic/claude-sonnet-4.5 anthropic/claude-sonnet-4.6 google/gemini-3.5-flash meta-llama/llama-3.3-70b-instruct meta-llama/llama-3.1-8b-instruct google/gemma-3-12b-it}"
CONDITIONS="${CONDITIONS:-window6 window10 window20 presup}"
RUNS="${RUNS:-1}"
WORKERS="${WORKERS:-4}"
REASONING="${REASONING:-low}"
SELECT="${SELECT:-}"            # e.g. "--ids easy_01,flight_easy_01" for a smoke

flags_for() {
  case "$1" in
    full)     echo "--full-context" ;;
    window6)  echo "--context-window 6" ;;
    window10) echo "--context-window 10" ;;
    window20) echo "--context-window 20" ;;
    presup)   echo "--full-context --probes both" ;;
    *) echo "unknown condition $1" >&2; exit 2 ;;
  esac
}

if [ "${RESCORE:-0}" != "1" ]; then
  for model in $MODELS; do
    label="${model##*/}"
    for cond in $CONDITIONS; do
      dir="$OUT/$cond/$label"
      if [ -f "$dir/raw_${label}_${cond}.json" ]; then echo "=== $label / $cond: done"; continue; fi
      echo "=== $label / $cond  $(date +%H:%M)"
      mkdir -p "$dir"
      # shellcheck disable=SC2046,SC2086
      trusttrajectory run --model "$model" --display-name "$label" --runs "$RUNS" --workers "$WORKERS" $SELECT \
        --scorer assertion --preset v4or_t13 --reasoning "$REASONING" --out "$dir" --tag "${label}_${cond}" --no-analysis \
        $(flags_for "$cond") > "$dir/run.log" 2>&1 || { echo "  FAILED (see $dir/run.log)"; continue; }
      grep -E "warning" "$dir/run.log" | sed 's/^/    /' | tail -2 || true
    done
  done
fi

# Replay-rescore every file with the current scorer, then summarise.
for f in "$OUT"/*/*/raw_*_*.json; do
  [ -f "$f" ] || continue
  d="$(dirname "$f")"
  trusttrajectory rescore "$f" --scorer assertion --replay --preset v4or_t13 --out "$d/rescored" | grep -E "warning" | sed "s|^|$f: |" || true
done
python scripts/pilot_report.py "$OUT" --rescored
