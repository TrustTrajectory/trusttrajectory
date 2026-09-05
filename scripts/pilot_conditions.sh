#!/usr/bin/env bash
# Pilot of the harder conditions on a 10-scenario subset (two per tier, both domains), N=1.
#
#   ./scripts/pilot_conditions.sh                 # default two models
#   MODELS="meta-llama/llama-3.1-8b-instruct" ./scripts/pilot_conditions.sh
#   CONDITIONS="baseline summary" ./scripts/pilot_conditions.sh
#
# Conditions (all scored with the assertion scorer; "baseline" is the paper's harness):
#   baseline      default preset (20-message window, direct probes)
#   full          full context, direct probes                         (--context-window 0)
#   window6       system prompt + last 5 messages                     (--context-window 6)
#   summary       rolling self-written summary, keep 4, fold every 2  (--memory summary, full context otherwise)
#   interference1 one distractor booking before the probes, full context
#   interference2 two distractor bookings, full context
#   presup        presupposition + direct probes, full context        (--probes both)
#   presup_window presupposition + direct probes under the 20-message window
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT="${OUT:-results/pilot}"
MODELS="${MODELS:-meta-llama/llama-3.3-70b-instruct anthropic/claude-sonnet-4.5}"
IDS="${IDS:-easy_01,flight_easy_01,medium_03,flight_medium_02,hard_01,flight_hard_03,conflict_03,flight_conflict_02,adversarial_01,flight_adversarial_01}"
CONDITIONS="${CONDITIONS:-baseline full window6 summary interference1 interference2 presup presup_window}"
RUNS="${RUNS:-1}"
WORKERS="${WORKERS:-4}"

flags_for() {
  case "$1" in
    baseline)      echo "" ;;
    full)          echo "--context-window 0" ;;
    window6)       echo "--context-window 6" ;;
    summary)       echo "--context-window 0 --memory summary --memory-keep 4 --memory-every 2" ;;
    interference1) echo "--context-window 0 --interference 1" ;;
    interference2) echo "--context-window 0 --interference 2" ;;
    presup)        echo "--context-window 0 --probes both" ;;
    presup_window) echo "--probes both" ;;
    *) echo "unknown condition $1" >&2; exit 2 ;;
  esac
}

for model in $MODELS; do
  label="${model##*/}"
  for cond in $CONDITIONS; do
    dir="$OUT/$cond/$label"
    if [ -f "$dir/raw_${label}_${cond}.json" ]; then
      echo "=== $label / $cond: done"; continue
    fi
    echo "=== $label / $cond"
    mkdir -p "$dir"
    # shellcheck disable=SC2046
    trusttrajectory run --model "$model" --display-name "$label" --ids "$IDS" --runs "$RUNS" --workers "$WORKERS" \
      --scorer assertion --preset v4or_t13 --out "$dir" --tag "${label}_${cond}" --no-analysis \
      $(flags_for "$cond") > "$dir/run.log" 2>&1 || { echo "  FAILED (see $dir/run.log)"; continue; }
    grep -E "^\s+►|warning" "$dir/run.log" | sed 's/^/    /' | tail -3
  done
done

echo "=== comparison"
for model in $MODELS; do
  label="${model##*/}"
  files=(); labels=""
  for cond in $CONDITIONS; do
    f="$OUT/$cond/$label/raw_${label}_${cond}.json"
    [ -f "$f" ] && { files+=("$f"); labels="${labels:+$labels,}$cond"; }
  done
  [ ${#files[@]} -gt 0 ] && trusttrajectory compare "${files[@]}" --labels "$labels" --out "$OUT/summary/$label" --n-boot 2000 --min-n 1 --no-figures || true
done

# Re-score every pilot file offline with the current scorer (replay), then summarise.
if [ "${RESCORE:-0}" = "1" ]; then
  for f in "$OUT"/*/*/raw_*_*.json; do
    d="$(dirname "$f")"
    trusttrajectory rescore "$f" --scorer assertion --replay --preset v4or_t13 --out "$d/rescored" | grep -E "warning" || true
  done
  python scripts/pilot_report.py "$OUT" --rescored
fi
