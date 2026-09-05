#!/usr/bin/env bash
# Launch the full paper suite with one driver process per model, in parallel,
# under caffeinate so the machine does not sleep. Then build the combined
# cross-model summary and the annotation sample.
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-meta-llama/llama-3.3-70b-instruct anthropic/claude-sonnet-4.5 anthropic/claude-sonnet-4.6 google/gemini-3.5-flash}"
OUT="${OUT:-results/paper}"
export RUNS="${RUNS:-3}" WORKERS="${WORKERS:-4}" JUDGE="${JUDGE:-openai/gpt-5.6-luna}" REASONING="${REASONING:-low}" OUT
mkdir -p "$OUT/logs"
pids=()
for m in $MODELS; do
  name="${m##*/}"
  MODELS="$m" ./scripts/run_paper_suite.sh > "$OUT/logs/$name.log" 2>&1 &
  pids+=($!)
  echo "started $m (pid $!) -> $OUT/logs/$name.log"
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "all model drivers finished (status $status)"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
raws=(); labels=()
for m in $MODELS; do name="${m##*/}"; f="$OUT/main/$name/raw_$name.json"; [[ -f "$f" ]] && { raws+=("$f"); labels+=("$name"); }; done
if [[ ${#raws[@]} -gt 0 ]]; then
  trusttrajectory compare "${raws[@]}" --labels "$(IFS=,; echo "${labels[*]}")" --out "$OUT/summary" --preset "${PRESET:-v4or_t13}"
  trusttrajectory annotate-export "${raws[@]}" --out "$OUT/annotation" --n 120
fi
echo "FULL SUITE DONE status=$status"
exit $status
