#!/usr/bin/env bash
# Re-run the paper's evaluation with matched settings, plus the two harness
# controls and LLM-judge rescoring, then build the cross-model summary.
#
#   OPENROUTER_API_KEY in .env or the environment.
#   SMOKE=1 ./scripts/run_paper_suite.sh          # 5 scenarios per model, N=1 (~$1)
#   ./scripts/run_paper_suite.sh                   # full suite
#   MODELS="google/gemini-3.5-flash" RUNS=1 ./scripts/run_paper_suite.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v trusttrajectory >/dev/null 2>&1 && [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
command -v trusttrajectory >/dev/null 2>&1 || { echo "trusttrajectory not installed: pip install -e ." >&2; exit 127; }

MODELS="${MODELS:-meta-llama/llama-3.3-70b-instruct anthropic/claude-sonnet-4.5 anthropic/claude-sonnet-4.6 google/gemini-3.5-flash}"
RUNS="${RUNS:-3}"
WORKERS="${WORKERS:-4}"
JUDGE="${JUDGE:-openai/gpt-5.6-luna}"
OUT="${OUT:-results/paper}"
PRESET="${PRESET:-v4or_t13}"
REASONING="${REASONING:-low}"          # matched inference: lowest effort every endpoint accepts (Gemini 3.5 Flash cannot disable reasoning)
CONTROLS="${CONTROLS:-1}"        # randomised pivot + no gate + state-card intervention (N=1 each)
LENGTH_CONTROL="${LENGTH_CONTROL:-1}"  # medium tier re-run with filler matching the hard tier's context
RESCORE="${RESCORE:-1}"
SELECT=""
if [[ "${SMOKE:-0}" == "1" ]]; then SELECT="--smoke"; RUNS=1; OUT="${OUT}_smoke"; fi

COMMON="--preset $PRESET --full-context --reasoning $REASONING --workers $WORKERS --no-analysis $SELECT"
MAIN_RAWS=(); LABELS=()

for m in $MODELS; do
  name="${m##*/}"
  echo "=== MAIN  $m  (N=$RUNS) ==="
  trusttrajectory run --model "$m" --display-name "$name" $COMMON --runs "$RUNS" --out "$OUT/main/$name" --tag "$name"
  MAIN_RAWS+=("$OUT/main/$name/raw_$name.json"); LABELS+=("$name")
  if [[ "$CONTROLS" == "1" ]]; then
    echo "=== CONTROL randomised pivot  $m ==="
    trusttrajectory run --model "$m" --display-name "$name" $COMMON --runs 1 --pivot-range 11 15 \
      --out "$OUT/pivot_random/$name" --tag "${name}_pivotrand"
    echo "=== CONTROL no gate  $m ==="
    trusttrajectory run --model "$m" --display-name "$name" $COMMON --runs 1 --gate none \
      --out "$OUT/nogate/$name" --tag "${name}_nogate"
    echo "=== INTERVENTION state card at pivot  $m ==="
    trusttrajectory run --model "$m" --display-name "$name" $COMMON --runs 1 --state-card \
      --out "$OUT/statecard/$name" --tag "${name}_statecard"
  fi
  if [[ "$LENGTH_CONTROL" == "1" && -z "$SELECT" ]]; then
    echo "=== LENGTH CONTROL medium tier padded to the hard tier's context  $m ==="
    filler=$(trusttrajectory context-gap "$OUT/main/$name/raw_$name.json" --source medium --target hard | sed -n 's/.*--filler-tokens \([0-9]*\).*/\1/p')
    trusttrajectory run --model "$m" --display-name "$name" $COMMON --runs "$RUNS" --tiers medium --filler-tokens "${filler:-0}" \
      --out "$OUT/length_control/$name" --tag "${name}_medium_padded"
  fi
done

echo "=== COMPARE main runs ==="
trusttrajectory compare "${MAIN_RAWS[@]}" --labels "$(IFS=,; echo "${LABELS[*]}")" --out "$OUT/summary" --preset "$PRESET"

if [[ "$CONTROLS" == "1" ]]; then
  for m in $MODELS; do
    name="${m##*/}"
    echo "=== COMPARE controls  $name ==="
    trusttrajectory compare "$OUT/main/$name/raw_$name.json" "$OUT/pivot_random/$name/raw_${name}_pivotrand.json" \
      "$OUT/nogate/$name/raw_${name}_nogate.json" "$OUT/statecard/$name/raw_${name}_statecard.json" \
      --labels "fixed pivot,random pivot,no gate,state card" \
      --out "$OUT/controls/$name" --preset "$PRESET" --no-figures
    if [[ "$LENGTH_CONTROL" == "1" && -z "$SELECT" ]]; then
      trusttrajectory compare "$OUT/main/$name/raw_$name.json" "$OUT/length_control/$name/raw_${name}_medium_padded.json" \
        --labels "main,medium padded" --out "$OUT/length_control/$name/summary" --preset "$PRESET" --no-figures
    fi
  done
fi

if [[ "$RESCORE" == "1" ]]; then
  for m in $MODELS; do
    name="${m##*/}"
    echo "=== RESCORE with judge $JUDGE  $name ==="
    trusttrajectory rescore "$OUT/main/$name/raw_$name.json" --scorer llm_judge --judge-model "$JUDGE" \
      --preset "$PRESET" --out "$OUT/judge/$name"
  done
fi
echo "=== ANNOTATION SAMPLE (blind, 120 turns across models x tiers x phase) ==="
trusttrajectory annotate-export "${MAIN_RAWS[@]}" --out "$OUT/annotation" --n 120
echo "=== DONE: $OUT ==="
