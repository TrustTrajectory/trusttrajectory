#!/usr/bin/env bash
# Re-score every raw trajectory file of a finished paper suite with the
# assertion scorer (v2) and rebuild the per-model / cross-model comparisons.
#
#   IN=results/paper OUT=results/paper_v2 ./scripts/rescore_suite.sh
#
# Judge files (judge/<model>/raw_rescored_judge_*.json) are re-scored with the
# stored-judge assertion scorer and compared against the regex assertion labels.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
IN="${IN:-results/paper}"
OUT="${OUT:-results/paper_v2}"
PRESET="${PRESET:-v4or_t13}"
LABELS="${LABELS:-Llama-3.3-70B,Claude S. 4.5,Claude S. 4.6,Gemini 3.5 F.}"
MODELS="${MODELS:-llama-3.3-70b-instruct claude-sonnet-4.5 claude-sonnet-4.6 gemini-3.5-flash}"
mkdir -p "$OUT"

rescore() {  # rescore <raw.json> <outdir>   (replay: harness state is recomputed from the stored turns)
  trusttrajectory rescore "$1" --scorer assertion --replay --preset "$PRESET" --out "$2" | grep -E 'rates|warning' | sed "s|^|    $(basename "$2") |"
}

mains=()
for m in $MODELS; do
  echo "=== $m"
  rescore "$IN/main/$m/raw_$m.json" "$OUT/main/$m"
  mains+=("$OUT/main/$m/raw_rescored_assertion.json")
  trusttrajectory analyze "$OUT/main/$m/raw_rescored_assertion.json" --out "$OUT/main/$m/analysis" --no-figures >/dev/null
  conds=("$OUT/main/$m/raw_rescored_assertion.json"); names=("main")
  for c in pivot_random:pivotrand nogate:nogate statecard:statecard; do
    d=${c%%:*}; tag=${c##*:}; f="$IN/$d/$m/raw_${m}_$tag.json"
    if [[ -f "$f" ]]; then rescore "$f" "$OUT/$d/$m"; conds+=("$OUT/$d/$m/raw_rescored_assertion.json"); names+=("$d"); fi
  done
  trusttrajectory compare "${conds[@]}" --labels "$(IFS=,; echo "${names[*]}")" --out "$OUT/controls/$m" --preset "$PRESET" --no-figures | tail -8
  lc="$IN/length_control/$m/raw_${m}_medium_padded.json"
  if [[ -f "$lc" ]]; then
    rescore "$lc" "$OUT/length_control/$m"
    trusttrajectory compare "$OUT/main/$m/raw_rescored_assertion.json" "$OUT/length_control/$m/raw_rescored_assertion.json" \
      --labels "main,medium padded" --out "$OUT/length_control/$m/summary" --preset "$PRESET" --no-figures | tail -6
  fi
  for j in "$IN"/judge/"$m"/raw_rescored_judge_*.json; do
    [[ -f "$j" ]] || continue
    echo "  --- judge (stored slots) vs assertion scorer"
    trusttrajectory rescore "$OUT/main/$m/raw_rescored_assertion.json" --scorer judge_assertion --judge-fields "$j" --preset "$PRESET" \
      --reference "$OUT/main/$m/raw_rescored_assertion.json" --out "$OUT/judge/$m" | grep -E 'agreement|rates|parsed'
  done
done

echo "=== cross-model summary"
trusttrajectory compare "${mains[@]}" --labels "$LABELS" --out "$OUT/summary" --preset "$PRESET" | tail -12
echo "RESCORE SUITE DONE -> $OUT"
