#!/usr/bin/env bash
# The four experiments: both datasets (persona 360, MMLU nutrition 306) against
# both corpora (the production catalog, `rag`, and the FSKG statements),
# reusing the answers recorded in October, then scored and combined into one
# result set.
#
# Each dataset x corpus is one run with a fixed id (full-<dataset>-<setting>),
# seeded with --reuse, so running a part again resumes it instead of starting
# over. Nothing already answered is asked again, except what the resume rules
# say must be: failed searches, unanswered clarifications, and follow-ups that
# got no thread memory (with their parent).
#
#   export GROQ_API_KEY=...                   # clarifier + local server
#   experiments/full-benchmark/run.sh prod    # catalog: persona + MMLU via the port-forward on :18001
#   experiments/full-benchmark/run.sh local   # statements: persona + MMLU via run_local.sh on :18011
#   experiments/full-benchmark/run.sh finish  # score MMLU (statements against catalog), combine all four
set -uo pipefail
cd "$(dirname "$0")/../.."

PROD_API="${PROD_API:-http://127.0.0.1:18001}"
LOCAL_API="${LOCAL_API:-http://127.0.0.1:18011}"
PERSONAS=20260904-personas-v2.json
PERSONA_QUESTIONS=20260904-questions-v2.json
MMLU_PERSONA=mmlu-persona.json
MMLU_QUESTIONS=mmlu-nutrition-test.json
RUNS=benchmark-runs

# Earlier answers per dataset x setting (later wins on overlap). 20261021T093220Z
# is left out: an abandoned half-run that 20261021T100127Z repeats in full.
declare -A REUSE=(
  [persona-rag]="20261021T085045Z 20261021T090814Z"
  [persona-statements]="20261021T105916Z"
  [mmlu-rag]="20261021T100127Z"
  [mmlu-statements]="20261021T105704Z"
)

case "${1:-}" in
  prod|local)
    if [[ -z "${GROQ_API_KEY:-}" ]]; then echo "error: export GROQ_API_KEY first" >&2; exit 2; fi ;;
  finish) ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac

failed=()

run() {  # run <dataset> <setting> <api> [extra runner args...]
  local dataset=$1 setting=$2 api=$3; shift 3
  local personas=$PERSONAS questions=$PERSONA_QUESTIONS
  if [[ $dataset == mmlu ]]; then personas=$MMLU_PERSONA questions=$MMLU_QUESTIONS; fi
  local args=(--api "$api" --personas "$personas" --questions "$questions" --run-id "full-$dataset-$setting")
  [[ $setting != rag ]] && args+=(--retriever "$setting")
  for run_id in ${REUSE[$dataset-$setting]:-}; do args+=(--reuse "$RUNS/$run_id"); done
  echo "=== $dataset / $setting" >&2
  python scripts/run_qa_benchmark.py "${args[@]}" "$@" || failed+=("$dataset/$setting")
}

case "${1:-}" in
  prod)
    for dataset in persona mmlu; do run "$dataset" rag "$PROD_API"; done ;;
  local)
    for dataset in persona mmlu; do run "$dataset" statements "$LOCAL_API"; done ;;
  finish)
    score=(python scripts/score_choice_benchmark.py)
    "${score[@]}" "$RUNS/full-mmlu-rag" --questions "$MMLU_QUESTIONS" || failed+=("score rag")
    "${score[@]}" "$RUNS/full-mmlu-statements" --questions "$MMLU_QUESTIONS" \
      --compare "$RUNS/full-mmlu-rag" || failed+=("score statements")
    python scripts/combine_runs.py --out "$RUNS/combined-full" \
      --dataset persona="$PERSONA_QUESTIONS" --dataset mmlu="$MMLU_QUESTIONS" \
      "$RUNS"/full-{persona,mmlu}-{rag,statements} \
      || failed+=("combine: incomplete, see coverage above") ;;
esac

if (( ${#failed[@]} )); then
  echo "needs attention: ${failed[*]} (re-run the same part to resume)" >&2
  exit 1
fi
