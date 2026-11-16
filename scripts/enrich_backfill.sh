#!/usr/bin/env bash
# Enrich every catalog article that is not enriched yet, using a self-hosted
# OpenAI-compatible model server (GPUStack serving gpt-oss-20b), and write the
# results back to the data catalog. The work is done by enrich_backfill.py.
#
#   scripts/enrich_backfill.sh start        preflight, then run detached in tmux
#   scripts/enrich_backfill.sh attach       watch the live panel (detach: Ctrl-b d)
#   scripts/enrich_backfill.sh status       progress without attaching
#   scripts/enrich_backfill.sh stop         graceful stop (finishes articles in flight)
#   scripts/enrich_backfill.sh run [ARGS]   run in the foreground (what tmux runs)
#   scripts/enrich_backfill.sh dry-run [N]  enrich N articles (default 5), write nothing
#   scripts/enrich_backfill.sh preflight    check model + catalog; writes nothing
#   scripts/enrich_backfill.sh selftest     model only: enrich a sample abstract, print it
#   scripts/enrich_backfill.sh count        total / enriched / pending articles
#
# Extra ARGS go to `enrich_backfill.py run` (see --help), e.g.
#   scripts/enrich_backfill.sh start --limit 500
#   scripts/enrich_backfill.sh start --retry-failed
#   scripts/enrich_backfill.sh start --fq 'venue:"Nutrients"'
#
# Settings: $ENRICH_HOME/enrich.env (default ~/foodscholar-enrich/enrich.env).
# Keep it chmod 600 and out of git. Keys:
#   LLM_API_KEY=...                                   required
#   CATALOG_CLIENT_ID=... CATALOG_CLIENT_SECRET=...   or CATALOG_USERNAME/CATALOG_PASSWORD
#   optional: LLM_BASE_URL LLM_MODEL CATALOG_URL ENRICH_CONCURRENCY ENRICH_MAX_ATTEMPTS
#             LLM_REASONING_EFFORT
# State (results.jsonl, enriched.jsonl with rollback data, run.log, progress.json):
#   $ENRICH_HOME/state; the dry run uses $ENRICH_HOME/dry-run-state.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
ENRICH_HOME="${ENRICH_HOME:-$HOME/foodscholar-enrich}"
ENV_FILE="${ENRICH_ENV_FILE:-$ENRICH_HOME/enrich.env}"
PY_SCRIPT="$REPO/scripts/enrich_backfill.py"

die() { echo "error: $*" >&2; exit 2; }

if [[ -f "$ENV_FILE" ]]; then
  if [[ -n "$(find "$ENV_FILE" -perm /077 2>/dev/null)" ]]; then
    echo "warning: $ENV_FILE is readable by others; run: chmod 600 $ENV_FILE" >&2
  fi
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

# After the env file, so it can set these too.
VENV="${ENRICH_VENV:-$ENRICH_HOME/venv}"
SESSION="${ENRICH_TMUX_SESSION:-enrich}"
STATE_DIR="${ENRICH_STATE_DIR:-$ENRICH_HOME/state}"

export LLM_BASE_URL="${LLM_BASE_URL:-http://test6.magellan2.imsi.athenarc.gr/v1}"
export LLM_MODEL="${LLM_MODEL:-gpt-oss-20b}"
export CATALOG_URL="${CATALOG_URL:-https://demo.wisefood-project.eu/dc}"
# Parallel articles in flight. GPUStack balances them over the model instances;
# each llama.cpp instance splits its context across parallel slots, and at 8+
# concurrent annotations outputs degraded (lost abstract, placeholders), so the
# default stays at ~2 per instance. Raise it only while the panel's
# "passed validation first time" stays high.
export ENRICH_CONCURRENCY="${ENRICH_CONCURRENCY:-4}"
export ENRICH_STATE_DIR="$STATE_DIR"
# Production serves the in-code prompts; never pick up a Langfuse key from the shell.
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY

ensure_venv() {
  if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c "import langchain, wisefood, httpx" 2>/dev/null; then
    return
  fi
  local py="${ENRICH_PYTHON:-}"
  if [[ -z "$py" ]]; then
    for cand in python3.11 /mnt/miniconda3/envs/foodscholar/bin/python python3; do
      if command -v "$cand" >/dev/null 2>&1; then py="$cand"; break; fi
    done
  fi
  [[ -n "$py" ]] || die "no python found; set ENRICH_PYTHON"
  echo "setting up $VENV with $py ..."
  mkdir -p "$(dirname "$VENV")"
  "$py" -m venv "$VENV"
  # Only what the reused enrichment modules import; versions as in requirements.txt.
  "$VENV/bin/pip" install -q --disable-pip-version-check \
    "langchain==0.3.27" "wisefood>=0.0.42" "httpx>=0.27,<0.29"
}

need_llm() {
  [[ -n "${LLM_API_KEY:-}" ]] || die "LLM_API_KEY is not set (put it in $ENV_FILE)"
}

need_catalog() {
  if [[ -n "${CATALOG_CLIENT_ID:-}" && -n "${CATALOG_CLIENT_SECRET:-}" ]]; then return; fi
  if [[ -n "${CATALOG_USERNAME:-}" && -n "${CATALOG_PASSWORD:-}" ]]; then return; fi
  die "catalog credentials missing: set CATALOG_CLIENT_ID + CATALOG_CLIENT_SECRET (or CATALOG_USERNAME + CATALOG_PASSWORD) in $ENV_FILE"
}

py() { "$VENV/bin/python" "$PY_SCRIPT" "$@"; }

preflight() {
  ensure_venv; need_llm; need_catalog
  echo "== model: $LLM_MODEL at $LLM_BASE_URL"
  py selftest --repeat 1 || die "the model selftest failed (see above)"
  echo "== catalog: $CATALOG_URL"
  py count || die "could not log in to / search the catalog"
}

cmd="${1:-}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
  start)
    command -v tmux >/dev/null || die "tmux is not installed"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      die "tmux session '$SESSION' already exists: '$0 attach' to watch it, '$0 stop' to stop it"
    fi
    if [[ "${ENRICH_SKIP_PREFLIGHT:-0}" == 1 ]]; then
      ensure_venv; need_llm; need_catalog
    else
      preflight
    fi
    mkdir -p "$STATE_DIR"
    # Pass our overrides explicitly: a running tmux server would not inherit them.
    inner="$(printf 'ENRICH_HOME=%q ENRICH_ENV_FILE=%q ENRICH_VENV=%q ENRICH_STATE_DIR=%q %q run' \
      "$ENRICH_HOME" "$ENV_FILE" "$VENV" "$STATE_DIR" "$SELF")"
    for a in "$@"; do inner+=" $(printf '%q' "$a")"; done
    # Keep the pane open after the run so the final summary stays readable.
    tmux new-session -d -s "$SESSION" -x 160 -y 40 \
      "$inner; code=\$?; echo; echo \"run ended with exit code \$code; press Enter to close\"; read -r _"
    echo
    echo "started in tmux session '$SESSION'"
    echo "  watch:  $0 attach      (detach with Ctrl-b d)"
    echo "  status: $0 status"
    echo "  stop:   $0 stop        (graceful)"
    echo "  log:    $STATE_DIR/run.log"
    ;;
  attach)
    exec tmux attach -t "$SESSION"
    ;;
  stop)
    tmux has-session -t "$SESSION" 2>/dev/null || die "no tmux session '$SESSION'"
    tmux send-keys -t "$SESSION" C-c
    echo "sent a graceful stop; articles in flight finish first (watch with: $0 attach)"
    ;;
  status)
    ensure_venv
    py status
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session '$SESSION' is open ($0 attach)"
    fi
    ;;
  run)
    ensure_venv; need_llm; need_catalog
    exec "$VENV/bin/python" "$PY_SCRIPT" run "$@"
    ;;
  dry-run)
    ensure_venv; need_llm; need_catalog
    n="${1:-5}"; [[ $# -gt 0 ]] && shift
    exec "$VENV/bin/python" "$PY_SCRIPT" --state-dir "$ENRICH_HOME/dry-run-state" \
      run --dry-run --limit "$n" "$@"
    ;;
  preflight)
    preflight
    ;;
  selftest)
    ensure_venv; need_llm
    py selftest --show "$@"
    ;;
  count)
    ensure_venv; need_catalog
    py count "$@"
    ;;
  ""|-h|--help|help)
    sed -n '2,32p' "$SELF" | sed 's/^# \{0,1\}//'
    ;;
  *)
    die "unknown command '$cmd' (try: $0 help)"
    ;;
esac
