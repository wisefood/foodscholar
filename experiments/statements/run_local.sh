#!/usr/bin/env bash
# Run FoodScholar from this checkout against the local statements stack
# (docker-compose.yaml next to this file), for retriever="statements".
#
# Configured like production where it matters for answers: answer cache off,
# no Langfuse (production serves the in-code prompts too), same models and
# ranking defaults. Background workers and the knowledge graph are off: the
# experiment only needs /qa/ask.
#
#   export GROQ_API_KEY=...            # never written anywhere by this script
#   experiments/statements/run_local.sh            # serves on 127.0.0.1:18011
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "error: export GROQ_API_KEY first" >&2
  exit 2
fi

commit="$(git rev-parse HEAD)"
git diff --quiet HEAD || commit="${commit}-dirty"

env -u LANGFUSE_PUBLIC_KEY -u LANGFUSE_SECRET_KEY \
  HOST=127.0.0.1 PORT="${PORT:-18011}" DEBUG=false \
  GIT_COMMIT="$commit" \
  ELASTIC_HOST=http://127.0.0.1:19200 \
  REDIS_HOST=127.0.0.1 REDIS_PORT=16379 \
  POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=15432 \
  POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres POSTGRES_DB=wisefood \
  CACHE_ENABLED=false KG_ENABLED=false \
  ENABLE_BACKGROUND_WORKER=false ENABLE_ENRICHMENT_JOB_WORKER=false \
  ENABLE_GUIDELINE_EXTRACTION_WORKER=false ENABLE_GUIDELINE_ENRICHMENT_WORKER=false \
  ENABLE_GUIDE_EXPORT_WORKER=false \
  python src/app.py
