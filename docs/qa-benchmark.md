# QA benchmark runs

`scripts/run_qa_benchmark.py` sends a persona question set through
`POST /api/v1/qa/ask` and records every answer exactly as returned, along with
what the deployment was running at the time. Scoring (citation checks,
LLM-as-a-judge, red-team) works from the recorded files and never calls the API
again, so one run can be scored many ways.

The runner talks to the FoodScholar API directly, not through the gateway the UI
uses, and imports nothing from `src/`: it runs from any machine that can reach
the API.

## Running

```bash
python scripts/run_qa_benchmark.py --api http://<foodscholar>:8001 \
    --personas personas.json --questions questions.json
```

| Option | Effect |
|---|---|
| `--persona P01`, `--type Factual`, `--only <question_id>`, `--limit N` | Run a subset (repeatable). The parent of a selected follow-up is always included. |
| `--run-id <id>` | Resume that run: questions already recorded as `ok` are skipped, errors are retried. |
| `--concurrency 2` | Questions in flight at once. Groq rate limits are the ceiling. |
| `--language-from question\|persona` | Answer language from the question row (default) or the persona's `api_mapping.language`. |
| `--retriever rag\|kggen\|no_rag` | Override the API's default retriever, e.g. to compare retrievers on the same set. |
| `--follow-up-delay 20` | Seconds a turn-2 question waits after its parent (see below). |
| `--dry-run` | Print the payloads that would be sent; no calls. |

The API must include `GET /qa/runtime-info`. The runner refuses to start without
it, because a deployment without that endpoint also lacks `bypass_cache`, and
silently ignores it, so re-runs would replay cached answers.

## What a run records

`benchmark-runs/<run_id>/` (git-ignored):

- **`manifest.json`**
  - Input file paths and sha256 hashes.
  - The selection and options.
  - `runtime_info_start` and `runtime_info_end`: two `GET /qa/runtime-info`
    snapshots, one before and one after the run.
  - `drift`: every setting that changed between the two snapshots.
  - `counts`: totals for `ok`, `error`, `skipped`, `needs_clarification`,
    `cache_hit` and `follow_up_without_context`. The last two should be 0.
- **`responses.jsonl`**: one line per question, with its ids, the request sent,
  `status` (`ok` / `error` / `skipped`), HTTP status, latency, and the complete
  `QAResponse`. That includes citations with quotes and context,
  `retrieved_sources`, `reasoning_steps`, `conversation_context`,
  `user_context` and `request_id`.

`runtime-info` reports:

- the git commit the image was built from (`GIT_COMMIT`, set by CI and the
  `Makefile`; `-dirty` if uncommitted changes were built);
- the model per pipeline role;
- every ranking and retrieval setting;
- the Langfuse version each prompt last served, or `fallback` for the in-code
  text.

Prompt versions are recorded per process, and a prompt appears only once it
has been used. That is why the end snapshot matters. A prompt that first
appears in the end snapshot is not drift; one whose version changed is.

## How each question is asked

For every question, the persona's `api_mapping` becomes request fields:

- `expertise_level`;
- `language`;
- `region` (`country_or_region`, e.g. `"Dublin, Ireland"`);
- `country` (the part after the last comma);
- `member_age_group`.

Every request also sets `bypass_cache: true` and
`user_id: "benchmark:<run_id>"`. That id selects a run's rows in
`GET /qa/requests?user_id=benchmark:<run_id>` and its traces in Langfuse.
Unfiltered, those rows still appear in the request list next to real users'
requests. `request_for()` in the script is the only place that knows the
question-set format.

`/qa/ask` accepts `country`, `region` and `member_age_group` from any caller
without a `member_id`. A member's own household context still takes precedence.

**Follow-ups** (`follow_up_of`, `turn: 2`) are sent in the parent's
`qa_thread_id`. The thread's summary and verbatim turns are written by a
background task after the parent's answer has been returned, so a follow-up
sent immediately would arrive with no conversation. That's why the runner waits
`--follow-up-delay` first. A follow-up answered without context
(`conversation_context.turn_count < 2`) is flagged and counted.

**Clarifications** are results, not obstacles. When FoodScholar asks one, the
runner records it and doesn't answer. Any follow-up in that thread is skipped,
because the pending clarification would only be served again.

## Not yet built

- **Citation verification (4A)**: quote fidelity against the catalog text,
  metadata against the catalog record, DOI resolution, and rendered-link
  integrity in the answer markdown.
- **LLM-as-a-judge (4B)**: a different model family at temperature 0.
  `retrieved_sources` carries no text, so the judge step must fetch each cited
  source's text from the catalog by URN to judge entailment.
- **Red-team (4C)**: a separate prompt set run through the same runner.
- **Langfuse (5)**: upload the question set as a dataset and attach scores to
  the traces of a run. This needs a reachable Langfuse; until then the run files
  are the record.
