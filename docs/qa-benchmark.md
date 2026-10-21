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
| `--run-id <id>` | Resume that run: questions already recorded as `ok` are skipped. Errors, degraded answers and the follow-ups of re-asked questions are asked again. |
| `--concurrency 2` | Questions in flight at once. Groq rate limits are the ceiling. |
| `--language-from question\|persona` | Answer language from the question row (default) or the persona's `api_mapping.language`. |
| `--retriever rag\|kggen\|no_rag` | Override the API's default retriever, e.g. to compare retrievers on the same set. |
| `--follow-up-delay 20` | Seconds a turn-2 question waits after its parent (see below). |
| `--clarify persona\|none` | Answer FoodScholar's clarifications as the persona (default), or record them and stop (see below). |
| `--groq-api-key <key>` | Groq key for `--clarify persona`; defaults to `GROQ_API_KEY`. |
| `--clarify-model`, `--clarification-answers <file>` | The Groq model that answers (default `openai/gpt-oss-20b`), and per-question scripted answers that override it. |
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
  - `counts`: totals for `ok`, `error`, `skipped`, `clarifications_asked`,
    `needs_clarification` (asked and left unanswered), `cache_hit`,
    `follow_up_without_context` and `degraded_retrieval`. The last three should
    be 0. `degraded_retrieval` counts answers where a search errored,
    for example an Elasticsearch timeout. The pipeline still answers from the
    other searches, so the answer is weaker but the request doesn't fail.
- **`responses.jsonl`**: one line per question, with its ids, the request sent,
  `status` (`ok` / `error` / `skipped`), HTTP status, latency, and the complete
  `QAResponse`. That includes citations with quotes and context,
  `retrieved_sources`, `reasoning_steps`, `conversation_context`,
  `user_context` and `request_id`. The file is append-only: a resumed run adds
  a new line for each question it asks again and keeps the earlier attempt.
  **The last line per `question_id` is the result**; earlier ones are history.

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

**Clarifications** are answered as the persona would answer them, within the
same run. They have to be: a pending clarification expires after 30 minutes.
FoodScholar asks at most one per thread, so one answer always leads to the
final answer.

- **How the answer is picked:** `openai/gpt-oss-20b` on Groq reads the persona's
  profile, the question, the clarification and its options, and must pick one
  of the offered options (a constrained JSON schema). For example, P01, a parent
  of school-age children, would answer "6-12".
- **Scripted answers win:** an entry in `--clarification-answers`
  (`{"question_id": "6-12"}`) overrides the model for that question.
- **Both turns are kept:** `clarification_turns` holds the clarification
  response exactly as returned, with the answer given, its source and the
  model's one-line reason. `response` is the final answer. The clarification
  category scores the first; the judge scores the second.
- **When no answer can be given** (an option outside the list, truncated
  output, an API error): the clarification stays the result, the error is
  recorded, and follow-ups in that thread are skipped.
- **`--clarify none`:** keeps the earlier behaviour for every question; the
  clarification is recorded and nothing is answered.

The Groq key comes from `--groq-api-key` or `GROQ_API_KEY` and is never
written to the run files. The runner checks the key and model before the run
starts.

## Closed-form benchmarks (MMLU)

These are benchmarks with known answers, run through the same runner and
scored without a judge.

1. **`scripts/mmlu_to_questions.py`** turns an MMLU split into a question set.
   - The question text is the stem, the four lettered options, and an
     instruction to lead with the letter.
   - The correct letter is stored as `gold` and never sent.
   - A neutral persona asks everything. It has no country or age group, so
     the clarifier answers "in general" if FoodScholar asks for detail.
   - Questions over 1000 characters are left out and reported. None of the
     306 `nutrition` test questions are.
2. **The runner, twice:** as-is, then with `--retriever no_rag`. Both use the
   same pipeline and answer model; only retrieval differs.
3. **`scripts/score_choice_benchmark.py`** reads the chosen letter from each
   final answer.
   - First it matches the answer's opening (`C.`, `**C**`, `Answer: C`).
     Otherwise gpt-oss-20b on Groq reads it, restricted to the four letters
     or `none`.
   - It writes `scores.jsonl` and `scores.json` into the run directory.
   - An error, an unanswered clarification or no clear choice all count as
     wrong in `accuracy`. `accuracy_when_answered` excludes them.
   - `--compare <other run>` pairs two scored runs of the same questions and
     runs an exact McNemar test on the questions only one of them got right.

```bash
python scripts/mmlu_to_questions.py --parquet <mmlu>/all/test-00000-of-00001.parquet \
    --subject nutrition --out-questions mmlu-nutrition-test.json --out-personas mmlu-persona.json
```

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
