# Guideline facets, context-aware extraction, and the retrieval gate

This document covers the guideline pipeline end to end: how a rule sentence gets
its context, how the already-extracted corpus is enriched after the fact, and
what has to happen before guideline retrieval is gated on `status:active`.

Run the phases in order. Each is independently deployable; the ordering matters
only where noted.

---

## The problem being solved

Guidelines were extracted from guide PDFs page by page, with each page processed
in isolation. Three consequences:

1. **Rules lost their context.** "Provide portions of red meat twice a week",
   extracted from *Eating guidelines for 1–4 year olds*, arrived in the catalog
   with no population, no life stage, no age range. The rule is unattributable
   on its own, and the QA retriever was boosting `target_populations` and
   `food_groups` — fields nothing ever populated.
2. **Structures spanning a page break were unrecoverable.** A table split across
   two pages lost its header on the second page, because that page was a fresh
   request with no memory of the first.
3. **Anything stored was retrievable.** Guideline retrieval excluded only
   `status:deleted`, so drafts and unreviewed rules could be cited in an answer
   or served as a daily tip.

---

## Phase 1 — Catalog (`wisefood-data-api`)

New optional fields on the guideline schema and ES mapping: `life_stage`,
`age_min_months`/`age_max_months`, `setting`, `health_conditions`, `nutrients`,
`guideline_type`, `topic`, `audience`, `applicable_regions`, plus extraction and
enrichment provenance (`extractor_name`, `extractor_run_id`, `extraction_model`,
`enrichment_version`, `enrichment_confidence`, `ai_generated_fields`,
`enhancements`) and an `embedding` vector.

All additive, so `_ensure_mapping_fields` picks them up at startup and the ~2700
stored guidelines validate unchanged.

**New endpoints**

| Endpoint | Purpose |
|---|---|
| `PATCH /api/v1/guidelines/{id}/enrich` | Write machine-derived facets onto one rule |
| `POST /api/v1/guidelines/enrich-batch` | Same for up to 200 rules; supports `dry_run` |
| `POST /api/v1/guidelines/editorial-policy` | Bulk lifecycle edit — this is how a guide's rules are activated |
| `POST /api/v1/guidelines/embeddings/backfill` | Queue stored rules for embedding |

**The no-clobber guard.** `Guideline.enrich` writes a field only when its current
value is empty or was itself machine-written (per `ai_generated_fields`). A
human edit through the console takes ownership of the field — `patch` removes it
from `ai_generated_fields` — so re-enrichment can never undo editorial work.
`force_fields` overrides this per field when a caller knowingly wants it.

**Verify**

```bash
# Schema and guard logic, no live Elasticsearch needed
cd wisefood-data-api && PYTHONPATH=src python -m pytest tests/ -q

# After restarting the API against a live cluster:
curl -s "$ES/guidelines/_mapping" | jq '.guidelines.mappings.properties | keys'
# expect life_stage, setting, enrichment_version, embedding, ...
```

---

## Phase 2 — Context-aware extraction (`foodscholar`)

Each page request now carries two things it previously lacked.

**Document identity.** Assembled from the catalog guide record and, where that
leaves the population unestablished, from a profile pass over the guide's own
opening pages (`profile_guide_document`). Catalog values always win; the
document only fills gaps, and `guide_context.derived_fields` records which
values came from the document.

This matters because catalog metadata is frequently thin, while a guide states
its scope plainly on its cover — which is exactly the fact a rule needs to
inherit. Disable per-run with `profile_document: false` on the extract request.

**The previous page.** A rolling summary plus a ~600-character text tail is
passed forward to every page. Triage additionally returns
`continues_from_previous`; when set, the previous page *image* is attached to
the extraction request, so a table header on the other side of a page break is
still visible.

Extraction output is now `schema_version: 2`: each rule is an object carrying
`section_label`, `source_snippet`, facet hints and `confidence` rather than a
bare string. Results stored under v1 keep their version and still import — they
are never migrated.

The import path (`import_latest_result_to_guide`) now maps all of it onto the
catalog: per-rule `action_type` (previously one constant for a whole batch),
`source_refs` with artifact and page, facets, and extractor provenance.

**Verify**

```bash
cd foodscholar && PYTHONPATH=src python -m pytest tests/test_guideline_extractor.py -q

# End to end, on a guide with a known multi-page table, into a DRAFT guide:
curl -X POST "$FS/api/v1/guidelines/extract/$ARTIFACT" \
  -H 'content-type: application/json' \
  -d '{"guide_id": "urn:guide:...", "profile_document": true}'
curl -s "$FS/api/v1/guidelines/extract/$ARTIFACT" | jq '{
  schema_version: .result.schema_version,
  context: .result.guide_context,
  derived: .result.guide_context.derived_fields,
  continuations: .result.continuation_pages }'
```

Check that `guide_context.population_note` is right and that the pages listed in
`continuation_pages` are the ones that actually continue a table. Then import
with `dry_run: true` and confirm `facets` and `source_refs` are populated.

---

## Phase 3 — Enriching the existing corpus

No re-extraction. `GuidelineEnricher` walks the stored guidelines guide by
guide, resolves that guide's context once, and asks the enrichment agent to
assign facets to each rule.

Context is resolved from three sources, in descending order of authority: the
catalog guide record, the guide context captured by a v2 extraction run, and a
profile pass over the guide's PDF. The third is what rescues the existing 31
guides — they were extracted before any context was captured.

Idempotency is by `enrichment_version`: rules at or above the current version
are skipped, so runs are resumable and repeatable, and bumping
`GUIDELINE_ENRICHMENT_VERSION` re-enriches everything.

**Runbook**

```bash
# 1. Preview one guide. Read `guide_context` and `context_sources` first —
#    if the context is wrong, every facet under that guide will be wrong the
#    same way.
curl -X POST "$FS/api/v1/guidelines/enrichment/preview" \
  -H 'content-type: application/json' \
  -d '{"guide_urn": "urn:guide:...", "limit": 10}' | jq

# 2. One guide for real, then spot-check in the console review UI.
curl -X POST "$FS/api/v1/guidelines/enrichment/enqueue" \
  -H 'content-type: application/json' \
  -d '{"guide_urns": ["urn:guide:..."]}'

# 3. The whole corpus.
curl -X POST "$FS/api/v1/guidelines/enrichment/enqueue" -d '{}'
curl -s "$FS/api/v1/guidelines/enrichment/status" | jq '.totals'

# 4. Confirm idempotency: a second run should report everything skipped.
```

Then verify no-clobber: edit one facet by hand in the console, re-run that
guide's enrichment with `force: true`, and confirm the edited value survived.

---

## Phase 4 — The retrieval gate (do the audit first)

Guideline retrieval is gated on `status:active` in `guideline_retrieval_filter`,
which every user-facing path now uses: QA answers, daily tips, and the tip
fallback. `qa_service` no longer builds its own query bodies, and
`tests/test_guideline_retrieval_gate.py` fails if a call site reintroduces one.

**This gate is only safe after an activation pass.** If most rules are `draft`,
enabling it empties guideline grounding silently — answers keep coming, just
without guideline evidence.

```bash
# 1. What does the corpus actually contain?
curl -s "$FS/api/v1/guidelines/corpus/audit" | jq '{
  total, retrievable, retrievable_share, status, review_status, warning }'

# 2. What would activation change, per guide?
curl -s "$FS/api/v1/guidelines/corpus/activation-plan" | jq

# 3. Activate one guide — dry run first (the default).
curl -X POST "$FS/api/v1/guidelines/corpus/activate/urn:guide:...?dry_run=true"
curl -X POST "$FS/api/v1/guidelines/corpus/activate/urn:guide:...?dry_run=false"
```

`require_verified=true` (the default) activates only rules an editor has
verified. Clearing it activates every non-deleted rule under the guide and
should follow a deliberate review of that guide.

After deploying, ask a QA question and confirm no non-active guideline id
appears in `retrieved_sources` or the citations, and that daily tips still
generate.

---

## Phase 5 — UI (`wisefood-ui`)

- `app/utils/guidelineFacets.ts` — chip building, labels, age-range formatting
  (months in, "1–4 yr" out, matching how guides state ages), and `fq` clause
  construction for facet filters.
- Facet chips on `GuidelineCard` and in the guide-detail rules pane. A facet
  that is currently machine-written is marked with a dot; hovering explains it.
- `GuidelineFacetFilters` — multi-select chip filters. A section renders nothing
  when its facet has no buckets, so pre-enrichment the UI is unchanged.
- **Citation deep links now land properly.** Previously `?guideline=<id>`
  resolved only if the rule happened to be on the loaded 8-item page. It now
  computes the right page, clears filters that would hide the rule, scrolls it
  into view and flashes it.
- `GuidelineCitationPeek` — clicking a guideline citation in an answer opens an
  in-place preview with the rule text and its facets; modifier-click keeps the
  old open-in-new-tab behaviour, and "Open in guide" goes to the highlighted
  rule.
- Strings are in all four locales (`en`, `el`, `sl`, `hu`) under `guidelines.*`.

---

## Phase 6 — Embeddings and hybrid retrieval (last)

Guidelines are embedded from their rule text plus the guide title and facet
labels — the short, context-free sentence alone embeds poorly, which is the same
reason the facets were derived.

Vectors are queued on create, import, enrich, and any patch that changes the
text or facets. The existing corpus needs one backfill:

```bash
curl -X POST "$DATA_API/api/v1/guidelines/embeddings/backfill?dry_run=true"
curl -X POST "$DATA_API/api/v1/guidelines/embeddings/backfill"
# resumable: only_missing=true (default) skips rules that already have a vector
```

Only once that completes, enable hybrid retrieval:

```
QA_GUIDELINE_RETRIEVAL_MODE=hybrid   # default: bm25
QA_GUIDELINE_KNN_BOOST=1.0
```

The gate applies to both legs. If the embedding call fails at query time,
retrieval degrades to keyword rather than failing the answer.

---

## Ordering

```
Phase 1 (catalog)  ──► deploy
                        ├─► Phase 2 (extraction)      ─┐
                        ├─► Phase 3 (enrichment)      ─┤
                        └─► Phase 4 code               │
                                                       ▼
                        Phase 4 audit + activation ──► enable the gate
                                                       │
                        Phase 5 (UI) ──────────────────┤
                        Phase 6 (embeddings → hybrid) ─┘
```

Phase 5's deep-link highlight fix is independent of everything and can ship at
any point; its facet chips and filters simply render nothing until Phase 3 has
populated data.

## Concurrent extractions

Extraction is queued, not synchronous. The model:

| Mechanism | Effect |
|---|---|
| Redis list `guidelines:queue`, `BLPOP` per worker | jobs spread across replicas |
| `SET NX EX 7200` per artifact | the same PDF is never extracted twice at once |
| Job id compared against the registered status | a superseded job is discarded, not run |
| Lock TTL refreshed on every page callback | a long PDF keeps its claim |
| One worker **thread** per replica | effective concurrency equals replica count |

There is no global concurrency cap beyond replica count. That is deliberate:
one thread per process bounds each replica to a single extraction, so scaling
out is the knob, and a long PDF cannot starve the others on the same box of
anything but queue position.

**Recovering an interrupted run.** Jobs are popped destructively, so a worker
that dies mid-extraction leaves a status of `running` with no queue entry and no
live lock. That combination is now detected: `is_orphaned` treats "running with
no lock held" as a dead job, `GET /extract/{artifact}` reports it as `stalled`
rather than spinning on `running` forever, and re-queueing clears it. Previously
the artifact could never be extracted again — the status blocked re-queueing and
the lock outlived the process by up to two hours, so it needed manual Redis
surgery.

`force: true` on the extract request re-queues regardless of state. The
per-artifact lock still applies, so the worst case is a superseded job that the
stale check discards.

Registering a job is a single conditional write, so two requests arriving
together for the same artifact cannot both decide they are the one enqueueing.

`tests/test_guideline_jobs.py` covers all of this against a fake Redis,
including simulating a worker death by expiring the lock.

## Routing: which endpoints go where

The UI has two backends and they are not interchangeable:

- `wisefoodApi` (`/dc/api`) reaches **wisefood-data-api directly** — articles,
  guidelines, artifacts, and the admin `/system/*` endpoints.
- `wisefoodRestApi` (`/rest/api/v1`) reaches **the wisefood-api gateway**, which
  is where everything FoodScholar lives. The gateway routes explicitly; there is
  no catch-all, so a new FoodScholar endpoint is unreachable from the browser
  until it is proxied.

Ported through the gateway (`wisefood-api/src/routers/foodscholar.py`):

| Route | Auth | Notes |
|---|---|---|
| `POST /foodscholar/guidelines/extract/{artifact}` | any | **Now accepts a body.** It previously posted `json={}`, so `guide_id` could never reach extraction and every rule lost its guide context. |
| `POST /foodscholar/guidelines/import/{artifact}` | any | Gained `import_facets`; `existing_scan_limit` is now optional (omit to scan all). |
| `POST /foodscholar/guidelines/enrichment/preview` | admin, expert | Curators need to see proposals before trusting them. |
| `POST /foodscholar/guidelines/enrichment/enqueue` | admin | Queues corpus-wide model work. |
| `GET /foodscholar/guidelines/enrichment/status` | admin, expert | |
| `GET /foodscholar/guidelines/enrichment/worker` | admin, expert | |
| `GET /foodscholar/guidelines/corpus/audit` | admin, expert | |
| `GET /foodscholar/guidelines/corpus/activation-plan` | admin, expert | |
| `POST /foodscholar/guidelines/corpus/activate/{guide_urn}` | admin | Activation is what puts a rule in front of users. |

When adding a FoodScholar endpoint, three things must change together: the
FoodScholar route, a method on `wisefood-api/src/backend/foodscholar.py`, and a
proxy route with the right auth. If the request has a body, the gateway needs a
matching schema in `wisefood-api/src/schemas.py` — a field missing there is
dropped silently rather than rejected, which is how the `guide_id` gap survived.

## Behaviour under load

Things that matter when this runs over a real corpus rather than a test guide.

**Backfills are resumable and never double-queue.** The scan deliberately does
not filter on `embedded_at`, even though that is what it is looking for. The
embedding worker sets that field asynchronously, so filtering on it makes the
result set shift under the paging cursor: pages get skipped, or — if the cursor
is reset to compensate — the same unprocessed documents are re-queued until the
cap is hit, flooding the queue while most of the corpus is never reached. The
scan walks a stable set sorted by a unique key, skips already-embedded documents
in Python, and guards with a seen set. `tests/test_embedding_backfill_paging.py`
drives this against a fake index that behaves the way Elasticsearch does.

**Bulk writes do not wait for refreshes.** `update_entity` defaults to
`refresh="wait_for"` because most callers re-read what they just wrote, but the
embedding worker passes `refresh=False`. `wait_for` blocks until the index's
next refresh cycle — a second by default — which turns a few thousand background
writes into an hour of waiting for nothing.

**Guide edits do not fan out into per-document writes.** Resyncing a guide's
region to its rules is two scripted `update_by_query` passes, not a read-modify-
write per rule. The second pass only touches `applicable_regions` where it still
mirrors the guide, so an editor's or the agent's deliberate value survives.

**Corpus scans use `search_after`.** `from`/`size` makes Elasticsearch re-sort
and discard everything before the cursor on every page and stops at
`max_result_window`. The enricher sorts on `(sequence_no, id)` — the tiebreaker
matters, because a non-unique sort silently skips or repeats rows.

**The embedding vector is never returned.** Every guideline read path sets
`_source.excludes`; once the corpus is embedded, 384 floats per hit would
otherwise dominate each response and be parsed only to be discarded.

**Model calls back off.** Every extraction call — triage, extraction, and the
document profile — retries transient failures with full-jitter exponential
backoff, honouring a server `Retry-After` when one is offered. Jitter matters:
without it, workers that failed together retry together and keep colliding,
which is the pileup the backoff exists to break. A whole-job failure is then
retried up to `GUIDELINE_EXTRACTION_MAX_ATTEMPTS`, so a rate limit that outlasts
the per-call backoff no longer leaves a run sitting failed for an operator to
notice. Unrecognised errors are treated as permanent — retrying a malformed
request only burns quota.

**Enrichment runs rules concurrently.** Rules are independent and each is one
mostly-waiting model call, so a bounded pool (`GUIDELINE_ENRICHMENT_CONCURRENCY`,
default 8) processes them in parallel while batches are still written in order
from the main thread. The bound is deliberate: the provider's rate limit is
shared with extraction and every other replica.

**Failures are contained.** A failed enrichment batch costs that batch, not the
guide — the run is resumable by `enrichment_version`. A per-rule agent failure
costs that rule. Enrichment holds preview proposals only for dry runs, capped at
50; a real 2700-rule run accumulates nothing.

**Admin introspection is constant-cost.** Index state is three cluster calls and
embedding coverage is one aggregation, regardless of how many indices exist —
previously about 49 sequential round trips per page load.

**Import is one request per 500 rules, not one per rule.** Three N+1 patterns
used to compound here:

- Reading existing rules sliced the SDK collection proxy, which returns *lazy*
  entities that fetch themselves on first attribute access — so reading
  `sequence_no` off 500 existing rules cost 500 HTTP GETs. It now uses a paged
  search requesting only the three fields dedupe needs.
- Rules were created individually, one POST each, and every POST re-resolved the
  guide, its artifacts, and the next sequence number server-side. The importer
  now uses the bulk endpoint that already existed.
- Inside the bulk endpoint, the guide's artifacts were re-fetched per item and
  each queued embedding re-read the parent guide for its title. Both are now
  resolved once per batch, and documents go in through the ES bulk API instead
  of one index call each.

Importing 300 rules into a guide with 500 existing went from roughly 800 HTTP
round trips and 2,000 Elasticsearch operations to 2 and a handful.
`tests/test_guideline_jobs.py` asserts the request counts directly, so a
regression to per-rule creation fails rather than merely getting slower.

## Console surfaces

| Page | What it does |
|---|---|
| `/console/assets` | Four panels: Guides, Articles, Recipes, **Textbooks** |
| `/console/assets/guides/enrichment` | Corpus-wide facet coverage, per-guide progress, backfill trigger |
| `/console/assets/guides/[urn]` | Extraction context the run assembled, plus per-guide enrichment |
| `/console/assets/guides/review/[urn]` | Rule-by-rule editor, including every enrichment facet |
| `/console/assets/textbooks` | Textbook library and creation |
| `/console/assets/textbooks/[urn]` | Metadata, editorial state, passage ingestion |
| `/console/system` | Index state and embedding coverage (admin only) |

**Textbook passages come from an external chunker.** The console does not
create them one at a time: it ingests a chunker's JSON output against one
artifact and replaces that artifact's passage set atomically, which is what
makes re-chunking a document repeatable instead of duplicative. The chunker
name and run id are stored on every passage.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GUIDELINE_ENRICHMENT_VERSION` | `1` | Bump to re-enrich the whole corpus |
| `ENABLE_GUIDELINE_ENRICHMENT_WORKER` | `true` | Runs queued enrichment jobs |
| `GUIDELINE_ENRICHMENT_QUEUE_KEY` | `guideline_enrichment:queue` | Redis queue |
| `QA_GUIDELINE_RETRIEVAL_MODE` | `bm25` | `hybrid` adds the vector leg |
| `QA_GUIDELINE_KNN_BOOST` | `1.0` | Vector weight against BM25 |
| `GUIDELINE_ENRICHMENT_CONCURRENCY` | `8` | Rules enriched at once per guide |
| `GUIDELINE_EXTRACTION_MAX_ATTEMPTS` | `3` | Attempts before an extraction is recorded failed |
