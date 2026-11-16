# QA ablation study

What each part of the agentic QA pipeline is worth, and whether the whole
beats simpler systems. Every configuration answers the same questions as the
full system, and the report compares them question by question.

Read `qa-benchmark.md` first: this study is built on the same runner, records
and scorer.

## How a configuration is applied

A request may carry `ablation`, a few pipeline settings to answer that one
request with:

```json
{"question": "...", "bypass_cache": true, "ablation": {"QA_PLANNER_ENABLED": false}}
```

- **Request-scoped.** The settings apply on top of the deployment's for that
  request only (a context-local overlay on `config.settings`). One API
  therefore answers every configuration, interleaved, from the same code,
  index and models.
- **Off by default.** Unless the API runs with `QA_ABLATION_ENABLED=true`, a
  request that carries `ablation` gets a 400. It is never answered silently
  with the defaults, which would compare the full system with itself.
  `GET /qa/runtime-info` reports `ablation.enabled` and every allowed setting
  with its default, and the runner refuses to start when they don't match.
- **Never cached for anyone else.** The overlay is part of the answer cache
  key.
- **Agentic pipeline only.** The switches act on the agentic pipeline.
  `QA_PIPELINE_MODE=legacy` selects the single-pass flow as a baseline, and
  the other switches don't reach it.

## What can be switched

| Area | Setting | Effect |
|---|---|---|
| Baseline | `QA_PIPELINE_MODE=legacy` | Single-pass flow: one retrieval, one answer |
| Baseline | `--retriever no_rag` (request field) | The same answer model with no sources |
| Planning | `QA_PLANNER_ENABLED=false` | The question as asked: one article search and one guideline search |
| Planning | `QA_MAX_SUBQUESTIONS` | Searches the planner may write |
| Planning | `QA_SEARCH_FILTERS_ENABLED=false` | Searches ignore the planner's filters (years, study design, population) |
| Agent rounds | `QA_MAX_REPAIR_ROUNDS` | 0 = no evidence check and no follow-up searches; 1 is the default |
| Agent rounds | `QA_GAP_RECHECK_ENABLED=false` | A first-round "corpus gap" ends the search instead of being rechecked with the evaluator's own suggested searches |
| Asker | `QA_CLARIFY_ENABLED=false` | Never ask for a detail |
| Asker | `QA_USER_CONTEXT_ENABLED=false` | Nothing known about the asker |
| Conversation | `QA_THREAD_MEMORY_ENABLED=false` | Follow-ups answered without earlier turns, summary or notes |
| Retrieval | `QA_RETRIEVAL_LEGS=lexical\|vector` | One search leg only |
| Retrieval | `QA_KNN_CANDIDATES=10` | The vector leg as it was before the kNN size fix |
| Retrieval | `QA_RRF_CANDIDATES`, `QA_RRF_K`, `QA_GUIDELINE_RETRIEVAL_MODE` | Hits per leg, fusion constant, guideline legs |
| Reranking | `QA_RANKING_PRIORS_ENABLED=false` | Fused relevance alone |
| Reranking | `QA_RANKING_PRIORS_OFF=recency,influence` | Only the named signals off. Signals: `tier`, `recency`, `influence`, `design`, `affinity` |
| Reranking | `QA_MIN_SCORE`, `QA_PER_DOC_CAP`, `QA_RECENCY_*`, `QA_INFLUENCE_WEIGHT`, `QA_EARNED_TIER_ENABLED` | Floor, per-guide cap, signal strengths |
| Evidence budget | `QA_ARTICLE_TOP_K`, `QA_GUIDELINE_TOP_K` | Articles and rules the answer reads |
| Models | `QA_DEFAULT_MODEL`, `QA_PLANNER_MODEL`, `QA_EVALUATOR_MODEL` | The model per role, limited to models the deployment is configured with |

Every switch is also an environment variable, with the default as deployed.
A new pipeline part should come with its switch, and with a line in
`experiments/ablation/study.json`, so its value is measured before it is kept.

## The study

`experiments/ablation/study.json` holds the full system, the baselines, the
ablations, and sweeps. A sweep is one setting tried over several values, for
example agent rounds 0, 1, 2 and 3. The study runs in two stages:

1. **Stage 1** (29 runs):
   - the full system three times, which gives the noise floor;
   - the baselines: closed book, single pass, and the production run
     recorded on 2026-11-04;
   - one-part-off ablations of the planner, the evidence check, the
     corpus-gap recheck, search filters, each search leg, reranking, the kNN fix, user context,
     clarification and thread memory.
2. **Stage 2** (34 more):
   - each reranking signal off on its own;
   - sweeps of agent rounds, sub-questions, candidates per leg, the per-guide
     cap and the article budget.

   Run stage 2 where stage 1 shows something worth decomposing.

```bash
# An API with ablation on, against the production catalog (see below)
QA_ABLATION_ENABLED=true CACHE_ENABLED=false ... python src/app.py

python scripts/run_ablation_study.py run    experiments/ablation/study.json --api http://127.0.0.1:18011 --stage 1
python scripts/run_ablation_study.py score  experiments/ablation/study.json      # MMLU; GROQ_API_KEY for unclear answers
python scripts/run_ablation_study.py report experiments/ablation/study.json      # benchmark-runs/qa-ablation-1/report.{md,json}
```

**Running.** Runs resume. Rerunning a command asks only what is missing, and
a run id can't be resumed under a different configuration. Configurations run
side by side (`--parallel`, default 3), so a slow hour at the provider falls
on all of them. `--dry-run` prints the runner commands.

**Where to run it.** Use a FoodScholar instance with `QA_ABLATION_ENABLED=true`
that reads the production catalog, either a separate deployment with no
ingress or `run_local.sh` against the catalog. Don't turn it on for the
public deployment.

## The report

The report has one table per dataset.

- **Compared to the full system.** Each configuration is compared with the
  full system on the questions both answered.
  - Δ: accuracy, or the judge score when the run has a `judgments.jsonl`.
  - A bootstrap 95% interval.
  - An exact McNemar test (a sign test for scores), Holm-corrected across the
    configurations of the dataset.
  - A plain verdict: "this part helps", "full system is better", "no
    detectable difference", and so on.
- **Noise floor.** This is how far apart three runs of the full system land.
  An effect no larger than the noise floor is reported as no difference.
- **Cost of each configuration:** latency at p50 and p95, searches, evidence
  checks, sources, the share of answers that fell back to `[general knowledge]`,
  and the clarification rate.
- **Sweep tables.** One table per sweep, showing where more rounds or a
  bigger budget stops paying for itself.
- **By question type.** A table of Δ per question type, because one change can
  help Safety guardrails and hurt Factual.

## What the numbers can and cannot say

**MMLU nutrition** (306 questions) is objective but measures what the answer
model already knows. Retrieval and reranking changes may barely move it. A
null result there is not a null result for grounding.

**The persona set** (120-question sample) is where the pipeline matters:
clarification, guidelines, citations, corpus-gap honesty. It needs a judge
that writes `judgments.jsonl` (`{"question_id", "score"}` per line); see the
judge plan on the board (foodscholar/llm-judge). Until then its report fills
the cost columns only.

**Power.** About 300 paired questions detect a difference of 4 to 5 points
when roughly 10% of answers disagree. About 120 detect only large effects.
Repeat the full system to know how large "large" is.
