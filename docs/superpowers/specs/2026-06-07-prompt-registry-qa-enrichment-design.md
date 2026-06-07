# Prompt Registry via Langfuse — QA + Enrichment (with fallbacks)

**Date:** 2026-06-07
**Status:** Approved for planning
**Scope:** Migrate the QA and Enrichment LLM prompts to Langfuse Prompt Management,
fronted by a centralized in-code registry that provides verbatim fallbacks.

## Goal

Make Langfuse the editable source of truth for the QA and enrichment prompts,
without coupling inference availability to Langfuse uptime. When Langfuse is
disabled, unreachable, or missing a prompt, the application uses an in-code
fallback and behaves exactly as it does today.

This builds on the opt-in observability already added in
`src/backend/langfuse.py` (gated on `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`).

## Prompts in scope

| Registry name              | Source today                                         | Type | Variables |
|----------------------------|------------------------------------------------------|------|-----------|
| `enrichment-annotation`    | `enrichment_agent.ANNOTATION_PROMPT`                 | text | `title`, `authors`, `abstract` |
| `enrichment-keywords-system` | `KEYWORD_EXTRACTION_PROMPT["system_prompt"]`        | text | — |
| `enrichment-keywords-user` | `KEYWORD_EXTRACTION_PROMPT["user_prompt"]`           | text | `abstract` |
| `qa-answer-rag-system`     | `qa_agent.py` RAG answer system f-string             | text | (static JSON skeleton) |
| `qa-answer-rag-user`       | `qa_agent.py` RAG answer human f-string              | text | `question`, `context` |
| `qa-answer-norag-system`   | `qa_agent.py` no-RAG answer system f-string          | text | (static JSON skeleton) |
| `qa-answer-norag-user`     | `qa_agent.py` no-RAG answer human f-string           | text | `question` |
| `qa-clarifier-system`      | `qa_clarifier.py` combined clarifier/safety system   | text | — |
| `qa-starter-questions`     | `qa_service.py:1581` starter-question generation     | text | `count` |
| `qa-tips-from-guidelines`  | `qa_service.py:1912` tips from guideline rules       | text | `candidate_count`, `guideline_context` |
| `qa-tips-from-articles`    | `qa_service.py:2321` tips from article abstracts     | text | `candidate_count`, `article_context` |
| `qa-tip-rewrite`           | `qa_service.py:2495` single-item evidence rewrite    | text | `text`, `style`, `article_context` |

**Note:** The qa_agent answer system prompts also interpolate
`expertise_level`, `complexity`, `language`, `answer_context` (not only the
human prompt). All become Langfuse variables. The f-strings use `{{{{ }}}}`
(quadruple braces) because they are f-strings nested in `ChatPromptTemplate`;
these become literal `{ }` in the stored prompt.

Out of scope for this pass: synthesis_agent, sessions chat prompt, guideline_extractor.
(They can follow the same pattern later.)

## Architecture

### New module: `src/backend/prompts.py` (the centralized registry)

This is the single place prompts live in code. Per the requirement to
"aggregate prompts per module so we have a centralized place in the code",
each prompt's canonical text (the fallback) is defined here, grouped by
subsystem, alongside the fetch logic.

```python
class _Prompt:
    def __init__(self, name, fallback, label="production", cache_ttl_seconds=60):
        ...
    def get(self):
        """Return the resolved prompt object (Langfuse-managed or fallback wrapper)."""
    def compile(self, **vars) -> str:
        """Resolve + substitute variables. Always returns a string."""
    def langchain(self, **precompiled):
        """Return get_langchain_prompt() form for ChatPromptTemplate use."""
```

`get()` behavior:
- If `langfuse_enabled()` is False → return a lightweight local wrapper around
  the fallback text (no network, no import cost beyond the guard).
- If enabled → `get_langfuse_client().get_prompt(name, fallback=<text>,
  label="production", cache_ttl_seconds=60)`. The SDK caches client-side
  (stale-while-revalidate) and uses `fallback` only when the local cache is
  empty AND the API is unreachable.

### Shared Langfuse client ("connection pool" analog)

The Langfuse Python SDK v3 is a **process-wide singleton** ("singleton per
public key") and the docs explicitly discourage per-request instantiation:
*"instantiating a handler or client per request is discouraged—create each
client/handler once and reuse it to avoid memory leaks."* A ChatGroq-style
multi-instance pool is therefore an anti-pattern here. The correct analog of a
connection pool is **one configured-once, cached, shared client** — a single
httpx connection and a single shared prompt cache reused across all requests
and threads.

Add to `src/backend/langfuse.py`:
```python
@lru_cache(maxsize=1)
def get_langfuse_client():
    """Process-wide Langfuse client (shared connection + prompt cache).
    Returns None when disabled."""
    if not langfuse_enabled():
        return None
    from langfuse import Langfuse
    return Langfuse()          # reads keys + base_url from env
```
- The existing `get_callback_handler()` is updated to rely on this client being
  configured (handler reads config from the singleton).
- Configured implicitly via env on first use; lazy (no startup pre-warm —
  first request pays a one-time ~40ms fetch per prompt, deemed acceptable).
- `flush_langfuse()` flushes this same singleton on shutdown.

The fallback text constants are grouped:
```python
# --- Enrichment ---
_ENRICHMENT_ANNOTATION_FALLBACK = """..."""   # moved verbatim from enrichment_agent
...
# --- QA ---
_QA_ANSWER_RAG_SYSTEM_FALLBACK = """..."""
...

ENRICHMENT_ANNOTATION   = _Prompt("enrichment-annotation", _ENRICHMENT_ANNOTATION_FALLBACK)
QA_ANSWER_RAG_SYSTEM    = _Prompt("qa-answer-rag-system", _QA_ANSWER_RAG_SYSTEM_FALLBACK)
...
```

### Variable syntax conversion (the careful part)

Current QA prompts are Python f-strings (`f"""...{question}..."""`) which also
contain literal JSON skeletons escaped as `{{ }}`. Conversion rules:

1. **Runtime variables** (`{question}`, `{abstract}`, `{title}`, `{authors}`,
   `{context}`) → Langfuse mustache `{{question}}`, supplied at `compile()` time.
2. **Escaped literal braces** in f-strings (`{{` / `}}` that render JSON skeletons)
   → plain `{` / `}` literals in the stored prompt (Langfuse only treats `{{name}}`
   as a variable; single braces are literal).
3. For LangChain call sites, use `prompt.langchain()` (wraps
   `get_langchain_prompt()`), which converts `{{var}}` → `{var}` for
   `ChatPromptTemplate`.

**Conversion is verified per prompt** (see Testing) because a recent commit
("Problematic f-string template fix") shows this is a real failure surface.

### Seeding: `scripts/seed_langfuse_prompts.py`

Imports the `_Prompt` registry objects from `backend/prompts.py` and calls
`langfuse.create_prompt(name=..., type="text", prompt=<fallback>, labels=["production"])`
for each. Idempotent: re-running only creates a new version when the text changed.
The in-code fallback constants are the single source of truth for the initial
seed; after seeding, Langfuse is the editable source and the constants remain
the safety net.

Run manually once per environment (and re-run after intentional fallback edits):
```
LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_BASE_URL=... \
  python scripts/seed_langfuse_prompts.py
```

## Call-site changes

- `src/agents/enrichment_agent.py`: `ANNOTATION_PROMPT` / `KEYWORD_EXTRACTION_PROMPT`
  move to `prompts.py`; call sites reference `prompts.ENRICHMENT_*`. Keep the
  `ChatPromptTemplate.from_messages` / `PromptTemplate` shapes via `.langchain()`.
- `src/agents/qa_agent.py`: replace the two answer f-strings (RAG + no-RAG,
  system + human) with registry `.langchain()` calls.
- `src/agents/qa_clarifier.py`: replace the concatenated system string with
  `prompts.QA_CLARIFIER_SYSTEM.compile()`.
- `src/services/qa_service.py`: replace starter-question and tips f-strings with
  registry `.compile(...)`.

## Error handling / resilience

- All resolution failures fall back to in-code text; failures are logged at
  WARNING, never raised into the request path.
- Langfuse disabled (no keys) → pure no-op, identical behavior to today.
- Network/API failure with empty cache → SDK `fallback` used.
- Prompt missing in Langfuse → SDK `fallback` used.

## Testing

1. **Round-trip equivalence** (TDD, per prompt): assert
   `registry.compile(**sample_vars)` byte-equals the original inline/f-string
   output for representative inputs. This guards the brace-conversion.
2. **Disabled-mode no-op**: with no Langfuse keys, every accessor returns the
   fallback; agents/services behave identically (unit-level, no network).
3. **Compile/import smoke**: `py_compile` all touched files; import
   `backend.prompts` and resolve every registered prompt.
4. **Seed script dry behavior**: importable and constructs the create_prompt
   payloads without requiring a live Langfuse (guarded).

## Non-goals

- No change to which models are used or to inference logic.
- No migration of synthesis/sessions/guideline prompts in this pass.
- No automatic seeding on app startup (seeding is an explicit script).

## Naming convention

Kebab-case, subsystem-prefixed: `enrichment-*`, `qa-*`. No environment in the
name (environments are handled by Langfuse labels: `production` vs others).
