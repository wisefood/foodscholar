# Integrating Langfuse into a WiseFood FastAPI LLM Service

A practical guide distilled from the FoodScholar integration. Covers tracing,
prompt management, prompt population, env vars, and the conventions that make
traces useful across the WiseFood platform.

Target: a Python **FastAPI** service that calls an LLM (typically **LangChain +
ChatGroq**, the WiseFood default). Applies to RecipeWrangler, FoodChat, and any
new LLM-powered service.

Reference implementation: `foodscholar/src/backend/langfuse.py`,
`src/backend/prompts.py`, `src/backend/groq.py`, `src/app.py`.

---

## 0. Design principles (why it's built this way)

1. **Strictly optional.** Langfuse must never be a hard dependency. If keys are
   absent or the package is missing, every helper degrades to a no-op and the
   app behaves exactly as before. Observability must not be able to take down
   a production service.
2. **Never block startup.** Prompt sync runs in a background thread. A slow or
   down Langfuse must not delay or fail pod boot.
3. **Never raise from a tracing path.** Every Langfuse call is wrapped in
   `try/except` and logged at WARNING. A tracing failure is not a request
   failure.
4. **Langfuse is the source of truth for prompt text; code holds a fallback.**
   UI edits always win; in-code text is a resilience net + one-time seed.
5. **No PII in trace metadata.** Only opaque IDs and feature tags.

---

## 1. Environment variables

Exactly three, read directly by the Langfuse SDK from the environment:

| Var | Required | Example | Notes |
|---|---|---|---|
| `LANGFUSE_PUBLIC_KEY` | yes (to enable) | `pk-lf-...` | Project public key |
| `LANGFUSE_SECRET_KEY` | yes (to enable) | `sk-lf-...` | Project secret key |
| `LANGFUSE_BASE_URL`   | optional | `http://langfuse-web:3000` | Self-hosted host; omit for cloud |

**Enablement rule:** tracing activates only when **both** keys are present
**and** `import langfuse` succeeds. One key alone = disabled.

### Kubernetes

Keys belong in a Secret, never in the Deployment YAML:

```yaml
env:
  - name: LANGFUSE_PUBLIC_KEY
    valueFrom:
      secretKeyRef: { name: langfuse-keys, key: public }
  - name: LANGFUSE_SECRET_KEY
    valueFrom:
      secretKeyRef: { name: langfuse-keys, key: secret }
  - name: LANGFUSE_BASE_URL
    value: "http://langfuse-web:3000"   # in-cluster service DNS
```

Use the **in-cluster service URL** for `LANGFUSE_BASE_URL` when Langfuse runs in
the same cluster — avoids egress and TLS overhead.

### Dependency

```
langfuse>=3.0
```
Pin the v3 line: the metadata key convention below (`langfuse_session_id` etc.)
is v3-specific.

---

## 2. The Langfuse module (`backend/langfuse.py`)

One module owns all Langfuse coupling. Nothing else imports the SDK directly.

### 2.1 Enablement guard

```python
def langfuse_enabled() -> bool:
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return False
    try:
        import langfuse  # noqa: F401
    except Exception as exc:
        logger.warning("Langfuse keys set but package not importable: %s", exc)
        return False
    return True
```

### 2.2 Singleton client + callback handler

The Langfuse SDK is a **singleton** — per-request instantiation leaks memory.
Cache both with `lru_cache(maxsize=1)`:

```python
@lru_cache(maxsize=1)
def get_callback_handler() -> Optional[Any]:
    """Shared LangChain CallbackHandler; None when disabled."""
    if not langfuse_enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception as exc:
        logger.warning("Failed to init CallbackHandler: %s", exc)
        return None

@lru_cache(maxsize=1)
def get_langfuse_client() -> Optional[Any]:
    """Process-wide client for prompt fetching + flushing."""
    if not langfuse_enabled():
        return None
    try:
        from langfuse import Langfuse
        return Langfuse()   # reads keys + base_url from env
    except Exception as exc:
        logger.warning("Failed to init Langfuse client: %s", exc)
        return None
```

The handler is **stateless** and reads credentials via the singleton client, so
one instance can safely be attached to every pooled LLM client and shared across
threads.

### 2.3 Flush on shutdown

Traces are buffered; without a flush the last requests are lost on pod
termination.

```python
def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("Failed to flush traces: %s", exc)
```

---

## 3. Tracing

### 3.1 Attach the handler once, at the LLM client

Do **not** pass callbacks at every call site. Attach in the LLM connection pool
so every call is traced automatically (`backend/groq.py`):

```python
from backend.langfuse import get_callback_handler

langfuse_handler = get_callback_handler()
if langfuse_handler is not None:
    callbacks = list(kwargs.pop("callbacks", []) or [])
    callbacks.append(langfuse_handler)
    kwargs["callbacks"] = callbacks
```

**Important:** callbacks are *excluded from the pool cache key* — they're
identity-bearing and would fragment the pool.

### 3.2 Enrich traces with `build_trace_config`

A bare handler gives you traces named after LangChain internals. To get
filterable, groupable traces, pass a `config` per invocation. This is the
canonical WiseFood helper:

```python
def build_trace_config(
    *,
    run_name: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {"run_name": run_name}
    metadata: Dict[str, Any] = {}
    if session_id is not None:
        metadata["langfuse_session_id"] = str(session_id)
    if user_id is not None:
        metadata["langfuse_user_id"] = str(user_id)
    if tags:
        metadata["langfuse_tags"] = list(tags)
    if extra_metadata:
        metadata.update(extra_metadata)
    if metadata:
        config["metadata"] = metadata
    return config
```

**Langfuse v3 metadata keys** (these exact names matter):

| Key | Purpose | Langfuse UI surface |
|---|---|---|
| `run_name` (top-level, not metadata) | descriptive trace name | trace list |
| `langfuse_session_id` | groups multi-turn conversations | **Sessions** view |
| `langfuse_user_id` | user/cost attribution | **Users** view, filters |
| `langfuse_tags` | per-feature analytics | tag filters |

Values must be **strings** (hence `str()` coercion). `None` values are omitted.

### 3.3 Call site

```python
from backend.langfuse import build_trace_config

config = build_trace_config(
    run_name="qa_answer",
    session_id=session_id,
    user_id=member_id,
    tags=["qa", "rag"],
)
response = self.llm.invoke(prompt.format_messages(**variables), config=config)
```

Works whether or not Langfuse is enabled: `run_name` is a standard LangChain
config key and the `langfuse_*` metadata keys are ignored with no handler
attached.

### 3.4 🔒 PII policy (non-negotiable)

**Only opaque identifiers and feature tags go in trace metadata.** Never put
personal data — allergies, dietary profile, member details, health conditions —
in `extra_metadata`.

The LLM message payload is the *only* place user context legitimately appears,
because it's the generation input by necessity. Metadata is not.

### 3.5 Naming conventions

- `run_name`: `snake_case`, verb-or-noun describing the step —
  `qa_answer`, `qa_clarify`, `memory_extract`, `synthesis`, `enrichment`.
  One stable name per logical LLM step so traces aggregate.
- `tags`: coarse feature buckets — `["qa"]`, `["memory"]`, `["tips"]`. Keep the
  vocabulary small or filtering becomes useless.
- `session_id`: the app's conversation/session ID, so multi-turn flows group in
  the Sessions view.

---

## 4. Prompt management

### 4.1 The registry pattern

Prompts live in one module (`backend/prompts.py`) as `_Prompt` objects, each
with Langfuse-managed text **and an in-code fallback**:

```python
class _Prompt:
    def __init__(self, name, fallback, label="production", cache_ttl_seconds=60):
        ...

    def _managed(self):
        client = get_langfuse_client()
        if client is None:
            return None
        try:
            return client.get_prompt(
                self.name,
                fallback=self.fallback,
                label=self.label,
                cache_ttl_seconds=self.cache_ttl_seconds,
            )
        except Exception as exc:
            logger.warning("get_prompt(%s) failed: %s", self.name, exc)
            return None

    def compile(self, **variables):
        managed = self._managed()
        if managed is not None:
            try:
                return managed.compile(**variables)
            except Exception as exc:
                logger.warning("compile(%s) failed; using fallback: %s", self.name, exc)
        return _compile_fallback(self.fallback, variables)
```

Key points:
- `fallback=` is passed to `get_prompt` **and** applied locally — belt and braces.
- `cache_ttl_seconds=60`: the SDK caches prompts in-process, so a UI edit
  propagates within ~1 min without a redeploy, at no per-request latency cost.
- `label="production"`: deploy prompt changes by moving the label in the UI.

### 4.2 Variable syntax

Langfuse uses **mustache** `{{var}}`. LangChain uses `{var}`. If you feed a
Langfuse prompt into a LangChain `ChatPromptTemplate`, convert:

```python
_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")
# Langfuse {{var}} -> LangChain {var}; escape stray braces first.
```

FoodScholar exposes both `compile(**vars)` (plain string) and a
`get_langchain_prompt()` path for template use. Getting this wrong produces
`KeyError` on unrelated braces (e.g. JSON examples in the prompt) — escape them.

### 4.3 Populating Langfuse (`sync_prompts`)

Auto-seed the registry into Langfuse, **creating only what's missing**:

```python
def sync_prompts(*, client=None, registry=None) -> Dict[str, int]:
    """Seed registry prompts into Langfuse, creating ONLY those missing."""
    ...
    for prompt in registry:
        existing = None
        try:
            existing = client.get_prompt(prompt.name, label=prompt.label, cache_ttl_seconds=0)
        except Exception:
            existing = None            # treat as missing
        if existing is not None:
            counts["skipped"] += 1     # UI is source of truth — never overwrite
            continue
        client.create_prompt(name=prompt.name, type="text", ...)
        counts["created"] += 1
    return counts
```

**Why never overwrite:** live text may be a deliberate UI edit. Overwriting on
every pod start would silently revert prompt-engineering work. This also makes
startup **idempotent** (`create_prompt` alone is not — it creates a new version
each call).

Use `cache_ttl_seconds=0` for the existence check so it isn't answered from a
stale cache.

### 4.4 Wire it into startup — non-blocking

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    def _seed_prompts() -> None:
        try:
            from backend.prompts import sync_prompts
            result = sync_prompts()
            if any(result.values()):
                logger.info("Langfuse prompt sync: %s", result)
        except Exception as exc:
            logger.warning("Langfuse prompt sync failed: %s", exc)

    threading.Thread(target=_seed_prompts, name="langfuse-prompt-sync",
                     daemon=True).start()

    yield

    from backend.langfuse import flush_langfuse
    flush_langfuse()
```

A **daemon thread**, so boot never waits on Langfuse. Safe on every pod start.

### 4.5 Manual seeding script

Ship a CLI for seeding outside app boot (`scripts/seed_langfuse_prompts.py`):

```python
from backend.prompts import sync_prompts, ALL_PROMPTS
result = sync_prompts()
print(f"Done ({len(ALL_PROMPTS)} registry prompts): {result}")
```

Useful for a fresh Langfuse project or CI.

---

## 5. Integration checklist for a new service

- [ ] Add `langfuse>=3.0` to requirements.
- [ ] Copy `backend/langfuse.py` (enablement guard, cached client + handler,
      `build_trace_config`, `flush_langfuse`).
- [ ] Attach `get_callback_handler()` in the LLM client/pool; exclude callbacks
      from any pool cache key.
- [ ] Add `build_trace_config(...)` at each LLM call site with a stable
      `run_name`, plus `session_id`/`user_id`/`tags` where available.
- [ ] Create a prompt registry with in-code fallbacks; route all prompt text
      through it.
- [ ] Call `sync_prompts()` in a daemon thread from the lifespan startup.
- [ ] Call `flush_langfuse()` on shutdown.
- [ ] Add the 3 env vars; keys via Secret, `LANGFUSE_BASE_URL` to the in-cluster
      service.
- [ ] Verify the disabled path: unset the keys, run the test suite — everything
      must pass with Langfuse off.

---

## 6. Testing

Langfuse must be invisible to tests. Two rules:

1. **Default-off in tests.** With no keys set, `langfuse_enabled()` is False and
   every helper no-ops — the suite runs without touching the network.
2. **Test `sync_prompts` with a mock client**, asserting the never-overwrite
   contract:

```python
def test_sync_skips_existing_prompts(self):
    client = MagicMock()
    client.get_prompt.return_value = object()      # exists
    result = sync_prompts(client=client, registry=[p])
    client.create_prompt.assert_not_called()       # UI wins
    self.assertEqual(result["skipped"], 1)
```

Also assert `build_trace_config` omits `None`s, coerces to `str`, and carries no
PII.

---

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| No traces at all | Only one key set; or `langfuse` not installed (check for the WARNING log) |
| Traces appear, no Sessions grouping | Missing/misspelled `langfuse_session_id` (v3 key name) |
| Traces named `ChatGroq`/`RunnableSequence` | `run_name` not passed in `config` |
| Last traces missing on redeploy | `flush_langfuse()` not called on shutdown |
| Prompt edits in UI not taking effect | Waiting on `cache_ttl_seconds` (≤60s); or label ≠ `production` |
| Prompt reverts after deploy | `sync_prompts` overwriting — it must skip existing |
| `KeyError` on prompt compile | `{{var}}` vs `{var}` mismatch, or unescaped literal braces |
| Slow startup | Prompt sync running inline instead of in a daemon thread |

---

## 8. WiseFood platform conventions (summary)

- **Three env vars**, keys via Secret, `LANGFUSE_BASE_URL` = in-cluster service.
- **v3 metadata keys**: `langfuse_session_id`, `langfuse_user_id`,
  `langfuse_tags`; `run_name` at config top level.
- **`build_trace_config`** is the only way to build trace config — don't
  hand-roll metadata dicts.
- **No PII in metadata**, ever.
- **Langfuse owns prompt text; code owns a fallback.** UI edits always win.
- **Optional and non-blocking**: no keys = no-op; sync in a daemon thread;
  never raise from a tracing path.
