"""
Per-model capability profiles.

A model id is not a drop-in replacement for another one. Two differences bite
in practice:

* **Reasoning families spend the completion budget on hidden reasoning.** Left
  at the provider default, ``openai/gpt-oss-*`` returns that reasoning inside
  ``content``, so the JSON payload the agents parse is either preceded by
  paragraphs of deliberation or truncated before it closes. That is what a
  "wrongly rendered" answer looks like from the outside.
* **Reasoning knobs are provider- and family-specific.** ``reasoning_format``
  and ``reasoning_effort`` are accepted by the Groq reasoning models and
  rejected by the Llama ones, so a call site that hardcodes them stops working
  the moment the model behind it is swapped.

Rather than teach every call site which family it is talking to, the quirks are
declared once here and applied by the Groq pool (``backend.groq``). Adding a
model to the deployment means adding a row here — not editing agents.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    """What one model family needs, and what it will refuse."""

    family: str
    # A reasoning model: needs its reasoning kept out of `content` and needs a
    # completion budget large enough for reasoning *plus* the payload.
    reasoning: bool = False
    # Groq-specific: 'hidden' keeps reasoning out of the response entirely.
    reasoning_format: Optional[str] = None
    reasoning_effort: Optional[str] = None
    # Floor, not an override: a call site asking for more keeps its value.
    min_max_tokens: Optional[int] = None
    # Whether the family accepts the Groq reasoning_* params at all. Passing
    # them to a model that does not is a 400, so they are dropped instead.
    supports_reasoning_params: bool = False
    # OpenAI's reasoning models reject any temperature other than the default.
    supports_temperature: bool = True
    # False for families we have no row for; drives the one-time warning.
    known: bool = True


# Matched as a substring against the lowercased model id, first hit wins, so
# more specific prefixes must come before more general ones.
_FAMILIES: Tuple[Tuple[str, ModelProfile], ...] = (
    (
        "gpt-oss",
        ModelProfile(
            family="gpt-oss",
            reasoning=True,
            reasoning_format="hidden",
            reasoning_effort="low",
            # gpt-oss truncates a ~12-item JSON payload at the 1024 default
            # because reasoning is charged against the same budget.
            min_max_tokens=2048,
            supports_reasoning_params=True,
        ),
    ),
    (
        "qwen3",
        ModelProfile(
            family="qwen3",
            reasoning=True,
            reasoning_format="hidden",
            # No reasoning_effort default: the family accepts the parameter but
            # its effort levels are not the same scale as gpt-oss's, so the
            # provider default is the honest choice until it is measured.
            min_max_tokens=2048,
            supports_reasoning_params=True,
        ),
    ),
    (
        "deepseek-r1",
        ModelProfile(
            family="deepseek-r1",
            reasoning=True,
            reasoning_format="hidden",
            min_max_tokens=2048,
            supports_reasoning_params=True,
        ),
    ),
    ("kimi-k2", ModelProfile(family="kimi-k2")),
    ("llama", ModelProfile(family="llama")),
    ("mixtral", ModelProfile(family="mixtral")),
    ("gemma", ModelProfile(family="gemma")),
    # OpenAI reasoning models, used by the guideline extractor through the
    # OpenAI SDK. reasoning_format is a Groq concept and is not accepted here.
    ("gpt-5", ModelProfile(family="openai-reasoning", reasoning=True, supports_temperature=False)),
    ("o3", ModelProfile(family="openai-reasoning", reasoning=True, supports_temperature=False)),
    ("o4", ModelProfile(family="openai-reasoning", reasoning=True, supports_temperature=False)),
    ("gpt-4", ModelProfile(family="openai-chat")),
)

# Model ids the provider has shut down, with the replacement it named. A
# retired id still matches its family row, so nothing else here would notice:
# the request simply fails at the provider. Warning at client-construction time
# puts the shutdown date and the replacement in the logs of the process that is
# about to fail, which is where an operator will be looking.
RETIRED = {
    "llama-3.1-8b-instant": ("2026-08-16", "openai/gpt-oss-20b"),
    "llama-3.3-70b-versatile": (
        "2026-08-16",
        "openai/gpt-oss-120b or qwen/qwen3.6-27b",
    ),
}

_UNKNOWN = ModelProfile(
    family="unknown",
    # An unregistered id gets nothing injected, but a knob the caller asked for
    # explicitly is still forwarded: the request was deliberate, and stripping
    # it would silently change behaviour. Leaked reasoning is caught downstream
    # by backend.model_output.normalize_model_text.
    supports_reasoning_params=True,
    known=False,
)

_warned_unknown = set()
_warned_retired = set()


def profile_for(model: str) -> ModelProfile:
    """Return the capability profile for ``model``."""
    name = (model or "").lower()

    if name in RETIRED and name not in _warned_retired:
        _warned_retired.add(name)
        shutdown, replacement = RETIRED[name]
        logger.warning(
            "Model '%s' was shut down by the provider on %s; calls to it will "
            "fail. Recommended replacement: %s.",
            model, shutdown, replacement,
        )

    for marker, profile in _FAMILIES:
        if marker in name:
            return profile

    if model not in _warned_unknown:
        _warned_unknown.add(model)
        logger.warning(
            "Model '%s' has no capability profile; calling it with caller "
            "defaults only. Add a row to backend.model_profiles if it is a "
            "reasoning model.",
            model,
        )
    return _UNKNOWN


def apply_profile(model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconcile caller-supplied kwargs with what ``model`` actually supports.

    Injects the family's defaults where the caller expressed no preference,
    raises ``max_tokens`` to the family floor, and drops parameters the family
    would reject. Returns a new dict; ``kwargs`` is left alone.
    """
    profile = profile_for(model)
    resolved = dict(kwargs)

    if not profile.supports_reasoning_params:
        for key in ("reasoning_format", "reasoning_effort"):
            if key in resolved:
                logger.debug(
                    "Dropping %s for %s: unsupported by the %s family",
                    key, model, profile.family,
                )
                resolved.pop(key)
    elif profile.reasoning:
        # 'hidden' is the whole point: it keeps deliberation out of `content`,
        # where it would otherwise be parsed as (or rendered as) the answer.
        if profile.reasoning_format and "reasoning_format" not in resolved:
            resolved["reasoning_format"] = profile.reasoning_format
        if profile.reasoning_effort and "reasoning_effort" not in resolved:
            resolved["reasoning_effort"] = profile.reasoning_effort

    if profile.min_max_tokens:
        current = resolved.get("max_tokens")
        if not isinstance(current, int) or current < profile.min_max_tokens:
            resolved["max_tokens"] = profile.min_max_tokens

    if not profile.supports_temperature and "temperature" in resolved:
        logger.debug(
            "Dropping temperature for %s: the %s family accepts only its default",
            model, profile.family,
        )
        resolved.pop("temperature")

    return resolved
