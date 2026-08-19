"""
Normalizing raw model output before anything tries to read it.

Every agent here asks for a JSON object and then parses ``response.content``.
Two provider-shaped surprises break that, and neither is the model being
"wrong":

* ``content`` is not always a string. Providers that return content blocks give
  a list of ``{"type": "text", "text": ...}`` dicts, and ``str()`` on that
  yields a Python repr that no JSON parser will accept.
* Reasoning models leak their deliberation into ``content`` when the provider
  is not told to hide it — as ``<think>...</think>``, or as the residue of
  OpenAI's harmony channel format (``analysis ... assistantfinal ...``) once
  the special tokens have been stripped.

``backend.model_profiles`` prevents the leak for families we know about; this
is the second line of defence for the ones we do not, and the reason a new
model can be pointed at the app without an agent-by-agent audit.
"""

import re
from typing import Any

# Reasoning wrappers emitted in-band by various families.
_THINK_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning|analysis)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# An unclosed opener means the completion was cut off mid-reasoning: there is no
# payload after it, but leaving the fragment in produces a garbage "answer".
_UNCLOSED_THINK_RE = re.compile(
    r"<(think|thinking|reasoning|analysis)\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)
# Harmony residue: the final channel marker survives token stripping even when
# the delimiters do not. Everything before the last marker is reasoning.
_HARMONY_FINAL_RE = re.compile(r"(?:<\|start\|>)?assistant\s*final\s*", re.IGNORECASE)
_HARMONY_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def normalize_model_text(content: Any) -> str:
    """
    Coerce a provider's ``content`` into the plain text the model meant to send.

    Flattens content blocks, strips reasoning that leaked in-band, and removes
    channel markers. Returns "" for anything with no text in it.
    """
    text = _flatten(content)
    if not text:
        return ""

    # Tokens first: the marker is only contiguous once <|channel|> and friends
    # are gone (`<|start|>assistant<|channel|>final<|message|>` -> `assistantfinal`).
    text = _HARMONY_TOKEN_RE.sub("", text)
    if "assistant" in text.lower():
        parts = _HARMONY_FINAL_RE.split(text)
        if len(parts) > 1:
            text = parts[-1]

    text = _THINK_BLOCK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)

    return text.strip()


def _flatten(content: Any) -> str:
    """Reduce a str / content-block list / anything else to a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, (list, tuple)):
        return "".join(_flatten(block) for block in content)
    return str(content)
