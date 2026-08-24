"""
Recovering JSON objects from raw LLM output.

Shared by every agent that asks a model for JSON. Models deviate from strict
JSON in predictable ways — Markdown fences, a sentence of reasoning before the
payload, trailing commas, stray control characters, leaked reasoning blocks —
and every one of those is recoverable rather than a reason to discard the
response.

This is the only JSON entry point the agents should use. When each call site
had its own parser, the weakest one decided which models the app could run: a
family whose output the robust parser recovered still broke the paths that only
split on code fences.
"""

import json
import re
from typing import Any, Dict, List, Optional

from backend.model_output import normalize_model_text


def _prepare(content: Any) -> str:
    """Reduce raw model output to the text a JSON parser should see."""
    text = normalize_model_text(content)

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()


def _extract_balanced(text: str, opener: str, closer: str) -> Optional[str]:
    """Return the first balanced ``opener...closer`` block in ``text``."""
    start = text.find(opener)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block in ``text``, if any."""
    return _extract_balanced(text, "{", "}")


def close_unclosed_json(content: Any) -> Optional[str]:
    """Repair model output whose only defect is missing closing brackets.

    gpt-oss sometimes stops cleanly (finish_reason "stop") before emitting
    the last closing brace(s) of a large JSON object — everything except the
    closers is present, so appending them recovers the payload losslessly.

    Returns the repaired JSON string, or ``None`` when the text is not a
    simple unclosed tail: no object at all, mismatched brackets, or a
    top-level object that does close (the defect is then something this
    repair cannot fix, like an unescaped quote mid-document).
    """
    prepared = _prepare(content)
    start = prepared.find("{")
    if start < 0:
        return None
    fragment = prepared[start:]

    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack or ("}" if stack[-1] == "{" else "]") != ch:
                return None
            stack.pop()
            if not stack:
                return None

    if not stack:
        return None

    repaired = fragment.rstrip()
    if in_string:
        repaired += '"'
    # A dangling comma before a synthesized closer would be invalid JSON.
    repaired = re.sub(r",\s*$", "", repaired)
    repaired += "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return repaired


def extract_first_json_array(text: str) -> Optional[str]:
    """Return the first balanced ``[...]`` block in ``text``, if any."""
    return _extract_balanced(text, "[", "]")


def parse_json_object(content: Any) -> Dict[str, Any]:
    """
    Parse a JSON object out of raw model output.

    Raises:
        ValueError: if no JSON object can be recovered from ``content``.
    """
    return _parse(content, dict, extract_first_json_object)


def parse_json_array(content: Any) -> List[Any]:
    """
    Parse a JSON array out of raw model output.

    Raises:
        ValueError: if no JSON array can be recovered from ``content``.
    """
    return _parse(content, list, extract_first_json_array)


def _parse(content: Any, expected: type, extractor) -> Any:
    text = _prepare(content)
    if not text:
        raise ValueError("model returned empty content")

    candidates = [text]
    extracted = extractor(text)
    if extracted and extracted != text:
        candidates.append(extracted)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        for variant in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                parsed = json.loads(variant)
            except Exception as e:  # noqa: BLE001 - any decode failure is a retry
                last_error = e
                continue
            if isinstance(parsed, expected):
                return parsed
            last_error = ValueError(
                f"expected a JSON {expected.__name__}, got {type(parsed).__name__}"
            )

    raise ValueError(f"unparseable model output: {last_error}")


def clean_str(value: Any) -> Optional[str]:
    """Return a stripped string, or None for anything empty or non-textual."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    # Models trained to always fill a field say "Not stated" instead of leaving
    # it out; that is an absent value, not a value.
    if cleaned.lower() in {"not stated", "unknown", "n/a", "none", "null"}:
        return None
    return cleaned


def clean_str_list(
    value: Any,
    *,
    allowed: Optional[list] = None,
    lowercase: bool = True,
    limit: Optional[int] = None,
) -> list:
    """
    Normalize a list of strings, dropping empties and out-of-vocabulary values.

    ``allowed`` enforces a closed vocabulary: anything the model invents is
    discarded rather than written to the catalog.
    """
    if not isinstance(value, list):
        return []

    cleaned: list = []
    for item in value:
        text = clean_str(item)
        if text is None:
            continue
        if lowercase:
            text = text.lower().replace(" ", "_")
        if allowed is not None and text not in allowed:
            continue
        if text not in cleaned:
            cleaned.append(text)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


def clean_int(value: Any, *, minimum: Optional[int] = None) -> Optional[int]:
    """Return an int, treating negatives (the "not stated" sentinel) as absent."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < 0:
        return None
    if minimum is not None and value < minimum:
        return None
    return value


def clean_confidence(value: Any) -> Optional[float]:
    """Clamp a model-reported confidence into [0, 1], or None if unusable."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return max(0.0, min(1.0, float(value)))
