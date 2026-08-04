"""Reusable guideline extraction service and local artifact workspace helpers."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from backend.prompts import (
    GUIDELINE_EXTRACTION,
    GUIDELINE_GUIDE_PROFILE,
    GUIDELINE_TRIAGE,
)
from services.model_backoff import call_with_backoff
from config import config
from utils import is_valid_uuid

if TYPE_CHECKING:
    import fitz
    from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_DPI = 144
TRIAGE_TEXT_LIMIT = 3500
DEFAULT_PDF_WORKSPACE = "/tmp/foodscholar/guideline_artifacts"
DEFAULT_ARTIFACT_FILENAME = "source.pdf"

# Result-bundle schema version. v1 stored `guidelines` as [{page, text}];
# v2 stores per-rule facet hints and provenance. Results already persisted in
# Postgres are never migrated, so the import path branches on this value.
EXTRACTION_SCHEMA_VERSION = 2

# How much of the previous page's raw text to carry forward. Enough to recover a
# table header or a sentence broken across the page boundary, short enough not
# to dominate the current page in the prompt.
PREVIOUS_PAGE_TAIL_CHARS = 600

# Sentinel the extraction schema uses for "age not stated". A strict JSON schema
# cannot express a nullable integer alongside `required`, so the model emits -1.
AGE_NOT_STATED = -1

# How many leading pages are read to profile the document before the page loop.
# Cover, imprint, foreword, and the head of the contents/introduction is where a
# guide states its own scope; five pages reaches that in practice.
DEFAULT_PROFILE_PAGE_COUNT = 5

LIFE_STAGE_VALUES = [
    "pregnancy",
    "lactation",
    "infancy",
    "early_childhood",
    "school_age",
    "adolescence",
    "adulthood",
    "older_adulthood",
]
SETTING_VALUES = [
    "school",
    "home",
    "clinical",
    "community",
    "workplace",
    "retail",
    "general",
]
GUIDELINE_TYPE_VALUES = [
    "food_based",
    "nutrient_based",
    "behavioral",
    "activity",
    "other",
]
ACTION_TYPE_VALUES = [
    "eat",
    "drink",
    "use",
    "do",
    "avoid",
    "prepare",
    "limit",
    "choose",
    "increase",
    "reduce",
]


class GuidelineExtractionError(RuntimeError):
    """Base error for guideline extraction failures."""


class GuidelineDependencyError(GuidelineExtractionError):
    """Raised when an optional extraction dependency is unavailable."""


class GuidelineConfigurationError(GuidelineExtractionError):
    """Raised when required runtime configuration is missing."""


class GuidelineArtifactNotFoundError(GuidelineExtractionError):
    """Raised when an artifact PDF is not available in the local workspace."""


@dataclass
class GuideContext:
    """
    What the guide is, drawn from its catalog record and from the document itself.

    Every page prompt carries this. Without it a rule sentence extracted from
    page 14 has no way to know it is addressed to caregivers of 1-4 year olds
    in Ireland, and the resulting record is unattributable.

    Catalog metadata is frequently thin or absent, and a guide's actual scope
    ("for children aged 1 to 4 years") usually lives on its cover or in its
    foreword rather than in a metadata field. So the context is assembled from
    two sources: the catalog record, and a profile pass that reads the opening
    pages of the PDF. Catalog values win where they exist — they are editorially
    curated — and the document fills every gap. ``derived_fields`` records which
    values came from the document so a reviewer can tell them apart.
    """

    guide_urn: str | None = None
    title: str | None = None
    region: str | None = None
    audience: str | None = None
    target_audiences: List[str] = field(default_factory=list)
    language: str | None = None
    publication_year: int | None = None
    issuing_authority: str | None = None
    # Read from the document, with no catalog counterpart.
    population_note: str | None = None
    age_min_months: int | None = None
    age_max_months: int | None = None
    scope_note: str | None = None
    evidence: List[str] = field(default_factory=list)
    derived_fields: List[str] = field(default_factory=list)

    @classmethod
    def from_guide(cls, guide: Dict[str, Any] | None) -> "GuideContext":
        """Build a context from a catalog guide record (or an empty one)."""
        if not guide:
            return cls()
        return cls(
            guide_urn=guide.get("urn"),
            title=guide.get("title") or guide.get("short_title"),
            region=guide.get("region"),
            audience=guide.get("audience"),
            target_audiences=list(guide.get("target_audiences") or []),
            language=guide.get("language"),
            publication_year=guide.get("publication_year"),
            issuing_authority=guide.get("issuing_authority"),
        )

    def is_empty(self) -> bool:
        return not any(
            [
                self.title,
                self.region,
                self.audience,
                self.target_audiences,
                self.language,
                self.publication_year,
                self.issuing_authority,
                self.population_note,
                self.age_min_months is not None,
                self.scope_note,
            ]
        )

    def needs_document_profile(self) -> bool:
        """
        Whether reading the document is worth a profile call.

        The bar is not "no metadata at all" but "nothing that establishes who
        the guidance is for": a record can carry a title and a region and still
        leave every rule population-less.
        """
        if not self.population_note and self.age_min_months is None:
            return True
        return not (self.title and self.region)

    def merge_document_profile(self, profile: "GuideDocumentProfile") -> "GuideContext":
        """
        Fill gaps from a document-derived profile, never overwriting the catalog.

        Returns a new context; the receiver is left untouched so the caller can
        report what the catalog alone provided.
        """
        merged = GuideContext(
            guide_urn=self.guide_urn,
            title=self.title,
            region=self.region,
            audience=self.audience,
            target_audiences=list(self.target_audiences),
            language=self.language,
            publication_year=self.publication_year,
            issuing_authority=self.issuing_authority,
            population_note=self.population_note,
            age_min_months=self.age_min_months,
            age_max_months=self.age_max_months,
            scope_note=self.scope_note,
            evidence=list(self.evidence),
            derived_fields=list(self.derived_fields),
        )

        for name in (
            "title",
            "region",
            "audience",
            "language",
            "publication_year",
            "issuing_authority",
            "population_note",
            "age_min_months",
            "age_max_months",
            "scope_note",
        ):
            if getattr(merged, name) not in (None, "", []):
                continue
            value = getattr(profile, name, None)
            if value in (None, "", []):
                continue
            setattr(merged, name, value)
            if name not in merged.derived_fields:
                merged.derived_fields.append(name)

        for quote in profile.evidence or []:
            if quote and quote not in merged.evidence:
                merged.evidence.append(quote)

        return merged

    def as_prompt_block(self) -> str:
        """Render the context for injection into a page prompt."""
        if self.is_empty():
            return (
                "Guide context: not available. Extract only what the page itself "
                "states; do not assume a population."
            )

        lines = ["Guide context (the document this page belongs to):"]
        if self.title:
            lines.append(f"- Title: {self.title}")
        if self.issuing_authority:
            lines.append(f"- Issuing authority: {self.issuing_authority}")
        if self.region:
            lines.append(f"- Region: {self.region}")
        if self.publication_year:
            lines.append(f"- Published: {self.publication_year}")
        if self.language:
            lines.append(f"- Language: {self.language}")
        audiences = ", ".join(
            [value for value in [self.audience, *self.target_audiences] if value]
        )
        if audiences:
            lines.append(f"- Intended audience: {audiences}")
        if self.population_note:
            lines.append(f"- Guidance is for: {self.population_note}")
        if self.age_min_months is not None or self.age_max_months is not None:
            low = self.age_min_months if self.age_min_months is not None else "unbounded"
            high = self.age_max_months if self.age_max_months is not None else "unbounded"
            lines.append(f"- Age range covered (months): {low} to {high}")
        if self.scope_note:
            lines.append(f"- Scope: {self.scope_note}")
        lines.append(
            "Treat this as the authority on who the guidance is for when the page "
            "itself does not say."
        )
        return "\n".join(lines)


@dataclass
class GuideDocumentProfile:
    """What the opening pages of the PDF say the document is."""

    title: str | None = None
    issuing_authority: str | None = None
    region: str | None = None
    language: str | None = None
    publication_year: int | None = None
    audience: str | None = None
    population_note: str | None = None
    age_min_months: int | None = None
    age_max_months: int | None = None
    scope_note: str | None = None
    evidence: List[str] = field(default_factory=list)
    pages_read: List[int] = field(default_factory=list)


@dataclass
class PageDecision:
    page: int
    decision: str
    reason: str
    continues_from_previous: bool = False


@dataclass
class ExtractedRule:
    """One rule sentence plus everything the page supported about it."""

    text: str
    section_label: str | None = None
    source_snippet: str | None = None
    target_population_hint: str | None = None
    age_min_months: int | None = None
    age_max_months: int | None = None
    life_stage: List[str] = field(default_factory=list)
    setting: List[str] = field(default_factory=list)
    health_conditions: List[str] = field(default_factory=list)
    nutrients: List[str] = field(default_factory=list)
    guideline_type: str | None = None
    topic: List[str] = field(default_factory=list)
    action_type_hint: str | None = None
    confidence: float | None = None


@dataclass
class PageExtraction:
    page: int
    page_summary: str
    guidelines: List[ExtractedRule]


@dataclass
class OutputBundle:
    source_pdf: str
    model: str
    dpi: int
    total_pages: int
    processed_pages: List[Dict[str, Any]]
    skipped_pages: List[Dict[str, Any]]
    guidelines: List[Dict[str, Any]]
    unique_guidelines: List[str]
    schema_version: int = EXTRACTION_SCHEMA_VERSION
    guide_context: Dict[str, Any] | None = None
    document_profile: Dict[str, Any] | None = None
    continuation_pages: List[int] = field(default_factory=list)


@dataclass
class ArtifactWorkspaceInfo:
    artifact_uuid: str
    workspace_root: str
    artifact_dir: str
    pdf_filename: str
    pdf_path: str
    pdf_exists: bool


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_default_model() -> str:
    """Return the configured model for guideline extraction."""
    return str(config.settings.get("GUIDELINE_EXTRACTION_MODEL", DEFAULT_MODEL))


def get_default_dpi() -> int:
    """Return the configured render DPI for guideline extraction."""
    return int(config.settings.get("GUIDELINE_RENDER_DPI", DEFAULT_DPI))


def get_artifact_pdf_filename() -> str:
    """Return the configured filename used inside each artifact workspace."""
    return str(config.settings.get("GUIDELINE_ARTIFACT_FILENAME", DEFAULT_ARTIFACT_FILENAME))


def get_pdf_workspace_root() -> Path:
    """Return the configured root folder for locally staged artifact PDFs."""
    workspace_root = Path(
        str(config.settings.get("GUIDELINE_PDF_WORKSPACE", DEFAULT_PDF_WORKSPACE))
    )
    if not workspace_root.is_absolute():
        workspace_root = PROJECT_ROOT / workspace_root
    return workspace_root.resolve()


def ensure_api_key() -> None:
    """Ensure the OpenAI API key is available before extraction starts."""
    if not os.getenv("OPENAI_API_KEY"):
        raise GuidelineConfigurationError("OPENAI_API_KEY is not set.")


def clean_text(text: str) -> str:
    """Normalize raw PDF text before prompting."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_guideline(text: str) -> str:
    """Normalize a guideline string for storage and deduplication."""
    text = text.strip()
    text = re.sub(r"^\s*[-*•]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def guideline_dedupe_key(text: str) -> str:
    """Normalized key used to detect near-duplicate rule sentences."""
    norm = normalize_guideline(text).lower()
    norm = re.sub(r"[^\w\s]", "", norm)
    return re.sub(r"\s+", " ", norm).strip()


def dedupe_guidelines(items: List[str]) -> List[str]:
    """Collapse exact and near-exact duplicate guideline strings."""
    seen = set()
    output = []

    for item in items:
        norm = guideline_dedupe_key(item)
        if norm and norm not in seen:
            seen.add(norm)
            output.append(normalize_guideline(item))

    return output


def image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """Encode page image bytes as a data URL for the Responses API."""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def schema_for_triage() -> Dict[str, Any]:
    """Return the strict JSON schema for page triage."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["relevant", "skip"],
            },
            "reason": {"type": "string"},
            "continues_from_previous": {
                "type": "boolean",
                "description": (
                    "True when this page continues a table, list, or sentence "
                    "begun on the previous page."
                ),
            },
        },
        "required": ["decision", "reason", "continues_from_previous"],
    }


def schema_for_extraction() -> Dict[str, Any]:
    """
    Return the strict JSON schema for page-level guideline extraction.

    Every property is `required` because OpenAI strict mode demands it; the
    model signals "not stated" with an empty string, an empty array, or the
    ``AGE_NOT_STATED`` sentinel, and :func:`_rule_from_payload` maps those back
    to ``None``.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page_summary": {"type": "string"},
            "guidelines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "section_label": {"type": "string"},
                        "source_snippet": {"type": "string"},
                        "target_population_hint": {"type": "string"},
                        "age_min_months": {"type": "integer"},
                        "age_max_months": {"type": "integer"},
                        "life_stage": {
                            "type": "array",
                            "items": {"type": "string", "enum": LIFE_STAGE_VALUES},
                        },
                        "setting": {
                            "type": "array",
                            "items": {"type": "string", "enum": SETTING_VALUES},
                        },
                        "health_conditions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "nutrients": {"type": "array", "items": {"type": "string"}},
                        "guideline_type": {
                            "type": "string",
                            "enum": [*GUIDELINE_TYPE_VALUES, ""],
                        },
                        "topic": {"type": "array", "items": {"type": "string"}},
                        "action_type_hint": {
                            "type": "string",
                            "enum": [*ACTION_TYPE_VALUES, ""],
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "text",
                        "section_label",
                        "source_snippet",
                        "target_population_hint",
                        "age_min_months",
                        "age_max_months",
                        "life_stage",
                        "setting",
                        "health_conditions",
                        "nutrients",
                        "guideline_type",
                        "topic",
                        "action_type_hint",
                        "confidence",
                    ],
                },
            },
        },
        "required": ["page_summary", "guidelines"],
    }


def schema_for_guide_profile() -> Dict[str, Any]:
    """Return the strict JSON schema for the document-level profile pass."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "issuing_authority": {"type": "string"},
            "region": {"type": "string"},
            "language": {"type": "string"},
            "publication_year": {"type": "integer"},
            "audience": {"type": "string"},
            "population_note": {"type": "string"},
            "age_min_months": {"type": "integer"},
            "age_max_months": {"type": "integer"},
            "scope_note": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "title",
            "issuing_authority",
            "region",
            "language",
            "publication_year",
            "audience",
            "population_note",
            "age_min_months",
            "age_max_months",
            "scope_note",
            "evidence",
        ],
    }


def _clean_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_str_list(value: Any, *, allowed: List[str] | None = None) -> List[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip().lower()
        if not item or (allowed is not None and item not in allowed):
            continue
        if item not in cleaned:
            cleaned.append(item)
    return cleaned


def _clean_age(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return None if value < 0 else value


def _rule_from_payload(payload: Any) -> ExtractedRule | None:
    """Map one raw model item to an :class:`ExtractedRule`, dropping empties."""
    if isinstance(payload, str):
        # Tolerate the v1 shape so a re-run against an older prompt version
        # still yields usable rules.
        text = normalize_guideline(payload)
        return ExtractedRule(text=text) if text else None

    if not isinstance(payload, dict):
        return None

    text = normalize_guideline(str(payload.get("text") or ""))
    if not text:
        return None

    age_min = _clean_age(payload.get("age_min_months"))
    age_max = _clean_age(payload.get("age_max_months"))
    if age_min is not None and age_max is not None and age_max < age_min:
        age_min, age_max = None, None

    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = None
    else:
        confidence = max(0.0, min(1.0, float(confidence)))

    guideline_type = _clean_optional_str(payload.get("guideline_type"))
    if guideline_type not in GUIDELINE_TYPE_VALUES:
        guideline_type = None
    action_type_hint = _clean_optional_str(payload.get("action_type_hint"))
    if action_type_hint not in ACTION_TYPE_VALUES:
        action_type_hint = None

    return ExtractedRule(
        text=text,
        section_label=_clean_optional_str(payload.get("section_label")),
        source_snippet=_clean_optional_str(payload.get("source_snippet")),
        target_population_hint=_clean_optional_str(
            payload.get("target_population_hint")
        ),
        age_min_months=age_min,
        age_max_months=age_max,
        life_stage=_clean_str_list(payload.get("life_stage"), allowed=LIFE_STAGE_VALUES),
        setting=_clean_str_list(payload.get("setting"), allowed=SETTING_VALUES),
        health_conditions=_clean_str_list(payload.get("health_conditions")),
        nutrients=_clean_str_list(payload.get("nutrients")),
        guideline_type=guideline_type,
        topic=_clean_str_list(payload.get("topic")),
        action_type_hint=action_type_hint,
        confidence=confidence,
    )


def safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse the model output and surface invalid JSON cleanly."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GuidelineExtractionError(
            f"Model did not return valid JSON. Raw output:\n{text}"
        ) from exc


def _load_pymupdf():
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GuidelineDependencyError(
            "PyMuPDF is required for guideline extraction. Install `PyMuPDF`."
        ) from exc
    return fitz


def _load_openai_client():
    # Use the Langfuse drop-in wrapper when observability is enabled so that
    # guideline-extraction OpenAI calls are traced. Falls back to the plain
    # SDK when Langfuse is disabled or unavailable.
    try:
        from backend.langfuse import langfuse_enabled

        if langfuse_enabled():
            from langfuse.openai import OpenAI  # type: ignore[import-not-found]

            return OpenAI
    except Exception:  # pragma: no cover - fall back to the plain SDK
        pass

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GuidelineDependencyError(
            "The OpenAI Python SDK is required for guideline extraction. Install `openai`."
        ) from exc
    return OpenAI


def render_page_to_png(page: Any, dpi: int = DEFAULT_DPI) -> bytes:
    """Render a PDF page to PNG bytes."""
    fitz = _load_pymupdf()
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes("png")


def extract_page_text(page: Any, limit: int = TRIAGE_TEXT_LIMIT) -> str:
    """Extract a bounded text preview from a PDF page."""
    text = page.get_text("text") or ""
    text = clean_text(text)
    if len(text) > limit:
        text = text[:limit]
    return text


def open_pdf(pdf_path: str):
    """Open a PDF using PyMuPDF."""
    fitz = _load_pymupdf()
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise GuidelineArtifactNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_file.is_file():
        raise GuidelineArtifactNotFoundError(f"PDF path is not a file: {pdf_path}")

    try:
        return fitz.open(pdf_path)
    except Exception as exc:
        raise GuidelineExtractionError(f"Failed to open PDF: {pdf_path}") from exc


@dataclass
class PreviousPage:
    """The slice of the previous page carried into the current one."""

    page: int
    summary: str = ""
    text_tail: str = ""
    image_bytes: bytes | None = None

    def as_prompt_block(self) -> str:
        lines = [f"Previous page ({self.page}) context:"]
        if self.summary:
            lines.append(f"- Summary: {self.summary}")
        if self.text_tail:
            lines.append(f"- Trailing text: {self.text_tail}")
        if len(lines) == 1:
            return ""
        lines.append(
            "Use this only to resolve structures continuing onto the current "
            "page (table headers, broken lists or sentences). Do not re-extract "
            "guidance that belongs to the previous page."
        )
        return "\n".join(lines)


def _user_content(
    page_number: int,
    page_text: str,
    image_url: str,
    previous: PreviousPage | None,
    *,
    include_previous_image: bool = False,
) -> List[Dict[str, Any]]:
    """Build the user turn: previous-page context, this page's text, images."""
    sections = [f"Page number: {page_number}"]

    previous_block = previous.as_prompt_block() if previous else ""
    if previous_block:
        sections.append(previous_block)

    sections.append(
        "Extracted text preview:\n"
        f"{page_text if page_text else '[NO EXTRACTABLE TEXT]'}"
    )

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": "\n\n".join(sections)}
    ]

    if include_previous_image and previous is not None and previous.image_bytes:
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"The previous page ({previous.page}) is shown first for "
                    "continuation context; the current page follows."
                ),
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": image_bytes_to_data_url(previous.image_bytes),
            }
        )

    content.append({"type": "input_image", "image_url": image_url})
    return content


def profile_guide_document(
    client: "OpenAI",
    model: str,
    pages: List[tuple[int, str, bytes]],
) -> GuideDocumentProfile:
    """
    Read the opening pages of a guide to establish what the document is.

    Catalog metadata is often missing the one thing that matters most for
    attribution — who the guidance is for — while the document states it plainly
    on its cover or in its foreword. This pass recovers it so every subsequent
    page extraction has a population to inherit.
    """
    if not pages:
        return GuideDocumentProfile()

    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Opening pages of the document, in order. Text preview per page "
                "followed by the page images.\n\n"
                + "\n\n".join(
                    f"--- Page {number} ---\n{text or '[NO EXTRACTABLE TEXT]'}"
                    for number, text, _ in pages
                )
            ),
        }
    ]
    for _, _, image_bytes in pages:
        content.append(
            {
                "type": "input_image",
                "image_url": image_bytes_to_data_url(image_bytes),
            }
        )

    response = call_with_backoff(
        lambda: client.responses.create(
            model=model,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": GUIDELINE_GUIDE_PROFILE.compile()}
                    ],
                },
                {"role": "user", "content": content},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "guide_profile",
                    "strict": True,
                    "schema": schema_for_guide_profile(),
                }
            },
        ),
        description="guide document profile",
    )

    data = safe_json_loads(response.output_text)

    region = _clean_optional_str(data.get("region"))
    if region:
        region = region.upper()
        if len(region) != 2 or not region.isalpha():
            region = None

    language = _clean_optional_str(data.get("language"))
    if language:
        language = language.lower()[:2] or None

    year = data.get("publication_year")
    if not isinstance(year, int) or isinstance(year, bool) or year < 1800:
        year = None

    age_min = _clean_age(data.get("age_min_months"))
    age_max = _clean_age(data.get("age_max_months"))
    if age_min is not None and age_max is not None and age_max < age_min:
        age_min, age_max = None, None

    return GuideDocumentProfile(
        title=_clean_optional_str(data.get("title")),
        issuing_authority=_clean_optional_str(data.get("issuing_authority")),
        region=region,
        language=language,
        publication_year=year,
        audience=_clean_optional_str(data.get("audience")),
        population_note=_clean_optional_str(data.get("population_note")),
        age_min_months=age_min,
        age_max_months=age_max,
        scope_note=_clean_optional_str(data.get("scope_note")),
        evidence=[
            quote.strip()
            for quote in (data.get("evidence") or [])
            if isinstance(quote, str) and quote.strip()
        ],
        pages_read=[number for number, _, _ in pages],
    )


def classify_page(
    client: "OpenAI",
    model: str,
    page_number: int,
    page_text: str,
    page_image_bytes: bytes,
    guide_context: GuideContext | None = None,
    previous: PreviousPage | None = None,
) -> PageDecision:
    """Decide whether a page should be used for guideline extraction."""
    image_url = image_bytes_to_data_url(page_image_bytes)
    context = guide_context or GuideContext()

    prompt = GUIDELINE_TRIAGE.compile(guide_context=context.as_prompt_block())

    response = call_with_backoff(
        lambda: client.responses.create(
            model=model,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": prompt}],
                },
                {
                    "role": "user",
                    "content": _user_content(
                        page_number, page_text, image_url, previous
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "page_triage",
                    "strict": True,
                    "schema": schema_for_triage(),
                }
            },
        ),
        description=f"page {page_number} triage",
    )

    data = safe_json_loads(response.output_text)
    return PageDecision(
        page=page_number,
        decision=data["decision"],
        reason=data["reason"],
        continues_from_previous=bool(data.get("continues_from_previous")),
    )


def extract_guidelines_from_page(
    client: "OpenAI",
    model: str,
    page_number: int,
    page_text: str,
    page_image_bytes: bytes,
    guide_context: GuideContext | None = None,
    previous: PreviousPage | None = None,
    continues_from_previous: bool = False,
) -> PageExtraction:
    """
    Extract guideline sentences from a relevant page.

    When ``continues_from_previous`` is set, the previous page image is sent
    alongside the current one so a table header or lead-in that lives on the
    other side of the page break is still visible to the model.
    """
    image_url = image_bytes_to_data_url(page_image_bytes)
    context = guide_context or GuideContext()

    prompt = GUIDELINE_EXTRACTION.compile(guide_context=context.as_prompt_block())

    response = call_with_backoff(
        lambda: client.responses.create(
            model=model,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": prompt}],
                },
                {
                    "role": "user",
                    "content": _user_content(
                        page_number,
                        page_text,
                        image_url,
                        previous,
                        include_previous_image=continues_from_previous,
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "guideline_extraction",
                    "strict": True,
                    "schema": schema_for_extraction(),
                }
            },
        ),
        description=f"page {page_number} extraction",
    )

    data = safe_json_loads(response.output_text)
    rules = []
    for item in data.get("guidelines", []):
        rule = _rule_from_payload(item)
        if rule is not None:
            rules.append(rule)

    return PageExtraction(
        page=page_number,
        page_summary=data.get("page_summary", "").strip(),
        guidelines=rules,
    )


class GuidelineExtractorService:
    """Service that extracts guideline statements from locally staged PDFs."""

    def __init__(self, workspace_root: str | Path | None = None):
        resolved_root = Path(workspace_root).resolve() if workspace_root else get_pdf_workspace_root()
        self.workspace_root = resolved_root

    def ensure_workspace_root(self) -> Path:
        """Create the shared artifact workspace if it does not yet exist."""
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        return self.workspace_root

    def get_artifact_workspace(self, artifact_uuid: str) -> ArtifactWorkspaceInfo:
        """Return the local staging paths for an artifact UUID."""
        if not is_valid_uuid(artifact_uuid):
            raise GuidelineExtractionError(
                "artifact_uuid must be a canonical UUID string."
            )

        workspace_root = self.ensure_workspace_root()
        artifact_dir = workspace_root / artifact_uuid
        artifact_dir.mkdir(parents=True, exist_ok=True)

        pdf_filename = get_artifact_pdf_filename()
        pdf_path = artifact_dir / pdf_filename

        return ArtifactWorkspaceInfo(
            artifact_uuid=artifact_uuid,
            workspace_root=str(workspace_root),
            artifact_dir=str(artifact_dir),
            pdf_filename=pdf_filename,
            pdf_path=str(pdf_path),
            pdf_exists=pdf_path.exists() and pdf_path.is_file(),
        )

    def process_artifact(
        self,
        artifact_uuid: str,
        model: str | None = None,
        dpi: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        guide_context: GuideContext | None = None,
        profile_document: bool = True,
        profile_page_count: int = DEFAULT_PROFILE_PAGE_COUNT,
    ) -> OutputBundle:
        """Extract guidelines from the local PDF resolved for an artifact UUID."""
        workspace = self.get_artifact_workspace(artifact_uuid)
        if not workspace.pdf_exists:
            raise GuidelineArtifactNotFoundError(
                f"No staged PDF found for artifact {artifact_uuid}."
            )

        return self.process_pdf(
            pdf_path=workspace.pdf_path,
            model=model or get_default_model(),
            dpi=dpi or get_default_dpi(),
            progress_callback=progress_callback,
            guide_context=guide_context,
            profile_document=profile_document,
            profile_page_count=profile_page_count,
        )

    def process_pdf(
        self,
        pdf_path: str,
        model: str | None = None,
        dpi: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        guide_context: GuideContext | None = None,
        profile_document: bool = True,
        profile_page_count: int = DEFAULT_PROFILE_PAGE_COUNT,
    ) -> OutputBundle:
        """
        Run the page-by-page guideline extraction pipeline for a PDF.

        Pages are still processed one at a time, but each call carries context
        the earlier pipeline lacked:

        - **Document identity.** Assembled from the catalog record and, when that
          leaves the population unestablished, from a profile pass over the
          opening pages of the PDF itself. A rule then knows it is addressed to
          carers of 1-4 year olds even when its sentence says nothing of the kind.
        - **The previous page.** A rolling summary and text tail, plus the page
          image itself when triage flags a continuation, so a table split across
          a page break keeps its header.
        """
        ensure_api_key()
        OpenAI = _load_openai_client()
        client = OpenAI()
        doc = open_pdf(pdf_path)

        requested_model = model or get_default_model()
        requested_dpi = dpi or get_default_dpi()
        context = guide_context or GuideContext()
        document_profile: GuideDocumentProfile | None = None
        total_pages = len(doc)
        processed_pages: List[Dict[str, Any]] = []
        skipped_pages: List[Dict[str, Any]] = []
        extracted_guidelines: List[Dict[str, Any]] = []
        all_guideline_texts: List[str] = []
        continuation_pages: List[int] = []
        previous: PreviousPage | None = None

        try:
            # Establish what the document is before mining it. The catalog record
            # is authoritative where it has values, but it is regularly missing
            # the population the rules need to inherit, and the document states
            # that on its cover.
            if profile_document and context.needs_document_profile():
                profile_pages = []
                for idx in range(min(profile_page_count, total_pages)):
                    page = doc[idx]
                    profile_pages.append(
                        (
                            idx + 1,
                            extract_page_text(page),
                            render_page_to_png(page, dpi=requested_dpi),
                        )
                    )
                try:
                    document_profile = profile_guide_document(
                        client=client,
                        model=requested_model,
                        pages=profile_pages,
                    )
                    # Several full-page PNGs; releasing them here keeps the
                    # peak out of the page loop that follows.
                    profile_pages.clear()
                    context = context.merge_document_profile(document_profile)
                    logger.info(
                        "Profiled guide document %s from pages %s; fields taken "
                        "from the document: %s",
                        pdf_path,
                        document_profile.pages_read,
                        context.derived_fields or "none",
                    )
                except GuidelineExtractionError:
                    # A failed profile must not cost us the extraction; we simply
                    # proceed with whatever the catalog provided.
                    logger.warning(
                        "Guide profiling failed for %s; continuing with catalog "
                        "metadata only.",
                        pdf_path,
                        exc_info=True,
                    )

            for idx, page in enumerate(doc, start=1):
                if progress_callback is not None:
                    progress_callback(idx, total_pages)

                page_text = extract_page_text(page)
                page_image = render_page_to_png(page, dpi=requested_dpi)

                decision = classify_page(
                    client=client,
                    model=requested_model,
                    page_number=idx,
                    page_text=page_text,
                    page_image_bytes=page_image,
                    guide_context=context,
                    previous=previous,
                )

                # The next page sees this one regardless of the decision: a
                # skipped page can still hold the header of a table that
                # continues, and its tail text is the cheapest way to recover it.
                next_previous = PreviousPage(
                    page=idx,
                    text_tail=page_text[-PREVIOUS_PAGE_TAIL_CHARS:] if page_text else "",
                    image_bytes=page_image,
                )

                if decision.decision == "skip":
                    skipped_pages.append(asdict(decision))
                    previous = next_previous
                    continue

                if decision.continues_from_previous:
                    continuation_pages.append(idx)

                extraction = extract_guidelines_from_page(
                    client=client,
                    model=requested_model,
                    page_number=idx,
                    page_text=page_text,
                    page_image_bytes=page_image,
                    guide_context=context,
                    previous=previous,
                    continues_from_previous=decision.continues_from_previous,
                )

                processed_pages.append(
                    {
                        "page": extraction.page,
                        "page_summary": extraction.page_summary,
                        "guideline_count": len(extraction.guidelines),
                        "continues_from_previous": decision.continues_from_previous,
                    }
                )

                for rule in extraction.guidelines:
                    row = asdict(rule)
                    row["page"] = idx
                    extracted_guidelines.append(row)
                    all_guideline_texts.append(rule.text)

                next_previous.summary = extraction.page_summary
                previous = next_previous
        finally:
            doc.close()

        unique_guidelines = dedupe_guidelines(all_guideline_texts)
        return OutputBundle(
            source_pdf=pdf_path,
            model=requested_model,
            dpi=requested_dpi,
            total_pages=total_pages,
            processed_pages=processed_pages,
            skipped_pages=skipped_pages,
            guidelines=extracted_guidelines,
            unique_guidelines=unique_guidelines,
            schema_version=EXTRACTION_SCHEMA_VERSION,
            guide_context=asdict(context) if not context.is_empty() else None,
            document_profile=asdict(document_profile) if document_profile else None,
            continuation_pages=continuation_pages,
        )


def write_json(path: str, data: OutputBundle) -> None:
    """Write extraction output as pretty JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asdict(data), handle, ensure_ascii=False, indent=2)


def write_markdown(path: str, data: OutputBundle) -> None:
    """Write a markdown report for extracted guidelines."""
    lines: List[str] = []
    lines.append(f"# Dietary guidelines extracted from `{data.source_pdf}`")
    lines.append("")
    lines.append(f"Model: `{data.model}`")
    lines.append("")
    lines.append("## Unique guidelines")
    lines.append("")

    if data.unique_guidelines:
        for item in data.unique_guidelines:
            lines.append(f"- {item}")
    else:
        lines.append("_No guidelines found._")

    lines.append("")
    lines.append("## Page-by-page extracted guidelines")
    lines.append("")

    grouped: Dict[int, List[str]] = {}
    for row in data.guidelines:
        grouped.setdefault(row["page"], []).append(row["text"])

    for page_num in sorted(grouped.keys()):
        lines.append(f"### Page {page_num}")
        lines.append("")
        for item in grouped[page_num]:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Skipped pages")
    lines.append("")
    if data.skipped_pages:
        for row in data.skipped_pages:
            lines.append(f"- Page {row['page']}: {row['reason']}")
    else:
        lines.append("_No pages were skipped._")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
