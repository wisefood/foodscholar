"""Reusable guideline extraction service and local artifact workspace helpers."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

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


class GuidelineExtractionError(RuntimeError):
    """Base error for guideline extraction failures."""


class GuidelineDependencyError(GuidelineExtractionError):
    """Raised when an optional extraction dependency is unavailable."""


class GuidelineConfigurationError(GuidelineExtractionError):
    """Raised when required runtime configuration is missing."""


class GuidelineArtifactNotFoundError(GuidelineExtractionError):
    """Raised when an artifact PDF is not available in the local workspace."""


@dataclass
class PageDecision:
    page: int
    decision: str
    reason: str


@dataclass
class PageExtraction:
    page: int
    page_summary: str
    guidelines: List[str]


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


@dataclass
class ArtifactWorkspaceInfo:
    artifact_uuid: str
    workspace_root: str
    artifact_dir: str
    pdf_filename: str
    pdf_path: str
    pdf_exists: bool


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


def dedupe_guidelines(items: List[str]) -> List[str]:
    """Collapse exact and near-exact duplicate guideline strings."""
    seen = set()
    output = []

    for item in items:
        norm = normalize_guideline(item).lower()
        norm = re.sub(r"[^\w\s]", "", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
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
        },
        "required": ["decision", "reason"],
    }


def schema_for_extraction() -> Dict[str, Any]:
    """Return the strict JSON schema for page-level guideline extraction."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page_summary": {"type": "string"},
            "guidelines": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["page_summary", "guidelines"],
    }


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


def classify_page(
    client: "OpenAI",
    model: str,
    page_number: int,
    page_text: str,
    page_image_bytes: bytes,
) -> PageDecision:
    """Decide whether a page should be used for guideline extraction."""
    image_url = image_bytes_to_data_url(page_image_bytes)

    prompt = (
        "You are classifying a single PDF page for a dietary-guideline extraction pipeline.\n\n"
        "Mark the page as 'skip' if it is any of the following:\n"
        "- table of contents\n"
        "- outline / section listing\n"
        "- cover page / title page\n"
        "- divider / decorative page\n"
        "- blank or nearly blank\n"
        "- index\n"
        "- glossary\n"
        "- references / bibliography\n"
        "- acknowledgements\n"
        "- appendix with no dietary guidance\n"
        "- page with only navigation elements or page furniture\n"
        "- page not containing dietary guidance, feeding advice, portion advice, nutrient advice, or meal guidance\n\n"
        "Mark the page as 'relevant' only if it contains explicit or clearly implied dietary guidance, "
        "feeding advice, portion guidance, nutrient recommendations, serving guidance, or meal recommendations.\n\n"
        "Be conservative: if the page is mostly navigational or structural, return 'skip'."
    )

    user_text = (
        f"Page number: {page_number}\n\n"
        f"Extracted text preview:\n{page_text if page_text else '[NO EXTRACTABLE TEXT]'}"
    )

    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": image_url},
                ],
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
    )

    data = safe_json_loads(response.output_text)
    return PageDecision(page=page_number, decision=data["decision"], reason=data["reason"])


def extract_guidelines_from_page(
    client: "OpenAI",
    model: str,
    page_number: int,
    page_text: str,
    page_image_bytes: bytes,
) -> PageExtraction:
    """Extract guideline sentences from a relevant page."""
    image_url = image_bytes_to_data_url(page_image_bytes)

    prompt = (
        "You extract dietary guidelines from a single page of a dietary or nutrition guide.\n\n"
        "Rules:\n"
        "- Extract only guidance supported by the page.\n"
        "- Include explicit guidance and clear implications directly supported by the page.\n"
        "- Do not invent facts.\n"
        "- Write each guideline as a standalone markdown-ready sentence string.\n"
        "- Keep the meaning faithful to the page.\n"
        "- If the page contains examples of child vs adult portions, convert them into sentence guidelines.\n"
        "- Do not include page numbers, headings, captions, or decorative text unless they convey guidance.\n"
        "- Return an empty list if no actual guidelines are present."
    )

    user_text = (
        f"Page number: {page_number}\n\n"
        f"Extracted text preview:\n{page_text if page_text else '[NO EXTRACTABLE TEXT]'}"
    )

    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": image_url},
                ],
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
    )

    data = safe_json_loads(response.output_text)
    guidelines = [
        normalize_guideline(item)
        for item in data.get("guidelines", [])
        if normalize_guideline(item)
    ]

    return PageExtraction(
        page=page_number,
        page_summary=data.get("page_summary", "").strip(),
        guidelines=guidelines,
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
        )

    def process_pdf(
        self,
        pdf_path: str,
        model: str | None = None,
        dpi: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OutputBundle:
        """Run the page-by-page guideline extraction pipeline for a PDF."""
        ensure_api_key()
        OpenAI = _load_openai_client()
        client = OpenAI()
        doc = open_pdf(pdf_path)

        requested_model = model or get_default_model()
        requested_dpi = dpi or get_default_dpi()
        total_pages = len(doc)
        processed_pages: List[Dict[str, Any]] = []
        skipped_pages: List[Dict[str, Any]] = []
        extracted_guidelines: List[Dict[str, Any]] = []
        all_guideline_texts: List[str] = []

        try:
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
                )

                if decision.decision == "skip":
                    skipped_pages.append(asdict(decision))
                    continue

                extraction = extract_guidelines_from_page(
                    client=client,
                    model=requested_model,
                    page_number=idx,
                    page_text=page_text,
                    page_image_bytes=page_image,
                )

                processed_pages.append(
                    {
                        "page": extraction.page,
                        "page_summary": extraction.page_summary,
                        "guideline_count": len(extraction.guidelines),
                    }
                )

                for guideline in extraction.guidelines:
                    extracted_guidelines.append(
                        {
                            "page": idx,
                            "text": guideline,
                        }
                    )
                    all_guideline_texts.append(guideline)
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
