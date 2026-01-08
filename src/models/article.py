"""Article chat and translation models."""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from models.search import Citation


class ArticleChatRequest(BaseModel):
    """Request for chatting with a specific article."""

    article_urn: str = Field(description="URN of the article to chat about")
    session_id: str = Field(description="Session ID for conversation continuity")
    message: str = Field(description="User's question about the article")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    max_history: int = Field(
        default=10, description="Maximum conversation history to maintain"
    )


class ArticleChatResponse(BaseModel):
    """Response from article chat."""

    article_urn: str = Field(description="Article URN")
    session_id: str = Field(description="Session ID")
    answer: str = Field(description="Natural language answer in markdown format")
    citations: List[Citation] = Field(
        description="Citations to specific sections of the article"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the answer"
    )
    follow_up_suggestions: Optional[List[str]] = Field(
        default=None, description="Suggested follow-up questions"
    )
    timestamp: str = Field(description="ISO timestamp of response")


class TranslationRequest(BaseModel):
    """Request for article translation."""

    article_urn: str = Field(description="URN of the article to translate")
    target_language: str = Field(
        description="Target language code (ISO 639-1, e.g., 'es', 'fr', 'de')"
    )
    sections: Optional[List[str]] = Field(
        default=None,
        description="Specific sections to translate (if None, translate all)",
    )
    user_id: Optional[str] = Field(default=None, description="User identifier")


class TranslatedSection(BaseModel):
    """A translated section of an article."""

    section_name: str = Field(description="Name of the section")
    original_text: str = Field(description="Original text")
    translated_text: str = Field(description="Translated text")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Translation quality confidence"
    )


class TranslationResponse(BaseModel):
    """Response from article translation."""

    article_urn: str = Field(description="Article URN")
    source_language: str = Field(description="Detected or known source language")
    target_language: str = Field(description="Target language")
    title_translation: str = Field(description="Translated title")
    sections: List[TranslatedSection] = Field(description="Translated sections")
    metadata: Dict[str, Any] = Field(
        description="Original article metadata (preserved)"
    )
    generated_at: str = Field(description="ISO timestamp")
    cache_hit: bool = Field(default=False, description="Whether cached")
