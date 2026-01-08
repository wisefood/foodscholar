"""Session and chat models (refactored from app.py)."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Reference(BaseModel):
    """Reference source for a food fact."""

    source_type: str = Field(
        description="Type of source (e.g., 'nutritional database', 'scientific study', 'culinary knowledge')"
    )
    description: str = Field(description="Brief description of the reference or source")


class FoodFact(BaseModel):
    """Individual food fact with context."""

    fact: str = Field(description="The specific food fact or information")
    category: str = Field(
        description="Category of the fact (e.g., 'nutrition', 'cooking', 'history', 'storage')"
    )
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")


class FoodFactsResponse(BaseModel):
    """Structured response from the food facts assistant."""

    answer: str = Field(description="Natural language response to the user's question")
    facts: List[FoodFact] = Field(description="List of specific food facts mentioned")
    references: List[Reference] = Field(
        description="Sources and references for the information provided"
    )
    follow_up_suggestions: Optional[List[str]] = Field(
        default=None, description="Suggested follow-up questions the user might ask"
    )


class SessionStartRequest(BaseModel):
    """Request to start a new session with context."""

    session_id: str
    user_id: str = Field(
        description="User identifier to track sessions across multiple conversations"
    )
    user_context: str = Field(
        description="Context about the user (e.g., dietary preferences, restrictions, goals)"
    )
    max_history: Optional[int] = 10


class SessionStartResponse(BaseModel):
    """Response when starting a new session."""

    session_id: str
    message: str
    greeting: FoodFactsResponse


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: Optional[str] = Field(
        default=None,
        description="Optional: User identifier (only needed if not using /start endpoint)",
    )
    user_context: Optional[str] = Field(
        default=None,
        description="Optional: User context (only needed for first message if not using /start endpoint)",
    )
    max_history: Optional[int] = 10


class ChatResponse(BaseModel):
    session_id: str
    response: FoodFactsResponse
    timestamp: str
    is_first_message: bool
    session_title: Optional[str] = None


class SessionMetadata(BaseModel):
    """Metadata about a user session."""

    session_id: str
    user_id: str
    created_at: str
    last_active: str
    title: Optional[str] = None
    message_count: int = 0


class UserSessionsResponse(BaseModel):
    """Response containing all sessions for a user."""

    user_id: str
    total_sessions: int
    sessions: List[Dict[str, Any]]
