"""Session management API endpoints.

Session state lives in the Redis-backed SESSION_STORE (see
services/session_store.py) with a sliding TTL, so sessions survive restarts,
work across replicas, and expire automatically — a requirement for ephemeral
guest users. Every session-scoped endpoint enforces that the supplied user_id
matches the session owner; sessions can no longer be created or read without
an owner.
"""
import logging
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import json

from backend.groq import GROQ_CHAT
from backend.langfuse import build_trace_config
from models.session import (
    SessionStartRequest,
    SessionStartResponse,
    ChatRequest,
    ChatResponse,
    FoodFactsResponse,
)
from services.session_store import SESSION_STORE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])

GREETING_MESSAGE = "Hello! I'd like help with food and nutrition questions."


def _new_session_doc(user_id: str, user_context: str = "") -> dict:
    now = datetime.now().isoformat()
    return {
        "user_id": user_id,
        "created_at": now,
        "last_active": now,
        "title": None,
        "user_context": user_context or "",
        "messages": [],
    }


def _require_owner(session_data: dict, user_id: Optional[str], session_id: str) -> None:
    """Reject any access where the caller's user_id does not match the owner."""
    if not user_id or session_data.get("user_id") != user_id:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Session '{session_id}' belongs to another user.",
        )


def _build_memory(
    messages: List[dict], max_history: int = 20
) -> ConversationBufferWindowMemory:
    """Rebuild a windowed conversation memory from stored messages."""
    memory = ConversationBufferWindowMemory(
        k=max_history, return_messages=True, memory_key="chat_history"
    )
    for msg in messages:
        if msg.get("type") == "human":
            memory.chat_memory.add_user_message(msg.get("content", ""))
        else:
            memory.chat_memory.add_ai_message(msg.get("content", ""))
    return memory


def _is_first_user_message(messages: List[dict]) -> bool:
    """Check if this is the first real user message (excluding automated greeting)."""
    return not any(
        msg.get("type") == "human" and msg.get("content") != GREETING_MESSAGE
        for msg in messages
    )


def generate_session_title(
    user_message: str,
    user_context: str = "",
    session_id: str = None,
    user_id: str = None,
) -> str:
    """Generate a short title for the session based on the first user message."""
    try:
        llm = GROQ_CHAT.get_client(model="llama-3.3-70b-versatile", temperature=0.3)

        context_info = f"\nUser context: {user_context}" if user_context else ""

        prompt = f"""Based on this user's first message in a food facts conversation, generate a very short, concise title (3-6 words max) that summarizes what they want to know about.{context_info}

User's message: "{user_message}"

Generate ONLY the title, nothing else. Examples of good titles:
- "Salmon Nutrition Benefits"
- "Vegan Protein Sources"
- "Meal Prep for Athletes"
- "Gluten-Free Baking Tips"

Title:"""

        response = llm.invoke(
            prompt,
            config=build_trace_config(
                run_name="session-title",
                session_id=session_id,
                user_id=user_id,
                tags=["session-chat", "title"],
            ),
        )
        title = response.content.strip().strip('"').strip("'")

        if len(title) > 50:
            title = title[:47] + "..."

        return title
    except Exception as e:
        logger.error(f"Error generating session title: {e}")
        return "Food Facts Conversation"


def create_structured_chain(session_data: dict, max_history: int = 20):
    """Create a chain that returns structured output."""
    llm = GROQ_CHAT.get_client(model="llama-3.3-70b-versatile", temperature=0.7)

    structured_llm = llm.with_structured_output(FoodFactsResponse)
    memory = _build_memory(session_data.get("messages", []), max_history)
    user_context = session_data.get("user_context", "")
    context_section = f"\n\nUser Context:\n{user_context}" if user_context else ""

    system_message = f"""You are a helpful food facts assistant, named FoodScholar. Provide accurate, interesting
information about food, nutrition, cooking, ingredients, and culinary topics.

IMPORTANT: You must respond with structured data including:
- A natural language answer, optionally in markdown format
- Specific food facts with categories (nutrition, cooking, history, storage, etc.) only if not greeting
- References for your information
- Optional follow-up question suggestions the user can ask next

STRUCTURE & FORMAT (STRICT):
- You MUST call the function FoodFactsResponse and provide ONLY its JSON arguments.
- Do NOT write any text outside the function call.
- Do NOT wrap JSON in code fences.
- Markdown is allowed ONLY inside the 'answer' field. No markdown anywhere else.

SCHEMA GUARDRAILS:
- facts[*].category: plain string such as "nutrition", "cooking", "history", "storage", "safety", "substitutions".
- facts[*].confidence: one of "high", "medium", "low".
- references[*].source_type: one of "nutritional database", "scientific study", "culinary knowledge", "textbook", "website".
- references[*].description: brief plain-text description.
- If unsure about a claim, lower confidence; do not invent sources.

Be honest about your confidence level. If you're not certain, mark confidence as 'medium' or 'low'.
For references, cite general knowledge sources like 'USDA Nutritional Database', 'culinary science',
'food chemistry research', etc.{context_section}

GREETING: If this is the first message from the user, greet them warmly and acknowledge their context
if provided without making them feel uncomfortable by repeating it. Ask them what food-related questions they have."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: memory.load_memory_variables({})["chat_history"]
        )
        | prompt
        | structured_llm
    )

    return chain


@router.post("/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """Start a new session with user context and get a greeting."""
    existing = SESSION_STORE.get(request.session_id)
    if existing is not None and existing.get("user_id") != request.user_id:
        raise HTTPException(
            status_code=409,
            detail=f"Session ID '{request.session_id}' already exists for another user.",
        )

    session_data = existing or _new_session_doc(request.user_id, request.user_context)
    session_data["user_context"] = request.user_context

    try:
        chain = create_structured_chain(session_data, request.max_history)

        greeting_response = chain.invoke(
            {"input": GREETING_MESSAGE},
            config=build_trace_config(
                run_name="session-greeting",
                session_id=request.session_id,
                user_id=request.user_id,
                tags=["session-chat", "greeting"],
            ),
        )

        session_data["messages"].append({"type": "human", "content": GREETING_MESSAGE})
        session_data["messages"].append(
            {"type": "ai", "content": greeting_response.model_dump_json()}
        )
        session_data["last_active"] = datetime.now().isoformat()
        SESSION_STORE.save(request.session_id, session_data)

        return SessionStartResponse(
            session_id=request.session_id,
            message="Session started successfully",
            greeting=greeting_response,
        )

    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a structured response from the food facts assistant."""
    session_data = SESSION_STORE.get(request.session_id)

    if session_data is None:
        if not request.user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required to start a session via /chat.",
            )
        session_data = _new_session_doc(request.user_id, request.user_context or "")
    else:
        _require_owner(session_data, request.user_id, request.session_id)
        if request.user_context and not session_data.get("user_context"):
            session_data["user_context"] = request.user_context

    session_data["last_active"] = datetime.now().isoformat()

    first_message = _is_first_user_message(session_data["messages"])
    session_title = None

    try:
        chain = create_structured_chain(session_data, request.max_history)
        response = chain.invoke(
            {"input": request.message},
            config=build_trace_config(
                run_name="session-chat",
                session_id=request.session_id,
                user_id=request.user_id,
                tags=["session-chat"],
            ),
        )

        session_data["messages"].append({"type": "human", "content": request.message})
        session_data["messages"].append(
            {"type": "ai", "content": response.model_dump_json()}
        )

        if first_message and request.message != GREETING_MESSAGE:
            session_title = generate_session_title(
                request.message,
                session_data.get("user_context", ""),
                session_id=request.session_id,
                user_id=request.user_id,
            )
            session_data["title"] = session_title

        SESSION_STORE.save(request.session_id, session_data)

        return ChatResponse(
            session_id=request.session_id,
            response=response,
            timestamp=datetime.now().isoformat(),
            is_first_message=first_message,
            session_title=session_title,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/users/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all sessions for a specific user."""
    sessions = []
    for session_id in SESSION_STORE.list_user_sessions(user_id):
        session_data = SESSION_STORE.get(session_id)
        if session_data is None:
            continue
        sessions.append(
            {
                "session_id": session_id,
                "metadata": {
                    "user_id": session_data.get("user_id"),
                    "created_at": session_data.get("created_at"),
                    "last_active": session_data.get("last_active"),
                    "title": session_data.get("title"),
                },
                "user_context": session_data.get("user_context") or None,
                "session_title": session_data.get("title"),
                "message_count": sum(
                    1
                    for m in session_data.get("messages", [])
                    if m.get("type") == "human"
                ),
            }
        )

    return {"user_id": user_id, "total_sessions": len(sessions), "sessions": sessions}


@router.get("/{session_id}/context")
async def get_context(session_id: str, user_id: str):
    """Retrieve user context for a session."""
    session_data = SESSION_STORE.get(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail="Session context not found")

    _require_owner(session_data, user_id, session_id)

    return {"session_id": session_id, "user_context": session_data.get("user_context", "")}


@router.get("/{session_id}/history")
async def get_history(session_id: str, user_id: str):
    """Retrieve conversation history for a session."""
    session_data = SESSION_STORE.get(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    _require_owner(session_data, user_id, session_id)

    messages = []
    for msg in session_data.get("messages", []):
        message_data = {"type": msg.get("type"), "content": msg.get("content")}

        if msg.get("type") == "ai":
            try:
                message_data["structured_response"] = json.loads(msg.get("content"))
            except (TypeError, json.JSONDecodeError):
                pass

        messages.append(message_data)

    return {
        "session_id": session_id,
        "user_context": session_data.get("user_context") or None,
        "session_title": session_data.get("title"),
        "messages": messages,
    }


@router.delete("/{session_id}/history")
async def clear_history(session_id: str, user_id: str):
    """Clear conversation history for a session."""
    session_data = SESSION_STORE.get(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    _require_owner(session_data, user_id, session_id)

    session_data["messages"] = []
    session_data["last_active"] = datetime.now().isoformat()
    SESSION_STORE.save(session_id, session_data)
    return {"message": f"History cleared for session {session_id}"}


@router.delete("/{session_id}")
async def delete_session(session_id: str, user_id: str):
    """Delete entire session including context."""
    session_data = SESSION_STORE.get(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    _require_owner(session_data, user_id, session_id)

    SESSION_STORE.delete(session_id, user_id=session_data.get("user_id"))
    return {
        "message": f"Session {session_id} deleted",
        "deleted": ["metadata", "memory", "context", "title"],
    }
