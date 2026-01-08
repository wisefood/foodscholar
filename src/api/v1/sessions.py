"""Session management API endpoints (refactored from app.py)."""
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, List
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import json

from backend.groq import GROQ_CHAT
from models.session import (
    SessionStartRequest,
    SessionStartResponse,
    ChatRequest,
    ChatResponse,
    FoodFactsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])

# In-memory storage for conversation memories and contexts
memories: Dict[str, ConversationBufferWindowMemory] = {}
user_contexts: Dict[str, str] = {}
user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
session_metadata: Dict[str, Dict] = {}  # session_id -> {user_id, created_at, last_active, title}
session_titles: Dict[str, str] = {}  # session_id -> title


def get_or_create_memory(
    session_id: str, max_history: int = 20
) -> ConversationBufferWindowMemory:
    """Get existing memory or create new one."""
    if session_id not in memories:
        memories[session_id] = ConversationBufferWindowMemory(
            k=max_history, return_messages=True, memory_key="chat_history"
        )
    return memories[session_id]


def generate_session_title(user_message: str, user_context: str = "") -> str:
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

        response = llm.invoke(prompt)
        title = response.content.strip().strip('"').strip("'")

        if len(title) > 50:
            title = title[:47] + "..."

        return title
    except Exception as e:
        logger.error(f"Error generating session title: {e}")
        return "Food Facts Conversation"


def is_first_user_message(session_id: str) -> bool:
    """Check if this is the first real user message (excluding automated greeting)."""
    if session_id not in memories:
        return True
    memory = memories[session_id]
    history = memory.load_memory_variables({})
    messages = history.get("chat_history", [])

    human_messages = [
        msg
        for msg in messages
        if msg.type == "human"
        and msg.content != "Hello! I'd like help with food and nutrition questions."
    ]

    return len(human_messages) == 0


def create_structured_chain(session_id: str, max_history: int = 20):
    """Create a chain that returns structured output."""
    llm = GROQ_CHAT.get_client(model="llama-3.3-70b-versatile", temperature=0.7)

    structured_llm = llm.with_structured_output(FoodFactsResponse)
    memory = get_or_create_memory(session_id, max_history)
    user_context = user_contexts.get(session_id, "")
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

    return chain, memory


@router.post("/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """Start a new session with user context and get a greeting."""
    if request.session_id in session_metadata:
        existing_user_id = session_metadata[request.session_id].get("user_id")
        if existing_user_id != request.user_id:
            raise HTTPException(
                status_code=409,
                detail=f"Session ID '{request.session_id}' already exists for another user.",
            )

    user_contexts[request.session_id] = request.user_context

    if request.user_id not in user_sessions:
        user_sessions[request.user_id] = []
    if request.session_id not in user_sessions[request.user_id]:
        user_sessions[request.user_id].append(request.session_id)

    session_metadata[request.session_id] = {
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat(),
        "last_active": datetime.now().isoformat(),
    }

    get_or_create_memory(request.session_id, request.max_history)

    try:
        chain, memory = create_structured_chain(request.session_id, request.max_history)

        greeting_response = chain.invoke(
            {"input": "Hello! I'd like help with food and nutrition questions."}
        )

        memory.save_context(
            {"input": "Hello! I'd like help with food and nutrition questions."},
            {"output": greeting_response.model_dump_json()},
        )

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
    if request.session_id in session_metadata and request.user_id:
        existing_user_id = session_metadata[request.session_id].get("user_id")
        if existing_user_id != request.user_id:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Session '{request.session_id}' belongs to another user.",
            )

    if request.user_context and request.session_id not in user_contexts:
        user_contexts[request.session_id] = request.user_context

    if request.user_id:
        if request.user_id not in user_sessions:
            user_sessions[request.user_id] = []
        if request.session_id not in user_sessions[request.user_id]:
            user_sessions[request.user_id].append(request.session_id)

        if request.session_id not in session_metadata:
            session_metadata[request.session_id] = {
                "user_id": request.user_id,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
            }

    if request.session_id in session_metadata:
        session_metadata[request.session_id]["last_active"] = datetime.now().isoformat()

    first_message = is_first_user_message(request.session_id)
    session_title = None

    try:
        chain, memory = create_structured_chain(request.session_id, request.max_history)
        response = chain.invoke({"input": request.message})

        memory.save_context(
            {"input": request.message}, {"output": response.model_dump_json()}
        )

        if (
            first_message
            and request.message != "Hello! I'd like help with food and nutrition questions."
        ):
            session_title = generate_session_title(
                request.message, user_contexts.get(request.session_id, "")
            )
            session_titles[request.session_id] = session_title

            if request.session_id in session_metadata:
                session_metadata[request.session_id]["title"] = session_title

        return ChatResponse(
            session_id=request.session_id,
            response=response,
            timestamp=datetime.now().isoformat(),
            is_first_message=first_message,
            session_title=session_title,
        )

    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/users/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all sessions for a specific user."""
    if user_id not in user_sessions:
        raise HTTPException(status_code=404, detail="User not found")

    sessions = []
    for session_id in user_sessions[user_id]:
        session_info = {
            "session_id": session_id,
            "metadata": session_metadata.get(session_id, {}),
            "user_context": user_contexts.get(session_id, None),
            "session_title": session_titles.get(session_id, None),
            "message_count": 0,
        }
        sessions.append(session_info)

    return {"user_id": user_id, "total_sessions": len(sessions), "sessions": sessions}


@router.get("/{session_id}/context")
async def get_context(session_id: str, user_id: str):
    """Retrieve user context for a session."""
    if session_id not in user_contexts:
        raise HTTPException(status_code=404, detail="Session context not found")

    if session_id in session_metadata:
        owner_id = session_metadata[session_id].get("user_id")
        if owner_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied.")

    return {"session_id": session_id, "user_context": user_contexts[session_id]}


@router.get("/{session_id}/history")
async def get_history(session_id: str, user_id: str):
    """Retrieve conversation history for a session."""
    if session_id not in memories:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_id in session_metadata:
        owner_id = session_metadata[session_id].get("user_id")
        if owner_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied.")

    memory = memories[session_id]
    history = memory.load_memory_variables({})

    messages = []
    for msg in history.get("chat_history", []):
        message_data = {"type": msg.type, "content": msg.content}

        if msg.type == "ai":
            try:
                structured_content = json.loads(msg.content)
                message_data["structured_response"] = structured_content
            except:
                message_data["content"] = msg.content

        messages.append(message_data)

    return {
        "session_id": session_id,
        "user_context": user_contexts.get(session_id, None),
        "session_title": session_titles.get(session_id, None),
        "messages": messages,
    }


@router.delete("/{session_id}/history")
async def clear_history(session_id: str, user_id: str):
    """Clear conversation history for a session."""
    if session_id not in memories:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_id in session_metadata:
        owner_id = session_metadata[session_id].get("user_id")
        if owner_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied.")

    memories[session_id].clear()
    return {"message": f"History cleared for session {session_id}"}


@router.delete("/{session_id}")
async def delete_session(session_id: str, user_id: str):
    """Delete entire session including context."""
    if session_id in session_metadata:
        owner_id = session_metadata[session_id].get("user_id")
        if owner_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied.")

    deleted_items = []

    if session_id in session_metadata:
        del session_metadata[session_id]
        deleted_items.append("metadata")

    if user_id and user_id in user_sessions:
        if session_id in user_sessions[user_id]:
            user_sessions[user_id].remove(session_id)
            if not user_sessions[user_id]:
                del user_sessions[user_id]

    if session_id in memories:
        del memories[session_id]
        deleted_items.append("memory")

    if session_id in user_contexts:
        del user_contexts[session_id]
        deleted_items.append("context")

    if session_id in session_titles:
        del session_titles[session_id]
        deleted_items.append("title")

    if deleted_items:
        return {"message": f"Session {session_id} deleted", "deleted": deleted_items}

    raise HTTPException(status_code=404, detail="Session not found")
