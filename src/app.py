from fastapi import FastAPI, HTTPException
import asyncio
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from routers.generic import install_error_handler
from datetime import datetime
import os
import json
import logging
import logsys
import uvicorn

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# Initialize logger
logger = logging.getLogger(__name__)
logsys.configure()


class Config:
    def __init__(self):
        self.settings = {}

    def setup(self):
        # Read environment variables and store them in the settings dictionary
        self.settings["HOST"] = os.getenv("HOST", "127.0.0.1")
        self.settings["PORT"] = int(os.getenv("PORT", 8000))
        self.settings["DEBUG"] = os.getenv("DEBUG", "true").lower() == "true"
        self.settings["CONTEXT_PATH"] = os.getenv("CONTEXT_PATH", "")
        self.settings["APP_EXT_DOMAIN"] = os.getenv(
            "APP_EXT_DOMAIN", "http://wisefood.gr"
        )
        self.settings["ELASTIC_HOST"] = os.getenv(
            "ELASTIC_HOST", "http://elasticsearch:9200"
        )
        self.settings["ES_DIM"] = int(os.getenv("ES_DIM", 384))
        self.settings["FOODSCHOLAR_URL"] = os.getenv(
            "FOODSCHOLAR_URL", "http://foodscholar:8001"
        )
        self.settings["RECIPEWRANGLER_URL"] = os.getenv(
            "RECIPEWRANGLER_URL", "http://recipewrangler:8001"
        )
        self.settings["FOODCHAT_URL"] = os.getenv(
            "FOODCHAT_URL", "http://foodchat:8001"
        )
        self.settings["MINIO_ENDPOINT"] = os.getenv(
            "MINIO_ENDPOINT", "http://minio:9000"
        )
        self.settings["MINIO_ROOT"] = os.getenv("MINIO_ROOT", "root")
        self.settings["MINIO_ROOT_PASSWORD"] = os.getenv(
            "MINIO_ROOT_PASSWORD", "minioadmin"
        )
        self.settings["MINIO_EXT_URL_CONSOLE"] = os.getenv(
            "MINIO_EXT_URL_CONSOLE", "https://s3.wisefood.gr/console"
        )
        self.settings["MINIO_EXT_URL_API"] = os.getenv(
            "MINIO_EXT_URL_API", "https://s3.wisefood.gr"
        )
        self.settings["MINIO_BUCKET"] = os.getenv("MINIO_BUCKET", "system")
        self.settings["KEYCLOAK_URL"] = os.getenv(
            "KEYCLOAK_URL", "http://keycloak:8080"
        )
        self.settings["KEYCLOAK_EXT_URL"] = os.getenv(
            "KEYCLOAK_EXT_URL", "https://auth.wisefood.gr"
        )
        self.settings["KEYCLOAK_ISSUER_URL"] = os.getenv(
            "KEYCLOAK_ISSUER_URL", "https://auth.wisefood.gr/realms/master"
        )
        self.settings["KEYCLOAK_REALM"] = os.getenv("KEYCLOAK_REALM", "master")
        self.settings["KEYCLOAK_CLIENT_ID"] = os.getenv(
            "KEYCLOAK_CLIENT_ID", "wisefood-api"
        )
        self.settings["KEYCLOAK_CLIENT_SECRET"] = os.getenv(
            "KEYCLOAK_CLIENT_SECRET", "secret"
        )
        self.settings["CACHE_ENABLED"] = (
            os.getenv("CACHE_ENABLED", "false").lower() == "true"
        )
        self.settings["REDIS_HOST"] = os.getenv("REDIS_HOST", "redis")
        self.settings["REDIS_PORT"] = int(os.getenv("REDIS_PORT", 6379))
        self.settings["POSTGRES_HOST"] = os.getenv("POSTGRES_HOST", "localhost")
        self.settings["POSTGRES_PORT"] = int(os.getenv("POSTGRES_PORT", 5432))
        self.settings["POSTGRES_USER"] = os.getenv("POSTGRES_USER", "postgres")
        self.settings["POSTGRES_PASSWORD"] = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.settings["POSTGRES_DB"] = os.getenv("POSTGRES_DB", "wisefood")
        self.settings["POSTGRES_POOL_SIZE"] = int(os.getenv("POSTGRES_POOL_SIZE", 10))
        self.settings["POSTGRES_MAX_OVERFLOW"] = int(
            os.getenv("POSTGRES_MAX_OVERFLOW", 20)
        )


# Configure application settings
config = Config()
config.setup()


app = FastAPI(title="FoodScholar Assistant API", debug=True)
install_error_handler(app)


# In-memory storage for conversation memories and contexts
memories: Dict[str, ConversationBufferWindowMemory] = {}
user_contexts: Dict[str, str] = {}
# Track which sessions belong to which users
user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
session_metadata: Dict[str, Dict] = (
    {}
)  # session_id -> {user_id, created_at, last_active, title}
session_titles: Dict[str, str] = {}  # session_id -> title


# Structured output models
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
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY"),
        )

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

        # Ensure title isn't too long
        if len(title) > 50:
            title = title[:47] + "..."

        return title
    except Exception as e:
        # Fallback to a generic title if generation fails
        return "Food Facts Conversation"


def is_first_user_message(session_id: str) -> bool:
    """Check if this is the first real user message (excluding automated greeting)."""
    if session_id not in memories:
        return True
    memory = memories[session_id]
    history = memory.load_memory_variables({})
    messages = history.get("chat_history", [])

    # Count only human messages, excluding the automated greeting
    human_messages = [
        msg
        for msg in messages
        if msg.type == "human"
        and msg.content != "Hello! I'd like help with food and nutrition questions."
    ]

    return len(human_messages) == 0


def create_structured_chain(session_id: str, max_history: int = 20):
    """Create a chain that returns structured output."""

    # Initialize LLM with structured output
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Create structured output LLM
    structured_llm = llm.with_structured_output(FoodFactsResponse)

    # Get or create memory for this session
    memory = get_or_create_memory(session_id, max_history)

    # Get user context if available
    user_context = user_contexts.get(session_id, "")
    context_section = f"\n\nUser Context:\n{user_context}" if user_context else ""

    # Create prompt template
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

    # Create chain
    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: memory.load_memory_variables({})["chat_history"]
        )
        | prompt
        | structured_llm
    )

    return chain, memory


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


@app.post("/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """Start a new session with user context and get a greeting."""

    # Validate session_id is not already used by another user
    if request.session_id in session_metadata:
        existing_user_id = session_metadata[request.session_id].get("user_id")
        if existing_user_id != request.user_id:
            raise HTTPException(
                status_code=409,
                detail=f"Session ID '{request.session_id}' already exists for another user. Please use a unique session ID.",
            )

    # Store user context
    user_contexts[request.session_id] = request.user_context

    # Track user-session relationship
    if request.user_id not in user_sessions:
        user_sessions[request.user_id] = []
    if request.session_id not in user_sessions[request.user_id]:
        user_sessions[request.user_id].append(request.session_id)

    # Store session metadata
    session_metadata[request.session_id] = {
        "user_id": request.user_id,
        "created_at": datetime.now().isoformat(),
        "last_active": datetime.now().isoformat(),
    }

    # Initialize memory
    get_or_create_memory(request.session_id, request.max_history)

    try:
        # Create chain and get greeting
        chain, memory = create_structured_chain(request.session_id, request.max_history)

        # Get greeting response
        greeting_response = chain.invoke(
            {"input": "Hello! I'd like help with food and nutrition questions."}
        )

        # Save to memory with full structured response
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a structured response from the food facts assistant."""

    # Validate session ownership if session already exists
    if request.session_id in session_metadata and request.user_id:
        existing_user_id = session_metadata[request.session_id].get("user_id")
        if existing_user_id != request.user_id:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Session '{request.session_id}' belongs to another user.",
            )

    # If context provided and it's a new session, store it
    if request.user_context and request.session_id not in user_contexts:
        user_contexts[request.session_id] = request.user_context

    # Track user-session relationship if user_id provided
    if request.user_id:
        if request.user_id not in user_sessions:
            user_sessions[request.user_id] = []
        if request.session_id not in user_sessions[request.user_id]:
            user_sessions[request.user_id].append(request.session_id)

        # Store/update session metadata
        if request.session_id not in session_metadata:
            session_metadata[request.session_id] = {
                "user_id": request.user_id,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
            }

    # Update last active timestamp
    if request.session_id in session_metadata:
        session_metadata[request.session_id]["last_active"] = datetime.now().isoformat()

    first_message = is_first_user_message(request.session_id)
    session_title = None

    try:
        # Create or get existing chain and memory
        chain, memory = create_structured_chain(request.session_id, request.max_history)

        # Get structured response
        response = chain.invoke({"input": request.message})

        # Save to memory with full structured response
        memory.save_context(
            {"input": request.message}, {"output": response.model_dump_json()}
        )

        # Generate title if this is the first user message (not the initial greeting)
        if (
            first_message
            and request.message
            != "Hello! I'd like help with food and nutrition questions."
        ):
            session_title = generate_session_title(
                request.message, user_contexts.get(request.session_id, "")
            )
            session_titles[request.session_id] = session_title

            # Update metadata with title
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


@app.get("/user/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    """Get all sessions and their histories for a specific user."""
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
            "messages": [],
        }

        # # Get conversation history if exists
        # if session_id in memories:
        #     memory = memories[session_id]
        #     history = memory.load_memory_variables({})
        #     messages_raw = history.get("chat_history", [])
        #     session_info["message_count"] = len(messages_raw)

        #     # Parse structured responses
        #     messages = []
        #     for msg in messages_raw:
        #         message_data = {"type": msg.type, "content": msg.content}

        #         # If it's an AI message, try to parse the JSON back to structured format
        #         if msg.type == "ai":
        #             try:
        #                 structured_content = json.loads(msg.content)
        #                 message_data["structured_response"] = structured_content
        #             except:
        #                 # If parsing fails, keep as plain text
        #                 message_data["content"] = msg.content

        #         messages.append(message_data)

        #     session_info["messages"] = messages

        sessions.append(session_info)

    return {"user_id": user_id, "total_sessions": len(sessions), "sessions": sessions}


@app.get("/user/{user_id}/session/{session_id}/context")
async def get_context(user_id: str, session_id: str):
    """Retrieve user context for a session."""
    if session_id not in user_contexts:
        raise HTTPException(status_code=404, detail="Session context not found")

    # Validate session ownership
    if session_id in session_metadata:
        owner_id = session_metadata[session_id].get("user_id")
        if owner_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied. You don't have permission to access this session.",
            )

    return {"session_id": session_id, "user_context": user_contexts[session_id]}


@app.get("/user/{user_id}/session/{session_id}/history")
async def get_history(user_id: str, session_id: str):
    """Retrieve conversation history for a session."""
    if session_id not in memories:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if session belongs to the user
    if session_id in session_metadata:
        owner_id = session_metadata[session_id].get("user_id")
        if owner_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied. This session does not belong to you.",
            )

    memory = memories[session_id]
    history = memory.load_memory_variables({})

    # Parse the JSON responses back to structured format
    messages = []
    for msg in history.get("chat_history", []):
        message_data = {"type": msg.type, "content": msg.content}

        # If it's an AI message, try to parse the JSON back to structured format
        if msg.type == "ai":
            try:
                structured_content = json.loads(msg.content)
                message_data["structured_response"] = structured_content
            except:
                # If parsing fails, keep as plain text
                message_data["content"] = msg.content

        messages.append(message_data)

    return {
        "session_id": session_id,
        "user_context": user_contexts.get(session_id, None),
        "session_title": session_titles.get(session_id, None),
        "messages": messages,
    }


@app.delete("/user/{user_id}/session/{session_id}/history")
async def clear_history(user_id: str, session_id: str):
    """Clear conversation history for a session."""
    if session_id not in memories:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if session belongs to the user
    if session_id in session_metadata:
        owner_id = session_metadata[session_id].get("user_id")
        if owner_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied. This session does not belong to you.",
            )

    memories[session_id].clear()
    return {"message": f"History cleared for session {session_id}"}


@app.delete("/user/{user_id}/session/{session_id}")
async def delete_session(user_id: str, session_id: str):
    """Delete entire session including context."""

    # Check if session exists and belongs to the user
    if session_id in session_metadata:
        owner_id = session_metadata[session_id].get("user_id")
        if owner_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied. This session does not belong to you.",
            )

    deleted_items = []

    # Delete metadata
    if session_id in session_metadata:
        del session_metadata[session_id]
        deleted_items.append("metadata")

    # Remove from user's session list
    if user_id and user_id in user_sessions:
        if session_id in user_sessions[user_id]:
            user_sessions[user_id].remove(session_id)
            # Clean up empty user entries
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


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "FoodScholar",
        "active_sessions": len(memories),
        "active_users": len(user_sessions),
    }


if __name__ == "__main__":

    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("PORT", 8005)), log_level="debug"
    )
