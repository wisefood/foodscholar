from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import os

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

app = FastAPI(title="WiseFood EU - FoodScholar Assistant")

# In-memory storage for conversation memories and contexts
memories: Dict[str, ConversationBufferWindowMemory] = {}
user_contexts: Dict[str, str] = {}
# Track which sessions belong to which users
user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
session_metadata: Dict[str, Dict] = {}  # session_id -> {user_id, created_at, last_active}


# Structured output models
class Reference(BaseModel):
    """Reference source for a food fact."""
    source_type: str = Field(description="Type of source (e.g., 'nutritional database', 'scientific study', 'culinary knowledge')")
    description: str = Field(description="Brief description of the reference or source")


class FoodFact(BaseModel):
    """Individual food fact with context."""
    fact: str = Field(description="The specific food fact or information")
    category: str = Field(description="Category of the fact (e.g., 'nutrition', 'cooking', 'history', 'storage')")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")


class FoodFactsResponse(BaseModel):
    """Structured response from the food facts assistant."""
    answer: str = Field(description="Natural language response to the user's question")
    facts: List[FoodFact] = Field(description="List of specific food facts mentioned")
    references: List[Reference] = Field(description="Sources and references for the information provided")
    follow_up_suggestions: Optional[List[str]] = Field(
        default=None, 
        description="Suggested follow-up questions the user might ask"
    )


def get_or_create_memory(session_id: str, max_history: int = 10) -> ConversationBufferWindowMemory:
    """Get existing memory or create new one."""
    if session_id not in memories:
        memories[session_id] = ConversationBufferWindowMemory(
            k=max_history,
            return_messages=True,
            memory_key="chat_history"
        )
    return memories[session_id]


def is_first_message(session_id: str) -> bool:
    """Check if this is the first message in the session."""
    if session_id not in memories:
        return True
    memory = memories[session_id]
    history = memory.load_memory_variables({})
    return len(history.get("chat_history", [])) == 0


def create_structured_chain(session_id: str, max_history: int = 10):
    """Create a chain that returns structured output."""
    
    # Initialize LLM with structured output
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Create structured output LLM
    structured_llm = llm.with_structured_output(FoodFactsResponse)
    
    # Get or create memory for this session
    memory = get_or_create_memory(session_id, max_history)
    
    # Get user context if available
    user_context = user_contexts.get(session_id, "")
    context_section = f"\n\nUser Context:\n{user_context}" if user_context else ""
    
    # Create prompt template
    system_message = f"""You are a helpful food facts assistant. Provide accurate, interesting 
information about food, nutrition, cooking, ingredients, and culinary topics.

IMPORTANT: You must respond with structured data including:
- A natural language answer
- Specific food facts with categories (nutrition, cooking, history, storage, etc.)
- References for your information
- Optional follow-up question suggestions

Be honest about your confidence level. If you're not certain, mark confidence as 'medium' or 'low'.
For references, cite general knowledge sources like 'USDA Nutritional Database', 'culinary science', 
'food chemistry research', etc.{context_section}

GREETING: If this is the first message from the user, greet them warmly and acknowledge their context 
if provided. Ask them what food-related questions they have."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
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
    user_id: str = Field(description="User identifier to track sessions across multiple conversations")
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
        description="Optional: User identifier (only needed if not using /start endpoint)"
    )
    user_context: Optional[str] = Field(
        default=None,
        description="Optional: User context (only needed for first message if not using /start endpoint)"
    )
    max_history: Optional[int] = 10


class ChatResponse(BaseModel):
    session_id: str
    response: FoodFactsResponse
    timestamp: str
    is_first_message: bool


@app.post("/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """Start a new session with user context and get a greeting."""
    
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
        "last_active": datetime.now().isoformat()
    }
    
    # Initialize memory
    get_or_create_memory(request.session_id, request.max_history)
    
    try:
        # Create chain and get greeting
        chain, memory = create_structured_chain(
            request.session_id,
            request.max_history
        )
        
        # Get greeting response
        greeting_response = chain.invoke({
            "input": "Hello! I'd like help with food and nutrition questions."
        })
        
        # Save to memory
        memory.save_context(
            {"input": "Hello! I'd like help with food and nutrition questions."},
            {"output": greeting_response.answer}
        )
        
        return SessionStartResponse(
            session_id=request.session_id,
            message="Session started successfully",
            greeting=greeting_response
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a structured response from the food facts assistant."""
    
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
                "last_active": datetime.now().isoformat()
            }
    
    # Update last active timestamp
    if request.session_id in session_metadata:
        session_metadata[request.session_id]["last_active"] = datetime.now().isoformat()
    
    first_message = is_first_message(request.session_id)
    
    try:
        # Create or get existing chain and memory
        chain, memory = create_structured_chain(
            request.session_id,
            request.max_history
        )
        
        # Get structured response
        response = chain.invoke({"input": request.message})
        
        # Save to memory
        memory.save_context(
            {"input": request.message},
            {"output": response.answer}
        )
        
        return ChatResponse(
            session_id=request.session_id,
            response=response,
            timestamp=datetime.now().isoformat(),
            is_first_message=first_message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/context/{session_id}")
async def get_context(session_id: str):
    """Retrieve user context for a session."""
    if session_id not in user_contexts:
        raise HTTPException(status_code=404, detail="Session context not found")
    
    return {
        "session_id": session_id,
        "user_context": user_contexts[session_id]
    }


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Retrieve conversation history for a session."""
    if session_id not in memories:
        raise HTTPException(status_code=404, detail="Session not found")
    
    memory = memories[session_id]
    history = memory.load_memory_variables({})
    
    return {
        "session_id": session_id,
        "user_context": user_contexts.get(session_id, None),
        "messages": [
            {"type": msg.type, "content": msg.content}
            for msg in history.get("chat_history", [])
        ]
    }


@app.delete("/history/{session_id}")
async def clear_history(session_id: str, user_id: str):
    """Clear conversation history for a session."""
    if session_id not in memories:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if session belongs to the user
    if session_id in session_metadata:
        owner_id = session_metadata[session_id].get("user_id")
        if owner_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied. This session does not belong to you."
            )
    
    memories[session_id].clear()
    return {"message": f"History cleared for session {session_id}"}


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
            "message_count": 0,
            "messages": []
        }
        
        # Get conversation history if exists
        if session_id in memories:
            memory = memories[session_id]
            history = memory.load_memory_variables({})
            messages = history.get("chat_history", [])
            session_info["message_count"] = len(messages)
            session_info["messages"] = [
                {"type": msg.type, "content": msg.content}
                for msg in messages
            ]
        
        sessions.append(session_info)
    
    return {
        "user_id": user_id,
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete entire session including context."""
    deleted_items = []
    
    # Get user_id before deleting metadata
    user_id = None
    if session_id in session_metadata:
        user_id = session_metadata[session_id].get("user_id")
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
    
    if deleted_items:
        return {"message": f"Session {session_id} deleted", "deleted": deleted_items}
    
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "FoodScholar Assistant",
        "active_sessions": len(memories),
        "active_users": len(user_sessions)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT", 8005))