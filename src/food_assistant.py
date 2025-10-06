from fastapi import FastAPI, HTTPException
import asyncio
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import os

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

app = FastAPI(title="Food Facts Assistant")

# In-memory storage for conversation memories
memories: Dict[str, ConversationBufferWindowMemory] = {}


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
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful food facts assistant. Provide accurate, interesting 
information about food, nutrition, cooking, ingredients, and culinary topics.

IMPORTANT: You must respond with structured data including:
- A natural language answer
- Specific food facts with categories (nutrition, cooking, history, storage, etc.)
- References for your information
- Optional follow-up question suggestions

Be honest about your confidence level. If you're not certain, mark confidence as 'medium' or 'low'.
For references, cite general knowledge sources like 'USDA Nutritional Database', 'culinary science', 
'food chemistry research', etc."""),
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


class ChatRequest(BaseModel):
    session_id: str
    message: str
    max_history: Optional[int] = 10


class ChatResponse(BaseModel):
    session_id: str
    response: FoodFactsResponse
    timestamp: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        chain, memory = create_structured_chain(request.session_id, request.max_history)

        # ✅ Correct async call
        response = await chain.ainvoke({"input": request.message})

        # Save memory (this part can remain sync)
        memory.save_context(
            {"input": request.message},
            {"output": response.answer}
        )

        return ChatResponse(
            session_id=request.session_id,
            response=response,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Retrieve conversation history for a session."""
    if session_id not in memories:
        raise HTTPException(status_code=404, detail="Session not found")
    
    memory = memories[session_id]
    history = memory.load_memory_variables({})
    
    return {
        "session_id": session_id,
        "messages": [
            {"type": msg.type, "content": msg.content}
            for msg in history.get("chat_history", [])
        ]
    }


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    if session_id in memories:
        memories[session_id].clear()
        return {"message": f"History cleared for session {session_id}"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete entire session."""
    if session_id in memories:
        del memories[session_id]
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Food Facts Assistant (LangChain + Structured Output)",
        "active_sessions": len(memories)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
