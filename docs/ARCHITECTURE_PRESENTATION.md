# FoodScholar Architecture Overview

## Part 1: Architecture Overview 

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT                               │
│              (Web/Mobile App)                           │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 API LAYER                               │
│            FastAPI (src/api/)                           │
│  • HTTP endpoints                                       │
│  • Request/Response validation                          │
│  • Authentication                                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              SERVICE LAYER                              │
│          (src/services/)                                │
│  • Business logic                                       │
│  • Caching strategy                                     │
│  • Orchestration                                        │
│  • Data transformation                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               AGENT LAYER                               │
│          (src/agents/)                                  │
│  • AI-powered synthesis                                 │
│  • LLM orchestration                                    │
│  • Prompt engineering                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Part 2: Layer 1 - API Layer

**Location:** `src/api/v1/`

**Responsibility:** HTTP interface to the application

**What it does:**
- ✅ Receives HTTP requests
- ✅ Validates input (Pydantic models)
- ✅ Handles authentication/authorization
- ✅ Returns HTTP responses
- ✅ Rate limiting

**What it does NOT do:**
- ❌ Business logic
- ❌ Data processing
- ❌ AI/LLM calls
- ❌ Caching decisions

**Example:**
```python
@router.post("/search/summarize", response_model=SearchSummaryResponse)
async def summarize_search(request: SearchSummaryRequest):
    """API endpoint - keeps it simple!"""
    service = SearchSummarizer(cache_enabled=True)
    return await service.summarize_search(request)
```



## Part 3: Layer 2 - Service Layer

**Location:** `src/services/`

**Responsibility:** Business logic orchestration

**What it does:**
- ✅ Caching strategy & management
- ✅ Data transformation
- ✅ Input validation & error handling
- ✅ Orchestrating multiple agents/components
- ✅ Monitoring & logging
- ✅ Post-processing & quality control

**What it does NOT do:**
- ❌ HTTP handling
- ❌ Direct LLM prompt engineering
- ❌ AI decision-making

**Example:**
```python
class SearchSummarizer:
    def __init__(self):
        self.synthesis_agent = SynthesisAgent()
        self.cache_manager = CacheManager()
        self.citation_validator = CitationValidator()

    async def summarize_search(self, request):
        # 1. Check cache
        # 2. Transform data
        # 3. Call agent
        # 4. Validate results
        # 5. Cache results
        # 6. Return
```



## Part 4: Layer 3 - Agent Layer

**Location:** `src/agents/`

**Responsibility:** AI-powered intelligence

**What it does:**
- ✅ LLM interactions (Groq, OpenAI, etc.)
- ✅ Prompt engineering
- ✅ Multi-step reasoning
- ✅ Tool orchestration
- ✅ Intelligent synthesis

**What it does NOT do:**
- ❌ Caching
- ❌ HTTP handling
- ❌ Business rules
- ❌ Data validation

**Example:**
```python
class SynthesisAgent:
    def synthesize_search_results(self, query, articles):
        # 1. Prepare article summaries
        # 2. Generate AI synthesis via LLM
        # 3. Extract findings
        # 4. Generate follow-ups
        # 5. Return structured response
```


## Part 5: Understanding Agents vs Tools

### **Tools = Individual Capabilities**
- Single-purpose functions
- No decision-making
- Deterministic execution
- Example: `synthesize_articles()`, `generate_follow_up_questions()`

### **Agents = Intelligent Orchestrators**
- Multi-step reasoning (powered by LLM)
- Decides which tools to use
- Adapts based on context
- Example: `SynthesisAgent.agent_executor`

### **Analogy:**
- **Tools** = Kitchen appliances (blender, oven, knife)
- **Agents** = Chef (decides when/how to use each appliance)


## Part 6: Data Flow Example - Search Summary

```
1. User Request
   ↓
2. API Layer (/api/v1/search.py)
   - Validates SearchSummaryRequest
   - Calls service
   ↓
3. Service Layer (services/search_summarizer.py)
   - Checks cache (HIT? Return cached)
   - Validates input (empty articles?)
   - Transforms data (ArticleResult → dict)
   - Calls agent
   ↓
4. Agent Layer (agents/synthesis_agent.py)
   - Prepares article summaries
   - Generates LLM synthesis
   - Extracts structured findings
   - Returns SearchSummaryResponse
   ↓
5. Service Layer (continued)
   - Validates citations
   - Caches result
   - Returns to API
   ↓
6. API Layer
   - Returns HTTP response to client
```


## Part 7: Key Design Principles

### 1. **Separation of Concerns**
Each layer has ONE clear responsibility

### 2. **Single Responsibility Principle**
- API = HTTP interface
- Service = Business logic
- Agent = AI intelligence

### 3. **Dependency Flow**
```
API → Service → Agent
(Never skip layers!)
```

### 4. **Testability**
Each layer can be tested independently with mocks

### 5. **Reusability**
Services and agents can be used from multiple API endpoints


## Part 8: Other Key Components Explained

### **Models** (`src/models/`)
Pydantic models for data validation and serialization

### **Utilities** (`src/utilities/`)
Reusable helper functions

### **Backend** (`src/backend/`)
External service integrations
- `GROQ_CHAT`: Groq LLM client pool
- Database connections
- Third-party APIs

## Part 9: LangChain Integration

### What is LangChain?
Framework for building LLM applications with:
- **Tools**: Callable functions agents can use
- **Agents**: LLM-powered decision makers
- **Chains**: Sequential operations
- **LangGraph**: Build stateful, multi-step workflows

### How We Use It:
```python
# Our agent exposes LangChain-compatible interfaces
agent = SynthesisAgent()

# Get as LangChain tools
tools = agent.get_tools()

# Get as agent executor (for LangGraph)
executor = agent.agent_executor

# Use in LangGraph workflows
runnable = agent.as_runnable()
```

## Part 10: Error Handling 

### Multi-Layer Error Handling:

**Agent Layer:**
```python
try:
    result = json.loads(llm_response)
except json.JSONDecodeError:
    # Fallback to json5 parser
    # Last resort: return error structure
```

**Service Layer:**
```python
try:
    result = agent.synthesize_search_results(...)
except GroqAPIError:
    # Retry with backoff
    # Use fallback model
except Exception:
    # Log for monitoring
    # Return graceful error
```

**API Layer:**
```python
try:
    result = service.summarize_search(request)
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```


## Part 11: Development Workflow

### Adding a New Feature:

1. **Define Models** (`src/models/`)
   - Request/Response Pydantic models

2. **Build Agent Logic** (`src/agents/`)
   - Implement AI-powered functionality
   - Add LLM interactions

3. **Create Service** (`src/services/`)
   - Add validation
   - Orchestrate agent calls

4. **Add API Endpoint** (`src/api/`)
   - Create route
   - Call service
   - Return response

5. **Write Tests**
   - Unit tests for agent
   - Integration tests for service
   - E2E tests for API


## Part 12: Best Practices

### DO 
- Keep layers separated
- Use type hints everywhere
- Log important operations
- Cache expensive operations
- Validate at boundaries
- Write tests for each layer
- Use Pydantic models for data

### DON'T 
- Skip layers (API → Agent directly)
- Put business logic in API endpoints
- Put caching logic in agents
- Mix HTTP concerns with AI logic
- Hardcode configuration
- Ignore errors silently


## Part 13: Common Pattern for Adding New Functionality

```python
# 1. Create agent class
class MyNewAgent:
    def __init__(self):
        self.llm = GROQ_CHAT.get_client(...)

    def do_something(self, input):
        # Agent logic
        return result

# 2. Create service
class MyService:
    def __init__(self):
        self.agent = MyNewAgent()

    def execute(self, request):
        # Check cache
        # Call agent
        # Process result

# 3. Add API endpoint
@router.post("/my-endpoint")
async def my_endpoint(request: MyRequest):
    service = MyService()
    return service.execute(request)
```
