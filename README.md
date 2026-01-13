# FoodScholar

## Overview

FoodScholar is an AI-powered scientific literature application that helps users discover and understand research about food and nutrition in an accessible and democritized way. 
## Architecture

### Three-Layer Architecture

FoodScholar follows a clean, layered architecture that separates concerns and promotes maintainability:

```
┌─────────────────────────────────────────────────────────┐
│                 Layer 1: API LAYER                      │
│              FastAPI HTTP Interface                     │
│  • Request/Response handling                            │
│  • Input validation (Pydantic)                          │
│  • Authentication & rate limiting                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 2: SERVICE LAYER                     │
│           Business Logic Orchestration                  │
│  • Caching strategy                                     │
│  • Data transformation                                  │
│  • Multi-component orchestration                        │
│  • Validation & error handling                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Layer 3: AGENT LAYER                      │
│          AI-Powered Intelligence (LangChain)            │
│  • LLM interactions (Groq)                              │
│  • Prompt engineering                                   │
│  • Multi-step reasoning                                 │
│  • Tool orchestration                                   │
└─────────────────────────────────────────────────────────┘
```

**Key Principles:**
- **Separation of Concerns**: Each layer has one clear responsibility
- **Dependency Flow**: API → Service → Agent (never skip layers)
- **Testability**: Each layer can be tested independently
- **Reusability**: Services and agents can be used across multiple endpoints

**[View Full Architecture Presentation](docs/ARCHITECTURE_PRESENTATION.md)** - Detailed slides for team onboarding

### Directory Structure

```
foodscholar/
├── src/
│   ├── app.py                          # Main FastAPI application
│   │
│   ├── api/                            # Layer 1: HTTP Interface
│   │   └── v1/
│   │       ├── search.py               # Search endpoints
│   │       └── sessions.py             # Session/chat endpoints
│   │
│   ├── services/                       # Layer 2: Business Logic
│   │   └── search_summarizer.py        # Search orchestration & caching
│   │
│   ├── agents/                         # Layer 3: AI Intelligence
│   │   ├── synthesis_agent.py          # Multi-document synthesis (LangChain)
│   │   └── search_agent.py             # Search agent (example)
│   │
│   ├── models/                         # Data Models (Pydantic)
│   │   ├── search.py                   # Search request/response models
│   │   ├── article.py                  # Article models
│   │   └── session.py                  # Session models
│   │
│   ├── utilities/                      # Utilities & Helpers
│   │   ├── cache.py                    # Redis caching manager
│   │   ├── citation_validator.py       # Citation validation
│   │   └── chunking.py                 # Text chunking
│   │
│   └── backend/                        # External Integrations
│       ├── groq.py                     # Groq LLM client pool
│       ├── elastic.py                  # Elasticsearch client
│       ├── redis.py                    # Redis client
│       └── platform.py                 # WiseFood API client
│
├── tests/                              # Test suite
├── docs/                               # Documentation
│   └── ARCHITECTURE_PRESENTATION.md    # Team onboarding slides
├── examples/                           # Code examples
│   └── synthesis_agent_langgraph_example.py
└── docker-compose.yaml                 # Docker services
```

### Understanding Agents vs Tools vs Services

**🤖 Agents** = Intelligent decision-makers powered by LLMs
- Make decisions about which actions to take
- Chain multiple operations together
- Adapt based on context
- Example: `SynthesisAgent` decides how to synthesize articles

**🔧 Tools** = Individual capabilities agents can use
- Single-purpose functions
- No decision-making
- Wrapped for LangChain compatibility
- Example: `synthesize_articles()`, `generate_follow_up_questions()`

**⚙️ Services** = Business logic orchestrators
- Manage caching, validation, error handling
- Coordinate multiple agents/components
- Transform data between layers
- Example: `SearchSummarizer` manages the full search workflow

**Analogy:**
- Tools = Kitchen appliances (blender, oven)
- Agents = Chef (decides which appliances to use)
- Services = Restaurant manager (coordinates kitchen, handles orders, manages inventory)
