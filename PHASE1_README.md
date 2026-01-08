# FoodScholar Phase 1: Search Result Summaries

## Overview

Phase 1 implements AI-powered search result summarization with citation tracking. This feature allows users to get comprehensive, synthesized summaries from multiple scientific articles with proper attribution.

## Architecture

### Directory Structure

```
src/
├── app.py                              # Main FastAPI application
├── api/
│   └── v1/
│       ├── search.py                   # Search summary endpoints
│       └── sessions.py                 # Session/chat endpoints (refactored)
├── models/
│   ├── search.py                       # Search summary data models
│   ├── article.py                      # Article models (Phase 2 prep)
│   └── session.py                      # Session models
├── services/
│   └── search_summarizer.py            # Search summarization orchestration
├── agents/
│   └── synthesis_agent.py              # Multi-document synthesis LLM agent
├── utils/
│   ├── cache.py                        # Caching utilities
│   ├── citation_validator.py           # Citation validation & tracking
│   └── chunking.py                     # Article text chunking
└── backend/
    ├── elastic.py                      # Elasticsearch client
    ├── redis.py                        # Redis client (updated)
    └── platform.py                     # WiseFood API client
```

## Key Features

### 1. Search Summarization (`POST /api/v1/search/summarize`)

**Input:**
```json
{
  "query": "effects of omega-3 fatty acids on cardiovascular health",
  "filters": {
    "year_from": 2015,
    "categories": ["meta-analysis", "randomized controlled trial"]
  },
  "max_articles": 10,
  "expertise_level": "intermediate"
}
```

**Output:**
- Comprehensive markdown summary synthesized from multiple articles
- Key findings with category classification
- Citations with article URNs, sections, and confidence levels
- Follow-up question suggestions
- Search metadata

**Features:**
- **Expertise-level adaptation**: Summaries adjust complexity for beginner/intermediate/expert users
- **Citation integrity**: Every claim is backed by specific article citations
- **Intelligent caching**: Results cached for 7 days with MD5 hash keys
- **Elasticsearch integration**: Leverages existing article index with faceted search

### 2. Citation System

Each finding includes detailed citations:

```python
Citation(
    article_urn="urn:article:12345",
    article_title="Study Title",
    authors=["Author 1", "Author 2"],
    year=2020,
    journal="Journal Name",
    section="results",
    quote="Direct quote if applicable",
    confidence="high",
    relevance_score=0.95
)
```

**Validation:**
- Ensures all findings have citations
- Tracks citation diversity
- Validates citation metadata completeness
- Flags low-confidence findings

### 3. Caching Layer

**Benefits:**
- 7-day TTL for search summaries
- MD5 hash-based cache keys
- Excludes user_id from cache key for sharing
- Pattern-based cache clearing

**Usage:**
```python
# Clear all search summary caches
DELETE /api/v1/search/cache/clear?pattern=search_summary:*
```

## API Endpoints

### Search Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/search/summarize` | Summarize search results |
| GET | `/api/v1/search/trending` | Get trending article summaries |
| DELETE | `/api/v1/search/cache/clear` | Clear cached summaries |
| GET | `/api/v1/search/health` | Health check |

### Session Endpoints (Refactored)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/sessions/start` | Start new chat session |
| POST | `/api/v1/sessions/chat` | Send chat message |
| GET | `/api/v1/sessions/users/{user_id}` | Get user sessions |
| GET | `/api/v1/sessions/{session_id}/history` | Get session history |
| DELETE | `/api/v1/sessions/{session_id}` | Delete session |

## Environment Variables

```bash
# Required
GROQ_API_KEY=your_groq_api_key          # For LLM synthesis
ELASTIC_HOST=http://elasticsearch:9200  # Elasticsearch endpoint
REDIS_HOST=redis                        # Redis for caching
REDIS_PORT=6379

# Optional
CACHE_ENABLED=true                      # Enable/disable caching
WISEFOOD_API_KEY=your_key              # WiseFood platform API key
ES_DIM=384                             # Embedding dimension
```

## How It Works

### Search Summary Flow

1. **Request Reception**: User submits query with filters
2. **Cache Check**: Check if summary exists in cache
3. **Article Search**: Query Elasticsearch with multi-match on title, abstract, keywords, content
4. **Synthesis**: LLM analyzes articles and generates structured findings
5. **Citation Extraction**: Extract citations from LLM response
6. **Validation**: Validate citation integrity
7. **Caching**: Store result in Redis
8. **Response**: Return summary with citations

### Synthesis Agent Prompt Strategy

The synthesis agent uses a structured prompt that:
- Adjusts language complexity based on expertise level
- Enforces strict citation requirements
- Distinguishes between individual study findings and consensus
- Notes contradictions and limitations
- Provides confidence levels for each finding
- Returns structured JSON with findings and supporting citations

### Citation Validator

Ensures quality by:
- Checking all findings have citations
- Calculating citation diversity (spread across articles)
- Validating metadata completeness
- Filtering by minimum confidence threshold

## Testing the Implementation

### 1. Start the Server

```bash
cd src
python app.py
```

Server runs on `http://localhost:8000`

### 2. Test Search Summary

```bash
curl -X POST http://localhost:8000/api/v1/search/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "query": "benefits of probiotics",
    "max_articles": 5,
    "expertise_level": "beginner"
  }'
```

### 3. Check API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI

### 4. Health Checks

```bash
# Main health check
curl http://localhost:8000/health

# Search service health
curl http://localhost:8000/api/v1/search/health
```

## Configuration Notes

### Elasticsearch Article Index

The implementation assumes an `articles` index with fields:
- `title` (text, boost: 3)
- `abstract` (text, boost: 2)
- `keywords` (text, boost: 2)
- `content` (text)
- `year` (integer)
- `journal` (keyword)
- `category` (keyword)
- `tags` (keyword array)
- `authors` (text array)
- `doi` (keyword)

### Cache TTLs

Defined in `utils/cache.py`:
- `TTL_SEARCH_SUMMARY`: 7 days (604800s)
- `TTL_TRANSLATION`: 30 days (2592000s)
- `TTL_ARTICLE_CHAT`: 1 day (86400s)
- `TTL_METADATA`: 1 hour (3600s)

## Known Limitations & Future Work

### Current Limitations

1. **Elasticsearch Index**: Assumes `articles` index exists. Need to verify index name and schema.
2. **WiseFood API Integration**: Not fully integrated. Currently uses direct Elasticsearch queries.
3. **Trending Articles**: Endpoint placeholder - needs trending detection algorithm.
4. **No Authentication**: API is open. Need to add auth middleware.
5. **Error Handling**: Basic error handling. Need more granular exception types.

### Planned Enhancements (Phase 2+)

- **Chat with Article**: Article-specific Q&A with RAG
- **Translation**: Full article translation
- **Auto-enhancement**: Background jobs for metadata enrichment
- **Vector Search**: Hybrid search combining keyword + semantic similarity
- **User Personalization**: Track user expertise and adjust responses
- **Citation Linking**: Deep links to specific article sections
- **Export**: PDF/Word export of summaries

## Code Quality Notes

### Type Safety
- All models use Pydantic for validation
- Type hints throughout codebase
- Strict schema enforcement

### Error Handling
- Logging at INFO/ERROR levels
- HTTPException with appropriate status codes
- Graceful degradation when cache unavailable

### Documentation
- Docstrings on all classes and functions
- OpenAPI/Swagger auto-generated from FastAPI
- Inline comments for complex logic

## Migration from Old Code

The refactoring maintains backward compatibility:

1. **Session endpoints**: Moved from `app.py` to `api/v1/sessions.py`
2. **Models**: Extracted to `models/session.py`
3. **In-memory storage**: Still uses in-memory dicts (future: move to Postgres)
4. **Legacy support**: Old endpoints still work at same paths

## Next Steps

1. **Verify Elasticsearch Schema**: Check actual article index structure
2. **Test with Real Data**: Run queries against production data
3. **Integrate WiseFood API**: Use WiseFood client instead of direct ES queries
4. **Add Authentication**: Implement JWT/OAuth
5. **Monitoring**: Add metrics and logging
6. **Deploy**: Containerize and deploy to staging

## Questions?

For issues or questions about Phase 1:
- Check logs: `src/` directory will have application logs
- API docs: `http://localhost:8000/docs`
- Code: See inline documentation in each module
