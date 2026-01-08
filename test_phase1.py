"""Test script for Phase 1 implementation."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing Phase 1 imports...")

try:
    # Test model imports
    print("✓ Testing models...")
    from models.search import (
        Citation,
        SynthesizedFinding,
        SearchSummaryRequest,
        SearchSummaryResponse,
        ArticleMetadata,
        ArticleChunk,
    )
    from models.article import (
        ArticleChatRequest,
        ArticleChatResponse,
        TranslationRequest,
        TranslationResponse,
    )
    from models.session import (
        SessionStartRequest,
        SessionStartResponse,
        ChatRequest,
        ChatResponse,
        FoodFactsResponse,
    )
    print("  ✓ All models imported successfully")

    # Test utility imports
    print("✓ Testing utilities...")
    from utils.cache import CacheManager, get_cache_manager
    from utils.citation_validator import CitationValidator, create_citation_from_article
    from utils.chunking import ArticleChunker, create_article_metadata
    print("  ✓ All utilities imported successfully")

    # Test agent imports
    print("✓ Testing agents...")
    from agents.synthesis_agent import SynthesisAgent
    print("  ✓ Synthesis agent imported successfully")

    # Test service imports
    print("✓ Testing services...")
    from services.search_summarizer import SearchSummarizer
    print("  ✓ Search summarizer imported successfully")

    # Test backend imports
    print("✓ Testing backend...")
    from backend.elastic import ELASTIC_CLIENT
    from backend.redis import RedisClientSingleton, REDIS
    from backend.platform import WisefoodClientSingleton
    print("  ✓ Backend clients imported successfully")

    # Test API router imports
    print("✓ Testing API routers...")
    from api.v1 import search, sessions
    print("  ✓ API routers imported successfully")

    # Test main app
    print("✓ Testing main app...")
    from app import app, config
    print("  ✓ Main app imported successfully")

    print("\n" + "="*60)
    print("✓ ALL IMPORTS SUCCESSFUL!")
    print("="*60)

    # Test model creation
    print("\n✓ Testing model instantiation...")

    test_request = SearchSummaryRequest(
        query="test query",
        max_articles=5,
        expertise_level="intermediate"
    )
    print(f"  ✓ Created SearchSummaryRequest: {test_request.query}")

    test_citation = Citation(
        article_urn="test:123",
        article_title="Test Article",
        section="abstract",
        confidence="high"
    )
    print(f"  ✓ Created Citation: {test_citation.article_title}")

    test_metadata = ArticleMetadata(
        urn="test:456",
        title="Test Metadata"
    )
    print(f"  ✓ Created ArticleMetadata: {test_metadata.title}")

    print("\n✓ Model instantiation successful!")

    # Test utility classes
    print("\n✓ Testing utility classes...")

    validator = CitationValidator()
    print("  ✓ Created CitationValidator")

    chunker = ArticleChunker()
    print("  ✓ Created ArticleChunker")

    cache_manager = get_cache_manager(enabled=False)  # Disabled to avoid Redis requirement
    print("  ✓ Created CacheManager (disabled mode)")

    print("\n" + "="*60)
    print("✓ PHASE 1 IMPLEMENTATION STRUCTURE VERIFIED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Set environment variables (GROQ_API_KEY, ELASTIC_HOST, etc.)")
    print("2. Ensure Elasticsearch is running with 'articles' index")
    print("3. Ensure Redis is running (if CACHE_ENABLED=true)")
    print("4. Run: cd src && python app.py")
    print("5. Visit: http://localhost:8000/docs")
    print("6. Test: POST /api/v1/search/summarize")

except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("\nThis might be due to missing dependencies. Make sure to install:")
    print("  pip install fastapi uvicorn langchain langchain-groq elasticsearch redis pydantic")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
