"""Test script for Groq connection pool."""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.groq import GROQ_CHAT


def test_connection_pool():
    """Test the Groq connection pool functionality."""
    print("Testing Groq Connection Pool")
    print("=" * 50)

    # Clear pool to start fresh
    GROQ_CHAT.clear_pool()
    print("\n1. Cleared pool")
    pool_size = GROQ_CHAT.get_pool_size()
    print(f"   Pool size: {pool_size}")
    assert pool_size == 0, "Pool should be empty"

    # Get first client
    print("\n2. Getting first client (llama-3.3-70b, temp=0.7)")
    llm1 = GROQ_CHAT.get_client(model="llama-3.3-70b-versatile", temperature=0.7)
    pool_size = GROQ_CHAT.get_pool_size()
    print(f"   Pool size: {pool_size}")
    print(f"   Connections: {GROQ_CHAT.get_pool_keys()}")
    assert pool_size == 1, "Pool should have 1 connection"

    # Get same configuration - should reuse
    print("\n3. Getting same client configuration (should reuse)")
    llm2 = GROQ_CHAT.get_client(model="llama-3.3-70b-versatile", temperature=0.7)
    pool_size = GROQ_CHAT.get_pool_size()
    print(f"   Pool size: {pool_size}")
    print(f"   Same instance? {llm1 is llm2}")
    assert llm1 is llm2, "Should return same instance from pool"
    assert pool_size == 1, "Pool should still have 1 connection"

    # Get different configuration - should create new
    print("\n4. Getting different client configuration (new instance)")
    llm3 = GROQ_CHAT.get_client(model="llama-3.3-70b-versatile", temperature=0.3)
    pool_size = GROQ_CHAT.get_pool_size()
    print(f"   Pool size: {pool_size}")
    print(f"   Connections: {GROQ_CHAT.get_pool_keys()}")
    print(f"   Different from llm1? {llm1 is not llm3}")
    assert llm1 is not llm3, "Should be different instance"
    assert pool_size == 2, "Pool should have 2 connections"

    # Get another different configuration
    print("\n5. Getting another different model (new instance)")
    llm4 = GROQ_CHAT.get_client(model="mixtral-8x7b-32768", temperature=0.0)
    pool_size = GROQ_CHAT.get_pool_size()
    print(f"   Pool size: {pool_size}")
    print(f"   Connections: {GROQ_CHAT.get_pool_keys()}")
    assert pool_size == 3, "Pool should have 3 connections"

    # Reuse llm3's configuration
    print("\n6. Reusing llm3's configuration (should reuse)")
    llm5 = GROQ_CHAT.get_client(model="llama-3.3-70b-versatile", temperature=0.3)
    pool_size = GROQ_CHAT.get_pool_size()
    print(f"   Pool size: {pool_size}")
    print(f"   Same as llm3? {llm3 is llm5}")
    assert llm3 is llm5, "Should return same instance as llm3"
    assert pool_size == 3, "Pool should still have 3 connections"

    # Clear pool
    print("\n7. Clearing pool")
    GROQ_CHAT.clear_pool()
    pool_size = GROQ_CHAT.get_pool_size()
    print(f"   Pool size: {pool_size}")
    assert pool_size == 0, "Pool should be empty after clear"

    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("\nConnection Pool Benefits:")
    print("  • Reuses existing ChatGroq instances")
    print("  • Avoids connection overhead on every request")
    print("  • Thread-safe for concurrent access")
    print("  • Automatic configuration-based pooling")


if __name__ == "__main__":
    try:
        test_connection_pool()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
