"""Caching utilities for FoodScholar."""
import hashlib
import json
import logging
from typing import Optional, Any, Dict
from datetime import timedelta
from backend.redis import RedisClientSingleton

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for search summaries, translations, and other expensive operations."""

    def __init__(self, enabled: bool = True, default_ttl: int = 604800):
        """
        Initialize cache manager.

        Args:
            enabled: Whether caching is enabled
            default_ttl: Default TTL in seconds (default: 7 days)
        """
        self.enabled = enabled
        self.default_ttl = default_ttl
        self.redis_client = None

        if self.enabled:
            try:
                self.redis_client = RedisClientSingleton().client
                logger.info("Cache manager initialized with Redis")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis client: {e}")
                self.enabled = False

    def generate_cache_key(
        self, prefix: str, data: Dict[str, Any], exclude_keys: list = None
    ) -> str:
        """
        Generate a cache key from data.

        Args:
            prefix: Key prefix (e.g., 'search_summary', 'translation')
            data: Data to hash
            exclude_keys: Keys to exclude from hashing (e.g., 'user_id')

        Returns:
            Cache key string
        """
        # Create a copy and exclude specified keys
        cache_data = data.copy()
        if exclude_keys:
            for key in exclude_keys:
                cache_data.pop(key, None)

        # Sort keys for consistent hashing
        sorted_data = json.dumps(cache_data, sort_keys=True)
        hash_digest = hashlib.md5(sorted_data.encode()).hexdigest()

        return f"{prefix}:{hash_digest}"

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            logger.debug(f"Cache miss: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (None = use default)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            self.redis_client.delete(key)
            logger.debug(f"Cache deleted: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.

        Args:
            pattern: Redis pattern (e.g., 'search_summary:*')

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache pattern: {e}")
            return 0

    def get_or_compute(
        self, key: str, compute_func: callable, ttl: Optional[int] = None
    ) -> Any:
        """
        Get value from cache or compute if not found.

        Args:
            key: Cache key
            compute_func: Function to call if cache miss
            ttl: TTL for computed value

        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute value
        value = compute_func()

        # Cache the result
        self.set(key, value, ttl)

        return value


# Predefined TTLs for different data types
TTL_SEARCH_SUMMARY = 604800  # 7 days
TTL_TRANSLATION = 2592000  # 30 days
TTL_ARTICLE_CHAT = 86400  # 1 day
TTL_METADATA = 3600  # 1 hour


def get_cache_manager(enabled: bool = True) -> CacheManager:
    """Factory function to get cache manager instance."""
    return CacheManager(enabled=enabled)
