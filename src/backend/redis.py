"""Redis client singleton for caching."""
import redis
import threading
import logging
import json
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


class RedisClientSingleton:
    """Thread-safe singleton Redis client."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize()
                    cls._instance = instance
        return cls._instance

    def _initialize(self):
        """Initialize Redis connection pool."""
        redis_host = config.settings["REDIS_HOST"]
        redis_port = config.settings["REDIS_PORT"]

        self._pool = redis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=5,
            decode_responses=True,
            max_connections=10,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        logger.info(f"Initialized Redis connection pool at {redis_host}:{redis_port}")

    @property
    def client(self) -> redis.Redis:
        """Get Redis client instance."""
        return self._client

    def set(self, key: str, value: any, ex: Optional[int] = None):
        """
        Set a value in Redis.

        Args:
            key: Redis key
            value: Value to store (will be JSON serialized if dict)
            ex: Expiration time in seconds
        """
        try:
            if isinstance(value, dict):
                value = json.dumps(value)
            self._client.set(key, value, ex=ex)
        except Exception as e:
            logger.error(f"Error setting Redis key {key}: {e}")
            raise

    def setex(self, key: str, time: int, value: any):
        """
        Set a value with expiration time.

        Args:
            key: Redis key
            time: Expiration time in seconds
            value: Value to store
        """
        try:
            if isinstance(value, dict):
                value = json.dumps(value)
            self._client.setex(key, time, value)
        except Exception as e:
            logger.error(f"Error setting Redis key {key} with expiration: {e}")
            raise

    def get(self, key: str) -> Optional[any]:
        """
        Get a value from Redis.

        Args:
            key: Redis key

        Returns:
            Value (JSON deserialized if applicable) or None
        """
        try:
            value = self._client.get(key)
            if value is None:
                return None

            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return value
        except Exception as e:
            logger.error(f"Error getting Redis key {key}: {e}")
            return None

    def delete(self, key: str) -> int:
        """
        Delete a key from Redis.

        Args:
            key: Redis key

        Returns:
            Number of keys deleted
        """
        try:
            return self._client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting Redis key {key}: {e}")
            raise

    def keys(self, pattern: str) -> list:
        """
        Get all keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., 'search_summary:*')

        Returns:
            List of matching keys
        """
        try:
            return self._client.keys(pattern)
        except Exception as e:
            logger.error(f"Error getting Redis keys with pattern {pattern}: {e}")
            return []

    def exists(self, key: str) -> bool:
        """
        Check if a key exists.

        Args:
            key: Redis key

        Returns:
            True if key exists, False otherwise
        """
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error(f"Error checking Redis key existence {key}: {e}")
            return False


# Legacy compatibility
class RedisClient:
    """Legacy Redis client for backward compatibility."""

    _pool = None

    @classmethod
    def _initialize_redis(cls):
        redis_host = config.settings["REDIS_HOST"]
        redis_port = config.settings["REDIS_PORT"]

        cls._pool = redis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=2,
            decode_responses=True,
            max_connections=10,
        )
        logger.info("Initialized Redis connection pool")
        return cls._pool

    def set(self, key, value):
        """Set a value in Redis using a connection from the pool."""
        try:
            if self._pool is None:
                self._initialize_redis()
            conn = redis.Redis(connection_pool=self._pool)
            if isinstance(value, dict):
                conn.set(key, json.dumps(value))
            else:
                conn.set(key, value)
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            raise

    def get(self, key):
        """Get a value from Redis using a connection from the pool."""
        try:
            if self._pool is None:
                self._initialize_redis()
            conn = redis.Redis(connection_pool=self._pool)
            value = conn.get(key)
            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return value
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            raise

    def delete(self, key):
        """Delete a value from Redis using a connection from the pool."""
        try:
            if self._pool is None:
                self._initialize_redis()
            conn = redis.Redis(connection_pool=self._pool)
            conn.delete(key)
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            raise


# Create legacy singleton instance
REDIS = RedisClient()
