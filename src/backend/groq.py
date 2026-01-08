"""Groq connection pool for efficient ChatGroq instance management."""
import os
import logging
from typing import Optional, Dict
from threading import Lock
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


class GroqConnectionPool:
    """
    Connection pool for ChatGroq instances.

    Maintains reusable ChatGroq instances keyed by configuration
    to avoid instantiating new connections for every request.
    """

    def __init__(self):
        """Initialize the connection pool."""
        self._pool: Dict[str, ChatGroq] = {}
        self._lock = Lock()
        self._api_key = os.getenv("GROQ_API_KEY")

        if not self._api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")

    def _get_pool_key(
        self,
        model: str,
        temperature: float,
        **kwargs
    ) -> str:
        """
        Generate a unique key for the pool based on configuration.

        Args:
            model: Model name
            temperature: Model temperature
            **kwargs: Additional configuration parameters

        Returns:
            Unique string key for this configuration
        """
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = "_".join(f"{k}={v}" for k, v in sorted_kwargs)

        key = f"{model}_{temperature}"
        if kwargs_str:
            key += f"_{kwargs_str}"

        return key

    def get_client(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ChatGroq:
        """
        Get or create a ChatGroq client from the pool.

        Args:
            model: Groq model name (default: llama-3.3-70b-versatile)
            temperature: Model temperature (default: 0.7)
            api_key: Optional API key override
            **kwargs: Additional ChatGroq configuration parameters

        Returns:
            ChatGroq instance from the pool
        """
        # Use provided API key or fall back to environment variable
        used_api_key = api_key or self._api_key

        # Generate pool key (excluding api_key from key for security)
        pool_key = self._get_pool_key(model, temperature, **kwargs)

        # Thread-safe check and creation
        with self._lock:
            if pool_key not in self._pool:
                logger.info(f"Creating new ChatGroq instance: {pool_key}")

                self._pool[pool_key] = ChatGroq(
                    model=model,
                    temperature=temperature,
                    api_key=used_api_key,
                    **kwargs
                )
            else:
                logger.debug(f"Reusing existing ChatGroq instance: {pool_key}")

        return self._pool[pool_key]

    def clear_pool(self):
        """Clear all cached connections."""
        with self._lock:
            self._pool.clear()
            logger.info("Cleared all ChatGroq connections from pool")

    def get_pool_size(self) -> int:
        """Get the current number of connections in the pool."""
        with self._lock:
            return len(self._pool)

    def get_pool_keys(self) -> list:
        """Get all pool keys (for debugging)."""
        with self._lock:
            return list(self._pool.keys())


# Global connection pool instance
GROQ_CHAT = GroqConnectionPool()
