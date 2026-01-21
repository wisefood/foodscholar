"""Wisefood DataClient connection pool for efficient client instance management."""
import os
import logging
from typing import Optional
from threading import Lock
from wisefood import DataClient, Credentials
from config import config
logger = logging.getLogger(__name__)


class DataClientPool:
    """
    Connection pool for DataClient instances.

    Maintains reusable DataClient instances to avoid creating new connections
    for every request.
    """

    def __init__(self, pool_size: int = 5):
        """
        Initialize the connection pool.
        
        Args:
            pool_size: Number of connections to maintain (default: 5)
        """
        self._pool = []
        self._lock = Lock()
        self._pool_size = pool_size
        self._available = []
        
        # Initialize pool with default connections
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the pool with default connections."""
        try:
            for _ in range(self._pool_size):
                client = DataClient(
                    base_url=config.settings["DATA_API_URL"],
                    credentials=Credentials(
                        client_id=config.settings["KEYCLOAK_CLIENT_ID"],
                        client_secret=config.settings["KEYCLOAK_CLIENT_SECRET"]
                    )
                )
                self._pool.append(client)
                self._available.append(client)
            logger.info(f"Initialized DataClient pool with {self._pool_size} connections")
        except Exception as e:
            logger.error(f"Failed to initialize DataClient pool: {e}")
            raise

    def get_client(self) -> DataClient:
        """
        Get an available DataClient from the pool.

        Returns:
            DataClient instance from the pool
        """
        with self._lock:
            if self._available:
                return self._available.pop()
            else:
                logger.warning("No available clients in pool, creating new one")
                return self._pool[0]  # Return first if all busy

    def return_client(self, client: DataClient):
        """
        Return a client to the pool.

        Args:
            client: DataClient instance to return
        """
        with self._lock:
            if client in self._pool and client not in self._available:
                self._available.append(client)

    def get_pool_size(self) -> int:
        """Get the pool size."""
        with self._lock:
            return len(self._available)

    def clear_pool(self):
        """Clear all connections from the pool."""
        with self._lock:
            self._pool.clear()
            self._available.clear()
            logger.info("Cleared DataClient pool")


# Global connection pool instance
WISEFOOD = DataClientPool()