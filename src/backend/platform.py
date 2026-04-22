"""Wisefood DataClient connection pool for efficient client instance management."""
import os
import logging
from typing import Optional
from threading import Lock
from wisefood import DataClient, Credentials
from wisefood.api_client import Client, Credentials as PlatformCredentials
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


class ClientPool:
    """
    Connection pool for the WiseFood platform Client.

    This is intentionally separate from DataClientPool. DataClient talks to the
    data-catalog API, while Client talks to the WiseFood platform API for
    systemic resources such as members and households.
    """

    def __init__(self, pool_size: int = 5):
        """
        Initialize the connection pool.

        Args:
            pool_size: Number of platform clients to maintain (default: 5)
        """
        self._pool = []
        self._lock = Lock()
        self._pool_size = pool_size
        self._available = []
        self._initialized = False

    def _initialize_pool(self):
        """Initialize the pool with WiseFood platform clients."""
        if self._initialized:
            return
        try:
            for _ in range(self._pool_size):
                client = Client(
                    base_url=config.settings["WISEFOOD_PLATFORM_API_URL"],
                    credentials=PlatformCredentials(
                        client_id=config.settings["KEYCLOAK_CLIENT_ID"],
                        client_secret=config.settings["KEYCLOAK_CLIENT_SECRET"],
                    ),
                )
                self._pool.append(client)
                self._available.append(client)
            logger.info(
                "Initialized WiseFood platform Client pool with %d connections",
                self._pool_size,
            )
            self._initialized = True
        except Exception as e:
            logger.error("Failed to initialize WiseFood platform Client pool: %s", e)
            raise

    def get_client(self) -> Client:
        """
        Get an available WiseFood platform Client from the pool.

        Returns:
            WiseFood platform Client instance from the pool.
        """
        with self._lock:
            if not self._initialized:
                self._initialize_pool()
            if self._available:
                return self._available.pop()
            logger.warning("No available platform clients in pool, reusing first one")
            return self._pool[0]

    def return_client(self, client: Client):
        """
        Return a platform client to the pool.

        Args:
            client: WiseFood platform Client instance to return.
        """
        with self._lock:
            if client in self._pool and client not in self._available:
                self._available.append(client)

    def get_pool_size(self) -> int:
        """Get the number of available platform clients."""
        with self._lock:
            return len(self._available)

    def clear_pool(self):
        """Clear all platform clients from the pool."""
        with self._lock:
            self._pool.clear()
            self._available.clear()
            self._initialized = False
            logger.info("Cleared WiseFood platform Client pool")


# Global connection pool instance
WISEFOOD = DataClientPool()
WISEFOOD_PLATFORM = ClientPool()
