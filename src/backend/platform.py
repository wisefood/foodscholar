import os
import threading
from wisefood import Client, DataClient

class WisefoodClientSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(WisefoodClientSingleton, cls).__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, api_key: str):
        self.api_key = api_key
        # Initialize other attributes or clients as needed

    def get_api_key(self) -> str:
        return self.api_key