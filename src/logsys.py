import logging
import logging.config
import os

import obs_context

_override = os.getenv("FASTAPI_DEBUG", "false").lower() in ["true", "1", "yes"]


def override_level(level: str):
    global _override
    if _override:
        return "DEBUG"
    return level


def log_format() -> str:
    """`json` emits one object per line and preserves `extra={...}` fields the
    text formatter silently drops. Default stays `text`."""
    return (os.getenv("LOG_FORMAT", "text") or "text").strip().lower()


def configure():
    _json = log_format() == "json"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {"()": "obs_context.JsonFormatter"},
                "standard": {
                    "()": "obs_context.ContextTextFormatter",
                    "format": "[%(asctime)s] %(name)s - %(levelname)s - [%(request_id)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "simple": {"format": "%(name)s:%(levelname)s:%(message)s"},
                "uvicorn": {
                    "()": "obs_context.ContextTextFormatter",
                    "format": "%(levelname)s: [%(request_id)s] %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "level": override_level("INFO"),
                    "formatter": "json" if _json else "standard",
                    "stream": "ext://sys.stdout",
                },
                "uvicorn_access": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "json" if _json else "uvicorn",
                    "stream": "ext://sys.stdout",
                },
                "uvicorn_error": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "json" if _json else "standard",
                    "stream": "ext://sys.stderr",
                },
            },
            "root": {
                "level": override_level("INFO"),
                "handlers": ["default"],
            },
            "loggers": {
                "uvicorn": {
                    "level": override_level("INFO"),
                    "handlers": ["uvicorn_error"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": override_level("INFO"),
                    "handlers": ["uvicorn_access"],
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": override_level("INFO"),
                    "handlers": ["uvicorn_error"],
                    "propagate": False,
                },
                "httpx": {"level": override_level("WARNING"), "propagate": False},
                "urllib3": {"level": override_level("WARNING"), "propagate": False},
                "fastapi": {"level": override_level("INFO"), "propagate": True},
            },
        }
    )