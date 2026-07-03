"""Redis-backed chat session store with TTL.

Replaces the module-level in-memory dicts that previously held all session
state in api/v1/sessions.py. Those dicts leaked memory (no TTL/eviction),
lost every session on restart, and broke with more than one replica because
a session only existed on the pod that created it.

Each session is a single JSON document:

    {
        "user_id":      str,
        "created_at":   iso8601 str,
        "last_active":  iso8601 str,
        "title":        str | None,
        "user_context": str,
        "messages":     [{"type": "human" | "ai", "content": str}, ...],
    }

A per-user Redis set indexes session ids for listing. Session keys carry a
sliding TTL (SESSION_TTL_SECONDS, refreshed on every save) so idle sessions —
including those of ephemeral guest users — are reaped automatically. The
user index is pruned lazily when listed.

If Redis is unreachable at startup the store degrades to a process-local
dict (single-replica development only; a warning is logged).
"""
import json
import logging
import threading
from typing import Dict, List, Optional

from config import config

logger = logging.getLogger(__name__)

_SESSION_KEY = "sessions:data:{session_id}"
_USER_INDEX_KEY = "sessions:by-user:{user_id}"


class SessionStore:
    def __init__(self):
        self._ttl = int(config.settings.get("SESSION_TTL_SECONDS", 7 * 24 * 3600))
        self._redis = None
        self._local: Dict[str, dict] = {}
        self._local_users: Dict[str, set] = {}
        self._lock = threading.Lock()
        try:
            from backend.redis import RedisClientSingleton

            client = RedisClientSingleton().client
            client.ping()
            self._redis = client
            logger.info("Session store using Redis (TTL %ss)", self._ttl)
        except Exception as e:
            logger.warning(
                "Redis unavailable for session store (%s) — falling back to "
                "process-local storage. Sessions will not survive restarts "
                "and multi-replica deployments will misbehave.",
                e,
            )

    # ------------------------------------------------------------------ #
    # CRUD                                                                #
    # ------------------------------------------------------------------ #

    def get(self, session_id: str) -> Optional[dict]:
        if self._redis is not None:
            raw = self._redis.get(_SESSION_KEY.format(session_id=session_id))
            if raw is None:
                return None
            try:
                return json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                logger.error("Corrupt session document for %s — dropping", session_id)
                self._redis.delete(_SESSION_KEY.format(session_id=session_id))
                return None
        with self._lock:
            data = self._local.get(session_id)
            return json.loads(json.dumps(data)) if data is not None else None

    def save(self, session_id: str, data: dict) -> None:
        """Persist a session document and refresh its TTL and user index."""
        user_id = data.get("user_id")
        if self._redis is not None:
            self._redis.setex(
                _SESSION_KEY.format(session_id=session_id),
                self._ttl,
                json.dumps(data),
            )
            if user_id:
                index_key = _USER_INDEX_KEY.format(user_id=user_id)
                self._redis.sadd(index_key, session_id)
                self._redis.expire(index_key, self._ttl)
            return
        with self._lock:
            self._local[session_id] = json.loads(json.dumps(data))
            if user_id:
                self._local_users.setdefault(user_id, set()).add(session_id)

    def delete(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a session document. Returns True if it existed."""
        if self._redis is not None:
            existed = bool(self._redis.delete(_SESSION_KEY.format(session_id=session_id)))
            if user_id:
                self._redis.srem(_USER_INDEX_KEY.format(user_id=user_id), session_id)
            return existed
        with self._lock:
            existed = self._local.pop(session_id, None) is not None
            if user_id and user_id in self._local_users:
                self._local_users[user_id].discard(session_id)
                if not self._local_users[user_id]:
                    del self._local_users[user_id]
            return existed

    def list_user_sessions(self, user_id: str) -> List[str]:
        """List live session ids for a user, pruning expired ones from the index."""
        if self._redis is not None:
            index_key = _USER_INDEX_KEY.format(user_id=user_id)
            session_ids = self._redis.smembers(index_key) or set()
            live = []
            for sid in session_ids:
                if self._redis.exists(_SESSION_KEY.format(session_id=sid)):
                    live.append(sid)
                else:
                    self._redis.srem(index_key, sid)
            return sorted(live)
        with self._lock:
            return sorted(self._local_users.get(user_id, set()))


SESSION_STORE = SessionStore()
