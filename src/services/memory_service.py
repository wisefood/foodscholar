"""
Consented memory for FoodScholar Q&A — "It seems you love lentils. Remember this?"

FoodScholar's port of FoodChat's consent-first memory: when a member phrases a
durable preference inside a nutrition question ("I'm vegetarian and love
lentils — is that enough protein?"), the extractor detects it, the nudge
policy filters it against what the profile already knows, and the suggestion
rides back on the QAResponse. Nothing is written until the user answers via
``POST /qa/memory`` — acceptance PATCHes the shared member profile with
``source: "foodscholar"`` provenance in ``properties.memory_log``; a decline
lands in ``properties.memory_optouts`` so neither app ever re-asks.

Policy (mirrors FoodChat's MemoryService, deliberately conservative):
  - only explicit, high-confidence statements nudge — EXCEPT allergy hints,
    which nudge at any confidence (safety data demands explicit consent);
  - same-kind dedupe: a dislike of something currently in the LIKES list
    still nudges (it's a contradiction the user should resolve);
  - declined values (memory_optouts, shared with FoodChat) never re-nudge;
  - at most MAX_SUGGESTIONS_PER_TURN per question.

Cross-app payoff: FoodChat plans and personalized tips read the same profile,
so an interest expressed here personalizes everything else with no extra work.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.groq import GROQ_CHAT
from backend.langfuse import build_trace_config
from backend.prompts import QA_MEMORY_EXTRACTOR

logger = logging.getLogger(__name__)

VALID_KINDS = {
    "like", "dislike", "cuisine", "allergy_hint", "goal", "dietary_pattern"
}
MAX_SUGGESTIONS_PER_TURN = 2
EXTRACTOR_MODEL = "llama-3.3-70b-versatile"


class MemoryService:
    """Suggestion policy + consented write-back for durable preferences."""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        """Lazy pooled Groq client (deterministic extraction)."""
        if self._llm is None:
            self._llm = GROQ_CHAT.get_client(
                model=EXTRACTOR_MODEL, temperature=0.0
            )
        return self._llm

    # ------------------------------------------------------------------ #
    # Suggestion (attached to QA responses by the router)                  #
    # ------------------------------------------------------------------ #

    def suggest(self, member_id: str, question: str) -> List[Dict[str, Any]]:
        """Nudge-worthy memory suggestions expressed in this question.

        Best-effort by design: any failure (LLM, profile lookup, parsing)
        returns [] — nudges must never break or slow down an answer path
        that already succeeded.
        """
        try:
            candidates = self._extract(question)
            if not candidates:
                return []
            profile = self._fetch_profile(member_id)
            return self._apply_policy(candidates, profile)
        except Exception as e:
            logger.warning("Memory suggestion failed for %s: %s", member_id, e)
            return []

    def _extract(self, question: str) -> List[Dict[str, Any]]:
        prompt = QA_MEMORY_EXTRACTOR.compile(question=question[:600])
        response = self.llm.invoke(
            prompt,
            config=build_trace_config(
                run_name="qa-memory-extractor",
                tags=["qa", "memory"],
            ),
        )
        text = str(getattr(response, "content", "") or "")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        parsed = json.loads(match.group(0))
        memories = parsed.get("memories", [])
        return memories if isinstance(memories, list) else []

    def _apply_policy(
        self, candidates: List[Dict[str, Any]], profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        prefs = profile.get("nutritional_preferences") or {}
        props = profile.get("properties") or {}
        likes = {str(v).lower() for v in (prefs.get("food_likes") or [])}
        dislikes = {str(v).lower() for v in (prefs.get("food_dislikes") or [])}
        allergies = {str(v).lower() for v in (profile.get("allergies") or [])}
        optouts = {str(v).lower() for v in (props.get("memory_optouts") or [])}
        # dietary_groups holds standing regimens FoodChat already reads.
        patterns = {str(v).lower() for v in (profile.get("dietary_groups") or [])}
        # Goals are stored as {"slug", "label"} dicts under properties.
        goals = {
            str((g or {}).get("slug", "")).lower()
            for g in (props.get("dietary_goals") or [])
        }
        # Same-KIND dedupe only — see the module docstring.
        known_by_kind = {
            "like": likes, "cuisine": likes,
            "dislike": dislikes,
            "allergy_hint": allergies,
            "goal": goals,
            "dietary_pattern": patterns,
        }

        suggestions: List[Dict[str, Any]] = []
        for cand in candidates:
            kind = cand.get("kind")
            value = str(cand.get("value", "")).strip().lower()
            if kind not in VALID_KINDS or not value:
                continue
            if value in known_by_kind.get(kind, set()) or value in optouts:
                continue
            if kind != "allergy_hint" and cand.get("confidence") != "high":
                continue
            statement = cand.get("statement") or (
                f"It seems “{value}” matters to you — want me to remember this?"
            )
            if kind == "dislike" and value in likes:
                statement = (
                    f"“{value}” is currently in your likes, but it sounds like "
                    f"you've gone off it — update your profile?"
                )
            elif kind in ("like", "cuisine") and value in dislikes:
                statement = (
                    f"“{value}” is currently in your dislikes, but it sounds "
                    f"like you enjoy it now — update your profile?"
                )
            suggestions.append({
                "id": str(uuid.uuid4()),
                "kind": kind,
                "value": value,
                "statement": statement,
            })
            if len(suggestions) >= MAX_SUGGESTIONS_PER_TURN:
                break

        if suggestions:
            logger.info(
                "%d memory suggestion(s) for question: %s",
                len(suggestions),
                [(s["kind"], s["value"]) for s in suggestions],
            )
        return suggestions

    # ------------------------------------------------------------------ #
    # Decision (POST /qa/memory)                                           #
    # ------------------------------------------------------------------ #

    def decide(
        self, member_id: str, kind: str, value: str, decision: str
    ) -> bool:
        """Apply an accepted suggestion or record a declined one.

        Returns True if a durable profile change was persisted. The SDK
        profile object auto-PATCHes the gateway on attribute assignment, and
        every write carries provenance (``source: "foodscholar"``).
        """
        value_norm = str(value).strip().lower()
        if kind not in VALID_KINDS or not value_norm:
            raise ValueError("Invalid memory suggestion payload")

        if decision == "accept":
            return self._apply(member_id, kind, value_norm)
        return self._record_optout(member_id, value_norm) and False

    def _apply(self, member_id: str, kind: str, value: str) -> bool:
        from backend.platform import WISEFOOD_PLATFORM

        client = WISEFOOD_PLATFORM.get_client()
        try:
            profile = client.members.get(member_id).profile
            prefs = dict(profile.nutritional_preferences or {})
            props = dict(profile.properties or {})

            if kind in ("like", "cuisine"):
                likes = list(prefs.get("food_likes") or [])
                if value not in [str(v).lower() for v in likes]:
                    likes.append(value)
                prefs["food_likes"] = likes
                # Contradiction resolution: liking removes from dislikes.
                prefs["food_dislikes"] = [
                    v for v in (prefs.get("food_dislikes") or [])
                    if str(v).lower() != value
                ]
                profile.nutritional_preferences = prefs
            elif kind == "dislike":
                dislikes = list(prefs.get("food_dislikes") or [])
                if value not in [str(v).lower() for v in dislikes]:
                    dislikes.append(value)
                prefs["food_dislikes"] = dislikes
                prefs["food_likes"] = [
                    v for v in (prefs.get("food_likes") or [])
                    if str(v).lower() != value
                ]
                profile.nutritional_preferences = prefs
            elif kind == "allergy_hint":
                allergies = list(profile.allergies or [])
                if value not in [str(a).lower() for a in allergies]:
                    allergies.append(value)
                profile.allergies = allergies
            elif kind == "goal":
                # value is a canonical slug (e.g. "reduce_fat"). Stored under
                # properties.dietary_goals as {slug, label} so FoodChat's meal
                # planner can switch on the slug. FoodChat MUST read this key.
                goals = list(props.get("dietary_goals") or [])
                if value not in [str((g or {}).get("slug", "")).lower() for g in goals]:
                    goals.append({"slug": value, "label": value.replace("_", " ")})
                props["dietary_goals"] = goals
            elif kind == "dietary_pattern":
                # Standing regimen (keto, mediterranean, vegan...). Stored in
                # dietary_groups, which FoodChat already reads.
                groups = list(profile.dietary_groups or [])
                if value not in [str(g).lower() for g in groups]:
                    groups.append(value)
                profile.dietary_groups = groups

            log = list(props.get("memory_log") or [])
            log.append({
                "kind": kind, "value": value,
                "source": "foodscholar",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            props["memory_log"] = log
            profile.properties = props
            logger.info(
                "Memory applied for member %s: %s=%r (foodscholar)",
                member_id, kind, value,
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to apply memory for member %s: %s",
                member_id, e, exc_info=True,
            )
            return False
        finally:
            WISEFOOD_PLATFORM.return_client(client)

    def _record_optout(self, member_id: str, value: str) -> bool:
        from backend.platform import WISEFOOD_PLATFORM

        client = WISEFOOD_PLATFORM.get_client()
        try:
            profile = client.members.get(member_id).profile
            props = dict(profile.properties or {})
            optouts = list(props.get("memory_optouts") or [])
            if value not in optouts:
                optouts.append(value)
                props["memory_optouts"] = optouts
                profile.properties = props
            return True
        except Exception as e:
            logger.error(
                "Failed to record opt-out for member %s: %s",
                member_id, e, exc_info=True,
            )
            return False
        finally:
            WISEFOOD_PLATFORM.return_client(client)

    def _fetch_profile(self, member_id: str) -> Dict[str, Any]:
        from backend.platform import WISEFOOD_PLATFORM

        client = WISEFOOD_PLATFORM.get_client()
        try:
            profile = client.members.get(member_id).profile
            if hasattr(profile, "to_dict"):
                data = profile.to_dict()
                if isinstance(data, dict):
                    return data
            data = getattr(profile, "_data", None)
            if isinstance(data, dict):
                return data
            return {
                "nutritional_preferences": dict(
                    getattr(profile, "nutritional_preferences", None) or {}
                ),
                "allergies": list(getattr(profile, "allergies", None) or []),
                "properties": dict(getattr(profile, "properties", None) or {}),
            }
        finally:
            WISEFOOD_PLATFORM.return_client(client)


MEMORY_SERVICE = MemoryService()
