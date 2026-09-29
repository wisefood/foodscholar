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
  - same-kind dedupe: a value the profile already holds comes back marked
    ``already_known`` so the UI can show it pre-selected rather than ask
    again, and it is never written twice; a dislike of something currently
    in the LIKES list still nudges (it's a contradiction the user should
    resolve);
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
from backend.model_output import normalize_model_text
from backend.prompts import QA_MEMORY_EXTRACTOR
from config import config

logger = logging.getLogger(__name__)

VALID_KINDS = {
    "like", "dislike", "cuisine", "allergy_hint", "goal", "dietary_pattern"
}
MAX_SUGGESTIONS_PER_TURN = 2
EXTRACTOR_MODEL = config.settings["MEMORY_EXTRACTOR_MODEL"]


def _known_statement(kind: str, value: str) -> str:
    """What the chip says for a value the profile already holds."""
    label = value.replace("_", " ")
    if kind == "goal":
        return f"“{label}” is already one of your goals."
    if kind == "dietary_pattern":
        return f"“{label}” is already part of your dietary profile."
    if kind == "allergy_hint":
        return f"“{label}” is already listed among your allergies."
    if kind == "dislike":
        return f"“{label}” is already in your dislikes."
    return f"“{label}” is already in your likes."


def _profile_fingerprint(prefs, props, allergies, groups) -> str:
    """The parts of a profile a memory write can touch, in a comparable form."""
    return json.dumps(
        {
            "likes": prefs.get("food_likes") or [],
            "dislikes": prefs.get("food_dislikes") or [],
            "goals": props.get("dietary_goals") or [],
            "allergies": allergies,
            "groups": groups,
        },
        sort_keys=True,
        default=str,
    )


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
            suggestions = self._apply_policy(candidates, profile)
            # The question is the provenance: a goal inferred here was inferred
            # because the member asked this. Echoed back on accept and stored
            # with the memory so the panel can answer "why am I seeing this?".
            source_text = " ".join(str(question or "").split())[:240]
            for suggestion in suggestions:
                suggestion["source_text"] = source_text
            return suggestions
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
        text = normalize_model_text(getattr(response, "content", ""))
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

        # What the model saw before the policy touched it: the one line that
        # separates "the model never said it" from "it said it at medium".
        logger.info(
            "Memory extractor candidates: %s",
            [(c.get("kind"), c.get("value"), c.get("confidence"))
             for c in candidates if isinstance(c, dict)],
        )

        fresh: List[Dict[str, Any]] = []
        known: List[Dict[str, Any]] = []
        for cand in candidates:
            kind = cand.get("kind")
            value = str(cand.get("value", "")).strip().lower()
            if kind not in VALID_KINDS or not value:
                continue
            if kind != "allergy_hint" and cand.get("confidence") != "high":
                continue
            if value in optouts:
                continue
            if value in known_by_kind.get(kind, set()):
                # Already on the profile: shown pre-selected so the user sees
                # it was recognised, never re-asked and never written twice.
                known.append({
                    "id": str(uuid.uuid4()),
                    "kind": kind,
                    "value": value,
                    "statement": _known_statement(kind, value),
                    "already_known": True,
                })
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
            fresh.append({
                "id": str(uuid.uuid4()),
                "kind": kind,
                "value": value,
                "statement": statement,
                "already_known": False,
            })

        # New nudges take the slots first; what is already known fills the rest.
        suggestions = (fresh + known)[:MAX_SUGGESTIONS_PER_TURN]

        if suggestions:
            logger.info(
                "%d memory suggestion(s) for question: %s",
                len(suggestions),
                [(s["kind"], s["value"], "known" if s["already_known"] else "new")
                 for s in suggestions],
            )
        return suggestions

    # ------------------------------------------------------------------ #
    # Decision (POST /qa/memory)                                           #
    # ------------------------------------------------------------------ #

    def decide(
        self, member_id: str, kind: str, value: str, decision: str,
        source_text: str = "",
    ) -> bool:
        """Apply an accepted suggestion or record a declined one.

        Returns True if a durable profile change was persisted; accepting a
        value the profile already holds writes nothing and returns False. The
        SDK profile object auto-PATCHes the gateway on attribute assignment,
        and every write carries provenance (``source: "foodscholar"``).
        """
        value_norm = str(value).strip().lower()
        if kind not in VALID_KINDS or not value_norm:
            raise ValueError("Invalid memory suggestion payload")

        if decision == "accept":
            return self._apply(member_id, kind, value_norm, source_text)
        return self._record_optout(member_id, value_norm) and False

    def _apply(
        self, member_id: str, kind: str, value: str, source_text: str = ""
    ) -> bool:
        from backend.platform import WISEFOOD_PLATFORM

        client = WISEFOOD_PLATFORM.get_client()
        try:
            profile = client.members.get(member_id).profile
            prefs = dict(profile.nutritional_preferences or {})
            props = dict(profile.properties or {})
            allergies = list(profile.allergies or [])
            groups = list(profile.dietary_groups or [])
            before = _profile_fingerprint(prefs, props, allergies, groups)

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
            elif kind == "dislike":
                dislikes = list(prefs.get("food_dislikes") or [])
                if value not in [str(v).lower() for v in dislikes]:
                    dislikes.append(value)
                prefs["food_dislikes"] = dislikes
                prefs["food_likes"] = [
                    v for v in (prefs.get("food_likes") or [])
                    if str(v).lower() != value
                ]
            elif kind == "allergy_hint":
                if value not in [str(a).lower() for a in allergies]:
                    allergies.append(value)
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
                if value not in [str(g).lower() for g in groups]:
                    groups.append(value)

            if _profile_fingerprint(prefs, props, allergies, groups) == before:
                # Nothing to remember twice: no PATCH, no second log entry.
                logger.info(
                    "Memory already on profile for member %s: %s=%r; not rewritten",
                    member_id, kind, value,
                )
                return False

            # Each assignment is a PATCH, so only the field this kind touches.
            if kind in ("like", "cuisine", "dislike"):
                profile.nutritional_preferences = prefs
            elif kind == "allergy_hint":
                profile.allergies = allergies
            elif kind == "dietary_pattern":
                profile.dietary_groups = groups

            log = list(props.get("memory_log") or [])
            entry = {
                "kind": kind, "value": value,
                "source": "foodscholar",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            # Omitted rather than stored empty, so entries written before this
            # and after it read the same way in the memory panel.
            if source_text:
                entry["source_text"] = source_text
            log.append(entry)
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
