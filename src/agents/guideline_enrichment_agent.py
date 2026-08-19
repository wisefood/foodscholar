"""
Enrichment agent for dietary guideline rules.

Assigns structured facets — life stage, age range, setting, nutrients, audience
and so on — to a rule sentence extracted from a dietary guide.

The problem this solves is context loss. Rules were extracted page by page, so
"Provide portions of red meat twice a week" arrives with no indication that it
came from a guide for 1-4 year olds. The guide is the authority on who a rule
applies to, so this agent is always given the guide's context alongside the
rule, and its main job is to push that context down onto the individual record.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate

from agents.json_output import (
    clean_confidence,
    clean_int,
    clean_str,
    clean_str_list,
    parse_json_object,
)
from backend.groq import GROQ_CHAT
from backend.langfuse import build_trace_config
from config import config
from backend.prompts import GUIDELINE_ENRICHMENT

logger = logging.getLogger(__name__)

_ENRICHMENT_MAX_TOKENS = 2048

# Closed vocabularies. Anything outside these is dropped rather than written:
# an invented facet value is worse than an absent one, because it pollutes the
# aggregations the catalog UI builds its filters from.
LIFE_STAGES = [
    "pregnancy",
    "lactation",
    "infancy",
    "early_childhood",
    "school_age",
    "adolescence",
    "adulthood",
    "older_adulthood",
]
SETTINGS = ["school", "home", "clinical", "community", "workplace", "retail", "general"]
GUIDELINE_TYPES = ["food_based", "nutrient_based", "behavioral", "activity", "other"]
AUDIENCES = [
    "caregiver",
    "individual",
    "health_professional",
    "policy_maker",
    "educator",
]
TARGET_POPULATIONS = [
    "general_population",
    "infants",
    "under_5_years",
    "ages_5_to_18",
    "adults",
    "elderly",
    "pregnant_people",
    "lactating_people",
    "other",
]
FOOD_GROUPS = [
    "none",
    "fruits",
    "vegetables",
    "grains",
    "dairy",
    "protein_foods",
    "fats_and_oils",
    "beverages",
    "salt",
    "sugars_and_sweets",
    "mixed",
    "other",
]
ACTION_TYPES = [
    "eat",
    "drink",
    "use",
    "do",
    "avoid",
    "prepare",
    "limit",
    "choose",
    "increase",
    "reduce",
]
FREQUENCIES = ["per_meal", "daily", "weekly", "monthly", "occasional"]

# Open vocabularies are capped so a verbose model cannot bloat a record.
MAX_OPEN_VOCAB_ITEMS = 5


class GuidelineEnrichmentAgent:
    """Assigns catalog facets to a single guideline rule."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        model = model or config.settings["GUIDELINE_ENRICHMENT_MODEL"]
        self.model = model
        self.llm = GROQ_CHAT.get_client(
            model=model,
            temperature=temperature,
            max_tokens=_ENRICHMENT_MAX_TOKENS,
            reasoning_effort="low",
        )

    @staticmethod
    def _format_rule_context(guideline: Dict[str, Any]) -> str:
        """
        Render what surrounded the rule on its source page.

        A rule sentence is short and often elliptical; the section it sat under
        and the summary of its page frequently carry the population or the food
        group the sentence itself omits.
        """
        lines = []
        section = guideline.get("section_label")
        if isinstance(section, str) and section.strip():
            lines.append(f"- Section: {section.strip()}")
        page_no = guideline.get("page_no")
        if page_no:
            lines.append(f"- Source page: {page_no}")
        summary = guideline.get("page_summary")
        if isinstance(summary, str) and summary.strip():
            lines.append(f"- What that page covered: {summary.strip()}")
        if not lines:
            return "- (no page context captured for this rule)"
        return "\n".join(lines)

    @staticmethod
    def _format_existing_facets(guideline: Dict[str, Any]) -> str:
        """Render the facets already on the record, for the prompt."""
        interesting = (
            "target_populations",
            "food_groups",
            "action_type",
            "frequency",
            "life_stage",
            "setting",
            "topic",
            "audience",
            "age_min_months",
            "age_max_months",
        )
        lines = []
        for name in interesting:
            value = guideline.get(name)
            if value in (None, "", [], {}):
                continue
            lines.append(f"- {name}: {value}")
        return "\n".join(lines) if lines else "- (none set)"

    def _normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map raw model output onto catalog fields, dropping anything unsupported.

        Absent is a valid answer throughout: a facet the model could not support
        is left out entirely rather than filled with a placeholder, so the
        catalog's "not set" and "the model said it doesn't apply" stay distinct.
        """
        facets: Dict[str, Any] = {}

        life_stage = clean_str_list(data.get("life_stage"), allowed=LIFE_STAGES)
        if life_stage:
            facets["life_stage"] = life_stage

        setting = clean_str_list(data.get("setting"), allowed=SETTINGS)
        if setting:
            facets["setting"] = setting

        audience = clean_str_list(data.get("audience"), allowed=AUDIENCES)
        if audience:
            facets["audience"] = audience

        target_populations = clean_str_list(
            data.get("target_populations"), allowed=TARGET_POPULATIONS
        )
        if target_populations:
            facets["target_populations"] = target_populations

        food_groups = clean_str_list(data.get("food_groups"), allowed=FOOD_GROUPS)
        if food_groups:
            facets["food_groups"] = food_groups

        health_conditions = clean_str_list(
            data.get("health_conditions"), limit=MAX_OPEN_VOCAB_ITEMS
        )
        if health_conditions:
            facets["health_conditions"] = health_conditions

        nutrients = clean_str_list(data.get("nutrients"), limit=MAX_OPEN_VOCAB_ITEMS)
        if nutrients:
            facets["nutrients"] = nutrients

        topic = clean_str_list(data.get("topic"), limit=3)
        if topic:
            facets["topic"] = topic

        guideline_type = clean_str(data.get("guideline_type"))
        if guideline_type:
            guideline_type = guideline_type.lower().replace(" ", "_")
            if guideline_type in GUIDELINE_TYPES:
                facets["guideline_type"] = guideline_type

        action_type = clean_str(data.get("action_type"))
        if action_type:
            action_type = action_type.lower()
            if action_type in ACTION_TYPES:
                facets["action_type"] = action_type

        frequency = clean_str(data.get("frequency"))
        if frequency:
            frequency = frequency.lower().replace(" ", "_")
            if frequency in FREQUENCIES:
                facets["frequency"] = frequency

        age_min = clean_int(data.get("age_min_months"))
        age_max = clean_int(data.get("age_max_months"))
        if age_min is not None and age_max is not None and age_max < age_min:
            age_min = age_max = None
        if age_min is not None:
            facets["age_min_months"] = age_min
        if age_max is not None:
            facets["age_max_months"] = age_max

        confidence = clean_confidence(data.get("confidence"))
        if confidence is not None:
            facets["enrichment_confidence"] = confidence

        return facets

    def enrich_guideline(
        self,
        guideline: Dict[str, Any],
        guide_context: str,
    ) -> Dict[str, Any]:
        """
        Propose catalog facets for one guideline rule.

        Returns a dict of catalog field names to values, containing only the
        facets the model could support. An empty dict is a valid outcome and
        means the rule carries no facetable information.
        """
        rule_text = (guideline.get("rule_text") or "").strip()
        if not rule_text:
            return {}

        prompt = ChatPromptTemplate.from_messages(
            [("system", GUIDELINE_ENRICHMENT.langchain()), ("human", "{instruction}")]
        )

        response = (prompt | self.llm).invoke(
            {
                "guide_context": guide_context,
                "rule_text": rule_text,
                "rule_context": self._format_rule_context(guideline),
                "existing_facets": self._format_existing_facets(guideline),
                "instruction": (
                    "Return the facets as a single JSON object with the keys "
                    "life_stage, age_min_months, age_max_months, setting, "
                    "health_conditions, nutrients, guideline_type, topic, "
                    "audience, target_populations, food_groups, action_type, "
                    "frequency, confidence. Omit a key or use an empty list "
                    "when the guide and rule do not support it. No prose."
                ),
            },
            config=build_trace_config(
                run_name="guideline-enrichment",
                tags=["enrichment", "guidelines"],
            ),
        )

        content = response.content

        try:
            data = parse_json_object(content)
        except ValueError as exc:
            logger.error(
                "Guideline enrichment JSON invalid for %s: %s | raw=%.500r",
                guideline.get("id"),
                exc,
                content,
            )
            return {}

        return self._normalize(data)
