"""Localized strings for the deterministic clarifier fallback.

The clarifier's normal path is the LLM (which is instructed to write user-facing
clarification text in the request language). This table only backs the
*deterministic* fallback in ``qa_clarifier.build_fallback_plan``, used when the
LLM call fails — so those rare fallbacks are still localized rather than always
English.

Structure: ``CLARIFICATION_I18N[clarification_id][lang]`` holds the user-facing
strings for one fallback clarification. Only user-READABLE text is translated:
the question, each option's label, and the reason. Machine-facing values
(clarification id, option ``value``s) stay canonical and live in
``qa_clarifier`` — they are never translated because they drive matching and
retrieval.

Adding a language: add its ISO 639-1 key under each clarification id. Any
language absent from the table falls back to English via ``localize``.
"""
from typing import Dict, List, Optional

DEFAULT_LANG = "en"

# clarification_id -> lang -> {"question", "reason", "labels": {value: label}}
CLARIFICATION_I18N: Dict[str, Dict[str, Dict[str, object]]] = {
    "target_age_group": {
        "en": {
            "question": "Who is the nutrition guidance for?",
            "reason": "Age materially changes safe nutrition advice.",
            "labels": {
                "infant": "Infant",
                "child": "Child",
                "teen": "Teen",
                "adult": "Adult",
            },
        },
        "sl": {
            "question": "Komu je namenjeno prehransko svetovanje?",
            "reason": "Starost bistveno spremeni varen prehranski nasvet.",
            "labels": {
                "infant": "Dojenček",
                "child": "Otrok",
                "teen": "Najstnik",
                "adult": "Odrasel",
            },
        },
    },
    "country_or_region": {
        "en": {
            "question": "Which country or guideline region should the answer use?",
            "reason": "Food-based recommendations can differ by country or guideline region.",
            "labels": {
                "US": "United States",
                "EU": "European Union",
                "UK": "United Kingdom",
                "GR": "Greece",
                "general": "No preference",
            },
        },
        "sl": {
            "question": "Katero državo ali regijo smernic naj upošteva odgovor?",
            "reason": "Priporočila glede živil se lahko razlikujejo med državami ali regijami smernic.",
            "labels": {
                "US": "Združene države",
                "EU": "Evropska unija",
                "UK": "Združeno kraljestvo",
                "GR": "Grčija",
                "general": "Brez posebne izbire",
            },
        },
    },
    "safety_context": {
        "en": {
            "question": "Is there any relevant safety context?",
            "reason": "Conditions and medications can change supplement safety advice.",
            "labels": {
                "none": "None",
                "pregnancy_or_breastfeeding": "Pregnant or breastfeeding",
                "chronic_condition": "Chronic condition",
                "medication_or_supplement": "Medication or supplement use",
            },
        },
        "sl": {
            "question": "Ali obstaja kakšen pomemben varnostni kontekst?",
            "reason": "Bolezni in zdravila lahko spremenijo varnostni nasvet glede prehranskih dopolnil.",
            "labels": {
                "none": "Nič od naštetega",
                "pregnancy_or_breastfeeding": "Nosečnost ali dojenje",
                "chronic_condition": "Kronično stanje",
                "medication_or_supplement": "Jemanje zdravil ali dopolnil",
            },
        },
    },
}


def _norm(language: Optional[str]) -> str:
    """Normalize an ISO 639-1-ish language code to a table key."""
    if not language:
        return DEFAULT_LANG
    return language.strip().lower().split("-")[0].split("_")[0]


def localize(clarification_id: str, language: Optional[str]) -> Dict[str, object]:
    """Return localized strings for a fallback clarification.

    Falls back to English when the clarification id or language is unknown, so a
    caller always gets a usable ``{"question", "reason", "labels"}`` dict.
    """
    by_lang = CLARIFICATION_I18N.get(clarification_id, {})
    if not by_lang:
        return {"question": "", "reason": None, "labels": {}}
    lang = _norm(language)
    return by_lang.get(lang) or by_lang.get(DEFAULT_LANG, {})


def label_for(clarification_id: str, value: str, language: Optional[str]) -> str:
    """Localized label for one option value; falls back to the value itself."""
    strings = localize(clarification_id, language)
    labels = strings.get("labels", {}) if isinstance(strings, dict) else {}
    return labels.get(value, value) if isinstance(labels, dict) else value
