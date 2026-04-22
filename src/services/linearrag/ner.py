import spacy
import scispacy
from collections import defaultdict
import json
import os
import logging
import re


logger = logging.getLogger(__name__)


def _sorted_unique(values):
    return sorted(set(values))


class SpacyNER:
    def __init__(self, spacy_model, nutrition_terms_file: str | None = "", nutrition_strict: bool = True, use_scispacy_only: bool = False):
        self.spacy_model = spacy.load(spacy_model)
        self.nutrition_terms = set()
        self.use_scispacy_only = bool(use_scispacy_only)
        self.nutrition_strict = False if self.use_scispacy_only else bool(nutrition_strict)

        # Common nutrition surface forms and abbreviations for canonicalization.
        self.alias_map = {
            "carb": "carbohydrate",
            "carbs": "carbohydrates",
            "vit c": "vitamin c",
            "vit d": "vitamin d",
            "vit e": "vitamin e",
            "vit k": "vitamin k",
            "vit b12": "vitamin b12",
            "vit b9": "vitamin b9",
            "folic acid": "folate",
            "omega 3": "omega-3 fatty acids",
            "omega 6": "omega-6 fatty acids",
        }

        # Lightweight keyword gate to keep entities nutrition-focused.
        self.nutrition_keywords = {
            "vitamin", "mineral", "nutrient", "diet", "dietary", "nutrition",
            "protein", "carbohydrate", "carb", "fat", "fiber", "sugar",
            "cholesterol", "calorie", "kcal", "omega", "folate", "sodium",
            "potassium", "calcium", "iron", "zinc", "magnesium", "iodine",
            "selenium", "food", "foods", "meal", "glycemic", "prebiotic",
            "probiotic", "microbiome", "hydration", "electrolyte",
        }

        nutrient_token = (
            r"(?:vitamin\s*(?:a|b(?:1|2|3|5|6|7|9|12)?|c|d|e|k)"
            r"|protein|carbohydrates?|carbs?|fat|fiber|sodium|potassium"
            r"|calcium|iron|zinc|magnesium|iodine|selenium"
            r"|cholesterol|sugar|folate|omega[-\s]?3|omega[-\s]?6|epa|dha)"
        )
        unit_token = r"(?:g|mg|mcg|ug|kcal|cal|iu|%)"
        self.amount_patterns = [
            re.compile(fr"\b{nutrient_token}\s*[:=]?\s*\d+(?:\.\d+)?\s*{unit_token}\b", re.IGNORECASE),
            re.compile(fr"\b\d+(?:\.\d+)?\s*{unit_token}\s*(?:of\s+)?{nutrient_token}\b", re.IGNORECASE),
        ]

        # Ensure sentence boundaries exist even for lightweight pipelines.
        if not any(name in self.spacy_model.pipe_names for name in ("parser", "senter", "sentencizer")):
            self.spacy_model.add_pipe("sentencizer")

        if self.use_scispacy_only:
            logger.info("Using sciSpaCy-only entity extraction; nutrition term rules and regex amount extraction are disabled.")
        else:
            self._load_nutrition_entity_ruler(nutrition_terms_file)

    def _load_nutrition_entity_ruler(self, nutrition_terms_file: str | None):
        if not nutrition_terms_file:
            return
        if not os.path.exists(nutrition_terms_file):
            logger.warning("Nutrition terms file not found: %s", nutrition_terms_file)
            return

        with open(nutrition_terms_file, "r", encoding="utf-8") as f:
            terms = json.load(f)

        if not isinstance(terms, list):
            logger.warning("Nutrition terms file must contain a JSON list: %s", nutrition_terms_file)
            return

        clean_terms = sorted({str(term).strip() for term in terms if str(term).strip()})
        if not clean_terms:
            logger.warning("Nutrition terms file is empty: %s", nutrition_terms_file)
            return
        self.nutrition_terms = {term.lower() for term in clean_terms}

        ruler = self.spacy_model.add_pipe(
            "entity_ruler",
            name="nutrition_entity_ruler",
            last=True,
            config={"overwrite_ents": False, "phrase_matcher_attr": "LOWER"}
        )
        patterns = [{"label": "NUTRITION_TERM", "pattern": term} for term in clean_terms]
        ruler.add_patterns(patterns)
        logger.info("Loaded %d nutrition term patterns", len(patterns))

    def _canonicalize_entity(self, text: str) -> str:
        text = re.sub(r"\s+", " ", str(text).strip().lower())
        text = text.strip(".,;:()[]{}\"'")
        return self.alias_map.get(text, text)

    def _is_nutrition_related(self, ent_text: str, ent_label: str = "") -> bool:
        text = self._canonicalize_entity(ent_text)
        if not text:
            return False

        if not self.nutrition_strict:
            return True

        if ent_label in {"NUTRITION_TERM", "NUTRITION_AMOUNT"}:
            return True

        if text in self.nutrition_terms:
            return True

        text_tokens = set(text.split())
        if self.nutrition_keywords.intersection(text_tokens):
            return True

        for pattern in self.amount_patterns:
            if pattern.search(text):
                return True

        return False

    def _extract_amount_entities(self, sentence_text: str):
        extracted = []
        for pattern in self.amount_patterns:
            for match in pattern.finditer(sentence_text):
                normalized = self._canonicalize_entity(match.group(0))
                if normalized:
                    extracted.append(normalized)
        return extracted

    def batch_ner(self, hash_id_to_passage, max_workers):
        passage_list = list(hash_id_to_passage.values())
        if not passage_list:
            return {}, defaultdict(list)

        workers = max(1, int(max_workers))
        batch_size = max(1, len(passage_list) // workers)
        docs_list = self.spacy_model.pipe(passage_list, batch_size=batch_size)
        passage_hash_ids = list(hash_id_to_passage.keys())

        passage_hash_id_to_entities = {}
        sentence_to_entities = defaultdict(list)
        for idx, doc in enumerate(docs_list):
            passage_hash_id = passage_hash_ids[idx]
            single_passage_hash_id_to_entities, single_sentence_to_entities = self.extract_entities_sentences(doc, passage_hash_id)
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
            for sent, ents in single_sentence_to_entities.items():
                for e in ents:
                    if e not in sentence_to_entities[sent]:
                        sentence_to_entities[sent].append(e)
        for sent, ents in sentence_to_entities.items():
            sentence_to_entities[sent] = _sorted_unique(ents)
        return passage_hash_id_to_entities, sentence_to_entities
            
    def extract_entities_sentences(self, doc, passage_hash_id):
        sentence_to_entities = defaultdict(list)
        unique_entities = []
        seen_entities = set()
        passage_hash_id_to_entities = {}

        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            sent_text = ent.sent.text
            ent_text = self._canonicalize_entity(ent.text)
            if not ent_text:
                continue
            if not self._is_nutrition_related(ent_text, ent.label_):
                continue
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            if ent_text not in seen_entities:
                seen_entities.add(ent_text)
                unique_entities.append(ent_text)

        # Add nutrient quantities (e.g., "protein 20 g", "20 g protein").
        if not self.use_scispacy_only:
            for sent in doc.sents:
                sent_text = sent.text
                for amount_entity in self._extract_amount_entities(sent_text):
                    if amount_entity not in sentence_to_entities[sent_text]:
                        sentence_to_entities[sent_text].append(amount_entity)
                    if amount_entity not in seen_entities:
                        seen_entities.add(amount_entity)
                        unique_entities.append(amount_entity)

        for sent_text, entities in sentence_to_entities.items():
            sentence_to_entities[sent_text] = _sorted_unique(entities)
        passage_hash_id_to_entities[passage_hash_id] = _sorted_unique(unique_entities)
        return passage_hash_id_to_entities, sentence_to_entities

    def question_ner(self, question: str):
        doc = self.spacy_model(question)
        question_entities = []
        seen_entities = set()
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            ent_text = self._canonicalize_entity(ent.text)
            if self._is_nutrition_related(ent_text, ent.label_):
                if ent_text not in seen_entities:
                    seen_entities.add(ent_text)
                    question_entities.append(ent_text)

        if not self.use_scispacy_only:
            for amount_entity in self._extract_amount_entities(question):
                if amount_entity not in seen_entities:
                    seen_entities.add(amount_entity)
                    question_entities.append(amount_entity)

        return _sorted_unique(question_entities)