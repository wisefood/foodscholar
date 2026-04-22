from services.linearrag.embedding_store import EmbeddingStore
from services.linearrag.utils import min_max_normalize
from services.linearrag.ner import SpacyNER
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import igraph as ig
import re
import logging
import torch
logger = logging.getLogger(__name__)

_PASSAGE_INDEX_PATTERN = re.compile(r'^(\d+):')


def _sorted_unique(values):
    return sorted(set(values))


def _normalize_entity_mapping(mapping):
    return {
        key: _sorted_unique(values)
        for key, values in sorted(mapping.items())
    }


def _edge_key(left, right):
    return tuple(sorted((left, right)))


class LinearRAG:
    def __init__(self, global_config):
        self.config = global_config
        logger.info(f"Initializing LinearRAG with config: {self.config}")
        retrieval_method = "Vectorized Matrix-based" if self.config.use_vectorized_retrieval else "BFS Iteration"
        logger.info(f"Using retrieval method: {retrieval_method}")
        
        # Setup device for GPU acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.use_vectorized_retrieval:
            logger.info(f"Using device: {self.device} for vectorized retrieval")
        
        self.dataset_name = global_config.dataset_name
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(
            self.config.spacy_model,
            getattr(self.config, "nutrition_terms_file", ""),
            getattr(self.config, "nutrition_strict", True),
            getattr(self.config, "use_scispacy_only", False),
        )

        self.graph_path = os.path.join(
            self.config.working_dir,
            self.dataset_name,
            "LinearRAG.graphml"
        )
        self.ner_results_path = os.path.join(
            self.config.working_dir,
            self.dataset_name,
            "ner_results.json"
        )
        self.batch_rebuild_pending_path = os.path.join(
            self.config.working_dir,
            self.dataset_name,
            "deferred_batch_index_pending.json"
        )

        if os.path.exists(self.graph_path):
            self.graph = ig.Graph.Read_GraphML(self.graph_path)
        else:
            self.graph = ig.Graph(directed=False)

        self.save_graph_every_batch = getattr(self.config, "save_graph_every_batch", False)
        self.entity_to_sentence = {}
        self.sentence_to_entity = {}
        self.entity_hash_id_to_sentence_hash_ids = {}
        self.sentence_hash_id_to_entity_hash_ids = {}
        self.passage_node_indices = []
        self.node_name_to_vertex_idx = {}
        self.vertex_idx_to_node_name = {}
        self.edge_name_pair_to_eid = {}
        self.passage_hash_id_to_entities = {}
        self.pending_passage_hash_ids = set()
        self.ner_state_loaded = False
        self.ner_state_dirty = False
        self._refresh_graph_lookup_cache()
        self.checkpoint_every_n_batches = max(0, int(getattr(self.config, "checkpoint_every_n_batches", 0) or 0))
        self.checkpoint_graph_on_checkpoint = bool(getattr(self.config, "checkpoint_graph_on_checkpoint", False))
        self.defer_graph_rebuild_during_batch_index = bool(
            getattr(self.config, "defer_graph_rebuild_during_batch_index", True)
        )
        self.deferred_batch_index_pending = os.path.exists(self.batch_rebuild_pending_path)

    def load_embedding_store(self):
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "entity_embedding.parquet"), batch_size=self.config.batch_size, namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir,self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence")

    def _refresh_graph_lookup_cache(self):
        if "name" not in self.graph.vs.attributes():
            self.node_name_to_vertex_idx = {}
            self.vertex_idx_to_node_name = {}
            self.edge_name_pair_to_eid = {}
            return

        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs}

        edge_lookup = {}
        for edge in self.graph.es:
            source_name = self.vertex_idx_to_node_name.get(edge.source)
            target_name = self.vertex_idx_to_node_name.get(edge.target)
            if source_name is None or target_name is None:
                continue
            edge_lookup[_edge_key(source_name, target_name)] = edge.index
        self.edge_name_pair_to_eid = edge_lookup

    def _build_entity_to_sentence_mapping(self, sentence_to_entity):
        entity_to_sentence = defaultdict(list)
        for sentence_text, entity_texts in sentence_to_entity.items():
            for entity_text in entity_texts:
                entity_to_sentence[entity_text].append(sentence_text)
        return _normalize_entity_mapping(entity_to_sentence)

    def _invalidate_retrieval_mappings(self):
        self.entity_hash_id_to_sentence_hash_ids = {}
        self.sentence_hash_id_to_entity_hash_ids = {}

    def _ensure_ner_state_loaded(self):
        if self.ner_state_loaded:
            return

        if os.path.exists(self.ner_results_path):
            with open(self.ner_results_path, "r", encoding="utf-8") as file_obj:
                existing_ner_results = json.load(file_obj)

            self.passage_hash_id_to_entities = _normalize_entity_mapping(
                existing_ner_results["passage_hash_id_to_entities"]
            )
            self.sentence_to_entity = _normalize_entity_mapping(
                existing_ner_results["sentence_to_entities"]
            )
            self.entity_to_sentence = self._build_entity_to_sentence_mapping(self.sentence_to_entity)
        else:
            self.passage_hash_id_to_entities = {}
            self.sentence_to_entity = {}
            self.entity_to_sentence = {}

        self.pending_passage_hash_ids = (
            set(self.passage_embedding_store.hash_id_to_text.keys())
            - set(self.passage_hash_id_to_entities.keys())
        )
        self.ner_state_loaded = True
        self.ner_state_dirty = False

    def _update_sentence_entity_mapping(self, sentence_text, ordered_entities):
        existing_entities = self.sentence_to_entity.get(sentence_text, [])
        existing_set = set(existing_entities)
        new_set = set(ordered_entities)

        if existing_set == new_set:
            self.sentence_to_entity[sentence_text] = ordered_entities
            return False

        for entity_text in existing_set - new_set:
            sentence_texts = self.entity_to_sentence.get(entity_text, [])
            updated_sentence_texts = [text for text in sentence_texts if text != sentence_text]
            if updated_sentence_texts:
                self.entity_to_sentence[entity_text] = updated_sentence_texts
            else:
                self.entity_to_sentence.pop(entity_text, None)

        for entity_text in new_set:
            sentence_texts = list(self.entity_to_sentence.get(entity_text, []))
            if sentence_text not in sentence_texts:
                sentence_texts.append(sentence_text)
            self.entity_to_sentence[entity_text] = _sorted_unique(sentence_texts)

        self.sentence_to_entity[sentence_text] = ordered_entities
        return True

    def _persist_index_state(self, write_graph=False):
        self.passage_embedding_store.save()
        self.entity_embedding_store.save()
        self.sentence_embedding_store.save()
        self.save_ner_results(persist=True)
        if write_graph:
            self._write_graph()

    def _mark_deferred_batch_index_pending(self):
        os.makedirs(os.path.dirname(self.batch_rebuild_pending_path), exist_ok=True)
        with open(self.batch_rebuild_pending_path, "w", encoding="utf-8") as file_obj:
            json.dump({"deferred_batch_index_pending": True}, file_obj)
        self.deferred_batch_index_pending = True

    def _clear_deferred_batch_index_pending(self):
        if os.path.exists(self.batch_rebuild_pending_path):
            os.remove(self.batch_rebuild_pending_path)
        self.deferred_batch_index_pending = False

    def _persist_batch_ingest_state(self):
        self.passage_embedding_store.save()
        self.save_ner_results(persist=True)
        self._mark_deferred_batch_index_pending()

    def _reset_deferred_index_artifacts(self):
        self.node_to_node_stats = defaultdict(dict)
        self.entity_embedding_store.reset()
        self.sentence_embedding_store.reset()
        self.graph = ig.Graph(directed=False)
        self._refresh_graph_lookup_cache()
        self._invalidate_retrieval_mappings()
        self.passage_node_indices = []

    def _finalize_deferred_batch_index(self, write_graph=True):
        self._ensure_ner_state_loaded()
        self._reset_deferred_index_artifacts()
        self._rebuild_index_graph_from_state(
            self.passage_hash_id_to_entities,
            self.sentence_to_entity,
            persist_state=False,
        )
        self._persist_index_state(write_graph=write_graph)
        self._clear_deferred_batch_index_pending()

    def _should_checkpoint_batch(self, batch_number):
        if self.save_graph_every_batch:
            return True
        return self.checkpoint_every_n_batches > 0 and batch_number % self.checkpoint_every_n_batches == 0

    def _can_extend_existing_graph(self, known_passage_hash_ids=None):
        if self.graph.vcount() == 0:
            return False
        if not self.node_name_to_vertex_idx:
            return False
        if known_passage_hash_ids is not None and not known_passage_hash_ids and self.passage_embedding_store.hash_ids:
            logger.warning(
                "Existing graph is present but no saved NER-backed passage baseline was found; falling back to a full rebuild"
            )
            return False
        if known_passage_hash_ids is None:
            return True

        missing_passage_hash_ids = [
            passage_hash_id
            for passage_hash_id in known_passage_hash_ids
            if passage_hash_id not in self.node_name_to_vertex_idx
        ]
        if missing_passage_hash_ids:
            logger.warning(
                "Existing graph is missing %d known passages; falling back to a full rebuild",
                len(missing_passage_hash_ids),
            )
            return False
        return True

    def _add_missing_graph_nodes(self, node_hash_ids):
        new_nodes = []

        for hash_id in node_hash_ids:
            if hash_id in self.node_name_to_vertex_idx:
                continue

            metadata = {}
            if hash_id in self.passage_embedding_store.hash_id_to_text:
                text = self.passage_embedding_store.hash_id_to_text[hash_id]
                metadata = self.passage_embedding_store.hash_id_to_metadata.get(hash_id, {})
            elif hash_id in self.entity_embedding_store.hash_id_to_text:
                text = self.entity_embedding_store.hash_id_to_text[hash_id]
            else:
                continue

            try:
                metadata_str = json.dumps(metadata)
            except Exception:
                metadata_str = json.dumps({})

            new_nodes.append({
                "name": hash_id,
                "content": text,
                "metadata": metadata_str,
            })

        if not new_nodes:
            return

        self.graph.add_vertices(len(new_nodes))
        start_idx = len(self.graph.vs) - len(new_nodes)

        for offset, node in enumerate(new_nodes):
            idx = start_idx + offset
            self.graph.vs[idx]["name"] = node["name"]
            self.graph.vs[idx]["content"] = node["content"]
            self.graph.vs[idx]["metadata"] = node["metadata"]
            self.node_name_to_vertex_idx[node["name"]] = idx
            self.vertex_idx_to_node_name[idx] = node["name"]

    def _upsert_graph_edge(self, source_hash_id, target_hash_id, weight):
        if source_hash_id == target_hash_id:
            return

        edge_id = self.edge_name_pair_to_eid.get(_edge_key(source_hash_id, target_hash_id))
        if edge_id is not None:
            self.graph.es[edge_id]["weight"] = weight
            return

        source_idx = self.node_name_to_vertex_idx.get(source_hash_id)
        target_idx = self.node_name_to_vertex_idx.get(target_hash_id)
        if source_idx is None or target_idx is None:
            return

        self.graph.add_edge(source_idx, target_idx, weight=weight)
        self.edge_name_pair_to_eid[_edge_key(source_hash_id, target_hash_id)] = self.graph.ecount() - 1

    def _delete_graph_edges(self, edge_keys):
        edge_ids = sorted(
            {
                self.edge_name_pair_to_eid[edge_key]
                for edge_key in edge_keys
                if edge_key in self.edge_name_pair_to_eid
            },
            reverse=True,
        )
        if not edge_ids:
            return

        self.graph.delete_edges(edge_ids)
        self._refresh_graph_lookup_cache()

    def _get_indexed_passage_items(self, passage_hash_ids=None):
        passage_id_to_text = self.passage_embedding_store.hash_id_to_text
        candidate_hash_ids = passage_hash_ids if passage_hash_ids is not None else passage_id_to_text.keys()

        indexed_items = []
        for node_key in candidate_hash_ids:
            text = passage_id_to_text.get(node_key)
            if text is None:
                continue
            match = _PASSAGE_INDEX_PATTERN.match(text.strip())
            if match:
                indexed_items.append((int(match.group(1)), node_key))

        indexed_items.sort(key=lambda item: (item[0], item[1]))
        return indexed_items

    def _build_adjacent_edge_keys(self, passage_hash_ids=None):
        indexed_items = self._get_indexed_passage_items(passage_hash_ids)
        return {
            _edge_key(indexed_items[i][1], indexed_items[i + 1][1])
            for i in range(len(indexed_items) - 1)
        }

    def _can_use_append_only_adjacent_update(self, old_passage_hash_ids, new_passage_hash_ids):
        old_indexed_items = self._get_indexed_passage_items(old_passage_hash_ids)
        new_indexed_items = self._get_indexed_passage_items(new_passage_hash_ids)

        if not new_indexed_items:
            return True
        if not old_indexed_items:
            return True

        return new_indexed_items[0][0] > old_indexed_items[-1][0]

    def _update_append_only_adjacent_edges(self, old_passage_hash_ids, new_passage_hash_ids):
        old_indexed_items = self._get_indexed_passage_items(old_passage_hash_ids)
        new_indexed_items = self._get_indexed_passage_items(new_passage_hash_ids)

        if not new_indexed_items:
            return

        if old_indexed_items:
            self._upsert_graph_edge(old_indexed_items[-1][1], new_indexed_items[0][1], 1.0)

        for i in range(len(new_indexed_items) - 1):
            self._upsert_graph_edge(new_indexed_items[i][1], new_indexed_items[i + 1][1], 1.0)

    def _compute_passage_entity_edge_weights(self, passage_hash_id_to_entities):
        edge_weights = defaultdict(dict)
        passage_to_entity_count = {}
        passage_to_all_score = defaultdict(int)

        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id.get(entity)
                if entity_hash_id is None:
                    continue
                count = passage.count(entity.lower())
                if count <= 0:
                    continue
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count

        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            total = passage_to_all_score.get(passage_hash_id, 0)
            if total <= 0:
                continue
            edge_weights[passage_hash_id][entity_hash_id] = count / total

        return edge_weights

    def _rebuild_index_graph_from_state(self, passage_hash_id_to_entities, sentence_to_entities, persist_state=True):
        entity_nodes, sentence_nodes, passage_map, entity_to_sentence, sentence_to_entity = self.extract_nodes_and_edges(
            passage_hash_id_to_entities,
            sentence_to_entities,
        )

        self.sentence_embedding_store.insert_text(sentence_nodes, persist=persist_state)
        self.entity_embedding_store.insert_text(entity_nodes, persist=persist_state)
        self._set_entity_sentence_mappings(entity_to_sentence, sentence_to_entity)

        self.add_entity_to_passage_edges(passage_map)
        self.add_adjacent_passage_edges()
        self._rebuild_graph()

    def _extend_existing_graph_state(
        self,
        old_passage_hash_ids,
        new_passage_hash_ids,
        new_passage_hash_id_to_entities,
        new_sentence_to_entities,
        persist_state=True,
    ):
        new_entity_nodes, new_sentence_nodes, _, _, _ = self.extract_nodes_and_edges(
            new_passage_hash_id_to_entities,
            new_sentence_to_entities,
        )

        if new_sentence_nodes:
            self.sentence_embedding_store.insert_text(new_sentence_nodes, persist=persist_state)
        if new_entity_nodes:
            self.entity_embedding_store.insert_text(new_entity_nodes, persist=persist_state)

        new_entity_hash_ids = [
            self.entity_embedding_store.text_to_hash_id[entity_text]
            for entity_text in new_entity_nodes
            if entity_text in self.entity_embedding_store.text_to_hash_id
        ]
        self._add_missing_graph_nodes(list(new_passage_hash_ids) + new_entity_hash_ids)

        edge_weights = self._compute_passage_entity_edge_weights(new_passage_hash_id_to_entities)
        for passage_hash_id, entity_weights in edge_weights.items():
            for entity_hash_id, weight in entity_weights.items():
                self._upsert_graph_edge(passage_hash_id, entity_hash_id, weight)

        if self._can_use_append_only_adjacent_update(old_passage_hash_ids, new_passage_hash_ids):
            self._update_append_only_adjacent_edges(old_passage_hash_ids, new_passage_hash_ids)
            return

        old_adjacent_edges = self._build_adjacent_edge_keys(old_passage_hash_ids)
        all_passage_hash_ids = set(self.passage_embedding_store.hash_id_to_text.keys())
        new_adjacent_edges = self._build_adjacent_edge_keys(all_passage_hash_ids)

        self._delete_graph_edges(old_adjacent_edges - new_adjacent_edges)
        for source_hash_id, target_hash_id in sorted(new_adjacent_edges - old_adjacent_edges):
            self._upsert_graph_edge(source_hash_id, target_hash_id, 1.0)

    def load_existing_data(self,passage_hash_ids):
        self._ensure_ner_state_loaded()
        ordered_passage_hash_ids = list(passage_hash_ids)
        new_passage_hash_ids = [
            passage_hash_id
            for passage_hash_id in ordered_passage_hash_ids
            if passage_hash_id in self.pending_passage_hash_ids
        ]
        return self.passage_hash_id_to_entities, self.sentence_to_entity, new_passage_hash_ids

    def _set_entity_sentence_mappings(self, entity_to_sentence, sentence_to_entity):
        self.entity_to_sentence = _normalize_entity_mapping(entity_to_sentence)
        self.sentence_to_entity = _normalize_entity_mapping(sentence_to_entity)

        entity_hash_id_to_sentence_hash_ids = defaultdict(list)
        for entity_text, sentence_texts in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id.get(entity_text)
            if entity_hash_id is None:
                continue
            for sentence_text in sentence_texts:
                sentence_hash_id = self.sentence_embedding_store.text_to_hash_id.get(sentence_text)
                if sentence_hash_id is not None:
                    entity_hash_id_to_sentence_hash_ids[entity_hash_id].append(sentence_hash_id)

        sentence_hash_id_to_entity_hash_ids = defaultdict(list)
        for sentence_text, entity_texts in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id.get(sentence_text)
            if sentence_hash_id is None:
                continue
            for entity_text in entity_texts:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id.get(entity_text)
                if entity_hash_id is not None:
                    sentence_hash_id_to_entity_hash_ids[sentence_hash_id].append(entity_hash_id)

        self.entity_hash_id_to_sentence_hash_ids = _normalize_entity_mapping(entity_hash_id_to_sentence_hash_ids)
        self.sentence_hash_id_to_entity_hash_ids = _normalize_entity_mapping(sentence_hash_id_to_entity_hash_ids)

    def _ensure_retrieval_graph_state(self):
        if not self.entity_hash_id_to_sentence_hash_ids or not self.sentence_hash_id_to_entity_hash_ids:
            if self.entity_to_sentence and self.sentence_to_entity:
                logger.info("Preparing entity/sentence hash mappings from in-memory NER results")
                self._set_entity_sentence_mappings(self.entity_to_sentence, self.sentence_to_entity)
            else:
                hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
                existing_passage_hash_id_to_entities, existing_sentence_to_entities, _ = self.load_existing_data(
                    hash_id_to_passage.keys()
                )
                _, _, _, entity_to_sentence, sentence_to_entity = self.extract_nodes_and_edges(
                    existing_passage_hash_id_to_entities,
                    existing_sentence_to_entities,
                )
                self._set_entity_sentence_mappings(entity_to_sentence, sentence_to_entity)
                logger.info("Rebuilt entity/sentence hash mappings from saved NER results")

        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_hash_id]
            for passage_hash_id in self.passage_hash_ids
            if passage_hash_id in self.node_name_to_vertex_idx
        ]

    def qa(self, questions):
        retrieval_results = self.retrieve(questions)
        system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""
        all_messages = []
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            prompt_user = """"""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)"
            ))
 
        for qa_result,question_info in zip(all_qa_results,retrieval_results):
            try:
                pred_ans = qa_result.split('Answer:')[1].strip()
            except:
                pred_ans = qa_result
            question_info["pred_answer"] = pred_ans
        return retrieval_results
        
    def retrieve(self, questions, metadata_filter=None):
        if self.deferred_batch_index_pending:
            logger.info("Finalizing deferred batch index state before retrieval")
            self._finalize_deferred_batch_index(write_graph=True)

        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self._refresh_graph_lookup_cache()
        self._ensure_retrieval_graph_state()

        # Precompute sparse matrices for vectorized retrieval if needed
        if self.config.use_vectorized_retrieval:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            e2s_total = e2s_shape[0] * e2s_shape[1]
            s2e_total = s2e_shape[0] * s2e_shape[1]
            e2s_sparsity = (1 - e2s_nnz / e2s_total) * 100 if e2s_total > 0 else 100.0
            s2e_sparsity = (1 - s2e_nnz / s2e_total) * 100 if s2e_total > 0 else 100.0
            logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
            logger.info(f"E2S Sparsity: {e2s_sparsity:.2f}% (nnz={e2s_nnz})")
            logger.info(f"S2E Sparsity: {s2e_sparsity:.2f}% (nnz={s2e_nnz})")
            logger.info(f"Device: {self.device}")

        filter_hash_ids = None
        has_metadata_filter = bool(metadata_filter)
        if has_metadata_filter:
            filter_hash_ids = set(
                self.passage_embedding_store.filter_hash_ids_by_metadata(metadata_filter)
            )
            logger.info(
                "Applying metadata filter %s matched %d passages",
                metadata_filter,
                len(filter_hash_ids),
            )

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            try:
                question = question_info["question"]
                question_embedding = self.config.embedding_model.encode(question,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
                seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores = self.get_seed_entities(question)
                if len(seed_entities) != 0:
                    sorted_passage_hash_ids,sorted_passage_scores = self.graph_search_with_seed_entities(question,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
                    if has_metadata_filter:
                        filtered_results = [
                            (passage_hash_id, passage_score)
                            for passage_hash_id, passage_score in zip(sorted_passage_hash_ids, sorted_passage_scores)
                            if passage_hash_id in filter_hash_ids
                        ]
                        if filtered_results:
                            sorted_passage_hash_ids, sorted_passage_scores = zip(*filtered_results)
                            sorted_passage_hash_ids = list(sorted_passage_hash_ids)
                            sorted_passage_scores = list(sorted_passage_scores)
                        else:
                            sorted_passage_hash_ids, sorted_passage_scores = [], []
                    final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                    final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                    final_passages = [self.passage_embedding_store.hash_id_to_text[passage_hash_id] for passage_hash_id in final_passage_hash_ids]
                else:
                    sorted_passage_indices,sorted_passage_scores = self.dense_passage_retrieval(question_embedding, filter_hash_ids)
                    final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                    final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                    final_passage_hash_ids = [self.passage_hash_ids[idx] for idx in final_passage_indices]
                    final_passages = [self.passage_embedding_store.texts[idx] for idx in final_passage_indices]
                final_sources = [
                    self.passage_embedding_store.hash_id_to_metadata.get(passage_hash_id, {})
                    for passage_hash_id in final_passage_hash_ids
                ]
                result = {
                    "question": question,
                    "sorted_passage": final_passages,
                    "sorted_passage_scores": final_passage_scores,
                    "sources": final_sources,
                    "gold_answer": question_info.get("answer", "")
                }
                retrieval_results.append(result)
            except Exception as e:
                logger.error(f"Error processing question: {question}. Error: {str(e)}")
                err_str = "Error"
                result = {
                    "question": err_str,
                    "sorted_passage": err_str,
                    "sorted_passage_scores": err_str,
                    "sources": err_str,
                    "gold_answer": err_str
                }
                retrieval_results.append(result)
                continue
        return retrieval_results
    
    def _precompute_sparse_matrices(self):
        """
        Precompute and cache sparse adjacency matrices for efficient vectorized retrieval using PyTorch.
        This is called once at the beginning of retrieve() to avoid rebuilding matrices per query.
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # Build entity-to-sentence matrix (Mention matrix) using COO format
        entity_to_sentence_indices = []
        entity_to_sentence_values = []
        
        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)
        
        # Build sentence-to-entity matrix
        sentence_to_entity_indices = []
        sentence_to_entity_values = []
        
        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)
        
        # Convert to PyTorch sparse tensors (COO format, then convert to CSR for efficiency)
        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.device
            )
        
        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.device
            )
            
    def graph_search_with_seed_entities(self, question, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores)
        passage_weights = self.calculate_passage_scores(question, question_embedding, actived_entities)
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices,ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices,ppr_sorted_passage_scores

    def run_ppr(self, node_weights):        
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]] 
            for i in sorted_indices_in_doc_scores
        ]
        
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for seed_entity_idx,seed_entity,seed_entity_hash_id,seed_entity_score in zip(seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score    
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        next_enitity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration+1)
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def calculate_entity_scores_vectorized(self,question_embedding,seed_entity_indices,seed_entities,seed_entity_hash_ids,seed_entity_scores):
        """
        GPU-accelerated vectorized version using PyTorch sparse tensors.
        Uses sparse representation for both matrices and entity score vectors for maximum efficiency.
        Now includes proper dynamic pruning to match BFS behavior:
        - Sentence deduplication (tracks used sentences)
        - Per-entity top-k sentence selection
        - Proper threshold-based pruning
        """
        # Initialize entity weights
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # Compute all sentence similarities with the question at once
        question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
        sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()
        
        # Convert to torch tensors and move to device
        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.device)
        
        # Track used sentences for deduplication (like BFS version)
        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
        
        # Initialize seed entity scores as sparse tensor
        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()
        
        # Also maintain a dense accumulator for total scores
        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
        entity_scores_dense.scatter_(0, torch.tensor(seed_entity_indices, device=self.device), 
                                     torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device))
        
        # Initialize actived_entities
        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        
        current_entity_scores_sparse = entity_scores_sparse
        
        # Iterative matrix-based propagation using sparse matrices on GPU
        for iteration in range(1, self.config.max_iterations):
            # Convert sparse tensor to dense for threshold operation
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()
            
            # Apply threshold to current scores
            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold, 
                current_entity_scores_dense, 
                torch.zeros_like(current_entity_scores_dense)
            )
            
            # Get non-zero indices for sparse representation
            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(nonzero_indices) == 0:
                break
            
            # Extract non-zero values and create sparse tensor
            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0), nonzero_values, (num_entities,), device=self.device
            ).coalesce()
            
            # Step 1: Sparse entity scores @ Sparse E2S matrix
            # Convert sparse vector to 2D for matrix multiplication
            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device
            ).coalesce()
            
            # E @ E2S -> sentence activation scores (sparse @ sparse = dense)
            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(),
                current_scores_2d
            )
            # Convert to dense before squeeze to avoid CUDA sparse tensor issues
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()
            
            # Apply sentence deduplication: mask out used sentences
            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation
            )
            
            # Step 2: Per-entity top-k sentence selection
            # This matches BFS behavior: each entity independently selects its top-k sentences
            selected_sentence_indices_list = []
            
            if len(nonzero_indices) > 0 and self.config.top_k_sentence > 0:
                # Iterate through each active entity
                for i, entity_idx in enumerate(nonzero_indices):
                    entity_score = nonzero_values[i]
                    
                    # Get sentences connected to this entity from the sparse matrix
                    # entity_to_sentence_sparse shape: (num_entities, num_sentences)
                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]  # Get column indices
                    
                    if len(entity_sentence_indices) == 0:
                        continue
                    
                    # Filter out already used sentences
                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]
                    
                    if len(available_sentence_indices) == 0:
                        continue
                    
                    # Get sentence similarities (for ranking)
                    sentence_sims = sentence_similarities[available_sentence_indices]
                    
                    # Select top-k sentences based ONLY on sentence similarity (matches BFS line 240)
                    # NOT weighted by entity_score at selection time
                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)
                
                # Merge all selected sentences (with deduplication via unique)
                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)
                    
                    # Mark selected sentences as used
                    used_sentence_mask[unique_selected_sentences] = True
                    
                    # Compute weighted sentence scores for propagation
                    # weighted_score = sentence_activation * sentence_similarity
                    weighted_sentence_scores = sentence_activation * sentence_similarities
                    
                    # Zero out non-selected sentences
                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores)
                    )
                else:
                    # No sentences selected, create zero vector
                    weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            else:
                # No active entities or top_k_sentence is 0
                weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            
            # Step 3: Weighted sentences @ S2E -> propagate to next entities
            # Convert to sparse for more efficient computation
            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack([weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device
                ).coalesce()
                
                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(),
                    weighted_scores_2d
                )
                # Convert to dense before squeeze to avoid CUDA sparse tensor issues
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
            
            # Update entity scores (accumulate in dense format)
            entity_scores_dense += next_entity_scores_dense
            
            # Update actived_entities dictionary (record last trigger like BFS)
            # This matches BFS behavior: unconditionally update for entities above threshold
            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.config.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]
                # Unconditionally update to record the last trigger (matches BFS line 252)
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)
            
            # Prepare sparse tensor for next iteration
            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0), next_nonzero_values, 
                    (num_entities,), device=self.device
                ).coalesce()
            else:
                break
        
        # Convert back to numpy for final processing
        entity_scores_final = entity_scores_dense.cpu().numpy()
        
        # Map entity scores to graph node weights (only for non-zero scores)
        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)
        
        return entity_weights, actived_entities

    def calculate_passage_scores(self, question, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        apply_attribute_boost = (
            getattr(self.config, "enable_hybrid_attribute_fallback", False)
            and self._is_attribute_query(question)
        )
        question_lower = question.lower()

        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus
            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)

            if apply_attribute_boost:
                overlap = self._attribute_keyword_overlap(question_lower, passage_text_lower)
                if overlap > 0:
                    passage_score += getattr(self.config, "attribute_keyword_boost", 0.25) * math.log(1 + overlap)

            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question_embedding, filter_hash_ids=None):
        question_emb = question_embedding.reshape(1, -1)

        if filter_hash_ids is not None:
            indices = [
                self.passage_embedding_store.hash_id_to_idx[h]
                for h in filter_hash_ids
            ]

            if not indices:
                return [], []

            passage_embeddings = self.passage_embeddings[indices]

            sims = np.dot(passage_embeddings, question_emb.T).flatten()
            order = np.argsort(sims)[::-1]

            return [indices[i] for i in order], sims[order].tolist()

        sims = np.dot(self.passage_embeddings, question_emb.T).flatten()
        order = np.argsort(sims)[::-1]

        return order, sims[order].tolist()

    def _is_attribute_query(self, question):
        tokens = set(re.findall(r"\w+", question.lower()))
        return any(keyword in tokens for keyword in getattr(self.config, "attribute_query_keywords", []))

    def _attribute_keyword_overlap(self, question_lower, passage_text_lower):
        overlap = 0
        for keyword in getattr(self.config, "attribute_query_keywords", []):
            if keyword in question_lower and keyword in passage_text_lower:
                overlap += 1
        return overlap
    
    def get_seed_entities(self, question):
        question_entities = self.spacy_ner.question_ner(question)
        if len(question_entities) == 0:
            return [],[],[],[]
        question_entity_embeddings = self.config.embedding_model.encode(question_entities,normalize_embeddings=True,show_progress_bar=False,batch_size=self.config.batch_size)
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []       
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def _ingest_passages_and_ner(self, passages, metadata_list=None, persist_state=True):
        if self.graph.vcount() > 0 and not self.node_name_to_vertex_idx:
            self._refresh_graph_lookup_cache()

        self._ensure_ner_state_loaded()
        old_passage_hash_ids = set(self.passage_hash_id_to_entities.keys())

        inserted_passage_hash_ids = self.passage_embedding_store.insert_text(
            passages,
            metadata_list,
            persist=persist_state,
        )
        self.pending_passage_hash_ids.update(inserted_passage_hash_ids)

        existing_p = self.passage_hash_id_to_entities
        existing_s = self.sentence_to_entity
        new_ids = sorted(self.pending_passage_hash_ids)

        if len(new_ids) > 0:
            new_map = {
                passage_hash_id: self.passage_embedding_store.hash_id_to_text[passage_hash_id]
                for passage_hash_id in new_ids
            }
            new_p, new_s = self.spacy_ner.batch_ner(new_map, self.config.max_workers)
            self.merge_ner_results(existing_p, existing_s, new_p, new_s)
        else:
            new_p, new_s = {}, {}
            logger.info("No new passages found for dataset %s", self.dataset_name)

        self.save_ner_results(persist=persist_state)
        return old_passage_hash_ids, existing_p, existing_s, new_ids, new_p, new_s

    def index(self, passages, metadata_list=None, persist_graph=True, persist_state=True):
        if self.deferred_batch_index_pending:
            logger.info("Finalizing deferred batch index state before standalone indexing")
            self._finalize_deferred_batch_index(write_graph=True)

        self.node_to_node_stats = defaultdict(dict)
        graph_changed = False

        old_passage_hash_ids, existing_p, existing_s, new_ids, new_p, new_s = self._ingest_passages_and_ner(
            passages,
            metadata_list,
            persist_state=persist_state,
        )

        if len(new_ids) == 0:
            if old_passage_hash_ids and not self._can_extend_existing_graph(old_passage_hash_ids):
                self._rebuild_index_graph_from_state(existing_p, existing_s, persist_state=persist_state)
                graph_changed = True
        elif self._can_extend_existing_graph(old_passage_hash_ids):
            self._extend_existing_graph_state(
                old_passage_hash_ids,
                new_ids,
                new_p,
                new_s,
                persist_state=persist_state,
            )
            graph_changed = True
        else:
            self._rebuild_index_graph_from_state(existing_p, existing_s, persist_state=persist_state)
            graph_changed = True

        if persist_graph and graph_changed:
            self._write_graph()

    def index_in_batches(self, passages, metadata_list=None, batch_size=100):
        if not self.defer_graph_rebuild_during_batch_index:
            total = len(passages)
            last_batch_number = 0
            last_checkpoint_batch = 0
            checkpoint_graph = self.save_graph_every_batch or self.checkpoint_graph_on_checkpoint

            for batch_number, i in enumerate(range(0, total, batch_size), start=1):
                last_batch_number = batch_number
                batch_p = passages[i:i + batch_size]
                batch_m = None if metadata_list is None else metadata_list[i:i + batch_size]

                logger.info(
                    "Indexing batch %d/%d with %d passages",
                    batch_number,
                    math.ceil(total / batch_size) if batch_size > 0 else 0,
                    len(batch_p),
                )
                print(f"Batch {batch_number}: {len(batch_p)} passages")
                self.index(
                    batch_p,
                    batch_m,
                    persist_graph=False,
                    persist_state=False,
                )
                if self._should_checkpoint_batch(batch_number):
                    logger.info("Persisting checkpoint after batch %d", batch_number)
                    self._persist_index_state(write_graph=checkpoint_graph)
                    last_checkpoint_batch = batch_number

            if last_batch_number == 0:
                return

            if last_checkpoint_batch != last_batch_number:
                self._persist_index_state(write_graph=True)
            elif not checkpoint_graph:
                self._write_graph()
            return

        total = len(passages)
        last_batch_number = 0
        self._ensure_ner_state_loaded()
        rebuild_required = self.deferred_batch_index_pending or bool(self.pending_passage_hash_ids)

        for batch_number, i in enumerate(range(0, total, batch_size), start=1):
            last_batch_number = batch_number
            batch_p = passages[i:i + batch_size]
            batch_m = None if metadata_list is None else metadata_list[i:i + batch_size]

            logger.info(
                "Indexing batch %d/%d with %d passages",
                batch_number,
                math.ceil(total / batch_size) if batch_size > 0 else 0,
                len(batch_p),
            )
            print(f"Batch {batch_number}: {len(batch_p)} passages")
            _, _, _, new_ids, _, _ = self._ingest_passages_and_ner(
                batch_p,
                batch_m,
                persist_state=False,
            )
            if new_ids:
                rebuild_required = True
                self._mark_deferred_batch_index_pending()
            if self._should_checkpoint_batch(batch_number):
                logger.info("Persisting batch ingest checkpoint after batch %d", batch_number)
                self._persist_batch_ingest_state()

        if last_batch_number == 0:
            if rebuild_required:
                logger.info("Finalizing deferred sentence/entity embeddings and graph")
                self._finalize_deferred_batch_index(write_graph=True)
            return

        if rebuild_required:
            logger.info("Finalizing deferred sentence/entity embeddings and graph")
            self._finalize_deferred_batch_index(write_graph=True)

    def batch_index(self, passages, metadata=None, batch_size=200):
        self.index_in_batches(passages, metadata_list=metadata, batch_size=batch_size)

    def _rebuild_graph(self):
        self.graph = ig.Graph(directed=False)
        self.augment_graph()
        self._refresh_graph_lookup_cache()

    def _write_graph(self):
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        self.graph.write_graphml(self.graph_path)

    def add_adjacent_passage_edges(self):
        indexed_items = self._get_indexed_passage_items()
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_node_names = set(
            v["name"] for v in self.graph.vs if "name" in v.attributes()
        )

        entity_map = self.entity_embedding_store.get_hash_id_to_text()
        passage_map = self.passage_embedding_store.get_hash_id_to_text()

        all_nodes = {**entity_map, **passage_map}

        new_nodes = []

        for hash_id, text in all_nodes.items():
            if hash_id in existing_node_names:
                continue

            metadata = {}

            if hash_id in self.passage_embedding_store.hash_id_to_metadata:
                metadata = self.passage_embedding_store.hash_id_to_metadata[hash_id]

            try:
                metadata_str = json.dumps(metadata)
            except Exception:
                metadata_str = json.dumps({})

            new_nodes.append({
                "name": hash_id,
                "content": text,
                "metadata": metadata_str
            })

        if new_nodes:
            self.graph.add_vertices(len(new_nodes))

            for i, node in enumerate(new_nodes):
                idx = len(self.graph.vs) - len(new_nodes) + i

                self.graph.vs[idx]["name"] = node["name"]
                self.graph.vs[idx]["content"] = node["content"]
                self.graph.vs[idx]["metadata"] = node["metadata"]

    def add_edges(self):
        edges = []
        weights = []
        
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        edge_weights = self._compute_passage_entity_edge_weights(passage_hash_id_to_entities)
        for passage_hash_id, entity_weights in edge_weights.items():
            for entity_hash_id, score in entity_weights.items():
                self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = []
        entity_seen = set()
        sentence_nodes = []
        passage_hash_id_to_entities = {}
        entity_to_sentence = defaultdict(list)
        sentence_to_entity = {}
        for passage_hash_id, entities in sorted(existing_passage_hash_id_to_entities.items()):
            ordered_entities = _sorted_unique(entities)
            passage_hash_id_to_entities[passage_hash_id] = ordered_entities
            for entity in ordered_entities:
                if entity not in entity_seen:
                    entity_seen.add(entity)
                    entity_nodes.append(entity)
        for sentence, entities in sorted(existing_sentence_to_entities.items()):
            ordered_entities = _sorted_unique(entities)
            sentence_nodes.append(sentence)
            sentence_to_entity[sentence] = ordered_entities
            for entity in ordered_entities:
                entity_to_sentence[entity].append(sentence)
                if entity not in entity_seen:
                    entity_seen.add(entity)
                    entity_nodes.append(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, dict(entity_to_sentence), sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        self.ner_state_loaded = True

        for passage_hash_id, entities in new_passage_hash_id_to_entities.items():
            ordered_entities = _sorted_unique(entities)
            existing_passage_hash_id_to_entities[passage_hash_id] = ordered_entities

        if new_passage_hash_id_to_entities:
            self.pending_passage_hash_ids.difference_update(new_passage_hash_id_to_entities.keys())

        sentence_mapping_changed = False
        for sentence_text, entities in new_sentence_to_entities.items():
            ordered_entities = _sorted_unique(entities)
            if self._update_sentence_entity_mapping(sentence_text, ordered_entities):
                sentence_mapping_changed = True

        if new_passage_hash_id_to_entities or sentence_mapping_changed:
            self.ner_state_dirty = True
            self._invalidate_retrieval_mappings()

        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, persist=True):
        if not persist:
            return

        self._ensure_ner_state_loaded()
        if not self.ner_state_dirty:
            return

        serialized_passage_hash_id_to_entities = _normalize_entity_mapping(self.passage_hash_id_to_entities)
        serialized_sentence_to_entities = _normalize_entity_mapping(self.sentence_to_entity)
        os.makedirs(os.path.dirname(self.ner_results_path), exist_ok=True)
        with open(self.ner_results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "passage_hash_id_to_entities": serialized_passage_hash_id_to_entities,
                    "sentence_to_entities": serialized_sentence_to_entities,
                },
                f,
                ensure_ascii=False,
                sort_keys=True,
            )
            self.passage_hash_id_to_entities = serialized_passage_hash_id_to_entities
            self.sentence_to_entity = serialized_sentence_to_entities
            self.entity_to_sentence = _normalize_entity_mapping(self.entity_to_sentence)
            self.ner_state_dirty = False
