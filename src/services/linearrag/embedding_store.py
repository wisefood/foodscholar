import os
import json
import pandas as pd
import numpy as np
from copy import deepcopy
from services.linearrag.utils import compute_mdhash_id


class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace

        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        self.metadata = []

        self.hash_id_to_text = {}
        self.text_to_hash_id = {}
        self.hash_id_to_idx = {}
        self.hash_id_to_metadata = {}
        self._dirty = False

        self._load_data()

    def _sort_records(self):
        if not self.hash_ids:
            return

        ordered = sorted(
            zip(self.hash_ids, self.texts, self.embeddings, self.metadata),
            key=lambda x: x[0]
        )

        self.hash_ids, self.texts, self.embeddings, self.metadata = map(list, zip(*ordered))

    def _rebuild_indexes(self):
        self.hash_id_to_idx = {h: i for i, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for h, t in zip(self.hash_ids, self.texts)}
        self.hash_id_to_metadata = {h: m for h, m in zip(self.hash_ids, self.metadata)}

    def _load_data(self):
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)

            self.hash_ids = df["hash_id"].tolist()
            self.texts = df["text"].tolist()
            self.embeddings = df["embedding"].tolist()

            if "metadata" in df.columns:
                self.metadata = [
                    json.loads(m) if isinstance(m, str) else m
                    for m in df["metadata"].tolist()
                ]
            else:
                self.metadata = [{} for _ in self.hash_ids]

            self._sort_records()
            self._rebuild_indexes()

        self._dirty = False

    def insert_text(self, text_list, metadata_list=None, persist=True):
        pending_nodes = {}

        for i, text in enumerate(text_list):
            meta = metadata_list[i] if metadata_list else {}
            hash_id = compute_mdhash_id(text, prefix=self.namespace + "-")
            if hash_id in self.hash_id_to_idx or hash_id in pending_nodes:
                continue
            pending_nodes[hash_id] = {"content": text, "metadata": meta}

        if not pending_nodes:
            return []

        missing_ids = list(pending_nodes.keys())
        texts = [pending_nodes[h]["content"] for h in missing_ids]
        metas = [pending_nodes[h]["metadata"] for h in missing_ids]

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        self.hash_ids.extend(missing_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metas)

        start_idx = len(self.hash_ids) - len(missing_ids)
        for offset, hash_id in enumerate(missing_ids):
            idx = start_idx + offset
            text = texts[offset]
            self.hash_id_to_idx[hash_id] = idx
            self.hash_id_to_text[hash_id] = text
            self.text_to_hash_id[text] = hash_id
            self.hash_id_to_metadata[hash_id] = metas[offset]

        self._dirty = True
        if persist:
            self.save()
        return missing_ids

    def save(self):
        if not self._dirty:
            return

        self._sort_records()
        self._rebuild_indexes()
        self._save_data()
        self._dirty = False

    def reset(self, persist=False):
        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        self.metadata = []
        self.hash_id_to_text = {}
        self.text_to_hash_id = {}
        self.hash_id_to_idx = {}
        self.hash_id_to_metadata = {}
        self._dirty = True
        if persist:
            self.save()

    def _save_data(self):
        df = pd.DataFrame({
            "hash_id": self.hash_ids,
            "text": self.texts,
            "embedding": self.embeddings,
            "metadata": [json.dumps(m) for m in self.metadata]
        })

        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        df.to_parquet(self.db_filename, index=False)

    def get_hash_id_to_text(self):
        return deepcopy(self.hash_id_to_text)

    # Metadata filtering
    def filter_hash_ids_by_metadata(self, filters: dict):
        return [
            h for h, m in self.hash_id_to_metadata.items()
            if all(m.get(k) == v for k, v in filters.items())
        ]

    def encode_texts(self, texts):
        return self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)
    
    def get_embeddings(self, hash_ids):
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings)[indices]
        return embeddings