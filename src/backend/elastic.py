import os
import threading
from elasticsearch import Elasticsearch, NotFoundError
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

ELASTIC_HOST = os.getenv("ELASTIC_HOST", "http://elasticsearch:9200")
ES_DIM = int(os.getenv("ES_DIM", 384))


class ElasticsearchClientSingleton:
    """Singleton around a single thread-safe Elasticsearch client."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._client = Elasticsearch(
                        hosts=ELASTIC_HOST,
                        # Optional tuning:
                        # request_timeout=10,
                        # max_retries=3,
                        # retry_on_timeout=True,
                    )
                    instance._bootstrap()
                    cls._instance = instance
        return cls._instance

    @property
    def client(self) -> Elasticsearch:
        return self._client

    # --- Simple helpers -----------------------------------------------------

    def index_exists(self, index_name: str) -> bool:
        return self.client.indices.exists(index=index_name)

    def get_entity(self, index_name: str, urn: str) -> Optional[Dict[str, Any]]:
        try:
            r = self.client.get(index=index_name, id=urn)
            return r["_source"]
        except NotFoundError:
            return None
        except Exception:
            logger.exception("Error fetching entity %s from %s", urn, index_name)
            raise

    def list_entities(
        self, index_name: str, size: int = 1000, offset: int = 0
    ) -> List[str]:
        body = {
            "from": offset,
            "size": size,
            "query": {"bool": {"must_not": {"term": {"status": "deleted"}}}},
        }
        r = self.client.search(index=index_name, body=body)
        return [h["_id"] for h in r["hits"]["hits"]]

    def fetch_entities(
        self, index_name: str, limit: int, offset: int
    ) -> List[Dict[str, Any]]:
        body = {
            "from": offset,
            "size": limit,
            "query": {"bool": {"must_not": {"term": {"status": "deleted"}}}},
        }
        r = self.client.search(index=index_name, body=body)
        return [hit["_source"] for hit in r["hits"]["hits"]]

    def index_entity(self, index_name: str, document: Dict[str, Any]) -> None:
        doc_id = document.get("urn", document.get("id"))
        self.client.index(
            index=index_name,
            id=doc_id,
            document=document,
            refresh="wait_for",
        )

    def delete_entity(self, index_name: str, urn: str) -> None:
        self.client.delete(index=index_name, id=urn, refresh="wait_for")

    def update_entity(self, index_name: str, document: Dict[str, Any]) -> None:
        # Avoid updating if only system fields are present
        if set(document.keys()) == {"updated_at", "urn"}:
            return

        existing = self.get_entity(index_name, document["urn"])
        if not existing:
            return

        merged = {**existing, **document}
        self.client.update(
            index=index_name,
            id=document["urn"],
            doc=merged,
            refresh="wait_for",
        )

    def delete_by_query(self, index_name: str, query: Dict[str, Any]) -> None:
        self.client.delete_by_query(
            index=index_name,
            body={"query": query},
            refresh=True,
        )

    # --- Vector search ------------------------------------------------------

    def knn_search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        num_candidates: int = 100,
        field: str = "embedding",
        filter_query: Optional[Dict[str, Any]] = None,
        source_excludes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform approximate kNN search using dense vectors.

        Args:
            index_name: ES index to search
            query_vector: Dense vector for similarity search
            k: Number of nearest neighbors to return
            num_candidates: Candidates to consider (higher = more accurate)
            field: Name of the dense_vector field
            filter_query: Optional ES filter to narrow search scope
            source_excludes: Fields to exclude from _source (e.g., ["embedding"])

        Returns:
            List of dicts with _id, _score, and all _source fields
        """
        knn_body = {
            "field": field,
            "query_vector": query_vector,
            "k": k,
            "num_candidates": num_candidates,
        }

        if filter_query:
            knn_body["filter"] = filter_query

        source_config = True
        if source_excludes:
            source_config = {"excludes": source_excludes}

        try:
            response = self.client.search(
                index=index_name,
                knn=knn_body,
                source=source_config,
            )

            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "_id": hit["_id"],
                    "_score": hit["_score"],
                    **hit["_source"],
                }
                results.append(result)

            logger.info(
                "kNN search on %s: k=%d, returned %d results",
                index_name, k, len(results),
            )
            return results

        except Exception:
            logger.exception("Error performing kNN search on %s", index_name)
            raise

    # --- Search with faceting ----------------------------------------------

    def parse_sort_string(self, sort_str: str):
        # Allow commas or spaces between fields
        tokens = sort_str.replace(",", " ").split()
        result = []

        i = 0
        while i < len(tokens):
            field = tokens[i]
            order = "asc"

            # If next token is asc/desc, use it
            if i + 1 < len(tokens) and tokens[i + 1].lower() in ("asc", "desc"):
                order = tokens[i + 1].lower()
                i += 2
            else:
                i += 1

            result.append((field, order))

        return result

ELASTIC_CLIENT = ElasticsearchClientSingleton()