from dataclasses import dataclass, field
import os
from services.linearrag.utils import LLM_Model


@dataclass
class LinearRAGConfig:
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_sci_sm"
    nutrition_terms_file: str | None = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "nutrition_terms.json")
    )
    nutrition_strict: bool = False
    use_scispacy_only: bool = False
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 3
    passage_ratio: float = 2
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.4
    use_vectorized_retrieval: bool = False  # True for vectorized matrix computation, False for BFS iteration
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])
    save_graph_every_batch: bool = False
    checkpoint_every_n_batches: int = 0
    checkpoint_graph_on_checkpoint: bool = False
    defer_graph_rebuild_during_batch_index: bool = True