from sentence_transformers import SentenceTransformer
from services.linearrag.config import LinearRAGConfig
from services.linearrag.LinearRAG import LinearRAG
import logging

logger = logging.getLogger(__name__)

_retriever: LinearRAG | None = None

def get_retriever() -> LinearRAG:
    global _retriever
    if _retriever is None:
        _retriever = _init_retriever()
    return _retriever

def _init_retriever() -> LinearRAG:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2") 
    config = LinearRAGConfig(
        dataset_name="linearrag",        
        embedding_model=embedding_model,
        working_dir="data",             
        retrieval_top_k=5,
        use_vectorized_retrieval=False,
    )
    logger.info("Loading LinearRAG retriever from disk...")
    return LinearRAG(config)

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    retriever = get_retriever()
    results = retriever.retrieve([{"question": question}])
    result = results[0]
    return [
        {"text": passage, "score": score, "source": source}
        for passage, score, source in zip(
            result["sorted_passage"],
            result["sorted_passage_scores"],
            result["sources"],
        )
    ]