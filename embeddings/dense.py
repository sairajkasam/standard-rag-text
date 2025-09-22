# embeddings/embed.py
import uuid
from typing import Any, Dict, List, Optional

from langchain_openai import OpenAIEmbeddings

from utils.logger import get_logger

logger = get_logger(__name__)


def dense_embedding(
    embedder: OpenAIEmbeddings,
    chunks: List[Dict[str, Any]],
    id_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate dense embeddings (OpenAI) and return chunks in unified format.
    Each chunk will look like:
      {
        "id": "...",
        "text": "...",
        "source": "...",
        "chunk_index": int,
        "embedding_type": "dense",
        "embedding": [float, float, ...]
      }
    """
    texts = [c.get("text", "") for c in chunks]
    try:
        vectors = embedder.embed_documents(texts)
    except Exception as e:
        logger.exception("Dense embedding batch failed: %s", e)
        raise

    results: List[Dict[str, Any]] = []
    for chunk, vec in zip(chunks, vectors):
        obj_id = chunk.get("id") or (
            f"{id_prefix}_{uuid.uuid4().hex}" if id_prefix else str(uuid.uuid4())
        )
        updated = {
            "id": obj_id,
            "text": [chunk.get("text", "")],
            "source": chunk.get("source"),
            "chunk_index": chunk.get("chunk_index"),
            "embedding_type": "dense",
            "embedding": [float(x) for x in vec],  # ensure list of floats
        }
        results.append(updated)

    logger.info("Dense embedding prepared for %d chunks", len(results))
    return results
