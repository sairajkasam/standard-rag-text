import uuid
from typing import Any, Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_openai import OpenAIEmbeddings

from utils.logger import get_logger

logger = get_logger(__name__)


def sparse_embedding(
    embedder: OpenAIEmbeddings,
    chunks: List[Dict[str, Any]],
    id_prefix: Optional[str] = None,
    max_features: int = 200000,
    ngram_range: tuple = (1, 1),
) -> List[Dict[str, Any]]:
    """
    Generate sparse TF-IDF embeddings and return chunks in unified format
    compatible with Weaviate's custom sparse vector input.
    Each chunk will look like:
      {
        "id": "...",
        "text": "...",
        "source": "...",
        "chunk_index": int,
        "embedding_type": "sparse",
        "embedding": {"indices": [int], "values": [float]}
      }
    """
    texts = [c.get("text", "") for c in chunks]
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)

    results: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        # Get the sparse matrix row for the current chunk
        row = X.getrow(i).tocoo()

        # Check if the sparse vector is empty (no non-zero values)
        if row.nnz > 0:
            indices = row.col.tolist()
            values = row.data.tolist()

            # ** Crucial Validation Step **
            if len(indices) != len(values):
                logger.error(
                    "Mismatched lengths for sparse vector (indices: %d, values: %d) at chunk %d: %s",
                    len(indices),
                    len(values),
                    i,
                    chunk.get("text", "No text provided."),
                )
                continue  # Skip this chunk

            obj_id = chunk.get("id") or (
                f"{id_prefix}_{uuid.uuid4().hex}" if id_prefix else str(uuid.uuid4())
            )

            emb = {"indices": indices, "values": values}
            dense_emb = embedder.embed_query(chunk.get("text"))
            updated = {
                "id": obj_id,
                "text": chunk.get("text", ""),
                "source": chunk.get("source"),
                "chunk_index": chunk.get("chunk_index"),
                "embedding_type": "sparse",
                "sparse_embedding": emb,
                "dense_embedding": [
                    float(x) for x in dense_emb
                ],  # ensure list of floats
            }
            results.append(updated)
        else:
            logger.warning(
                "Skipping chunk with empty sparse embedding (row %d): %s",
                i,
                chunk.get("text", "No text provided."),
            )

    logger.info(
        "Sparse TF-IDF embedding prepared for %d chunks (vocab=%d)",
        len(results),
        len(vectorizer.get_feature_names_out()),
    )
    return results
