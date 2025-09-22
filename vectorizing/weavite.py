from typing import Any, Dict, List

from utils.logger import get_logger
from weaviate.exceptions import WeaviateInvalidInputError

logger = get_logger(__name__)


def insert_data_batches(
    client: Any, input_payload: Dict[str, Any], data: List[Dict[str, Any]]
) -> Dict[str, int]:
    collection = client.collections.get(input_payload["index_name"])
    embedding_type = input_payload.get("embedding_type")
    try:
        with collection.batch.dynamic() as batch:
            for d in data:
                properties = {"chunks": d.get("text")}

                # --- FIX: The vector assignments are now correct ---
                if embedding_type == "dense":
                    vector = {"dense_vector": d.get("embedding")}
                elif embedding_type == "sparse" or embedding_type == "hybrid":
                    # The sparse embedding dict goes to the 'sparse_vector'
                    # The dense embedding list goes to the 'dense_vector'
                    vector = {
                        "dense_vector": d.get("dense_embedding"),
                        "sparse_vector": d.get("sparse_embedding"),
                    }
                else:
                    continue

                batch.add_object(properties=properties, vector=vector)
    except WeaviateInvalidInputError as e:
        logger.error(f"Weaviate input error: {e}")

    # Check for batch errors
    if collection.batch.failed_objects:
        logger.error(
            f"Failed to import {len(collection.batch.failed_objects)} objects."
        )
        for failed in collection.batch.failed_objects:
            logger.error(f"Failed object: {failed.uuid}, error: {failed.message}")
