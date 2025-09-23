from altair import DataType
from weaviate.classes.config import Configure
from weaviate.classes.config import DataType as dt
from weaviate.classes.config import Property
from weaviate.config import AdditionalConfig, Timeout

from app.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import weaviate
except Exception:
    weaviate = None


def create_client():
    client = weaviate.connect_to_custom(
        http_host=Config.WEAVIATE_HOST,
        http_port=int(Config.WEAVIATE_PORT),
        http_secure=False,
        grpc_host=Config.WEAVIATE_HOST,
        grpc_port=int(Config.WEAVIATE_GRPC_PORT),
        grpc_secure=False,
        skip_init_checks=True,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30000, query=30000, insert=32000)
        ),
    )
    logger.info(f"Weaviate ready: {client.is_ready()}")
    logger.info("Client connection is established")
    return client


def ensure_collection_simple(
    client,
    collection_name: str,
    vector_type: str,  # "dense", "sparse", or "hybrid"
    recreate: bool = False,
):
    """
    Create a collection that contains only the required named vector(s).
    vector_type: "dense" | "sparse" | "hybrid"
    """
    if weaviate is None:
        raise RuntimeError("weaviate client not installed")

    vector_type = (vector_type or "dense").lower()
    if vector_type not in {"dense", "sparse", "hybrid"}:
        raise ValueError("vector_type must be 'dense', 'sparse' or 'hybrid'")

    # check existing collection
    index_exists = client.collections.exists(collection_name)
    existing = False
    try:
        if index_exists:
            existing = True
        else:
            existing = False
    except Exception as e:
        existing = False

    if existing and not recreate:
        logger.info("Collection exists and recreate=False -> skipping create")
        return existing

    if existing and recreate:
        try:
            client.collections.delete(collection_name)
            logger.info("Deleted existing collection (recreate=True)")
        except Exception as e:
            logger.warning("Failed to delete existing collection: %s", e)

    # Define common properties for the collection.
    collection_properties = [
        Property(name="chunks", data_type=dt.TEXT_ARRAY),
        Property(name="source", data_type=dt.TEXT),
        Property(name="chunk_id", data_type=dt.TEXT),
        Property(name="chunk_index", data_type=dt.INT),
        Property(name="embedding_type", data_type=dt.TEXT),
        Property(name="story_title", data_type=dt.TEXT),
        Property(name="chapter_index", data_type=dt.INT),
        Property(name="paragraph_range", data_type=dt.TEXT),
        Property(name="char_start", data_type=dt.INT),
        Property(name="char_end", data_type=dt.INT),
    ]

    vectorizer_config = []

    # --- Dense Vector Configuration ---
    if vector_type == "dense":
        vectorizer_config.append(
            Configure.NamedVectors.none(
                name="dense_vector",
                vector_index_config=Configure.VectorIndex.hnsw(),
            )
        )
    # --- Sparse and Hybrid Vector Configuration ---
    # Both require a dense vector with an HNSW index.
    elif vector_type in ["sparse", "hybrid"]:
        # Add dense vector configuration (required for sparse/hybrid)
        vectorizer_config.append(
            Configure.NamedVectors.none(
                name="dense_vector",
                vector_index_config=Configure.VectorIndex.hnsw(),
            )
        )
        # Add sparse vector configuration
        vectorizer_config.append(
            Configure.NamedVectors.none(
                name="sparse_vector",
            )
        )

    # Create the collection with the specified configuration.
    collection = client.collections.create(
        name=collection_name,
        vectorizer_config=vectorizer_config,
        properties=collection_properties,
        inverted_index_config=Configure.inverted_index(
            bm25_b=0.7,
            bm25_k1=1.25,
        ),
    )

    logger.info(f"Successfully created collection: '{collection_name}'")
    return collection
