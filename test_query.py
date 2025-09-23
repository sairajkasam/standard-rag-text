import weaviate
import os
from dotenv import load_dotenv

# --- Configuration ---
# Make sure your .env file is in the same directory or provide the URL directly
load_dotenv()
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "9090")
COLLECTION_NAME = "new_test"  # Use the exact name of your collection
from weaviate.config import AdditionalConfig, Timeout

try:
    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=int(WEAVIATE_PORT),
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=int(WEAVIATE_GRPC_PORT),
        grpc_secure=False,
        skip_init_checks=True,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30000, query=30000, insert=32000)
        ),
    )
    collection = client.collections.use(COLLECTION_NAME)

    # Fetch 3 random objects from the collection
    response = collection.query.fetch_objects(
        limit=3, include_vector=True
    )  # Ask for the vector

    print(f"--- Inspecting 3 Objects from '{COLLECTION_NAME}' Collection ---")

    if not response.objects:
        print(
            "\nERROR: No objects found in the collection! The ingestion process may have failed."
        )
    else:
        breakpoint()
        for i, obj in enumerate(response.objects):
            print(f"\n--- OBJECT {i+1} ---")
            print(f"UUID: {obj.uuid}")

            # Check the text content
            chunk_text = obj.properties.get("chunks")
            print(f"Chunk Text: {chunk_text[:200]}...")  # Print first 200 chars

            # VERY IMPORTANT: Check the vectors
            dense_vector = obj.vectors.get("dense_vector")
            sparse_vector = obj.vectors.get(
                "sparse_vector"
            )  # Assuming you might have this

            print(f"Dense Vector exists: {dense_vector is not None}")
            if dense_vector:
                print(f"Dense Vector length: {len(dense_vector)}")

            print(f"Sparse Vector exists: {sparse_vector is not None}")
            if sparse_vector:
                print(
                    f"Sparse Vector sample: {dict(list(sparse_vector.items())[:3])}..."
                )


except Exception as e:
    print(f"\nAn error occurred: {e}")
