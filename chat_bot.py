import os
import logging
from typing import Any, Dict, List

import openai
import weaviate
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from weaviate.config import AdditionalConfig, Timeout
from langchain_openai import OpenAIEmbeddings
from weaviate.classes.query import MetadataQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# --- Configuration ---
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "9090")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

logger.info(
    f"Configuration loaded - Weaviate Host: {WEAVIATE_HOST}, Port: {WEAVIATE_PORT}"
)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hybrid Search RAG Chatbot",
    description="A chatbot that uses hybrid search with Weaviate, reranking, and OpenAI's LLM.",
    version="1.0.0",
)


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    index_name: str
    dense_k: int = 20
    sparse_k: int = 25
    top_k_rerank: int = 7
    top_k_llm: int = 4
    dense_w: float = 0.7
    sparse_w: float = 0.4
    llm_temperature: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


logger.info(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL}")
reranker = CrossEncoder(CROSS_ENCODER_MODEL)
logger.info("Cross-encoder model loaded successfully")


def get_client():
    try:
        logger.info("Initializing Weaviate client...")
        weaviate_client = weaviate.connect_to_custom(
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
        logger.info("Weaviate client initialized successfully")

        openai.api_key = OPENAI_API_KEY
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        logger.info("OpenAI API key configured successfully")

        return weaviate_client
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        raise RuntimeError(f"Configuration error: {ve}") from ve
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}", exc_info=True)
        # This will prevent the app from starting if essential services are not available
        raise RuntimeError(f"Failed to initialize clients: {e}") from e


def hybrid_search(
    client: weaviate.Client,
    collection_name: str,
    query: str,
    query_vector: List[float],
    dense_k: int,
    sparse_k: int,
    dense_w: float,
    sparse_w: float,
) -> List[Dict[str, Any]]:
    """Performs a hybrid search on a Weaviate collection."""
    try:
        logger.info(f"Starting hybrid search on collection: {collection_name}")
        logger.debug(
            f"Search parameters - dense_k: {dense_k}, sparse_k: {sparse_k}, dense_w: {dense_w}, sparse_w: {sparse_w}"
        )
        collection = client.collections.get(collection_name)

        # Note: Weaviate Python client v4 handles the embedding for the query.
        # You just need to pass the query text.
        response = collection.query.near_vector(
            near_vector=[float(x) for x in query_vector],
            target_vector="dense_vector",
            limit=dense_k,
            return_metadata=MetadataQuery(score=True),
        )

        results = []
        for obj in response.objects:
            result = obj.properties
            result["uuid"] = obj.uuid
            results.append(result)

        logger.info(f"Hybrid search completed. Found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Error in hybrid_search: {e}", exc_info=True)
        raise RuntimeError(f"Hybrid search failed: {e}") from e


def rerank_results(
    query: str, candidates: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """Reranks search results using a cross-encoder model."""
    try:
        logger.info(
            f"Starting reranking with {len(candidates)} candidates, top_k: {top_k}"
        )

        pairs = []
        reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        for result in candidates:
            chunk_text = result.get("chunks")

            # --- FIX: Ensure chunk_text is a string ---
            # The reranker expects a [query, document] pair where both are strings.
            # If the retrieved chunk is a list (e.g., ['text']), extract the string.
            if isinstance(chunk_text, list) and chunk_text:
                document = chunk_text[0]
            else:
                document = str(chunk_text or "")  # Convert to string and handle None

            pairs.append((query, document))

        if not pairs:
            logger.warning("No pairs available for reranking")
            return []

        logger.debug(f"Created {len(pairs)} query-document pairs for reranking")
        scores = reranker.predict(pairs)

        # Combine results with scores and sort
        for i in range(len(scores)):
            candidates[i]["rerank_score"] = float(scores[i])

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        reranked_results = candidates[:top_k]
        logger.info(
            f"Reranking completed. Returning top {len(reranked_results)} results"
        )
        return reranked_results

    except Exception as e:
        logger.error(f"Error in rerank_results: {e}", exc_info=True)
        raise RuntimeError(f"Reranking failed: {e}") from e


def get_llm_response(
    query: str, context_chunks: List[Dict[str, Any]], temperature: float
) -> str:
    """Generates a response from an LLM using the provided context."""
    try:
        logger.info(
            f"Generating LLM response with {len(context_chunks)} context chunks"
        )
        logger.debug(f"LLM temperature: {temperature}")

        context = "\n---\n".join(
            [chunk.get("chunks", "")[0] for chunk in context_chunks]
        )

        prompt = f"""
        You are a helpful assistant. Answer the user's query based on the following context.
        If the context does not contain the answer, state that you don't know.

        Context:
        {context}

        Query: {query}

        Answer:
        """

        logger.debug("Sending request to OpenAI API")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )

        answer = response.choices[0].message.content.strip()
        logger.info("LLM response generated successfully")
        logger.debug(f"Response length: {len(answer)} characters")
        return answer

    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"Error in get_llm_response: {e}", exc_info=True)
        raise RuntimeError(f"LLM response generation failed: {e}") from e


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives a user query, performs hybrid search, reranks results,
    and generates a final answer using an LLM.
    """
    try:
        logger.info(
            f"Received chat request for query: '{request.query}' on index: {request.index_name}"
        )
        logger.debug(f"Request parameters: {request.dict()}")

        # Initialize embedder
        logger.debug("Initializing OpenAI embeddings")
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-large", api_key=OPENAI_API_KEY
        )
        weaviate_client = get_client()
        # Generate dense vector for the query
        logger.debug("Generating dense vector for query")
        dense_vector = embedder.embed_query(request.query)
        logger.debug(f"Generated dense vector with {len(dense_vector)} dimensions")

        # 1. Retrieval
        logger.info("Starting retrieval phase")
        retrieval_candidates = hybrid_search(
            client=weaviate_client,
            collection_name=request.index_name,
            query=request.query,
            query_vector=dense_vector,
            dense_k=request.dense_k,
            sparse_k=request.sparse_k,
            dense_w=request.dense_w,
            sparse_w=request.sparse_w,
        )

        if not retrieval_candidates:
            logger.warning("No retrieval candidates found")
            return ChatResponse(
                answer="I couldn't find any relevant information.", sources=[]
            )

        # 2. Reranking
        logger.info("Starting reranking phase")
        reranked_chunks = rerank_results(
            query=request.query,
            candidates=retrieval_candidates,
            top_k=request.top_k_rerank,
        )

        # 3. Prepare for LLM
        llm_context_chunks = reranked_chunks[: request.top_k_llm]
        logger.info(f"Prepared {len(llm_context_chunks)} chunks for LLM context")

        # 4. LLM Generation
        logger.info("Starting LLM generation phase")
        answer = get_llm_response(
            query=request.query,
            context_chunks=llm_context_chunks,
            temperature=request.llm_temperature,
        )

        # Close client connection
        try:
            weaviate_client.close()
            logger.debug("Weaviate client connection closed")
        except Exception as close_e:
            logger.warning(f"Error closing Weaviate client: {close_e}")

        logger.info("Chat request completed successfully")
        return ChatResponse(answer=answer, sources=llm_context_chunks)

    except HTTPException as he:
        # Re-raise HTTP exceptions as they are already properly formatted
        logger.error(f"HTTP exception in chat endpoint: {he.detail}")
        try:
            weaviate_client.close()
        except Exception:
            pass
        raise he
    except ValueError as ve:
        logger.error(f"Validation error in chat endpoint: {ve}", exc_info=True)
        try:
            weaviate_client.close()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Invalid request: {ve}")
    except RuntimeError as re:
        logger.error(f"Runtime error in chat endpoint: {re}", exc_info=True)
        try:
            weaviate_client.close()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Service error: {re}")
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        try:
            weaviate_client.close()
        except Exception:
            pass
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting chat bot server...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=5000)
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise
