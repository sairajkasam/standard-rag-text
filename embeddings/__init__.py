from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings

from app.constants import EmbeddingType, ModelProvider
from app.schema import RagPayload
from embeddings.dense import dense_embedding
from embeddings.model_provider import get_open_ai_embedder
from embeddings.sparse import sparse_embedding
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbedderModel:
    def __init__(self, payload: RagPayload):
        self.payload = payload

    def get_embedder(self):
        providers = {
            ModelProvider.OPENAI: get_open_ai_embedder,
        }
        provider = providers.get(self.payload.model_provider)
        logger.info(f"Selected model provider: {self.payload.model_provider}")
        if not provider:
            raise ValueError(
                f"Unsupported model provider: {self.payload.model_provider}"
            )
        return provider(self.payload.model_name)


class ProcessEmbedding:
    def __init__(
        self,
        embedder: OpenAIEmbeddings,
        payload: RagPayload,
    ):
        self.embedder = embedder
        self.payload = payload

    def embed_texts(self, chunks):
        embedd_types = {
            EmbeddingType.SPARSE: sparse_embedding,
            EmbeddingType.DENSE: dense_embedding,
        }
        emb_func = embedd_types.get(self.payload.embedding_type)
        if not emb_func:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported embedding type: {self.payload.embedding_type}",
            )
        logger.info(f"Using embedding type: {self.payload.embedding_type}")
        return emb_func(self.embedder, chunks)
