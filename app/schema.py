from typing import Generic, Optional, TypeVar, Union

from pydantic import BaseModel, Field

from app.constants import ChunkType, GenericChunkType

DataT = TypeVar("DataT")


class FixedChunkRequest(BaseModel):
    type: str = Field(ChunkType.FIXED)
    chunk_size: int
    overlap: int


class GenericChunkRequest(BaseModel):
    type: GenericChunkType


class SlidingWindowChunkRequest(BaseModel):
    type: str = Field(ChunkType.SLIDING_WINDOW)
    window_size: int
    stride: int


class HybridChunkRequest(BaseModel):
    type: str = Field(ChunkType.HYBRID)
    max_chars: Optional[int] = Field(1000)
    max_sentences: Optional[int] = Field(10)
    overlap_sentences: Optional[int] = Field(1)
    min_sentences: Optional[int] = Field(1)


class RagRequest(BaseModel, Generic[DataT]):
    chunking: DataT


# A generic payload that allows either Fixed or Variable chunk request
ChunkRequest = Union[
    FixedChunkRequest,
    GenericChunkRequest,
    SlidingWindowChunkRequest,
    HybridChunkRequest,
]


class RagPayload(RagRequest[ChunkRequest]):
    model_provider: str
    model_name: str = None
    embedding_type: str
    index_name: str
    vector_database: str
    recreate_index: bool = False
