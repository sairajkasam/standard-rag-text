from typing import Generic, Optional, TypeVar, Union

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class FixedChunkRequest(BaseModel):
    type: str
    chunk_size: int
    overlap: int


class GenericChunkRequest(BaseModel):
    type: str


class SlidingWindowChunkRequest(BaseModel):
    type: str
    window_size: int
    stride: int


class HybridChunkRequest(BaseModel):
    type: str
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
    model_name: str
