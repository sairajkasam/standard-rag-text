from dataclasses import dataclass
from enum import Enum


@dataclass
class ChunkType:
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SLIDING_WINDOW = "sliding_window"
    HYBRID = "hybrid"
    SLIDING_PARAGRAPH = "sliding_paragraph"


class GenericChunkType(Enum):
    SENTENCE = ChunkType.SENTENCE
    PARAGRAPH = ChunkType.PARAGRAPH


@dataclass
class ModelProvider:
    OPENAI = "openai"
    NVIDIA = "nvidia"
    HUGGINGFACE = "huggingface"


@dataclass
class EmbeddingType:
    SPARSE = "sparse"
    DENSE = "dense"
