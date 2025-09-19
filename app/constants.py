from dataclasses import dataclass


@dataclass
class ChunkType:
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SLIDING_WINDOW = "sliding_window"
    HYBRID = "hybrid"


@dataclass
class ModelProvider:
    OPENAI = "openai"
    NVIDIA = "nvidia"
    HUGGINGFACE = "huggingface"
