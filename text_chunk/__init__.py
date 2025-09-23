from pathlib import Path

from fastapi import HTTPException

from app.constants import ChunkType
from app.schema import RagPayload
from text_chunk.fixed import FixedChunkProcessor
from text_chunk.hybrid import HybridChunkProcessor
from text_chunk.paragraph import ParagraphChunkProcessor
from text_chunk.sentance import SentenceChunkProcessor
from text_chunk.sliding_paragraph import ParagraphSlidingChunkProcessor
from text_chunk.sliding_window import SlidingWindowChunkProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


class ChunkPlugin:
    def process_chunk(self, payload: RagPayload, file_path: Path):
        logger.info(f"Processing chunk request: {payload.chunking}")
        logger.info(f"File path: {file_path}")

        # 2. Pick chunk processor
        chunk_types = {
            ChunkType.FIXED: FixedChunkProcessor,
            ChunkType.SENTENCE: SentenceChunkProcessor,
            ChunkType.PARAGRAPH: ParagraphChunkProcessor,
            ChunkType.SLIDING_WINDOW: SlidingWindowChunkProcessor,
            ChunkType.HYBRID: HybridChunkProcessor,
            ChunkType.SLIDING_PARAGRAPH: ParagraphSlidingChunkProcessor,
        }
        processor_cls = chunk_types.get(payload.chunking.type)

        if not processor_cls:
            logger.error(f"Unsupported chunk type: {payload.chunking.type}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported chunk type: {payload.chunking.type}",
            )

        # 3. Create processor with tokenizer
        chunk_processor = processor_cls(payload.chunking)
        return chunk_processor.process(file_path)
