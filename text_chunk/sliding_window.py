from pathlib import Path
from typing import Any, Dict, List

from app.schema import SlidingWindowChunkRequest
from utils.logger import get_logger

logger = get_logger(__name__)


class SlidingWindowChunkProcessor:
    def __init__(self, payload: SlidingWindowChunkRequest):
        self.payload = payload

    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Reads the given file and splits it into overlapping chunks
        using a sliding window approach.
        """
        logger.info(
            f"Sliding window chunking file: {file_path} "
            f"window_size={self.payload.window_size}, stride={self.payload.stride}"
        )

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = file_path.read_text(encoding="latin-1", errors="ignore")

        text_len = len(text)
        all_chunks: List[Dict[str, Any]] = []

        window_size = int(self.payload.window_size)
        stride = int(self.payload.stride)

        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        start = 0
        chunk_idx = 0

        while start < text_len:
            end = min(start + window_size, text_len)
            chunk_text = text[start:end]

            all_chunks.append(
                {
                    "id": f"slide_chunk_{chunk_idx}",
                    "source": file_path.name,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                }
            )

            chunk_idx += 1
            start += stride

        logger.info(
            f"Sliding window chunking finished for file: {file_path.name} -> chunks: {len(all_chunks)}"
        )
        return all_chunks
