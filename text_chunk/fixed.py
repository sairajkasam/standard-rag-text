from pathlib import Path
from typing import Any, Dict, List

from app.schema import FixedChunkRequest
from utils.logger import get_logger

logger = get_logger(__name__)


class FixedChunkProcessor:
    def __init__(self, payload: FixedChunkRequest):
        self.payload = payload

    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Reads the given file and splits it into overlapping character chunks.
        Returns the list of chunk dicts.
        """
        logger.info(
            f"Chunking file: {file_path} with fixed chunk size {self.payload.chunk_size} "
            f"and overlap {self.payload.overlap}"
        )

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = file_path.read_text(encoding="latin-1", errors="ignore")

        all_chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_idx = 0
        text_len = len(text)

        chunk_size = int(self.payload.chunk_size)
        overlap = int(self.payload.overlap)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]

            all_chunks.append(
                {
                    "id": f"chunk_{chunk_idx}",
                    "source": file_path.name,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                }
            )

            chunk_idx += 1
            # advance with overlap, ensure progress
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start

        logger.info(
            f"Chunk processing finished for file: {file_path.name} -> chunks: {len(all_chunks)}"
        )
        return all_chunks
