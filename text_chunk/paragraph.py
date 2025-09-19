import re
from pathlib import Path
from typing import Any, Dict, List

from app.schema import GenericChunkRequest
from utils.logger import get_logger

logger = get_logger(__name__)


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs by blank lines.
    Keeps paragraphs that contain non-whitespace characters.
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on two or more newlines (allow spaces on blank lines)
    parts = re.split(r"\n\s*\n+", text)
    paragraphs = [p.strip() for p in parts if p and p.strip()]
    return paragraphs


class ParagraphChunkProcessor:
    def __init__(self, payload: GenericChunkRequest):
        self.payload = payload

    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Reads the file, splits into paragraphs, and returns chunk dicts.
        Each chunk corresponds to one paragraph.
        """
        logger.info(f"Paragraph-based chunking file: {file_path}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = file_path.read_text(encoding="latin-1", errors="ignore")

        paragraphs = split_into_paragraphs(text)

        all_chunks: List[Dict[str, Any]] = []
        for idx, para in enumerate(paragraphs):
            all_chunks.append(
                {
                    "id": f"para_chunk_{idx}",
                    "source": file_path.name,
                    "chunk_index": idx,
                    "text": para,
                }
            )

        logger.info(
            f"Paragraph chunking finished for file: {file_path.name} -> chunks: {len(all_chunks)}"
        )
        return all_chunks
