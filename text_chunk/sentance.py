import re
from pathlib import Path
from typing import Any, Dict, List

from app.schema import GenericChunkRequest
from utils.logger import get_logger

logger = get_logger(__name__)


def regex_sentence_split(text: str) -> List[str]:
    """
    Split text into sentences using a regex.

    This uses a conservative rule: split at punctuation (.!?)
    followed by whitespace and a capital letter or quote, or line breaks.
    It's not perfect (no tokenizer is) but works for normal prose
    without needing external dependencies.
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Basic sentence splitter:
    # split on punctuation (?!.), followed by space and a capital letter / quote,
    # or when followed by a newline.
    pattern = r"(?<=[\.\!\?])\s+(?=(?:[\"\'“”]?[A-Z0-9]))|(?<=\n)"
    parts = re.split(pattern, text)

    # Clean up whitespace and remove empty strings
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences


class SentenceChunkProcessor:
    def __init__(self, payload: GenericChunkRequest):
        self.payload = payload

    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Reads the given file, splits it into sentences using regex,
        and returns a list of chunk dicts (one sentence per chunk).
        """
        logger.info(f"Sentence-based chunking (regex) file: {file_path}")

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = file_path.read_text(encoding="latin-1", errors="ignore")

        sentences = regex_sentence_split(text)

        all_chunks: List[Dict[str, Any]] = []
        for idx, sentence in enumerate(sentences):
            all_chunks.append(
                {
                    "id": f"sent_chunk_{idx}",
                    "source": file_path.name,
                    "chunk_index": idx,
                    "text": sentence,
                }
            )

        logger.info(
            f"Sentence-based chunking finished for file: {file_path.name} -> chunks: {len(all_chunks)}"
        )
        logger.debug(f"Chunks: {all_chunks[:3]}...")  # log first 3 chunks
        return all_chunks
