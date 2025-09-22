import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from app.schema import HybridChunkRequest
from utils.logger import get_logger

logger = get_logger(__name__)


def _make_id(prefix: str = "hyb") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def regex_sentence_split(text: str) -> List[str]:
    """
    Lightweight sentence splitter using regex. Good for normal prose.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = r"(?<=[\.\!\?])\s+(?=(?:[\"\'“”]?[A-Z0-9]))|(?<=\n)"
    parts = re.split(pattern, text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences


class HybridChunkProcessor:
    """
    Hybrid sentence-based chunker:
      - groups whole sentences into chunks
      - stops grouping when max_chars or max_sentences is reached
      - supports overlap by number of sentences (overlap_sentences)
    Payload (HybridChunkRequest) may include optional fields:
      - max_chars: int (default 1000)
      - max_sentences: int (default 10)
      - overlap_sentences: int (default 1)
      - min_sentences: int (default 1)
    """

    def __init__(self, payload: HybridChunkRequest):
        # payload must exist for compatibility; read optional fields with safe defaults
        self.payload = payload
        # defaults
        self.max_chars = int(getattr(payload, "max_chars", 1000))
        self.max_sentences = int(getattr(payload, "max_sentences", 10))
        self.overlap_sentences = int(getattr(payload, "overlap_sentences", 1))
        self.min_sentences = int(getattr(payload, "min_sentences", 1))

        if self.max_chars <= 0:
            raise ValueError("max_chars must be > 0")
        if self.max_sentences <= 0:
            raise ValueError("max_sentences must be > 0")
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        if self.min_sentences < 1:
            raise ValueError("min_sentences must be >= 1")

    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Read file, split into sentences, group sentences into hybrid chunks.
        Returns list of chunk dicts.
        """
        logger.info(
            f"Hybrid chunking file: {file_path} max_chars={self.max_chars} "
            f"max_sentences={self.max_sentences} overlap_sentences={self.overlap_sentences}"
        )

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw = file_path.read_text(encoding="latin-1", errors="ignore")

        sentences = regex_sentence_split(raw)
        total_sentences = len(sentences)
        logger.info(f"Total sentences found: {total_sentences}")

        all_chunks: List[Dict[str, Any]] = []
        i = 0
        chunk_idx = 0

        while i < total_sentences:
            # Start forming a chunk from sentence i
            current_sentences = []
            current_chars = 0
            s_count = 0
            j = i
            while j < total_sentences:
                sent = sentences[j]
                # if adding this sentence would exceed either limit, break
                if (s_count + 1 > self.max_sentences) or (
                    current_chars + len(sent) > self.max_chars
                    and s_count >= self.min_sentences
                ):
                    break
                current_sentences.append(sent)
                current_chars += len(sent) + 1  # +1 for space/newline
                s_count += 1
                j += 1

            # If we couldn't add even min_sentences due to a very long single sentence,
            # force include one sentence to make progress.
            if s_count == 0 and j < total_sentences:
                current_sentences.append(sentences[j])
                j += 1
                s_count = 1

            chunk_text = " ".join(current_sentences).strip()
            all_chunks.append(
                {
                    "id": _make_id("hyb"),
                    "source": file_path.name,
                    "chunk_index": chunk_idx,
                    "text": chunk_text
                }
            )

            chunk_idx += 1

            # compute next start index using overlap_sentences
            next_i = j - self.overlap_sentences
            # ensure progress
            if next_i <= i:
                next_i = j
            i = next_i

        logger.info(
            f"Hybrid chunking finished for file: {file_path.name} -> chunks: {len(all_chunks)}"
        )
        logger.debug(f"Chunks: {all_chunks[:3]}...")  # log first 3 chunks
        return all_chunks
