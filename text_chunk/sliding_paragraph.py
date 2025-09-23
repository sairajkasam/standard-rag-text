import itertools
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.schema import SlidingWindowChunkRequest  # assume same payload shape
from utils.logger import get_logger

logger = get_logger(__name__)


def paragraphs_from_text(text: str) -> List[str]:
    # split on blank lines (handles Gutenberg paragraphs)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # merge extremely short paras with previous to avoid tiny chunks
    merged = []
    for p in paras:
        if len(p) < 30 and merged:
            merged[-1] = merged[-1] + "\n\n" + p
        else:
            merged.append(p)
    return merged


def concat_paras(
    paras: List[str], start_idx: int, end_idx: int
) -> Tuple[str, int, int]:
    """Concatenate paras[start_idx:end_idx] and return text and char offsets relative to full text."""
    # We'll reconstruct char offsets by joining with double newline
    joined = "\n\n".join(paras[start_idx:end_idx])
    return joined


def extract_chapter_and_title(file_path: Path):
    """
    Extracts chapter index and human-readable title from a filename.
    Example: '03_a_case_of_identity.txt' -> (3, 'A Case of Identity')
    """
    name = file_path.stem  # filename without extension, e.g. "03_a_case_of_identity"

    # Split off leading digits for chapter index
    match = re.match(r"(\d+)[-_ ]?(.*)", name)
    if not match:
        return None, name.replace("_", " ").replace("-", " ").title()

    chapter_str, raw_title = match.groups()
    chapter_index = int(chapter_str)

    # Clean up raw_title -> human title
    title = raw_title.replace("_", " ").replace("-", " ").strip()
    title = title.title()

    return chapter_index, title


class ParagraphSlidingChunkProcessor:
    """
    Build overlapping chunks by concatenating paragraph units until max_chars reached,
    then slide back by stride characters (approximated by paragraph boundaries).
    Stores paragraph index range and approximate char offsets.
    """

    def __init__(self, payload: SlidingWindowChunkRequest):
        self.payload = payload

    def process(
        self, file_path: Path, story_title: str = None, chapter_index: int = 0
    ) -> List[Dict[str, Any]]:
        chapter_index, story_title = extract_chapter_and_title(file_path)
        logger.info(
            f"Paragraph sliding chunking file: {file_path} "
            f"window={self.payload.window_size}, stride={self.payload.stride}"
        )

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            full_text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            full_text = file_path.read_text(encoding="latin-1", errors="ignore")

        paras = paragraphs_from_text(full_text)
        if len(paras) == 0:
            return []

        # Precompute cumulative char offsets for paragraphs to get accurate char_start/end
        para_texts = paras
        cum_offsets = []
        offset = 0
        for p in para_texts:
            cum_offsets.append(offset)
            offset += len(p) + 2  # account for the two newline characters we join with

        max_chars = int(self.payload.window_size)
        stride = int(self.payload.stride)
        if max_chars <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be > 0")

        all_chunks: List[Dict[str, Any]] = []
        n = len(para_texts)
        start_para = 0
        chunk_idx = 0

        while start_para < n:
            # expand end_para until concatenated length >= max_chars or end of paras
            end_para = start_para
            built = ""
            while end_para < n:
                candidate = (
                    para_texts[end_para]
                    if built == ""
                    else built + "\n\n" + para_texts[end_para]
                )
                if len(candidate) > max_chars and end_para > start_para:
                    break
                built = candidate
                end_para += 1

            # fallback if single paragraph is longer than max_chars: truncate
            if built == "":
                # single paragraph exceeds max_chars; take slice
                built = para_texts[start_para][:max_chars]
                char_start = cum_offsets[start_para]
                char_end = char_start + len(built)
            else:
                char_start = cum_offsets[start_para]
                # compute end offset = start offset of end_para (if end_para <= n) + length of built
                char_end = char_start + len(built)

            chunk = {
                "chunk_id": f"para_slide_chunk_{chunk_idx}",
                "source": file_path.name,
                "story_title": story_title or "",
                "chapter_index": chapter_index,
                "chunk_index": chunk_idx,
                "paragraph_range": (start_para, max(start_para, end_para - 1)),
                "char_start": int(char_start),
                "char_end": int(char_end),
                "text": built,
            }
            all_chunks.append(chunk)
            chunk_idx += 1

            # Advance start_para according to stride (in chars). Find the paragraph index corresponding to (char_start + stride)
            target_char = char_start + stride
            # find smallest para idx such that cum_offsets[idx] >= target_char (or >)
            next_para = None
            for idx in range(start_para + 1, n):
                if cum_offsets[idx] >= target_char:
                    next_para = idx
                    break
            if next_para is None:
                # if not found, just move to end_para (progress)
                next_para = end_para
            if next_para <= start_para:
                next_para = end_para  # ensure progress
            start_para = next_para

        logger.info(f"Paragraph sliding chunking finished: {len(all_chunks)} chunks")
        return all_chunks
