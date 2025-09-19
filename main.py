import concurrent.futures
import os
import pathlib
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.schema import RagPayload
from text_chunk import ChunkPlugin
from utils.logger import get_logger

app = FastAPI()
logger = get_logger(__name__)


def _process_file(payload: RagPayload, text_file: pathlib.Path) -> Dict[str, Any]:
    """
    Worker executed in threadpool for each file.
    Returns a dict with filename, chunk_count, and optional error.
    """
    try:
        logger.info(f"Worker starting for file: {text_file.name}")

        chunk_plugin = ChunkPlugin()
        result = chunk_plugin.process_chunk(payload, text_file)
        count = len(result) if result is not None else 0

        return {"filename": text_file.name, "chunks": count, "error": None}
    except Exception as exc:
        logger.exception(f"Error processing file {text_file.name}: {exc}")
        return {"filename": text_file.name, "chunks": 0, "error": str(exc)}


@app.post("/standard-rag-text")
async def standard_rag_text(payload: RagPayload):
    logger.info("Processing RAG text request")

    data_dir = pathlib.Path("data")
    if not data_dir.exists():
        logger.warning("data directory does not exist - creating it")
        data_dir.mkdir(parents=True, exist_ok=True)

    text_files = list(data_dir.glob("*.txt"))
    logger.info(f"Found text files length: {len(text_files)}")

    if not text_files:
        logger.warning("No .txt files found in data directory")
        return JSONResponse({"detail": "no text files found"}, status_code=404)

    # choose a reasonable max_workers
    cpu_count = os.cpu_count() or 1
    max_workers = min(32, len(text_files), cpu_count * 5)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit all file-processing tasks
        futures = {executor.submit(_process_file, payload, f): f for f in text_files}

        # collect results as they finish
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)

    # build a summary to return
    total_chunks = sum(r["chunks"] for r in results)
    errors = [r for r in results if r["error"]]

    logger.info(
        f"All files processed. Total chunks: {total_chunks}. Errors: {len(errors)}"
    )

    response = {
        "file_results": results,
        "total_files": len(text_files),
        "total_chunks": total_chunks,
        "errors": errors,
    }

    status = 200 if not errors else 207  # 207 Partial Success if some files failed
    return JSONResponse(response, status_code=status)
