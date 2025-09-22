from langchain_openai import OpenAIEmbeddings

from app.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


def get_open_ai_embedder(model: str):
    logger.info(f"Creating OpenAI embedder for model: {model}")
    return OpenAIEmbeddings(model=model, api_key=Config.OPENAI_API_KEY)
