import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    WEAVIATE_HOST: str = os.getenv("WEAVIATE_HOST")
    WEAVIATE_PORT: str = os.getenv("WEAVIATE_PORT", "8080")
    WEAVIATE_GRPC_PORT: str = os.getenv("WEAVIATE_GRPC_PORT", "8090")
