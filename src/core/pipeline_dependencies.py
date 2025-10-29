from functools import lru_cache

from src.core.pipeline import Pipeline
from src.embedding.controller import EmbeddingController
from src.embedding.service import EmbeddingService
from src.llm.llm import LLM


@lru_cache(maxsize=None)
def get_embedding_service() -> EmbeddingService:
    print("🚀 正在初始化 EmbeddingService (只應出現一次)")
    return EmbeddingService()


@lru_cache(maxsize=None)
def get_llm() -> LLM:
    print("🚀 正在初始化 LLM (只應出現一次)")
    return LLM()


@lru_cache(maxsize=None)
def get_embedding_controller() -> EmbeddingController:
    print("🚀 正在初始化 EmbeddingController (只應出現一次)")
    return EmbeddingController(embedding_service=get_embedding_service())


@lru_cache(maxsize=None)
def get_pipeline() -> Pipeline:
    print("🚀 正在初始化 Pipeline (只應出現一次)")
    return Pipeline(
        embedding_controller=get_embedding_controller(),
        embedding_service=get_embedding_service(),
        llm=get_llm()
    )
