from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.core.pipeline import Pipeline
from src.core.pipeline_dependencies import get_pipeline
from src.embedding.controller import EmbeddingController
from src.embedding.service import EmbeddingService
from src.llm.llm import LLM


class MessagePayload(BaseModel):
    message: str


router = APIRouter()


async def get_ai_reply(payload: MessagePayload):
    embedding_service = EmbeddingService()
    embedding_controller = EmbeddingController(embedding_service)
    llm = LLM()
    pipeline = Pipeline(embedding_controller, embedding_service, llm)
    return pipeline.call_pipeline(payload.message)


@router.post(path="/chat/ai", tags=["ai"])
async def post_chat_to_ai(payload: MessagePayload, pipeline: Pipeline = Depends(get_pipeline)):
    # content = await get_ai_reply(payload)
    content = pipeline.call_pipeline(payload.message)
    return {"response": content}
