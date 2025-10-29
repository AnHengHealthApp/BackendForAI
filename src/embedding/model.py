import os

import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .base import EmbeddingModel


class LocalModel(EmbeddingModel):
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"LocalModel 使用的設備: {self.device}")
        print(f"正在載入本地模型: {model_name}...")

        model_kwargs = {"attn_implementation": "flash_attention_2"} if self.device == "cuda" else {}
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs=model_kwargs
        ).to(self.device)
        print("本地模型載入成功！")

    def encode(self, texts: list[str]) -> np.ndarray:
        # SentenceTransformer 的 encode 方法已經回傳 numpy array，直接使用即可
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )


class OpenAIModel(EmbeddingModel):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def encode(self, texts: list[str]) -> np.ndarray:
        """透過 API 呼叫 OpenAI 並取得 embedding vector。"""
        if not texts:
            return np.array([])

        texts = [t.replace("\n", " ") for t in texts]

        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
