import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        filepath = "./data/medicine_with_embeddings.jsonl"
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        self.df = pd.DataFrame(data)

        print(self.df)

        self.df['embedding'] = self.df['embedding'].apply(np.array)

    def cosine_similarity(self, a, b):
        """計算兩個向量的cosine similarity"""
        embedding = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return embedding

    def get_embedding(self, text, model="text-embedding-3-small"):
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def search_docs(self, user_query, top_n=3):
        embedding_user = self.get_embedding(user_query)

        self.df["similarities"] = self.df.embedding.apply(lambda x: self.cosine_similarity(x, embedding_user))
        res = self.df.sort_values("similarities", ascending=False).head(top_n)
        return res
