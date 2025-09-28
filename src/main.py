from dotenv import load_dotenv

from src.core.pipeline import Pipeline
from src.embedding.controller import EmbeddingController
from src.embedding.service import EmbeddingService
from src.llm.llm import LLM


def main():
    load_dotenv()

    try:
        embedding_service = EmbeddingService()
        embedding_controller = EmbeddingController(embedding_service)
        llm = LLM()
        pipline = Pipeline(embedding_controller, embedding_service, llm)

        queries = """
          我的身高為:178.00公分，體重為:77.00公斤，年齡:21歲，性別:男。
            最近 7 天血糖紀錄：
            2025-09-20 20:35:00 血糖值為:90.00 測量情境:餐後
            2025-09-20 20:35:00 血糖值為:65.00 測量情境:空腹
            2025-09-21 20:30:00 血糖值為:102.00 測量情境:餐後
            2025-09-21 20:30:00 血糖值為:85.00 測量情境:空腹
            最近 7 天血壓紀錄：
            2025-09-20 20:35:00 舒張壓:145 收縮壓為:95 心跳速率:90
            用戶問題：天氣會影響血壓嗎？
            """

        ans = pipline.call_pipeline(queries)
        print(ans)


    except Exception as e:
        print(f"程式執行時發生錯誤: {e}")


if __name__ == "__main__":
    main()
