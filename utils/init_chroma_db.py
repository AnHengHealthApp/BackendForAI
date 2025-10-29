"""
初始化 Chroma Vector Database
讀取 medicine_completion.jsonl 並建立向量索引
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "medicine_completion.jsonl"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

def get_embeddings_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )

    return [data.embedding for data in response.data]

def initialize_chroma_db():
    print("🚀 開始初始化 Chroma Vector Database...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"找不到資料檔案: {DATA_FILE}\n"
            f"請確認 medicine_completion.jsonl 已放入 data/ 資料夾"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    try:
        chroma_client.delete_collection(name="medicine_db")
        print("🗑️  已刪除舊的 collection")
    except:
        pass

    collection = chroma_client.create_collection(
        name="medicine_db",
        metadata={"description": "醫藥資料向量資料庫"}
    )

    print(f"📖 讀取資料檔案: {DATA_FILE}")
    medicines = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                medicines.append(json.loads(line))

    print(f"✅ 共載入 {len(medicines)} 筆醫藥資料")

    batch_size = 100 # 500~1000
    total_batches = (len(medicines) + batch_size - 1) // batch_size

    for i in range(0, len(medicines), batch_size):
        batch = medicines[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"🔄 處理批次 {batch_num}/{total_batches} ({len(batch)} 筆資料)...")

        ids = []
        documents = []
        metadatas = []
        texts_for_embedding = []

        for idx, medicine in enumerate(batch):
            medicine_name = medicine.get("medicine", "")
            completion = medicine.get("completion", "")

            combined_text = f"{medicine_name}: {completion}"

            ids.append(f"med_{i + idx}")
            documents.append(combined_text)
            metadatas.append({
                "medicine": medicine_name,
                "completion": completion
            })
            texts_for_embedding.append(combined_text)

        embeddings = get_embeddings_batch(texts_for_embedding, client)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    print(f"✨ 成功建立向量資料庫！")
    print(f"📊 總計儲存: {len(medicines)} 筆資料")
    print(f"💾 資料庫位置: {CHROMA_DB_PATH}")

    count = collection.count()
    print(f"✅ 驗證完成，資料庫內有 {count} 筆記錄")

    return collection

if __name__ == "__main__":
    try:
        initialize_chroma_db()
        print("\n🎉 初始化完成！現在可以啟動 API 服務了。")
    except Exception as e:
        print(f"\n❌ 初始化失敗: {e}")
        raise