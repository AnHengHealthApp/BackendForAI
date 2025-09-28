import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

# 輸入的原始 .jsonl 檔案路徑
INPUT_FILE_PATH = "D:\Files\\3.Coding\Python\AnHengHealthAppAI\data\medicine_completion_test.jsonl"

# 處理完後要輸出的 .jsonl 檔案路徑
# **建議使用與輸入不同的檔名，避免覆蓋原始檔案**
OUTPUT_FILE_PATH = "../data/medicine_with_embeddings.jsonl"

# 要使用的 Embedding 模型
EMBEDDING_MODEL = "text-embedding-3-small"

# 每次送出給 API 處理的資料筆數 (Batch Size)
BATCH_SIZE = 500


# --- 2. 主程式邏輯 ---

def get_embeddings_in_batches(client, texts, model, batch_size):
    """
    以批次方式取得 embeddings，並顯示進度條。
    """
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="正在產生 Embeddings"):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"處理批次 {i // batch_size} 時發生錯誤: {e}")
            all_embeddings.extend([None] * len(batch))
            time.sleep(5)

    return all_embeddings


def main():
    """
    主執行函式
    """
    print("--- 開始前處理腳本 (JSONL 版本) ---")

    if not API_KEY:
        raise ValueError("OpenAI API 金鑰未設定，請在腳本中或環境變數中設定 API_KEY。")
    client = OpenAI(api_key=API_KEY)

    # --- 主要修改：讀取 .jsonl 檔案 ---
    try:
        print(f"正在從 {INPUT_FILE_PATH} 讀取資料...")
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]

        df = pd.DataFrame(data)
        print(f"成功讀取 {len(df)} 筆資料。")
    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 {INPUT_FILE_PATH}。請確認檔案路徑和名稱是否正確。")
        return
    except json.JSONDecodeError as e:
        print(f"錯誤：解析 JSONL 檔案時發生錯誤。請檢查檔案格式是否正確，每一行都應為一個獨立的 JSON 物件。錯誤訊息: {e}")
        return
    # --- 修改結束 ---

    df.dropna(subset=['medicine', 'completion'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("正在合併 'medicine' 與 'completion' 欄位...")
    df['combined'] = df.apply(
        lambda row: f"藥品名稱：{row['medicine']}\n主要用途與說明：{row['completion']}",
        axis=1
    )

    texts_to_embed = df['combined'].tolist()
    embeddings = get_embeddings_in_batches(client, texts_to_embed, EMBEDDING_MODEL, BATCH_SIZE)

    df['embedding'] = embeddings
    df.dropna(subset=['embedding'], inplace=True)
    df.drop(columns=['combined'], inplace=True)

    print(f"正在將結果儲存至 {OUTPUT_FILE_PATH}...")
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        for record in df.to_dict('records'):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"--- 處理完成！---")
    print(f"成功處理並儲存了 {len(df)} 筆資料到 {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main()
