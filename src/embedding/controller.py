class EmbeddingController:
    """embedding 控制器，負責處理 user 回應並呼叫 embedding 服務"""

    def __init__(self, embedding_service):
        self.similarity_threshold = 0
        self.embedding_service = embedding_service

    def data_search(self, user_question):
        """搜尋用戶提問並回應"""
        results = []
        user_question = "".join(user_question)

        res = self.embedding_service.search_docs(user_question, top_n=3)

        for index, row in res.iterrows():
            similarity = row['similarities']
            medicine = row['medicine']
            completion = row['completion']

            if similarity > self.similarity_threshold:
                reply = f"藥品名稱：{medicine}\n說明：{completion}"
                results.append(f"{len(results) + 1}. {reply}")

        if results:
            output = "\n".join(results)
            all_output = f"問題：{user_question}\n回覆參考\n{output}"

            return all_output
        else:
            output = "請重新詢問！"

            return output
