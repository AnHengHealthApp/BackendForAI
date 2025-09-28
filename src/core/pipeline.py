class Pipeline(object):
    def __init__(self, embedding_controller, embedding_service, llm):
        self.embedding_controller = embedding_controller
        self.embedding_service = embedding_service
        self.llm = llm

    def call_pipeline(self, message):
        """整體流程 rag -> llm"""
        rag_result = self.embedding_controller.data_search(message)
        llm_response = self.llm.chat(rag_result)
        return llm_response
