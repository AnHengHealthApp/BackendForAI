from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """定義所有嵌入模型都必須遵守的標準介面。"""
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """將文本列表轉換為 numpy 向量陣列。"""
        pass
