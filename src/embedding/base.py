from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """將文本列表轉換為 numpy 向量陣列。"""
        pass
