"""可插拔的文本向量化封装

优先使用 sentence-transformers，若不可用则返回 None，由调用方回退。
模型名称可通过环境变量 EMBEDDING_MODEL 配置，默认 BAAI/bge-small-en-v1.5。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Iterable

from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_embedder() -> Callable[[str], Iterable[float]] | None:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        logger.info(f"加载 embedding 模型: {model_name}")
        model = SentenceTransformer(model_name)

        def _embed(text: str) -> list[float]:
            if not text:
                return []
            vec = model.encode([text])[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()

        return _embed
    except Exception as e:
        logger.warning(f"无法加载 embedding 模型 {model_name}，将回退默认嵌入: {e}")
        return None
