"""engram_memory 向量操作：embedding 编码与余弦相似度等纯函数。"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.app.plugin_system.api import llm_api
from src.app.plugin_system.api.log_api import get_logger

logger = get_logger("engram_memory.vector_ops")


async def embed_texts(
    texts: list[str],
    *,
    task_name: str,
    request_name: str,
) -> list[list[float]]:
    """批量编码文本为嵌入向量。

    Args:
        texts: 待编码的文本列表。
        task_name: 模型任务名（model.toml 的 model_tasks 节）。
        request_name: LLM 请求名称，用于统计。

    Returns:
        向量列表，顺序与 ``texts`` 一致。

    Raises:
        RuntimeError: 返回条数不足或调用失败时抛出。
    """
    if not texts:
        return []
    model_set = llm_api.get_model_set_by_task(task_name)
    request = llm_api.create_embedding_request(
        model_set=model_set,
        request_name=request_name,
        inputs=texts,
    )
    response = await request.send()
    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings or len(embeddings) < len(texts):
        raise RuntimeError("embedding 返回条数不足")
    return [[float(value) for value in vec] for vec in embeddings]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度，范围 [0.0, 1.0]。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    dot_sum = float(left_array @ right_array)
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot_sum / (left_norm * right_norm)


def vector_norm_sq(vector: list[float]) -> float:
    """计算向量的 L² 平方范数（平方和）。"""
    if not vector:
        return 0.0
    vector_array = np.asarray(vector, dtype=np.float64)
    return float(vector_array @ vector_array)


def normalize_vector(vector: list[float]) -> list[float]:
    """将向量归一化为单位向量（L² 范数 = 1.0），零向量返回全零。"""
    norm_sq = vector_norm_sq(vector)
    if norm_sq <= 1e-12:
        return [0.0 for _ in vector]
    vector_array = np.asarray(vector, dtype=np.float64)
    return (vector_array / np.sqrt(norm_sq)).tolist()


def to_float_vector(
    values: Any,
    *,
    expected_dim: int | None = None,
    source: str = "unknown",
    collection_name: str = "unknown",
) -> list[float]:
    """将向量结构转换为一维 float 列表，并严格校验维度。

    Args:
        values: 待转换的向量数据（None、list、numpy 数组等）。
        expected_dim: 期望维度，None 表示不校验。
        source: 来源标识，用于错误日志。
        collection_name: 集合名称，用于错误日志。

    Returns:
        一维 float 列表；输入无效时返回空列表。

    Raises:
        ValueError: 高维输入或维度不匹配时抛出。
    """
    if values is None:
        return []
    try:
        array = np.asarray(values, dtype=np.float64)
    except Exception:  # noqa: BLE001
        return []
    if array.size <= 0:
        return []

    if array.ndim == 0:
        vector = [float(array)]
    elif array.ndim == 1:
        vector = array.tolist()
    else:
        if int(array.shape[0]) == 1:
            vector = np.asarray(array[0], dtype=np.float64).reshape(-1).tolist()
        else:
            raise ValueError(
                f"向量维度异常 collection={collection_name} source={source}: "
                f"期望 1 维，实际 shape={array.shape}"
            )

    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(
            f"向量维度不匹配 collection={collection_name} source={source}: "
            f"期望 {expected_dim}，实际 {len(vector)}"
        )
    return vector


def sanitize_vector_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """清洗写入向量库的 metadata，仅保留 str/int/float/bool 标量字段。

    ChromaDB 不支持列表、字典等复杂类型，且 None 不是合法的 MetadataValue，
    本函数会一并过滤掉。

    Args:
        metadata: 原始元数据字典，可能包含任意类型值。

    Returns:
        仅含标量字段的新字典。
    """
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
    return cleaned
