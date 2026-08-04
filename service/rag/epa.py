"""EPA 向量重塑：局部 SVD 子空间 + 标签三角动力学。

从 booku_memory 移植的纯函数算法，用于检索时对查询向量按
三元标签（core/diffusion/opposing）进行语义重塑，以及写入时
计算新颖度能量比（去重决策）。
"""

from __future__ import annotations

import math

import numpy as np

from .vector_ops import normalize_vector, vector_norm_sq


def power_iteration(
    matrix: list[list[float]],
    *,
    iterations: int = 24,
) -> tuple[float, list[float]]:
    """对称矩阵幂迭代，返回最大特征值与特征向量。"""
    size = len(matrix)
    if size == 0:
        return 0.0, []
    matrix_array = np.asarray(matrix, dtype=np.float64)
    vector = np.ones(size, dtype=np.float64)
    for _ in range(iterations):
        next_vector = matrix_array @ vector
        norm_sq = float(next_vector @ next_vector)
        if norm_sq <= 1e-12:
            return 0.0, [0.0 for _ in range(size)]
        vector = next_vector / math.sqrt(norm_sq)
    mv = matrix_array @ vector
    eigenvalue = float(vector @ mv)
    return eigenvalue, vector.tolist()


def build_local_svd_basis(vectors: list[list[float]]) -> list[list[float]]:
    """通过邻域向量构建局部 SVD 子空间正交基。

    对输入向量集做 SVD，提取累计能量前 90% 的奇异向量（归一化）作为子空间基。

    Args:
        vectors: 输入向量列表，长度须一致且非零。

    Returns:
        正交归一化局部基向量列表；输入为空或无效时返回空列表。
    """
    if not vectors:
        return []
    first_dim = len(vectors[0]) if vectors[0] else 0
    valid_vectors = [
        vector
        for vector in vectors
        if vector
        and len(vector) == first_dim
        and vector_norm_sq(vector) > 1e-12
    ]
    if not valid_vectors:
        return []

    matrix = np.asarray(valid_vectors, dtype=np.float64)
    _, singular_values, vh_matrix = np.linalg.svd(matrix, full_matrices=False)
    singular_energy = singular_values * singular_values
    total_trace = float(np.sum(singular_energy))
    if total_trace <= 1e-12:
        return []

    basis: list[list[float]] = []
    explained = 0.0
    for index, energy in enumerate(singular_energy.tolist()):
        if energy <= 1e-8:
            break
        direction = vh_matrix[index]
        normalized = normalize_vector(direction.tolist())
        if vector_norm_sq(normalized) <= 1e-12:
            break
        basis.append(normalized)
        explained += float(energy)
        if explained / total_trace >= 0.9:
            break
    return basis


def projection_entropy_logic_depth(
    query_vector: list[float],
    evidence_vectors: list[list[float]],
) -> float:
    """基于投影熵计算检索子空间的逻辑深度（L = 1 - H/log₂K）。

    Args:
        query_vector: 查询向量。
        evidence_vectors: 初始检索得到的证据向量集合。

    Returns:
        逻辑深度，范围 [0.0, 1.0]；证据为空时返回 0.0。
    """
    basis = build_local_svd_basis(evidence_vectors)
    if not basis:
        return 0.0
    basis_matrix = np.asarray(basis, dtype=np.float64)
    query_array = np.asarray(query_vector, dtype=np.float64)
    coefficients = basis_matrix @ query_array
    energies = np.maximum(0.0, coefficients * coefficients)
    total_energy = float(np.sum(energies))
    if total_energy <= 1e-12:
        return 0.0
    probs = energies / total_energy
    probs = probs[probs > 1e-12]
    if probs.size <= 1:
        return 1.0
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(int(probs.size))
    if max_entropy <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, 1.0 - entropy / max_entropy))


def estimate_resonance(
    query_text: str,
    query_core_tags: set[str],
    query_diffusion_tags: set[str],
    query_opposing_tags: set[str],
) -> bool:
    """估算当前查询是否具有跨域共振特征。

    共振判定 = 多标签组同时涉及，或文本中含有跨域标志词。

    Args:
        query_text: 查询字符串，用于检测跨域标志词。
        query_core_tags: 核心标签集合。
        query_diffusion_tags: 扩散标签集合。
        query_opposing_tags: 对立标签集合。

    Returns:
        True 表示存在跨域共振，否则 False。
    """
    explicit_domain_count = sum(
        1
        for tag_set in (query_core_tags, query_diffusion_tags, query_opposing_tags)
        if len(tag_set) > 0
    )
    if explicit_domain_count >= 2:
        return True
    markers = ("并且", "同时", "以及", "cross", "across", "对比")
    lower_text = query_text.lower()
    return any(marker in lower_text for marker in markers)


def weighted_centroid(
    query_vector: list[float],
    vectors_with_weight: list[tuple[list[float], float]],
) -> list[float]:
    """计算带权语义中心。

    Args:
        query_vector: 查询向量，用于长度对齐校验。
        vectors_with_weight: (embedding, weight) 元组列表。

    Returns:
        带权平均向量；无有效向量时返回全零向量。
    """
    if not query_vector:
        return []
    valid_vectors: list[np.ndarray] = []
    valid_weights: list[float] = []
    for vector, weight in vectors_with_weight:
        if len(vector) != len(query_vector):
            continue
        if weight <= 1e-12:
            continue
        valid_vectors.append(np.asarray(vector, dtype=np.float64))
        valid_weights.append(float(weight))
    if not valid_vectors:
        return [0.0 for _ in query_vector]
    matrix = np.vstack(valid_vectors)
    weight_array = np.asarray(valid_weights, dtype=np.float64)
    total_weight = float(np.sum(weight_array))
    if total_weight <= 1e-12:
        return [0.0 for _ in query_vector]
    centroid = (weight_array @ matrix) / total_weight
    return centroid.tolist()


def reshape_query_vector(
    query_vector: list[float],
    *,
    beta: float,
    core_vectors: list[tuple[list[float], float]],
    diffusion_vectors: list[tuple[list[float], float]],
    opposing_vectors: list[tuple[list[float], float]],
    energy_cutoff: float,
) -> list[float]:
    """根据 TAG 三角标签动力学将查询向量重塑为更精确的方向。

    重塑公式：``reshaped = (1-beta)*query + beta*(core + diffusion_residual - opposing)``。
    扩散向量采用残差能量领导的正交化策略，避免线性相关扩散方向膨胀。

    Args:
        query_vector: 原始查询向量。
        beta: 重塑强度 [0.0, 1.0]。
        core_vectors: 核心标签匹配记忆的 (embedding, weight) 列表。
        diffusion_vectors: 扩散标签匹配记忆的 (embedding, weight) 列表。
        opposing_vectors: 对立标签匹配记忆的 (embedding, weight) 列表。
        energy_cutoff: 扩散向量展入阈值，残差能量比低于此则忽略。

    Returns:
        幂-2 归一化后的重塑向量；输入为空或归一化失败时返回空列表。
    """
    if not query_vector:
        return []
    query_array = np.asarray(query_vector, dtype=np.float64)
    core_term = weighted_centroid(query_vector, core_vectors)
    opposing_term = weighted_centroid(query_vector, opposing_vectors)
    core_array = np.asarray(core_term, dtype=np.float64)
    opposing_array = np.asarray(opposing_term, dtype=np.float64)

    diffusion_array = np.zeros_like(query_array)
    basis_arrays: list[np.ndarray] = []
    for vector, weight in diffusion_vectors:
        if len(vector) != len(query_vector) or weight <= 1e-12:
            continue
        vector_array = np.asarray(vector, dtype=np.float64)
        if basis_arrays:
            basis_matrix = np.vstack(basis_arrays)
            projection = basis_matrix.T @ (basis_matrix @ vector_array)
        else:
            projection = np.zeros_like(vector_array)
        residual = vector_array - projection
        residual_energy = float(residual @ residual)
        total_energy = float(vector_array @ vector_array)
        if total_energy <= 1e-12:
            continue
        ratio = residual_energy / total_energy
        if ratio < energy_cutoff:
            continue
        residual_norm = math.sqrt(residual_energy)
        if residual_norm <= 1e-12:
            continue
        normalized_residual = residual / residual_norm
        basis_arrays.append(normalized_residual)
        diffusion_array += normalized_residual * float(weight)

    reshaped = (1.0 - beta) * query_array + beta * (
        core_array + diffusion_array - opposing_array
    )
    return normalize_vector(reshaped.tolist())


def novelty_energy_ratio(
    new_vector: list[float],
    basis_vectors: list[list[float]],
) -> float:
    """计算新向量相对已有向量局部子空间的新颖度能量比。

    值越近 1.0 表示差异大（内容新），越近 0.0 表示内容重复。

    Args:
        new_vector: 待评估的输入向量。
        basis_vectors: 代表已有内容的邻域向量采样集合。

    Returns:
        新颖度能量比，范围 [0.0, 1.0]；向量集为空时返回 1.0（视为全新）。
    """
    if not basis_vectors:
        return 1.0
    svd_basis = build_local_svd_basis(basis_vectors)
    if not svd_basis:
        return 1.0
    basis_matrix = np.asarray(svd_basis, dtype=np.float64)
    vector_array = np.asarray(new_vector, dtype=np.float64)
    projection = basis_matrix.T @ (basis_matrix @ vector_array)
    residual = vector_array - projection
    residual_energy = float(residual @ residual)
    total_energy = float(vector_array @ vector_array)
    if total_energy <= 1e-12:
        return 0.0
    return residual_energy / total_energy
