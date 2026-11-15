# CompressionBench - Benchmark weight-compression methods: ratio vs error
# Part of the Kapildhinwa2010-dev Toolkit.

"""CompressionBench - compression benchmark suite.

Lineage: all toolkit repos. Benchmarks four compression strategies (SVD
rank, quantization, 1-D Fourier fit, hybrid SVD+quantize) on a matrix and
reports relative reconstruction error and estimated storage. The storage
figures are parameter-count estimates, not measured file sizes.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "svd_compress",
    "quantize",
    "fourier_compress",
    "hybrid_compress",
    "benchmark",
]


def _fourier_basis(n_cols: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def _check(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    return matrix


def svd_compress(matrix: np.ndarray, rank: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if rank < 1:
        raise ValueError("rank must be >= 1")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def quantize(matrix: np.ndarray, bits: int = 8) -> np.ndarray:
    matrix = _check(matrix)
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo, hi = float(matrix.min()), float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
    return codes * scale + lo


def fourier_compress(matrix: np.ndarray, num_coeffs: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if num_coeffs < 1:
        raise ValueError("num_coeffs must be >= 1")
    basis = _fourier_basis(matrix.shape[1], num_coeffs)
    coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
    return coeffs.T @ basis.T


def hybrid_compress(matrix: np.ndarray, rank: int = 4, bits: int = 8) -> np.ndarray:
    """SVD low-rank followed by quantization."""
    return quantize(svd_compress(matrix, rank=rank), bits=bits)


def _storage_bytes(matrix: np.ndarray, method: str) -> int:
    """Rough parameter-count storage estimate; not a measured file size."""
    m, n = matrix.shape
    if method == "svd_rank4":
        return (4 * (m + n + 1)) * 8
    if method == "quantize8":
        return matrix.size
    if method == "fourier4":
        return (2 * 4 + 1) * m * 8
    if method == "hybrid":
        return 4 * (m + n + 1)
    raise ValueError("unknown method {}".format(method))


def benchmark(matrix: np.ndarray) -> list:
    """Run all methods; return error, storage estimate and ratio per method."""
    matrix = _check(matrix)
    results = []
    methods = {
        "svd_rank4": svd_compress(matrix, rank=4),
        "quantize8": quantize(matrix, bits=8),
        "fourier4": fourier_compress(matrix, num_coeffs=4),
        "hybrid": hybrid_compress(matrix, rank=4, bits=8),
    }
    norm = float(np.linalg.norm(matrix))
    original = matrix.size * 8
    for name, approx in methods.items():
        rel_error = float(np.linalg.norm(matrix - approx)) / norm if norm else 0.0
        stored = _storage_bytes(matrix, name)
        results.append({
            "method": name,
            "relative_error": rel_error,
            "estimated_storage_bytes": stored,
            "compression_ratio": original / max(stored, 1),
            "passed": rel_error < 0.5,
        })
    return results
