"""Tests for CompressionBench: ratio vs error benchmark."""

import numpy as np
import pytest

from compressionbench import benchmark, fourier_compress, hybrid_compress, quantize, svd_compress


def _matrix():
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    return np.outer(np.sin(t), np.cos(t))


def test_svd_compress_low_rank_exact():
    W = _matrix()
    assert np.max(np.abs(svd_compress(W, rank=2) - W)) < 1e-9


def test_quantize_roundtrip_error_bounded():
    W = _matrix()
    q = quantize(W, bits=8)
    assert float(np.max(np.abs(q - W))) < 0.05
    with pytest.raises(ValueError):
        quantize(W, bits=0)


def test_fourier_compress_smooth_matrix():
    W = _matrix()
    approx = fourier_compress(W, num_coeffs=4)
    assert float(np.max(np.abs(approx - W))) < 1e-6


def test_hybrid_combines_both():
    W = _matrix()
    approx = hybrid_compress(W, rank=2, bits=8)
    assert approx.shape == W.shape


def test_benchmark_returns_all_methods_with_storage():
    rows = benchmark(_matrix())
    names = {r["method"] for r in rows}
    assert names == {"svd_rank4", "quantize8", "fourier4", "hybrid"}
    for r in rows:
        assert "relative_error" in r and "estimated_storage_bytes" in r
        assert r["compression_ratio"] > 0


def test_check_rejects_bad_input():
    with pytest.raises(ValueError):
        benchmark(np.zeros(4))
