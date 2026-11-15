# CompressionBench

[![CI](https://github.com/kapildhinwa2010-dev/CompressionBench/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/CompressionBench/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/CompressionBench/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/CompressionBench/actions/workflows/codeql.yml)

Benchmark weight-compression methods: reconstruction error vs estimated storage.

## What it does

Runs four compression strategies on a matrix - SVD rank-4, 8-bit
quantization, a 1-D Fourier fit per column, and hybrid SVD+quantize - and
reports relative reconstruction error, an estimated storage size and the
implied compression ratio for each.

    from compressionbench import benchmark

    for row in benchmark(W):
        print(row["method"], row["relative_error"], row["compression_ratio"])

## API

- svd_compress(matrix, rank) - truncated SVD reconstruction
- quantize(matrix, bits) - uniform quantization round-trip
- fourier_compress(matrix, num_coeffs) - 1-D Fourier basis fit per column
- hybrid_compress(matrix, rank, bits) - SVD then quantize
- benchmark(matrix) - one row per method with error, storage, ratio

## Honest results and limitations

- Storage numbers are parameter-count estimates (e.g. rank times shape for
  SVD), not measured serialized file sizes; real formats add headers and
  entropy coding.
- Errors are relative Frobenius-norm errors against the given matrix only;
  nothing here predicts downstream model accuracy after compression.
- Fixed defaults (rank 4, 8 bits, 4 coefficients) favour smooth matrices;
  on random matrices every method shows large error.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
