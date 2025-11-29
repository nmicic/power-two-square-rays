# Theta Toolkit - CUDA & CPU Implementation

**Version**: v1.2
**Date**: November 2025 (Spec finalized 2025-11-28)
**Repository**: https://github.com/nmicic/power-two-square-rays/

> This README describes the **Theta Toolkit** (CUDA + CPU), which is the supported
> implementation part of this repository. The visualization/math-art part is documented
> in the root [`README.md`](../README.md).

---

## Overview

Integer-native angular encoding via 2-adic decomposition. Every positive integer `n` decomposes uniquely as `n = 2^v2(n) × core(n)`, where `theta_key(n) = bit_reverse(core(n))`.

**Key Result**: theta_key matches xxHash/Murmur3 performance on GPU (~30,000 M ops/sec) while preserving mathematical structure.

---

## Files

### Specification

| File | Description |
|------|-------------|
| `cuda/THETA_SPEC_v1.2.md` | Complete specification with Ray Embedding (Prism View) |

### Python

| File | Description |
|------|-------------|
| `python/theta_toolkit.py` | CPU reference toolkit: primitives, theta_key, theta_bucket, ray embedding |

### CUDA

| File | Description |
|------|-------------|
| `cuda/theta_cuda_v1.2.cuh` | Header-only GPU library |
| `cuda/theta_cuda_benchmark_v1.2.cu` | GPU benchmark with pattern modes |
| `cuda/THETA_CUDA_BENCHMARK_README_v1.2` | Benchmark user guide |

---

## Quick Start

### Python

```python
from python.theta_toolkit import theta_key, v2, odd_core, theta_bucket

n = 12
print(f"n={n}, v2={v2(n)}, core={odd_core(n)}, theta_key={theta_key(n)}")
# n=12, v2=2, core=3, theta_key=3

# Bucketing (uniform distribution)
bucket = theta_bucket(n, num_buckets=64)
```

### CUDA

```bash
# Compile
nvcc -O3 -arch=sm_89 cuda/theta_cuda_benchmark_v1.2.cu -o theta_bench

# Run benchmark
./theta_bench 10 64 1 0    # 10M, 64 buckets, full-range, random
./theta_bench 10 64 1 1    # Powers of two pattern (2-adic test)
```

### In Your Code

```cpp
#include "theta_cuda_v1.2.cuh"

__global__ void my_kernel(uint32_t* data, uint32_t* buckets, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buckets[idx] = theta::theta_bucket_32(data[idx], 64);
    }
}
```

---

## Benchmark Results (RTX 4000 Ada, 10M flows)

| Hash | 32-bit (M/s) | 64-bit (M/s) | Ratio |
|------|--------------|--------------|-------|
| theta_key | 30,030 | 15,082 | 1.00x |
| xxhash | 30,002 | 15,071 | 1.00x |
| murmur3 | 30,141 | 15,036 | 1.00x |
| crc32 | 29,593 | - | 0.99x |

**Bucket Uniformity** (chi-square, 64 buckets):
- theta_bucket: 73.1 - GOOD ✓
- xxhash32: 50.4 - GOOD ✓

---

## Version History

- **v1.2**: Ray Embedding (Prism View), pattern modes in benchmark
- **v1.1**: Fixed theta_bucket uniformity, core_bits for reconstruction
- **v1.0**: Initial release

---

## Disclaimer

This is an AI-assisted exploratory project. Educational and experimental only—not peer-reviewed.

**Not cryptographic**: theta_key is deterministic and trivially reversible. Do NOT use for protecting secrets or in security-critical applications. Use real, vetted crypto libraries (e.g., libsodium, OpenSSL) for security.

Portions of this code include AI-assisted generation (ChatGPT, Claude). All work reviewed and validated by the author.
