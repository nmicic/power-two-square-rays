# ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
# This directory remains for reference only.
# Do NOT use in production. No support, no guarantees.

# Old CUDA Experiments

> **Archive Notice**
>
> This directory contains early CUDA experiments related to theta order, prime scanning,
> and natural order analysis.
>
> These files are **not maintained** and were superseded by the later `cuda/theta_cuda_v1.2.cuh`
> and `cuda/theta_cuda_benchmark_v1.2.cu` implementation.
>
> They are kept only for reference and historical interest.

---

## Files

| File | Description |
|------|-------------|
| `theta_order_complete.cu` | Complete theta order implementation (experimental) |
| `theta_order_prime_analysis.cu` | Prime analysis using TRUE theta order iteration |
| `theta_order_multiprec.cu` | Multi-precision theta order analysis (32/64-bit) |
| `natural_order_analysis.cu` | Natural order prime segment analyzer (see note below) |
| `theta_order_batch.sh` | Batch processing script |
| `THETA_ORDER_HOWTO.md` | Original documentation |

---

## Historical Note: natural_order_analysis.cu

This file has an interesting history. It was the **first attempt** to implement
theta-order prime analysis. However, we discovered during development that the
iteration logic was actually processing integers in **natural order** (consecutive
integers n, n+1, n+2...), NOT in theta order (by angular position).

**How the bug was found:** The CSV output showed `min_shell == max_shell` for every
segment. True theta order iteration would mix integers from MULTIPLE shells within
each segment.

**Why we kept it:** Despite the bug, the code works correctly as a natural-order
prime analyzer:
- Miller-Rabin primality testing is correct
- Prime counting matches known values (e.g., π(2^32) = 203,280,221)
- The implementation is useful for natural-order prime analysis

The TRUE theta-order implementation was subsequently built in the `theta_order_*.cu`
files, which correctly iterate by angular position and mix shells within segments.

---

## Canonical Implementation

For the current, supported CUDA implementation, see:

- `cuda/theta_cuda_v1.2.cuh` - Header-only CUDA library
- `cuda/theta_cuda_benchmark_v1.2.cu` - Benchmark suite
- `cuda/THETA_SPEC_v1.2.md` - Canonical specification
