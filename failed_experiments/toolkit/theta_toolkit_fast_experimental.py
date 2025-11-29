# ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
# This file remains for reference only.
# Do NOT use in production. No support, no guarantees.
#
# For the canonical Theta Toolkit implementation and spec, see:
#   - cuda/theta_cuda_v1.2.cuh
#   - cuda/THETA_SPEC_v1.2.md
"""
Theta Toolkit - Optimized Version with Numba JIT
================================================

This module provides JIT-compiled versions of theta operations
for maximum performance.

Requires: numba (pip install numba)

Falls back to pure Python if numba not available.
"""

import numpy as np
from typing import Tuple

# Try to import numba, fall back gracefully
try:
    from numba import jit, njit, prange, vectorize, int64, uint64, float64
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create no-op decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    jit = njit
    def vectorize(*args, **kwargs):
        def decorator(func):
            return np.vectorize(func)
        return decorator
    def prange(*args):
        return range(*args)
    int64 = int
    uint64 = int
    float64 = float

print(f"Numba available: {HAS_NUMBA}")


# =============================================================================
# JIT-COMPILED CORE PRIMITIVES
# =============================================================================

@njit(cache=True)
def v2_fast(n: int) -> int:
    """JIT-compiled 2-adic valuation."""
    if n == 0:
        return 0
    if n < 0:
        n = -n
    count = 0
    while (n & 1) == 0:
        count += 1
        n >>= 1
    return count


@njit(cache=True)
def odd_core_fast(n: int) -> int:
    """JIT-compiled odd core extraction."""
    if n == 0:
        return 0
    if n < 0:
        n = -n
    while (n & 1) == 0:
        n >>= 1
    return n


@njit(cache=True)
def bit_length_fast(n: int) -> int:
    """JIT-compiled bit length."""
    if n == 0:
        return 0
    if n < 0:
        n = -n
    length = 0
    while n > 0:
        length += 1
        n >>= 1
    return length


@njit(cache=True)
def shell_fast(n: int) -> int:
    """JIT-compiled shell computation."""
    bl = bit_length_fast(n)
    return bl - 1 if bl > 0 else 0


@njit(cache=True)
def bit_reverse_fast(val: int, bits: int) -> int:
    """JIT-compiled bit reversal."""
    if bits <= 0 or val == 0:
        return 0
    if val < 0:
        val = -val
    result = 0
    for _ in range(bits):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result


@njit(cache=True)
def theta_key_fast(n: int) -> int:
    """JIT-compiled theta key computation."""
    if n <= 0:
        return 0
    core = odd_core_fast(n)
    bits = bit_length_fast(core)
    return bit_reverse_fast(core, bits)


# =============================================================================
# JIT-COMPILED VECTORIZED OPERATIONS
# =============================================================================

@njit(parallel=True, cache=True)
def theta_key_vec_fast(arr: np.ndarray) -> np.ndarray:
    """
    Parallel JIT-compiled theta key computation.
    
    Uses numba's prange for parallel execution.
    """
    n = len(arr)
    result = np.zeros(n, dtype=np.int64)
    
    for i in prange(n):
        val = arr[i]
        if val < 0:
            val = -val
        if val > 0:
            result[i] = theta_key_fast(val)
    
    return result


@njit(parallel=True, cache=True)
def shell_vec_fast(arr: np.ndarray) -> np.ndarray:
    """Parallel JIT-compiled shell computation."""
    n = len(arr)
    result = np.zeros(n, dtype=np.int64)
    
    for i in prange(n):
        val = arr[i]
        if val < 0:
            val = -val
        if val > 0:
            result[i] = shell_fast(val)
    
    return result


@njit(parallel=True, cache=True)
def v2_vec_fast(arr: np.ndarray) -> np.ndarray:
    """Parallel JIT-compiled v2 computation."""
    n = len(arr)
    result = np.zeros(n, dtype=np.int64)
    
    for i in prange(n):
        val = arr[i]
        if val < 0:
            val = -val
        if val > 0:
            result[i] = v2_fast(val)
    
    return result


@njit(parallel=True, cache=True)
def odd_core_vec_fast(arr: np.ndarray) -> np.ndarray:
    """Parallel JIT-compiled odd core computation."""
    n = len(arr)
    result = np.zeros(n, dtype=np.int64)
    
    for i in prange(n):
        val = arr[i]
        if val < 0:
            val = -val
        if val > 0:
            result[i] = odd_core_fast(val)
    
    return result


# =============================================================================
# JIT-COMPILED DISTANCE AND SIMILARITY
# =============================================================================

@njit(cache=True)
def theta_distance_int_fast(a: int, b: int) -> int:
    """JIT-compiled integer distance in theta space."""
    key_a = theta_key_fast(a) if a > 0 else 0
    key_b = theta_key_fast(b) if b > 0 else 0
    sh_a = shell_fast(a) if a > 0 else 0
    sh_b = shell_fast(b) if b > 0 else 0
    
    diff_key = key_a - key_b
    if diff_key < 0:
        diff_key = -diff_key
    
    diff_sh = sh_a - sh_b
    if diff_sh < 0:
        diff_sh = -diff_sh
    
    return diff_key + diff_sh


@njit(cache=True)
def theta_hamming_fast(a: int, b: int) -> int:
    """JIT-compiled Hamming distance between theta keys."""
    key_a = theta_key_fast(a) if a > 0 else 0
    key_b = theta_key_fast(b) if b > 0 else 0
    
    xor = key_a ^ key_b
    count = 0
    while xor > 0:
        count += xor & 1
        xor >>= 1
    return count


@njit(parallel=True, cache=True)
def theta_pairwise_distance_fast(arr: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix using theta distance.
    
    Returns: (N, N) distance matrix
    
    Parallelized for large arrays.
    """
    n = len(arr)
    result = np.zeros((n, n), dtype=np.int64)
    
    # Pre-compute theta keys and shells
    keys = np.zeros(n, dtype=np.int64)
    shells = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        val = arr[i]
        if val < 0:
            val = -val
        if val > 0:
            keys[i] = theta_key_fast(val)
            shells[i] = shell_fast(val)
    
    # Compute pairwise distances (symmetric)
    for i in prange(n):
        for j in range(i + 1, n):
            diff_key = keys[i] - keys[j]
            if diff_key < 0:
                diff_key = -diff_key
            diff_sh = shells[i] - shells[j]
            if diff_sh < 0:
                diff_sh = -diff_sh
            dist = diff_key + diff_sh
            result[i, j] = dist
            result[j, i] = dist
    
    return result


# =============================================================================
# JIT-COMPILED EMBEDDING
# =============================================================================

@njit(parallel=True, cache=True)
def theta_embed_batch_fast(arr: np.ndarray) -> np.ndarray:
    """
    Fast batch embedding to (N, 5) integer array.
    
    Columns: [theta_key, shell, v2, quadrant, sign]
    """
    n = len(arr)
    result = np.zeros((n, 5), dtype=np.int64)
    
    for i in prange(n):
        val = arr[i]
        sign = 1 if val >= 0 else 0
        if val < 0:
            val = -val
        
        if val > 0:
            key = theta_key_fast(val)
            sh = shell_fast(val)
            v = v2_fast(val)
            
            # Compute quadrant from top 2 bits
            bits = bit_length_fast(key)
            quad = ((key >> (bits - 2)) & 0x3) if bits >= 2 else 0
            
            result[i, 0] = key
            result[i, 1] = sh
            result[i, 2] = v
            result[i, 3] = quad
        
        result[i, 4] = sign
    
    return result


# =============================================================================
# JIT-COMPILED K-MEANS
# =============================================================================

@njit(parallel=True, cache=True)
def theta_kmeans_step_fast(points: np.ndarray, 
                           centroids: np.ndarray) -> np.ndarray:
    """
    Fast k-means assignment using theta distance.
    
    Returns cluster assignments.
    """
    n_points = len(points)
    n_centroids = len(centroids)
    
    # Pre-compute theta info for centroids
    centroid_keys = np.zeros(n_centroids, dtype=np.int64)
    centroid_shells = np.zeros(n_centroids, dtype=np.int64)
    
    for j in range(n_centroids):
        val = centroids[j]
        if val < 0:
            val = -val
        if val > 0:
            centroid_keys[j] = theta_key_fast(val)
            centroid_shells[j] = shell_fast(val)
    
    assignments = np.zeros(n_points, dtype=np.int32)
    
    for i in prange(n_points):
        val = points[i]
        if val < 0:
            val = -val
        
        if val > 0:
            pt_key = theta_key_fast(val)
            pt_shell = shell_fast(val)
        else:
            pt_key = 0
            pt_shell = 0
        
        min_dist = 2**62  # Large value
        best_j = 0
        
        for j in range(n_centroids):
            diff_key = pt_key - centroid_keys[j]
            if diff_key < 0:
                diff_key = -diff_key
            diff_sh = pt_shell - centroid_shells[j]
            if diff_sh < 0:
                diff_sh = -diff_sh
            dist = diff_key + diff_sh
            
            if dist < min_dist:
                min_dist = dist
                best_j = j
        
        assignments[i] = best_j
    
    return assignments


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================

def theta_key_batch(arr, use_fast=True):
    """Compute theta keys for array, auto-selecting fast path."""
    arr = np.asarray(arr, dtype=np.int64)
    if use_fast and HAS_NUMBA:
        return theta_key_vec_fast(arr)
    else:
        # Fallback
        from theta_toolkit import theta_key_vec
        return theta_key_vec(arr)


def theta_distance_matrix(arr, use_fast=True):
    """Compute full pairwise distance matrix."""
    arr = np.asarray(arr, dtype=np.int64)
    if use_fast and HAS_NUMBA:
        return theta_pairwise_distance_fast(arr)
    else:
        # Fallback to loop
        n = len(arr)
        result = np.zeros((n, n), dtype=np.int64)
        from theta_toolkit import theta_distance_int
        for i in range(n):
            for j in range(i+1, n):
                d = theta_distance_int(int(arr[i]), int(arr[j]))
                result[i, j] = d
                result[j, i] = d
        return result


# =============================================================================
# BENCHMARK COMPARISON
# =============================================================================

def benchmark_optimized():
    """Compare optimized vs base implementations."""
    import time
    
    print("=" * 70)
    print("OPTIMIZED THETA TOOLKIT BENCHMARK")
    print("=" * 70)
    print(f"Numba JIT available: {HAS_NUMBA}")
    
    np.random.seed(42)
    
    # Test sizes
    sizes = [1000, 10000, 100000]
    
    print(f"\n{'Size':>10} | {'theta_key_vec_fast':>20} | {'shell_vec_fast':>18} | {'embed_fast':>15}")
    print("-" * 75)
    
    for size in sizes:
        arr = np.random.randint(1, 1000000, size=size).astype(np.int64)
        
        # Warmup
        if HAS_NUMBA:
            _ = theta_key_vec_fast(arr[:100])
            _ = shell_vec_fast(arr[:100])
            _ = theta_embed_batch_fast(arr[:100])
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            _ = theta_key_vec_fast(arr)
        key_time = (time.perf_counter() - start) / 10 * 1000
        
        start = time.perf_counter()
        for _ in range(10):
            _ = shell_vec_fast(arr)
        shell_time = (time.perf_counter() - start) / 10 * 1000
        
        start = time.perf_counter()
        for _ in range(10):
            _ = theta_embed_batch_fast(arr)
        embed_time = (time.perf_counter() - start) / 10 * 1000
        
        print(f"{size:>10} | {key_time:>17.2f}ms | {shell_time:>15.2f}ms | {embed_time:>12.2f}ms")
    
    # Pairwise distance benchmark
    print(f"\nPairwise distance matrix (N×N):")
    for n in [100, 500, 1000]:
        arr = np.random.randint(1, 100000, size=n).astype(np.int64)
        
        # Warmup
        if HAS_NUMBA:
            _ = theta_pairwise_distance_fast(arr[:50])
        
        start = time.perf_counter()
        _ = theta_pairwise_distance_fast(arr)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"  N={n:>4}: {elapsed:>8.2f}ms ({n*n:>10,} pairs)")
    
    # K-means benchmark
    print(f"\nK-means assignment (1000 points, K clusters):")
    points = np.random.randint(1, 100000, size=1000).astype(np.int64)
    
    for k in [5, 10, 50, 100]:
        centroids = np.random.randint(1, 100000, size=k).astype(np.int64)
        
        # Warmup
        if HAS_NUMBA:
            _ = theta_kmeans_step_fast(points[:100], centroids)
        
        start = time.perf_counter()
        for _ in range(100):
            _ = theta_kmeans_step_fast(points, centroids)
        elapsed = (time.perf_counter() - start) / 100 * 1000
        
        print(f"  K={k:>3}: {elapsed:>8.3f}ms")


if __name__ == "__main__":
    benchmark_optimized()
