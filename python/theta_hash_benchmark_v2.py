#!/usr/bin/env python3
# Copyright (c) 2025
# Author: Nenad Micic <nenad@micic.be>
# License: MIT License
#
# This file contains benchmark code for the Theta Toolkit.
# See repository root README for details.
#
# DISCLAIMER:
#     This is an AI-assisted exploratory project.
#     Educational and experimental only - not peer-reviewed.
#     Results are observations from author's test environment.
#     Community validation on different hardware is welcome.
#
# Portions of this file include AI-assisted code generation
# (ChatGPT, Claude). All work reviewed and validated by the author.
"""
theta_hash_benchmark_v2.py - Comprehensive Hash/Bucket Performance Benchmark

Compares theta_key and theta_bucket against industry-standard hash functions:
- xxhash (XXH32, XXH64)
- crc32 (zlib)
- murmur3 (mmh3)
- theta_key (this project)
- theta_bucket (this project)

Metrics:
1. Bucket uniformity (chi-square test)
2. Stability when #buckets changes
3. Speed on CPU (ops/sec)

Repository: https://github.com/nmicic/power-two-square-rays/
"""

import sys
import time
import math
import argparse
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from collections import Counter

#==============================================================================
# OPTIMIZED THETA IMPLEMENTATION
#==============================================================================

# Precompute 8-bit reverse table
_REV8 = [int(bin(i)[2:].zfill(8)[::-1], 2) for i in range(256)]

def v2(n: int) -> int:
    """2-adic valuation: count trailing zeros."""
    if n == 0: return 0
    return (n & -n).bit_length() - 1

def odd_core(n: int) -> int:
    """Extract odd core."""
    if n == 0: return 0
    return n >> v2(n)

def shell(n: int) -> int:
    """Shell index."""
    if n == 0: return 0
    return n.bit_length() - 1

def theta_key(n: int) -> int:
    """
    Compute theta key using lookup table for bit reversal.
    ~4x faster than naive loop implementation.
    """
    if n == 0: return 0
    
    # Get odd core
    v = (n & -n).bit_length() - 1
    core = n >> v
    bits = core.bit_length()
    
    # Byte-by-byte bit reversal using lookup table
    if bits <= 8:
        return _REV8[core] >> (8 - bits)
    elif bits <= 16:
        b0, b1 = core & 0xFF, (core >> 8) & 0xFF
        rev = (_REV8[b0] << 8) | _REV8[b1]
        return rev >> (16 - bits)
    elif bits <= 24:
        b0, b1, b2 = core & 0xFF, (core >> 8) & 0xFF, (core >> 16) & 0xFF
        rev = (_REV8[b0] << 16) | (_REV8[b1] << 8) | _REV8[b2]
        return rev >> (24 - bits)
    else:
        b0, b1 = core & 0xFF, (core >> 8) & 0xFF
        b2, b3 = (core >> 16) & 0xFF, (core >> 24) & 0xFF
        rev = (_REV8[b0] << 24) | (_REV8[b1] << 16) | (_REV8[b2] << 8) | _REV8[b3]
        return rev >> (32 - bits)

def theta_bucket(n: int, num_buckets: int, secret: int = 0) -> int:
    """
    Theta-based bucket assignment (FIXED v2).
    
    Key fix: Use 64-bit multiply and take UPPER bits to ensure
    uniform distribution even though theta_key is always odd.
    """
    key = theta_key(n)
    v = v2(n)
    sh = shell(n)
    
    # Combine with different primes
    h = key
    h = (h * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    h = h ^ (v * 0x517CC1B727220A95)
    h = h ^ (sh * 0x2545F4914F6CDD1D)
    h = h ^ secret
    
    # Final mix: upper bits are better distributed
    h = ((h * 0x9E3779B97F4A7C15) >> 32) & 0xFFFFFFFF
    
    return h % num_buckets

def theta_bucket_simple(n: int, num_buckets: int) -> int:
    """Simple theta_key mod (for comparison - will fail uniformity)."""
    return theta_key(n) % num_buckets


#==============================================================================
# EXTERNAL HASH FUNCTIONS
#==============================================================================

import zlib

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: NumPy not available.")

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    print("WARNING: xxhash not available. Install: pip install xxhash")

try:
    import mmh3
    HAS_MURMUR = True
except ImportError:
    HAS_MURMUR = False
    print("WARNING: mmh3 not available. Install: pip install mmh3")


def hash_crc32(n: int) -> int:
    return zlib.crc32(n.to_bytes(4, 'little')) & 0xFFFFFFFF

def hash_xxh32(n: int) -> int:
    if not HAS_XXHASH: return hash_crc32(n)
    return xxhash.xxh32(n.to_bytes(4, 'little')).intdigest()

def hash_xxh64(n: int) -> int:
    if not HAS_XXHASH: return hash_crc32(n)
    return xxhash.xxh64(n.to_bytes(4, 'little')).intdigest()

def hash_murmur3(n: int) -> int:
    if not HAS_MURMUR: return hash_crc32(n)
    return mmh3.hash(n.to_bytes(4, 'little'), signed=False)

def bucket_crc32(n: int, num_buckets: int) -> int:
    return hash_crc32(n) % num_buckets

def bucket_xxh32(n: int, num_buckets: int) -> int:
    return hash_xxh32(n) % num_buckets

def bucket_xxh64(n: int, num_buckets: int) -> int:
    return hash_xxh64(n) % num_buckets

def bucket_murmur3(n: int, num_buckets: int) -> int:
    return hash_murmur3(n) % num_buckets


#==============================================================================
# BENCHMARK DATA STRUCTURES
#==============================================================================

@dataclass
class UniformityResult:
    name: str
    num_buckets: int
    chi_square: float
    p_value: float
    max_deviation_pct: float
    verdict: str

@dataclass
class StabilityResult:
    name: str
    buckets_from: int
    buckets_to: int
    consistency_ratio: float
    verdict: str

@dataclass
class SpeedResult:
    name: str
    count: int
    time_sec: float
    ops_per_sec: float
    ns_per_op: float

@dataclass 
class BenchmarkSummary:
    name: str
    uniformity_score: float
    stability_score: float
    speed_mops: float
    overall_score: float


#==============================================================================
# BENCHMARK FUNCTIONS
#==============================================================================

def generate_flow_ids(count: int, seed: int = 42) -> List[int]:
    if HAS_NUMPY:
        np.random.seed(seed)
        return np.random.randint(1, 0xFFFFFFFF, size=count, dtype=np.uint32).tolist()
    else:
        import random
        random.seed(seed)
        return [random.randint(1, 0xFFFFFFFE) for _ in range(count)]


def chi_square_uniformity(counts: List[int], expected: float) -> Tuple[float, float]:
    chi_sq = sum((c - expected) ** 2 / expected for c in counts)
    df = len(counts) - 1
    if df > 100:
        z = (chi_sq - df) / math.sqrt(2 * df)
        p_value = 0.5 * (1 + math.erf(-z / math.sqrt(2)))
    else:
        p_value = math.exp(-chi_sq / 2) if chi_sq > df * 2 else 0.5
    return chi_sq, p_value


def uniformity_verdict(chi_sq: float, df: int) -> str:
    expected = df
    std_dev = math.sqrt(2 * df)
    deviation = abs(chi_sq - expected) / std_dev
    if deviation < 2:
        return "GOOD"
    elif deviation < 3:
        return "MARGINAL"
    else:
        return "POOR"


def benchmark_uniformity(bucket_func, name: str, data: List[int], num_buckets: int) -> UniformityResult:
    counts = [0] * num_buckets
    for n in data:
        counts[bucket_func(n, num_buckets)] += 1
    
    expected = len(data) / num_buckets
    chi_sq, p_value = chi_square_uniformity(counts, expected)
    max_dev = max(abs(max(counts) - expected), abs(min(counts) - expected))
    max_dev_pct = (max_dev / expected) * 100
    verdict = uniformity_verdict(chi_sq, num_buckets - 1)
    
    return UniformityResult(name, num_buckets, chi_sq, p_value, max_dev_pct, verdict)


def benchmark_stability(bucket_func, name: str, data: List[int], b_from: int, b_to: int) -> StabilityResult:
    consistent = 0
    for n in data:
        b1 = bucket_func(n, b_from)
        b2 = bucket_func(n, b_to)
        if b_to == b_from * 2:
            if b2 == b1 or b2 == b1 + b_from:
                consistent += 1
        elif b_to * 2 == b_from:
            if b1 % b_to == b2:
                consistent += 1
        else:
            exp = b1 * b_to // b_from
            if abs(b2 - exp) <= 1:
                consistent += 1
    
    ratio = consistent / len(data)
    if b_to == b_from * 2:
        verdict = "EXCELLENT" if ratio > 0.9 else "GOOD" if ratio > 0.7 else "MARGINAL" if ratio > 0.5 else "POOR"
    else:
        verdict = "EXCELLENT" if ratio > 0.8 else "GOOD" if ratio > 0.5 else "MARGINAL"
    
    return StabilityResult(name, b_from, b_to, ratio, verdict)


def benchmark_speed(hash_func, name: str, data: List[int]) -> SpeedResult:
    # Warmup
    for i in range(min(10000, len(data))):
        _ = hash_func(data[i])
    
    start = time.perf_counter()
    for n in data:
        _ = hash_func(n)
    elapsed = time.perf_counter() - start
    
    return SpeedResult(
        name=name,
        count=len(data),
        time_sec=elapsed,
        ops_per_sec=len(data) / elapsed,
        ns_per_op=(elapsed * 1e9) / len(data)
    )


#==============================================================================
# MAIN BENCHMARK
#==============================================================================

def run_full_benchmark(count: int = 1_000_000, bucket_sizes: List[int] = None):
    if bucket_sizes is None:
        bucket_sizes = [16, 64, 128, 256, 1024]
    
    print("=" * 70)
    print("  THETA HASH BENCHMARK v2 (FIXED)")
    print("=" * 70)
    print(f"  Flow IDs: {count:,}")
    print(f"  Bucket sizes: {bucket_sizes}")
    print(f"  NumPy: {'Yes' if HAS_NUMPY else 'No'}")
    print(f"  xxhash: {'Yes' if HAS_XXHASH else 'No'}")
    print(f"  mmh3: {'Yes' if HAS_MURMUR else 'No'}")
    print("=" * 70)
    print()
    
    print("Generating synthetic flow IDs...", end=" ", flush=True)
    start = time.perf_counter()
    data = generate_flow_ids(count)
    print(f"done ({time.perf_counter() - start:.2f}s)")
    print()
    
    # Hash functions
    hash_funcs = [
        ("crc32", hash_crc32),
        ("theta_key", theta_key),
    ]
    if HAS_XXHASH:
        hash_funcs.insert(1, ("xxh32", hash_xxh32))
        hash_funcs.insert(2, ("xxh64", hash_xxh64))
    if HAS_MURMUR:
        hash_funcs.insert(-1, ("murmur3", hash_murmur3))
    
    # Bucket functions
    bucket_funcs = [
        ("crc32", bucket_crc32),
        ("theta_bucket", theta_bucket),
        ("theta_simple", theta_bucket_simple),
    ]
    if HAS_XXHASH:
        bucket_funcs.insert(1, ("xxh32", bucket_xxh32))
        bucket_funcs.insert(2, ("xxh64", bucket_xxh64))
    if HAS_MURMUR:
        bucket_funcs.insert(-2, ("murmur3", bucket_murmur3))
    
    #--------------------------------------------------------------------------
    # 1. UNIFORMITY
    #--------------------------------------------------------------------------
    print("=" * 70)
    print("  1. BUCKET UNIFORMITY TEST")
    print("=" * 70)
    print()
    
    uniformity_results: Dict[str, List[UniformityResult]] = {}
    
    for num_buckets in bucket_sizes:
        print(f"Testing {num_buckets} buckets:")
        print(f"  {'Hash':<15} {'Chi-Sq':>12} {'Max Dev %':>10} {'Verdict':>10}")
        print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10}")
        
        for name, func in bucket_funcs:
            result = benchmark_uniformity(func, name, data, num_buckets)
            if name not in uniformity_results:
                uniformity_results[name] = []
            uniformity_results[name].append(result)
            print(f"  {name:<15} {result.chi_square:>12.1f} {result.max_deviation_pct:>9.2f}% {result.verdict:>10}")
        print()
    
    #--------------------------------------------------------------------------
    # 2. STABILITY
    #--------------------------------------------------------------------------
    print("=" * 70)
    print("  2. BUCKET STABILITY TEST")
    print("=" * 70)
    print()
    
    stability_tests = [(64, 128), (128, 256), (256, 128), (64, 96)]
    stability_results: Dict[str, List[StabilityResult]] = {}
    
    for b_from, b_to in stability_tests:
        print(f"Testing {b_from} → {b_to} buckets:")
        print(f"  {'Hash':<15} {'Consistency':>12} {'Verdict':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12}")
        
        for name, func in bucket_funcs:
            result = benchmark_stability(func, name, data[:100000], b_from, b_to)
            if name not in stability_results:
                stability_results[name] = []
            stability_results[name].append(result)
            print(f"  {name:<15} {result.consistency_ratio:>11.2%} {result.verdict:>12}")
        print()
    
    #--------------------------------------------------------------------------
    # 3. SPEED
    #--------------------------------------------------------------------------
    print("=" * 70)
    print("  3. CPU SPEED TEST")
    print("=" * 70)
    print()
    
    speed_data = data[:min(count, 1_000_000)]
    print(f"Hashing {len(speed_data):,} values:")
    print(f"  {'Hash':<15} {'Time (s)':>10} {'M ops/s':>10} {'ns/op':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    
    speed_results: Dict[str, SpeedResult] = {}
    for name, func in hash_funcs:
        result = benchmark_speed(func, name, speed_data)
        speed_results[name] = result
        mops = result.ops_per_sec / 1_000_000
        print(f"  {name:<15} {result.time_sec:>10.3f} {mops:>10.2f} {result.ns_per_op:>10.1f}")
    print()
    
    #--------------------------------------------------------------------------
    # 4. SUMMARY
    #--------------------------------------------------------------------------
    print("=" * 70)
    print("  4. SUMMARY")
    print("=" * 70)
    print()
    
    summaries: List[BenchmarkSummary] = []
    
    for name in set(r[0] for r in bucket_funcs):
        uni_results = uniformity_results.get(name, [])
        uni_score = sum(100 if r.verdict == "GOOD" else 50 if r.verdict == "MARGINAL" else 0 
                       for r in uni_results) / len(uni_results) if uni_results else 0
        
        stab_results = stability_results.get(name, [])
        stab_score = sum(r.consistency_ratio * 100 for r in stab_results) / len(stab_results) if stab_results else 0
        
        hash_name = name.replace("_bucket", "").replace("_simple", "")
        if hash_name == "theta":
            hash_name = "theta_key"
        speed_res = speed_results.get(hash_name)
        speed_mops = (speed_res.ops_per_sec / 1_000_000) if speed_res else 0
        
        speed_score = min(100, speed_mops * 20)
        overall = 0.4 * uni_score + 0.3 * stab_score + 0.3 * speed_score
        
        summaries.append(BenchmarkSummary(name, uni_score, stab_score, speed_mops, overall))
    
    summaries.sort(key=lambda s: s.overall_score, reverse=True)
    
    print(f"  {'Hash':<15} {'Uniform':>10} {'Stable':>10} {'Speed':>10} {'Overall':>10}")
    print(f"  {'':15} {'(0-100)':>10} {'(0-100)':>10} {'(M op/s)':>10} {'(0-100)':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for s in summaries:
        print(f"  {s.name:<15} {s.uniformity_score:>10.1f} {s.stability_score:>10.1f} {s.speed_mops:>10.2f} {s.overall_score:>10.1f}")
    print()
    
    #--------------------------------------------------------------------------
    # 5. VERDICT
    #--------------------------------------------------------------------------
    print("=" * 70)
    print("  5. VERDICT")
    print("=" * 70)
    print()
    
    theta_bucket_summary = next((s for s in summaries if s.name == "theta_bucket"), None)
    theta_simple_summary = next((s for s in summaries if s.name == "theta_simple"), None)
    best_traditional = next((s for s in summaries if "theta" not in s.name), None)
    
    if theta_bucket_summary and best_traditional:
        print(f"  theta_bucket vs {best_traditional.name}:")
        print(f"    Uniformity: {theta_bucket_summary.uniformity_score:.1f} vs {best_traditional.uniformity_score:.1f}")
        print(f"    Stability:  {theta_bucket_summary.stability_score:.1f} vs {best_traditional.stability_score:.1f}")
        print(f"    Speed:      {theta_bucket_summary.speed_mops:.2f} vs {best_traditional.speed_mops:.2f} M ops/s")
        print(f"    Overall:    {theta_bucket_summary.overall_score:.1f} vs {best_traditional.overall_score:.1f}")
        print()
        
        # Calculate relative performance
        speed_ratio = theta_bucket_summary.speed_mops / best_traditional.speed_mops if best_traditional.speed_mops > 0 else 0
        overall_ratio = theta_bucket_summary.overall_score / best_traditional.overall_score if best_traditional.overall_score > 0 else 0
        
        if theta_bucket_summary.uniformity_score >= 80 and overall_ratio >= 0.8:
            print("  ✓ THETA IS COMPETITIVE")
            print(f"    Uniformity: PASS ({theta_bucket_summary.uniformity_score:.0f}%)")
            print(f"    Overall score within {100*overall_ratio:.0f}% of {best_traditional.name}")
            verdict = "PASS"
        elif theta_bucket_summary.uniformity_score >= 80:
            print("  ~ THETA IS USABLE")
            print(f"    Uniformity: PASS ({theta_bucket_summary.uniformity_score:.0f}%)")
            print(f"    Speed penalty: {1/speed_ratio:.1f}x slower than {best_traditional.name}")
            verdict = "MARGINAL"
        else:
            print("  ✗ THETA NEEDS WORK")
            print(f"    Uniformity: FAIL ({theta_bucket_summary.uniformity_score:.0f}%)")
            verdict = "FAIL"
        
        # Additional analysis
        print()
        if theta_simple_summary:
            print(f"  Note: theta_simple (raw mod) uniformity: {theta_simple_summary.uniformity_score:.0f}%")
            print(f"        This fails because theta_key is always ODD (LSB=1).")
            print(f"        theta_bucket fixes this with proper mixing.")
    else:
        verdict = "INCOMPLETE"
    
    print()
    print("=" * 70)
    print(f"  BENCHMARK COMPLETE - Verdict: {verdict}")
    print("=" * 70)
    
    return {
        "uniformity": uniformity_results,
        "stability": stability_results,
        "speed": speed_results,
        "summaries": summaries,
        "verdict": verdict
    }


def main():
    parser = argparse.ArgumentParser(description="Theta Hash Benchmark v2")
    parser.add_argument("--count", "-n", type=int, default=1_000_000)
    parser.add_argument("--buckets", "-b", type=str, default="16,64,128,256,1024")
    parser.add_argument("--quick", "-q", action="store_true")
    args = parser.parse_args()
    
    if args.quick:
        count, bucket_sizes = 100_000, [16, 64, 256]
    else:
        count = args.count
        bucket_sizes = [int(x) for x in args.buckets.split(",")]
    
    results = run_full_benchmark(count=count, bucket_sizes=bucket_sizes)
    sys.exit(0 if results["verdict"] in ["PASS", "MARGINAL"] else 1)


if __name__ == "__main__":
    main()
