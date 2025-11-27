# theta_order_analysis.md - GPU Prime Segment Analyzer

Unified CUDA tool for counting primes and computing integer-native statistics.

##  **Disclaimer**  
> This is an **AI-assisted exploratory visualization project**.  
> Educational and experimental only. Not peer-reviewed.  
> Not mathematical research. Visual patterns are artifacts of the construction.  

## Quick Start

```bash
# Compile (A4000 / Ampere)
nvcc -O3 -gencode arch=compute_86,code=sm_86 theta_order_analysis.md -o theta_order_analysis

# Count all primes up to 2^32 (should find exactly 203,280,221)
./theta_order_analysis --exp 32

# Test a known prime (M127 = 2^127 - 1)
./theta_order_analysis --test 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
```

## Build

### Compiler Requirements

- CUDA Toolkit 11.0+ (for `__umul64hi` intrinsic)
- C++11 or later

### Compile Commands by GPU

```bash
# NVIDIA A4000 / RTX 3080 / RTX 3090 (Ampere, sm_86)
nvcc -O3 -gencode arch=compute_86,code=sm_86 theta_order_analysis.md -o theta_order_analysis

# RTX 4090 / RTX 4080 (Ada Lovelace, sm_89)
nvcc -O3 -gencode arch=compute_89,code=sm_89 theta_order_analysis.md -o theta_order_analysis

# H100 (Hopper, sm_90)
nvcc -O3 -gencode arch=compute_90,code=sm_90 theta_order_analysis.md -o theta_order_analysis

# RTX 2080 / T4 (Turing, sm_75)
nvcc -O3 -gencode arch=compute_75,code=sm_75 theta_order_analysis.md -o theta_order_analysis

# Generic (works on most GPUs but slower)
nvcc -O3 theta_order_analysis.md -o theta_order_analysis
```

### Check Your GPU

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Usage

### Basic Commands

```bash
# Count primes up to 2^N
./theta_order_analysis --exp 32          # π(2^32) = 203,280,221
./theta_order_analysis --exp 20          # π(2^20) = 82,025

# Count primes in a specific range
./theta_order_analysis --start 1000000 --size 1000000

# Use hex for start value
./theta_order_analysis --start 0x100000000 --exp 30

# Test single number for primality
./theta_order_analysis --test 131071              # 2^17 - 1 (prime)
./theta_order_analysis --test 0x7FFFFFFFFFFFFFFF  # 2^63 - 1 (composite)
```

### 256-bit Mode

```bash
# Force 256-bit mode
./theta_order_analysis --mode 256 --segments 1000 --per-seg 1000

# Test Mersenne prime M127
./theta_order_analysis --test 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

# Scan near a large prime
./theta_order_analysis --mode 256 --start 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF --segments 100
```

### Output Control

```bash
# Custom output file
./theta_order_analysis --exp 20 --output my_results.csv

# Verbose (show every segment)
./theta_order_analysis --exp 20 --verbose

# Quiet (minimal output)
./theta_order_analysis --exp 32 --quiet
```

### Segmentation

```bash
# Larger segments = less CSV overhead, more memory
./theta_order_analysis --exp 32 --segment-exp 24   # 2^24 = 16M per segment

# Smaller segments = more granular data
./theta_order_analysis --exp 32 --segment-exp 16   # 2^16 = 64K per segment
```

## CLI Reference

```
theta_order_analysis - Unified GPU Prime Segment Analyzer

Range specification:
  --exp N             Process 2^N integers from start (default start: 0)
  --size N            Process N integers from start
  --start N           Start value (decimal or 0x hex)

Segmentation:
  --segment-exp S     Segment size = 2^S (default: 20 = 1M integers)
  --segments N        Number of segments (256-bit mode only)
  --per-seg N         Candidates per segment (256-bit mode only)

Mode selection:
  --mode 64           Force 64-bit mode
  --mode 256          Force 256-bit mode
  (auto-detected from --start if not specified)

Output:
  --output FILE       CSV file (default: analysis.csv)
  --quiet             Minimal output
  --verbose           Show every segment

Special:
  --test N            Test single number for primality
  --help              Show help
```

## Output Format

### 64-bit Mode CSV Columns

| Column | Description |
|--------|-------------|
| `segment_id` | Segment number (0, 1, 2, ...) |
| `key_start` | First integer in segment |
| `key_end` | First integer AFTER segment |
| `total_integers` | Count of integers in segment |
| `prime_count` | Number of primes found |
| `composite_count` | Number of composites |
| `prime_density` | prime_count / total_integers |
| `min_shell`, `max_shell`, `avg_shell` | floor(log2(n)) statistics |
| `min_popcount`, `max_popcount`, `avg_popcount` | Hamming weight statistics |
| `min_v2`, `max_v2`, `avg_v2` | 2-adic valuation statistics |
| `first_prime` | Smallest prime in segment |
| `last_prime` | Largest prime in segment |
| `twin_prime_count` | Primes p where p+2 is also prime |
| `primes_mod6_1` | Primes ≡ 1 (mod 6) |
| `primes_mod6_5` | Primes ≡ 5 (mod 6) |
| `compute_time_ms` | Kernel execution time |

### 256-bit Mode CSV Columns

| Column | Description |
|--------|-------------|
| `segment` | Segment number |
| `primes_found` | Count of primes |
| `first_offset` | Index of first prime (0 = segment start) |
| `last_offset` | Index of last prime |
| `avg_popcount` | Average Hamming weight of primes |

## Performance

### Throughput by Range

| Range | Witnesses | Throughput | Notes |
|-------|-----------|------------|-------|
| n < 2^32 | 3 | ~100-150 M/sec | Deterministic |
| n ≥ 2^32 | 12 | ~30-50 M/sec | Deterministic |
| 256-bit | 20 | ~150K tests/sec | Probabilistic (error < 10^-12) |

### Expected Times (A4000)

| Command | Primes Found | Time |
|---------|--------------|------|
| `--exp 20` | 82,025 | ~0.5 sec |
| `--exp 24` | 1,077,871 | ~2 sec |
| `--exp 28` | 14,630,843 | ~15 sec |
| `--exp 32` | 203,280,221 | ~40 sec |

### Optimization Tips

1. **Use appropriate segment size**: `--segment-exp 20` (default) is good for most cases
2. **For 256-bit mode**: Use `--segments >= 96` to saturate GPU (A4000 has 48 SMs)
3. **Reduce CSV overhead**: Use larger `--segment-exp` for big ranges

## Validation

### Known Prime Counts

```bash
# These should match exactly:
./theta_order_analysis --exp 10   # π(1024) = 172
./theta_order_analysis --exp 16   # π(65536) = 6,542
./theta_order_analysis --exp 20   # π(1048576) = 82,025
./theta_order_analysis --exp 24   # π(16777216) = 1,077,871
./theta_order_analysis --exp 32   # π(4294967296) = 203,280,221
```

### Known Mersenne Primes

```bash
# All should return "PRIME"
./theta_order_analysis --test 0x1FFFF                                    # M17 = 2^17 - 1
./theta_order_analysis --test 0x7FFFFFFFFFFFFFFF                         # M61 = 2^61 - 1
./theta_order_analysis --test 0x1FFFFFFFFFFFFFFFFFFFFF                   # M89 = 2^89 - 1
./theta_order_analysis --test 0x7FFFFFFFFFFFFFFFFFFFFFFFFFF               # M107 = 2^107 - 1
./theta_order_analysis --test 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF         # M127 = 2^127 - 1
```

### Known Composites

```bash
# Should return "COMPOSITE"
./theta_order_analysis --test 561      # Carmichael number
./theta_order_analysis --test 1105     # Carmichael number
./theta_order_analysis --test 0x8000000000000001  # 2^63 + 1 (divisible by 3)
```

## Metrics Explanation

### What We Compute (INTEGER-NATIVE)

All metrics use only integer/bitwise operations - no floating point in the computation:

| Metric | Formula | Meaning |
|--------|---------|---------|
| `shell` | floor(log2(n)) | Bit-length minus 1; which power-of-2 shell |
| `popcount` | count of 1-bits | Hamming weight |
| `v2` | max k where 2^k divides n | 2-adic valuation (trailing zeros) |
| `mod6` | n mod 6 | All primes > 3 are 1 or 5 mod 6 |

### What We DON'T Compute (and why)

| Metric | Why Excluded |
|--------|--------------|
| `gap` (p2 - p1) | Natural-order distance, meaningless in theta-order |
| `theta_rad` | Would require atan2(), violates integer-native |
| `delta_prime` | Natural-order metric |

## Relationship to power-two-square-rays

This tool supports the [power-two-square-rays](https://github.com/nmicic/power-two-square-rays/) project which explores:

```
n = core × 2^v2
```

Where:
- `v2(n)` = 2-adic valuation (trailing zeros)
- `core` = odd factor (the "odd-core")
- `key` = theta-order position

**Key insight**: Primes (except 2) have v2 = 0, meaning they ARE their own odd-core. Every prime > 2 defines exactly one odd-core ray in the visualization.

## Troubleshooting

### "No CUDA devices found"

```bash
# Check if CUDA driver is loaded
nvidia-smi

# Check CUDA toolkit
nvcc --version
```

### Compilation Errors

```bash
# If sm_XX not supported, find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Use matching architecture
nvcc -O3 -arch=sm_XX theta_order_analysis.md -o theta_order_analysis
```

### Wrong Prime Count

If you get different numbers than expected:
1. Check the range: `--exp 32` means [0, 2^32), not [1, 2^32]
2. For validation, π(N) counts primes ≤ N, we count primes < N
3. Segment boundaries might cause off-by-one; check `key_start` and `key_end`

## License

MIT License

## Author

@nmicic / Claude collaboration

Part of the power-two-square-rays project.
