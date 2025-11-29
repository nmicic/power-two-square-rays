# ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
# This directory remains for reference only.
# Do NOT use in production. No support, no guarantees.

> **Archive Notice – Experimental / Legacy Code**
>
> This directory contains early Python experiments for the Theta Toolkit, including:
> - prototype `theta_toolkit.py` implementations
> - fast / vectorized experiments
> - toy crypto-style benchmarks
> - plotting / ray visualization scripts
>
> These files are **not maintained**, **not production-ready**, and in particular,
> anything under `theta_crypto_*.py` or involving Feistel/digests is **not cryptographic
> security** and **must not be used to protect secrets**.
>
> The canonical specification and CUDA implementation live under `cuda/` in this repo.

---

# Theta Python Toolkit (Legacy)

**Version**: 1.2
**Status**: Archived / Experimental
**Repository**: https://github.com/nmicic/power-two-square-rays/

Integer-native angular encoding via 2-adic decomposition.

## What is Theta?

Every positive integer `n` decomposes uniquely:
```
n = 2^v2(n) × core(n)
```

Where:
- `v2(n)` = count of trailing zeros (2-adic valuation)
- `core(n)` = odd part after removing all factors of 2
- `theta_key(n)` = bit_reverse(core(n))

This creates a **deterministic angular encoding** that preserves mathematical structure while achieving hash-like distribution.

---

## Files

| File | Description |
|------|-------------|
| `theta_toolkit.py` | Core library: primitives, codec, ray embedding, ML features |
| `theta_toolkit_fast.py` | NumPy/Numba optimized vectorized operations |
| `theta_codec.py` | Binary encoding/decoding with stdin/stdout support |
| `theta_benchmark.py` | CPU benchmark vs xxhash/crc32/murmur3 |
| `ThetaSpec_v1.2.md` | Complete specification |

---

## Quick Start

### Core Functions

```python
from theta_toolkit import v2, odd_core, theta_key, shell, theta_bucket

n = 12  # = 4 × 3 = 2² × 3

print(f"v2({n}) = {v2(n)}")           # 2 (trailing zeros)
print(f"odd_core({n}) = {odd_core(n)}")  # 3 (odd part)
print(f"theta_key({n}) = {theta_key(n)}")  # 3 (bit_reverse of core)
print(f"shell({n}) = {shell(n)}")       # 3 (floor(log2(12)))

# Bucketing (uniform distribution)
bucket = theta_bucket(n, num_buckets=64)
print(f"bucket = {bucket}")
```

### Full Decomposition

```python
from theta_toolkit import decompose

d = decompose(12)
print(f"n={d.n}, v2={d.v2}, core={d.core}, theta_key={d.theta_key}")
print(f"shell={d.shell}, edge={d.edge}, quadrant={d.quadrant}")

# Reconstruct
from theta_toolkit import recompose_from_core
reconstructed = recompose_from_core(d.v2, d.core)
assert reconstructed == d.n
```

### Ray Embedding (v1.2)

```python
from theta_toolkit import theta_ray_coords, theta_ray_features

# Get 2D coordinates
x, y = theta_ray_coords(12)
print(f"Ray coords: ({x:.3f}, {y:.3f})")

# Get full feature vector for ML
features = theta_ray_features(12)
# [theta_key, shell, v2, quadrant, sign, x, y, slope]
```

### ML Embedding

```python
from theta_toolkit import theta_embed

# 8-element feature vector
features = theta_embed(12, max_shell=32)
print(features)  # Normalized values for ML

# Batch processing with NumPy
import numpy as np
from theta_toolkit import ThetaEmbedding

embedder = ThetaEmbedding(max_shell=32)
X = embedder.fit_transform(np.array([1, 12, 100, 1000]))
print(X.shape)  # (4, 8)
```

---

## Command-Line Tools

### theta_codec.py - Binary Encoding

```bash
# Encode file
python3 theta_codec.py encode input.bin output.theta

# Decode file
python3 theta_codec.py decode output.theta recovered.bin

# Stdin/stdout (use - for stdin/stdout)
cat input.bin | python3 theta_codec.py encode - - > output.theta
cat output.theta | python3 theta_codec.py decode - - > recovered.bin

# With encryption
python3 theta_codec.py encode input.bin output.theta --key "secret"

# Text format (like uuencode)
python3 theta_codec.py encode input.bin output.txt --text

# Verify integrity
python3 theta_codec.py verify output.theta

# Demo
python3 theta_codec.py demo
```

### theta_benchmark.py - CPU Performance

```bash
# Run benchmark
python3 theta_benchmark.py

# Expected output:
# theta_bucket: 1.80 M ops/sec, chi-sq 59.9 - GOOD
# xxhash32:     6.18 M ops/sec, chi-sq 72.6 - GOOD
```

### theta_toolkit_fast.py - Vectorized Benchmark

```bash
python3 theta_toolkit_fast.py

# Output:
#       Size |   theta_key_vec_fast |     shell_vec_fast
# -----------|--------------------|------------------
#       1000 |              3.68ms |            1.44ms
#      10000 |             37.86ms |           15.39ms
```

---

## API Reference

### Core Primitives

```python
v2(n: int) -> int
    # 2-adic valuation (trailing zeros)
    # v2(12) = 2, v2(8) = 3, v2(7) = 0

odd_core(n: int) -> int
    # Odd part: n / 2^v2(n)
    # odd_core(12) = 3, odd_core(8) = 1

bit_length(n: int) -> int
    # Number of bits: floor(log2(n)) + 1

shell(n: int) -> int
    # Shell index: floor(log2(n))

bit_reverse(val: int, bits: int) -> int
    # Reverse bits within width

theta_key(n: int) -> int
    # Angular encoding: bit_reverse(odd_core(n))
```

### Decomposition

```python
decompose(n: int) -> ThetaDecomposition
    # Full decomposition with all fields
    
recompose_from_core(v2: int, core: int) -> int
    # Reconstruct: core << v2
    
recompose_from_theta(v2: int, theta_key: int, core_bits: int) -> int
    # Reconstruct from theta encoding
```

### Bucketing

```python
theta_bucket(n: int, num_buckets: int, secret: int = 0) -> int
    # Uniform bucket assignment (fixed mixing)
    # IMPORTANT: Simple theta_key % buckets has POOR uniformity
    # because theta_key is always odd. This function uses proper mixing.
```

### Ray Embedding (v1.2)

```python
theta_ray_coords(n: int, X: float = 1.0, Y: float = 0.0) -> Tuple[float, float]
    # 2D coordinates on ray chart
    
theta_ray_features(n: int) -> List[float]
    # [theta_key, shell, v2, quadrant, sign, x, y, slope]
    
slope_for_core(core: int) -> float
    # Default slopes: core=1→-1, core=3→+1, core=5→0, core=7→4/3
```

### ML Features

```python
theta_embed(n: int, max_shell: int = 32) -> List[float]
    # 8-element normalized feature vector

class ThetaEmbedding(BaseEstimator, TransformerMixin):
    # sklearn-compatible transformer
    def fit_transform(X) -> np.ndarray
```

---

## Key Properties

### Mathematical Invariant

**theta_key is ALWAYS ODD** for n > 0:
- core is odd by definition (LSB = 1)
- core has MSB = 1 in minimal width
- bit_reverse swaps MSB ↔ LSB
- Therefore theta_key has LSB = 1

This causes simple `theta_key % buckets` to fail (only fills odd buckets). Use `theta_bucket()` which applies proper 64-bit mixing.

### Benchmark Results

CPU (Python, 10M values):
- theta_bucket: 1.80 M ops/sec
- xxhash32: 6.18 M ops/sec
- Uniformity: Both achieve GOOD chi-square

GPU (CUDA, RTX 4000 Ada):
- theta_key: 30,030 M ops/sec
- xxhash32: 30,002 M ops/sec
- **Ratio: 1.00x** (parity!)

---

## Use Cases

1. **Flow-ID Hashing**: Deterministic bucket assignment for network flows
2. **Spatial Indexing**: Shell-based locality for geometric data
3. **ML Feature Encoding**: Angular features preserve structure
4. **Binary Encoding**: Reversible codec with built-in checksum
5. **Visualization**: Ray embedding for geometric plots

---

## Installation

No dependencies for core functionality:
```bash
# Just copy theta_toolkit.py
```

Optional dependencies:
```bash
pip install numpy          # For vectorized operations
pip install scikit-learn   # For ThetaEmbedding transformer
pip install numba          # For JIT-compiled fast version
pip install xxhash         # For benchmark comparison
```

---

## Version History

- **v1.2**: Ray Embedding (Prism View), stdin/stdout codec support
- **v1.1**: Fixed theta_bucket uniformity, added core_bits for reconstruction
- **v1.0**: Initial release

---

## Disclaimer

This is an AI-assisted exploratory project. Educational and experimental only - not peer-reviewed. The "encryption" in theta_codec is XOR-based obfuscation, NOT cryptographically secure.

---

## License

MIT License

Repository: https://github.com/nmicic/power-two-square-rays/
