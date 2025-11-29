# Theta Toolkit Specification v1.2

> **Repository**: https://github.com/nmicic/power-two-square-rays/  
> **Status**: Specification  
> **Version**: 1.2  
> **Date**: 2025-11-28

---

# Part I: Mathematical Specification

## 1. The 2-Adic Decomposition

### 1.1 Definition

Every positive integer `n > 0` has a **unique** decomposition:

```
n = 2^v2(n) × core(n)
```

Where:
- `v2(n)` ∈ ℕ₀ — the **2-adic valuation** (count of trailing zeros in binary)
- `core(n)` ∈ ℕ_odd — the **odd core** (n with all factors of 2 removed)

**Formal definitions:**
```
v2(n)   = max { k ∈ ℕ₀ : 2^k divides n }
core(n) = n / 2^v2(n)
```

**Uniqueness Theorem**: For any n > 0, there exists exactly one pair (v2, core) such that:
1. n = core × 2^v2
2. core is odd (core mod 2 = 1)

### 1.2 Shell Definition (Primary)

The **shell** of an integer n > 0 is:

```
shell(n) = floor(log2(n)) = bit_length(n) - 1
```

Where `bit_length(n)` is the number of bits needed to represent n in binary (excluding leading zeros).

**Interpretation**: Shell k contains all integers in the range [2^k, 2^(k+1)).

| Shell | Range | Count |
|-------|-------|-------|
| 0 | [1, 2) | 1 |
| 1 | [2, 4) | 2 |
| 2 | [4, 8) | 4 |
| k | [2^k, 2^(k+1)) | 2^k |

### 1.3 The Theta Key

The **theta key** encodes angular position using bit reversal:

```
theta_key(n) = bit_reverse(core(n), bit_length(core(n)))
```

Where `bit_reverse(val, k)` reverses the k least significant bits of val.

**Example**:
```
n = 12 = 0b1100
v2(12) = 2 (two trailing zeros)
core(12) = 3 = 0b11
bit_length(3) = 2
theta_key(12) = bit_reverse(0b11, 2) = 0b11 = 3
```

### 1.4 Invariant: Theta Key is Always Odd

**Claim**: For all n > 0, theta_key(n) is odd (LSB = 1).

**Proof sketch**:
1. core(n) is odd by definition, so its LSB = 1
2. core(n) > 0, so its MSB = 1 (in minimal bit-width representation)
3. bit_reverse swaps MSB ↔ LSB
4. Therefore theta_key has LSB = 1 (from original MSB)

∎

**Corollary**: theta_key(n) ≥ 1 for all n > 0.

### 1.5 Edge/Quadrant Encoding

The top 2 bits of theta_key encode the **edge** (quadrant):

| Bits | Index | Name | Geometric Position |
|------|-------|------|-------------------|
| 00 | 0 | TOP | 12 o'clock, moving right |
| 01 | 1 | RIGHT | 3 o'clock, moving down |
| 10 | 2 | BOTTOM | 6 o'clock, moving left |
| 11 | 3 | LEFT | 9 o'clock, moving up |

**Extraction** (for theta_key with k = bit_length(theta_key) ≥ 2):
```
quadrant = (theta_key >> (k - 2)) & 0x3
```

For k < 2, quadrant defaults to 0 (TOP).

### 1.6 Shell Discontinuity Warning

**CRITICAL**: Theta keys behave like angles within a shell, but shell boundaries introduce **discontinuities**.

```
n = 255: shell=7, core=255, theta_key=255 (8 bits)
n = 256: shell=8, core=1,   theta_key=1   (1 bit)  ← DISCONTINUITY
n = 257: shell=8, core=257, theta_key=257 (9 bits)
```

At each shell boundary:
- Bit-length of core can change dramatically
- Angular resolution doubles (2^k positions → 2^(k+1) positions)
- Normalized theta values are NOT continuous

**Implication**: Do not treat theta_key as a continuous variable across shells.

---

## 2. Reconstruction and Reversibility

### 2.1 Full Decomposition Tuple

For complete reversibility, we need:

```
encode(n) → (v2, core, core_bits)
```

Where:
- `v2` = v2(n)
- `core` = core(n) = odd_core(n)
- `core_bits` = bit_length(core)

### 2.2 Decomposition Data Structure

The `ThetaDecomposition` stores **both** core and theta_key:

```python
@dataclass
class ThetaDecomposition:
    n: int              # Original value
    sign: int           # 1 or -1
    v2: int             # 2-adic valuation
    core: int           # Odd core (PRIMARY for reconstruction)
    core_bits: int      # bit_length(core)
    shell: int          # floor(log2(n))
    theta_key: int      # bit_reverse(core, core_bits) (DERIVED)
    quadrant: int       # 0-3
    edge: str           # 'TOP', 'RIGHT', 'BOTTOM', 'LEFT'
    
    @property
    def core_shell(self) -> int:
        """Shell of the core (= core_bits - 1). Used in codec."""
        return self.core_bits - 1 if self.core_bits > 0 else 0
```

**Design note**: `core` is the primary stored value; `theta_key` is derived. Reconstruction uses `recompose_from_core()`.

### 2.3 Reconstruction Functions (v1.2)

Two explicit reconstruction functions:

```python
def recompose_from_core(v2: int, core: int) -> int:
    """
    Reconstruct n from v2 and odd core.
    
    This is the PRIMARY reconstruction path.
    Assumes core is already validated as odd.
    
    Returns:
        n = core << v2
    """
    if core == 0:
        return 0
    return core << v2


def recompose_from_theta(v2: int, theta_key: int) -> int:
    """
    Reconstruct n from v2 and theta_key.
    
    This is the CODEC reconstruction path.
    theta_key == 0 is treated as sentinel for n == 0.
    
    Returns:
        n = bit_reverse(theta_key, bit_length(theta_key)) << v2
    """
    if theta_key == 0:
        return 0
    
    core_bits = bit_length(theta_key)
    core = bit_reverse(theta_key, core_bits)
    return core << v2
```

**Usage guideline**:
- Use `recompose_from_core()` when working with decomposition structs
- Use `recompose_from_theta()` when decoding from codec format

### 2.4 Proof of Bijection

**Theorem**: The mapping n ↔ (v2, theta_key) is a bijection for n > 0.

**Proof**:
1. **n → (v2, theta_key)**: Unique by 2-adic decomposition + bit reversal
2. **(v2, theta_key) → n**: 
   - core_bits = bit_length(theta_key) is uniquely determined
   - core = bit_reverse(theta_key, core_bits) recovers the odd core
   - n = core << v2 reconstructs the original
3. **No collisions**: Different n have different (v2, core) pairs, and bit_reverse is bijective for fixed bit-width

∎

---

## 3. Zero Handling and Padding Semantics

### 3.1 Mathematical Definition

The 2-adic decomposition is **formally defined only for n > 0**.

For n = 0:
- v2(0) is undefined (or conventionally ∞)
- core(0) is undefined
- theta_key(0) is undefined

### 3.2 Implementation Convention

For practical implementations, we define:

```
v2(0) = 0
core(0) = 0
theta_key(0) = 0
shell(0) = 0
```

This is a **sentinel value convention**, not a mathematical definition.

### 3.3 Codec Zero Handling (v1.2 Clarification)

**Encoding requirement** (MUST):
- Encoders MUST produce `(theta_key=0, core_shell=0, v2=0)` for zero values.

**Decoding requirement** (MUST/SHOULD):
- Decoders MUST accept `(0, 0, 0)` and return 0.
- Decoders SHOULD treat any chunk with `theta_key=0` but `core_shell≠0` or `v2≠0` as **suspicious** (log warning, optionally reject).

**Rationale**: This prevents ambiguity. If padding or special markers are needed, use a separate mechanism outside the theta chunk format.

### 3.4 Test Vector for Zero

```
n = 0:
  v2 = 0 (by convention)
  core = 0 (by convention)
  core_bits = 0
  theta_key = 0
  shell = 0
  quadrant = 0
  edge = "TOP" (default)
  
Codec encoding: (0, 0, 0)
```

---

## 4. Core API Summary

### 4.1 Primitive Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `v2(n)` | int → int | 2-adic valuation |
| `odd_core(n)` | int → int | Odd factor |
| `bit_length(n)` | int → int | floor(log2(n)) + 1 |
| `bit_reverse(val, bits)` | (int, int) → int | Reverse bits |
| `shell(n)` | int → int | floor(log2(n)) |
| `popcount(n)` | int → int | Hamming weight |

### 4.2 Theta Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `theta_key(n)` | int → int | bit_reverse(core, bit_length(core)) |
| `theta_quadrant(n)` | int → int | 0-3 from top 2 bits |
| `theta_edge(n)` | int → str | "TOP"/"RIGHT"/"BOTTOM"/"LEFT" |

### 4.3 Decomposition & Reconstruction

| Function | Signature | Description |
|----------|-----------|-------------|
| `decompose(n)` | int → ThetaDecomposition | Full decomposition |
| `recompose_from_core(v2, core)` | (int, int) → int | Primary reconstruction |
| `recompose_from_theta(v2, theta_key)` | (int, int) → int | Codec reconstruction |

### 4.4 Codec Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `encode_chunk(n)` | int → (int, int, int) | Returns (theta_key, core_shell, v2) |
| `decode_chunk(theta_key, core_shell, v2)` | (int, int, int) → int | With validation |

**Note**: `encode_theta(n)` and `decode_theta(v2, theta_key)` are conceptual names for the bijection. The codec uses `encode_chunk`/`decode_chunk` which include the `core_shell` validation field.

---

## 5. Algorithmic Complexity

All core operations are **O(log n)** = O(bit_length):

| Operation | Naive | With Intrinsic |
|-----------|-------|----------------|
| v2(n) | O(log n) | O(1) via `ctz` |
| core(n) | O(log n) | O(1) via shift |
| bit_length(n) | O(log n) | O(1) via `clz` |
| bit_reverse(n, k) | O(k) | O(1) via `brev` |
| theta_key(n) | O(log n) | O(1) combined |
| shell(n) | O(log n) | O(1) via `clz` |

**CUDA Intrinsics**:
- `__ffs(n)` — find first set bit (for v2)
- `__clz(n)` — count leading zeros (for bit_length)
- `__brev(n)` — bit reverse 32 bits
- `__popc(n)` — population count

---

# Part II: Codec Specification

## 6. File Format

### 6.1 Overview

The theta codec transforms binary files to theta representation with:
- Built-in integrity checking via checksums
- Optional XOR-based obfuscation (NOT encryption)
- Shell statistics for analysis

### 6.2 Header Format

```
Offset  Size    Field           Description
------  ----    -----           -----------
0       6       magic           "THETA1" (ASCII)
6       4       header_len      Length of JSON header (little-endian uint32)
10      var     header_json     JSON-encoded header fields
```

**Header JSON fields**:
```json
{
  "version": 1,
  "chunk_bits": 32,
  "original_size": 12345,
  "encrypted": false,
  "key_hash": "",
  "xor_checksum": 1234567890,
  "mix_checksum": 9876543210,
  "num_chunks": 100
}
```

| Field | Type | Description |
|-------|------|-------------|
| version | int | Format version (currently 1) |
| chunk_bits | int | Original data chunk size: 8, 16, 32, or 64 |
| original_size | int | Original file size in bytes |
| encrypted | bool | Whether XOR obfuscation was applied |
| key_hash | string | SHA256 prefix of key (for verification), empty if not encrypted |
| xor_checksum | int | XOR of all theta_keys |
| mix_checksum | int | Rotate-XOR-add checksum |
| num_chunks | int | Number of encoded chunks |

### 6.3 Chunk Format

Each chunk is 10 bytes:

```
Offset  Size    Field           Description
------  ----    -----           -----------
0       8       theta_key       uint64, little-endian
8       1       core_shell      uint8 = bit_length(theta_key) - 1
9       1       v2              uint8
```

**Note on `core_shell`**: This field stores the shell of the **core**, not the shell of n:
```
core_shell = bit_length(core) - 1 = bit_length(theta_key) - 1
```

This is **redundant** for reconstruction (since bit_length(theta_key) can be computed) but serves two purposes:
1. **Validation**: Decoders can verify `core_shell == bit_length(theta_key) - 1`
2. **Analytics**: Enables shell distribution analysis without recomputing

### 6.4 Relationship: chunk_bits vs theta_key Storage

The `chunk_bits` header field specifies the **original data width** (8/16/32/64 bits).

The `theta_key` is always stored as uint64, but:
- For chunk_bits=8: theta_key ≤ 255, upper 56 bits are zero
- For chunk_bits=16: theta_key ≤ 65535, upper 48 bits are zero
- For chunk_bits=32: theta_key ≤ 2^32-1, upper 32 bits are zero
- For chunk_bits=64: theta_key uses full 64 bits

**Decoder behavior**: Use `chunk_bits` to validate that decoded values fit in the original width.

### 6.5 Encoding Algorithm

```python
def encode_chunk(n: int) -> Tuple[int, int, int]:
    """
    Encode a single integer to codec format.
    
    Returns: (theta_key, core_shell, v2)
    """
    if n == 0:
        return (0, 0, 0)  # MUST be all zeros for n=0
    
    v2_val = v2(n)
    core = odd_core(n)
    core_bits = bit_length(core)
    theta_key = bit_reverse(core, core_bits)
    core_shell = core_bits - 1
    
    return (theta_key, core_shell, v2_val)
```

### 6.6 Decoding Algorithm (with Validation)

```python
def decode_chunk(theta_key: int, core_shell: int, v2: int, 
                 validate: bool = True) -> int:
    """
    Decode a single chunk with integrity validation.
    
    Raises ThetaFormatError if core_shell doesn't match theta_key.
    """
    # Special case: zero
    if theta_key == 0:
        if validate and (core_shell != 0 or v2 != 0):
            # SHOULD warn: suspicious zero encoding
            warnings.warn(f"Suspicious zero chunk: core_shell={core_shell}, v2={v2}")
        return 0
    
    # Compute expected core_shell from theta_key
    computed_core_bits = bit_length(theta_key)
    expected_core_shell = computed_core_bits - 1
    
    # Validation: MUST match
    if validate and core_shell != expected_core_shell:
        raise ThetaFormatError(
            f"Shell mismatch: stored {core_shell}, computed {expected_core_shell}. "
            f"Possible corruption."
        )
    
    # Reconstruct using recompose_from_theta
    return recompose_from_theta(v2, theta_key)
```

**Validation requirement levels**:
- **MUST**: Strict implementations raise error on mismatch
- **SHOULD**: Lenient implementations log warning but continue
- **MAY**: Permissive implementations ignore (not recommended)

### 6.7 Checksum Algorithms

**XOR checksum**:
```python
xor_checksum = 0
for chunk in chunks:
    xor_checksum ^= chunk.theta_key
```

**MIX checksum** (rotate-XOR-add):
```python
mix_checksum = 0x5A5A5A5A5A5A5A5A  # Initial seed
for chunk in chunks:
    mix_checksum = rotate_left_64(mix_checksum, 5)
    mix_checksum ^= chunk.theta_key
    mix_checksum = (mix_checksum + chunk.core_shell) & 0xFFFFFFFFFFFFFFFF
```

---

## 7. Obfuscation (NOT Encryption)

### 7.1 Security Disclaimer

**WARNING**: The following operations are **obfuscation only**, NOT cryptographic encryption.

They are:
- Deterministic (same input → same output)
- Trivially reversible with known key
- Vulnerable to chosen-plaintext attacks
- NOT suitable for protecting sensitive data

**Use only for**:
- Reversible ID scrambling
- Non-security obfuscation
- Deterministic pseudorandom mapping

### 7.2 XOR Masking

```python
def theta_key_masked(n: int, mask: int) -> int:
    return theta_key(n) ^ mask
```

### 7.3 Feistel Scrambling

```python
def theta_feistel(n: int, secret: int, bits: int = 32, rounds: int = 4) -> int:
    result = theta_key(n)
    for r in range(rounds):
        round_key = secret ^ (r * 0x9E3779B9)
        result = feistel_round(result, round_key, bits)
    return result
```

### 7.4 Digest (Renamed from theta_hash)

```python
def digest_theta_key(n: int, secret: int = 0, algo: str = 'sha256') -> str:
    """
    Cryptographic digest of theta_key.
    
    NOTE: This is NOT a theta-native hash. It is simply:
        hash(theta_key(n) ^ secret)
    
    NOT reversible. NOT a theta primitive.
    """
    import hashlib
    data = (theta_key(n) ^ secret).to_bytes(8, 'little')
    return hashlib.new(algo, data).hexdigest()
```

---

# Part III: ML Embedding Guidelines

## 8. Feature Extraction

### 8.1 Feature Vector Definition

**Basic features (5 elements)**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | theta_key | int | bit_reverse(core) |
| 1 | shell | int | floor(log2(n)) |
| 2 | v2 | int | 2-adic valuation |
| 3 | quadrant | int | 0-3, from top 2 bits of theta_key |
| 4 | sign | int | 1 if n > 0, 0 if n = 0 |

**Extended features (10 elements)**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0-4 | (basic) | | Same as above |
| 5 | core | int | Odd core |
| 6 | core_bits | int | bit_length(core) |
| 7 | popcount | int | Hamming weight of theta_key |
| 8 | lo_byte | int | theta_key & 0xFF |
| 9 | hi_byte | int | (theta_key >> 8) & 0xFF |

### 8.2 Normalization

**MAX_SHELL parameter**: The maximum expected shell value. Must be chosen based on:
- Maximum expected input value: MAX_SHELL = ceil(log2(max_n))
- Or a fixed hyperparameter (e.g., 64 for 64-bit integers)

**Normalized embedding**:
```python
def theta_embed_normalized(n: int, max_shell: int = 64) -> Tuple[float, float]:
    key = theta_key(n)
    sh = shell(n)
    
    if key == 0:
        return (0.0, 0.0)
    
    # Normalize theta_key to [0, 1) within its bit-width
    bits = bit_length(key)
    norm_theta = key / (2 ** bits)
    
    # Normalize shell to [0, 1)
    norm_shell = sh / max_shell
    
    return (norm_theta, norm_shell)
```

**WARNING**: `norm_theta` has discontinuities at shell boundaries. For continuous embeddings, consider using (theta_key, shell) separately with appropriate scaling.

### 8.3 Distance Metrics

**Naive Euclidean distance is NOT recommended** due to shell discontinuities.

**Shell-aware distance**:
```python
def theta_distance_safe(a: int, b: int, 
                        shell_weight: float = 1.0,
                        angle_weight: float = 0.1) -> float:
    sh_a, sh_b = shell(a), shell(b)
    
    # Shell component (discrete jump penalty)
    shell_diff = abs(sh_a - sh_b)
    
    # Angular component (only meaningful within same shell)
    angle_diff = angular_distance(a, b) / (2**16)  # Normalize to [0, 1]
    
    return (shell_weight * shell_diff) + (angle_weight * angle_diff)
```

### 8.4 Recommendations for ML

1. **For tree-based models** (XGBoost, Random Forest): Use raw integer features
2. **For neural networks**: Use normalized features with shell as separate input
3. **For clustering**: Use shell-aware distance or cluster within shells
4. **For embeddings**: Consider (theta_key, shell) as 2D integer embedding
5. **For visualization**: Use ray embedding (Part V) for geometric view

---

# Part IV: Implementation Roadmap

## 9. Quick-Win Implementation Tasks

### 9.1 Python Core (Priority: HIGH)

**Required functions**:
```python
# Primitives
def v2(n: int) -> int: ...
def odd_core(n: int) -> int: ...
def bit_length(n: int) -> int: ...
def bit_reverse(val: int, bits: int) -> int: ...
def shell(n: int) -> int: ...
def popcount(n: int) -> int: ...

# Theta operations
def theta_key(n: int) -> int: ...
def theta_quadrant(n: int) -> int: ...
def theta_edge(n: int) -> str: ...

# Decomposition & Reconstruction (v1.2)
def decompose(n: int) -> ThetaDecomposition: ...
def recompose_from_core(v2: int, core: int) -> int: ...
def recompose_from_theta(v2: int, theta_key: int) -> int: ...

# Codec
def encode_chunk(n: int) -> Tuple[int, int, int]: ...
def decode_chunk(theta_key: int, core_shell: int, v2: int) -> int: ...
```

### 9.2 NumPy Vectorization (Priority: HIGH)

```python
def v2_vec(arr: np.ndarray) -> np.ndarray: ...
def odd_core_vec(arr: np.ndarray) -> np.ndarray: ...
def shell_vec(arr: np.ndarray) -> np.ndarray: ...
def theta_key_vec(arr: np.ndarray) -> np.ndarray: ...
def features_vec(arr: np.ndarray, extended: bool = False) -> np.ndarray: ...
```

### 9.3 sklearn Transformer (Priority: MEDIUM)

```python
from sklearn.base import BaseEstimator, TransformerMixin

class ThetaFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, extended=False, normalize=False, max_shell=64): ...
    def fit(self, X, y=None): return self
    def transform(self, X): ...
    def get_feature_names_out(self, input_features=None): ...
```

### 9.4 CUDA Bulk Kernel (Priority: HIGH)

```cuda
__global__ void theta_decompose_bulk(
    const uint32_t* input,
    uint32_t* theta_key,
    uint32_t* shell,
    uint32_t* v2,
    uint32_t* popcount,
    int n
);
```

### 9.5 Required Unit Tests (Priority: HIGH)

1. **Reversibility tests**: `recompose_from_core(d.v2, d.core) == n`
2. **Theta reversibility**: `recompose_from_theta(d.v2, d.theta_key) == n`
3. **Edge cases**: Powers of 2, Mersenne numbers, zero
4. **CPU vs CUDA agreement**
5. **Codec round-trip** with validation
6. **Codec validation catches corruption**
7. **Shell mixing** in theta order

---

# Part V: Ray Embedding (Prism View)

## 10. Overview

The **Ray Embedding** provides a second geometric "view" of integers, complementing the theta_key angular view. This is a **view layer** that does NOT change the underlying primitive:

```
n = 2^v2 × core
theta_key = bit_reverse(core, bit_length(core))
```

Instead, it defines an additional 2D coordinate system where:
- Each **shell k** maps to a specific x-coordinate approaching a limit line
- Each **odd core c** defines a straight line (ray) with slope m_c
- Integers with the same core lie along the same ray

**Use cases**:
- Visualization of integer structure
- ML feature augmentation
- Spatial indexing / hashing
- Geometric analysis of number patterns

### 10.1 Normative Status

The ray embedding:
- **MUST NOT** change the definition of theta_key, v2, core, or shell
- **MAY** be used for visualization, embeddings, or spatial indexing
- **MUST** be documented as a view-layer, not a new primitive

---

## 11. Shell Geometry

### 11.1 Shell X-Coordinates

For shell index k ≥ 0, the x-coordinate is:

```
x_k = X × (2 - 2^(1-k))
```

Where X is a scale parameter (default X = 1.0).

**Computed values**:

| Shell k | x_k (X=1) | Interpretation |
|---------|-----------|----------------|
| 0 | 0.0 | Origin (n=1 only) |
| 1 | 1.0 | First shell |
| 2 | 1.5 | |
| 3 | 1.75 | |
| 4 | 1.875 | |
| ... | ... | Converging |
| ∞ | 2.0 | Limit line |

**Key property**: As k → ∞, x_k → 2X. Shells **compress** towards the vertical limit line x = 2X.

### 11.2 Shell Midline

All shell midpoints lie on a horizontal line:

```
y_mid = Y
```

Where Y is a vertical offset parameter (default Y = 0.0).

---

## 12. Ray Structure

### 12.1 Ray Definition

Each odd core c defines a **ray** passing through the origin (conceptually):

```
Ray(c): y = Y + m_c × x
```

Where m_c is the **slope** assigned to that core.

### 12.2 Default Slope Assignments

The specification defines slopes for the four smallest odd cores:

| Core c | Slope m_c | Angle | Interpretation |
|--------|-----------|-------|----------------|
| 1 | -1 | -45° | Descending diagonal |
| 3 | +1 | +45° | Ascending diagonal |
| 5 | 0 | 0° | Horizontal midline |
| 7 | +4/3 | ~53° | Steep ascending |

**Symmetry observations**:
- Cores 1 and 3 form symmetric 45° lines around y = Y
- Core 5 lies exactly on the midline
- Core 7 breaks symmetry, diverging upward more steeply

### 12.3 Slope Function

```python
def slope_for_core(c: int) -> float:
    """
    Default slope assignment for odd cores.
    
    Defined cores: 1, 3, 5, 7
    Other cores: Implementation-defined (MAY use theta_key-based formula)
    """
    DEFAULT_SLOPES = {
        1: -1.0,
        3: +1.0,
        5:  0.0,
        7: +4/3,
    }
    
    if c in DEFAULT_SLOPES:
        return DEFAULT_SLOPES[c]
    
    # Implementation-defined extension for other cores
    # One option: use normalized theta_key to distribute slopes
    key = theta_key(c)
    bits = bit_length(key)
    normalized = key / (2 ** bits)  # [0.5, 1.0)
    return (normalized - 0.75) * 4  # Maps to [-1, +1] roughly
```

**Note**: Implementations MAY define different slope functions for cores > 7. The default slopes for 1, 3, 5, 7 are normative for v1.2.

---

## 13. Ray Coordinates

### 13.1 Coordinate Function

```python
def theta_ray_coords(n: int, X: float = 1.0, Y: float = 0.0) -> Tuple[float, float]:
    """
    Compute 2D ray embedding coordinates for integer n.
    
    Args:
        n: Input integer (n > 0)
        X: Horizontal scale (shells approach x = 2X)
        Y: Vertical offset (midline y = Y)
    
    Returns:
        (x, y) coordinates in the ray embedding space
    """
    if n == 0:
        return (0.0, Y)  # Convention: zero at origin on midline
    
    k = shell(n)
    
    # Shell x-coordinate
    if k == 0:
        x = 0.0
    else:
        x = X * (2 - 2 ** (1 - k))
    
    # Ray y-coordinate
    c = odd_core(n)
    m = slope_for_core(c)
    y = Y + m * x
    
    return (x, y)
```

### 13.2 Examples

| n | shell | core | x (X=1) | slope | y (Y=0) |
|---|-------|------|---------|-------|---------|
| 1 | 0 | 1 | 0.0 | -1 | 0.0 |
| 2 | 1 | 1 | 1.0 | -1 | -1.0 |
| 3 | 1 | 3 | 1.0 | +1 | +1.0 |
| 4 | 2 | 1 | 1.5 | -1 | -1.5 |
| 5 | 2 | 5 | 1.5 | 0 | 0.0 |
| 6 | 2 | 3 | 1.5 | +1 | +1.5 |
| 7 | 2 | 7 | 1.5 | +4/3 | +2.0 |
| 8 | 3 | 1 | 1.75 | -1 | -1.75 |

---

## 14. Qualitative Properties

### 14.1 Shell Compression

As shell index k increases, x_k approaches 2X:

```
lim(k→∞) x_k = 2X
```

**Interpretation**: Deeper shells (larger integers) live progressively closer to the vertical limit line. The "resolution" increases as we approach the limit.

### 14.2 Ray Structure

All integers with the same odd core lie along a single ray:

```
{n : core(n) = c} → Ray(c)
```

For example, {1, 2, 4, 8, 16, ...} all lie on Ray(1) with slope -1.

### 14.3 Symmetry and Asymmetry

- **Symmetric pair**: Core 1 (m=-1) and Core 3 (m=+1) form mirror images across y = Y
- **Midline**: Core 5 (m=0) lies exactly on y = Y
- **Asymmetric outlier**: Core 7 (m=+4/3) breaks the pattern, rising more steeply

This asymmetry in core 7 reflects the non-uniform distribution of odd numbers in the angular (theta_key) space.

### 14.4 Macro vs Micro Structure

- **Macro scale**: Shells determine distance to the limit line (x-coordinate)
- **Micro scale**: Cores determine directional spread (ray slope)

Together, they provide a "prism" decomposition: the same integer can be viewed through the **theta chart** (v2, core, theta_key, shell) or the **ray chart** (x, y).

---

## 15. Ray Features for ML

### 15.1 Combined Feature Vector

```python
def theta_ray_features(n: int, X: float = 1.0, Y: float = 0.0) -> List[float]:
    """
    Combined theta + ray feature vector (8 features).
    
    Returns:
        [theta_key, shell, v2, quadrant, sign, x, y, slope]
    """
    d = decompose(n)
    x, y = theta_ray_coords(n, X, Y)
    m = slope_for_core(d.core)
    
    return [
        float(d.theta_key),
        float(d.shell),
        float(d.v2),
        float(d.quadrant),
        float(1 if n > 0 else 0),
        x,
        y,
        m
    ]
```

### 15.2 Feature Table

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | theta_key | float | Theta key (integer cast) |
| 1 | shell | float | Shell index |
| 2 | v2 | float | 2-adic valuation |
| 3 | quadrant | float | 0-3 |
| 4 | sign | float | 1 if n > 0, else 0 |
| 5 | x | float | Ray x-coordinate |
| 6 | y | float | Ray y-coordinate |
| 7 | slope | float | Ray slope m_c |

### 15.3 Use Cases

1. **Visualization**: Plot integers in (x, y) space to see shell/ray structure
2. **Clustering**: Use (x, y) as spatial features for k-means or DBSCAN
3. **Spatial hashing**: Quantize (x, y) into grid cells for locality-sensitive hashing
4. **Anomaly detection**: Identify integers whose (x, y) deviates from expected ray

---

## 16. API Additions (v1.2)

### 16.1 Python Functions

```python
# Ray embedding (Part V)
def slope_for_core(c: int) -> float: ...
def theta_ray_coords(n: int, X: float = 1.0, Y: float = 0.0) -> Tuple[float, float]: ...
def theta_ray_features(n: int, X: float = 1.0, Y: float = 0.0) -> List[float]: ...

# Vectorized (optional)
def theta_ray_coords_vec(arr: np.ndarray, X: float = 1.0, Y: float = 0.0) -> np.ndarray: ...
```

### 16.2 CUDA Extension (Optional)

```cuda
__device__ __host__ inline
void theta_ray_coords_32(uint32_t n, float X, float Y, float* out_x, float* out_y);
```

---

# Appendices

## Appendix A: Test Vectors

### A.1 Basic Test Vectors

```
n       v2  core    core_bits  theta_key  shell  quad  edge
------  --  ------  ---------  ---------  -----  ----  ------
1       0   1       1          1          0      0     TOP
2       1   1       1          1          1      0     TOP
3       0   3       2          3          1      3     LEFT
4       2   1       1          1          2      0     TOP
5       0   5       3          5          2      2     BOTTOM
6       1   3       2          3          2      3     LEFT
7       0   7       3          7          2      3     LEFT
8       3   1       1          1          3      0     TOP
12      2   3       2          3          3      3     LEFT
100     2   25      5          19         6      2     BOTTOM
```

### A.2 Ray Embedding Test Vectors (X=1, Y=0)

```
n       shell  core   x       slope   y
------  -----  ----   ------  ------  ------
1       0      1      0.0     -1      0.0
2       1      1      1.0     -1      -1.0
3       1      3      1.0     +1      +1.0
4       2      1      1.5     -1      -1.5
5       2      5      1.5     0       0.0
6       2      3      1.5     +1      +1.5
7       2      7      1.5     +4/3    +2.0
8       3      1      1.75    -1      -1.75
16      4      1      1.875   -1      -1.875
```

### A.3 Reconstruction Test Vectors

```
n=12:   recompose_from_core(2, 3) = 3 << 2 = 12 ✓
        recompose_from_theta(2, 3) = bit_reverse(3,2)=3, 3<<2 = 12 ✓

n=100:  recompose_from_core(2, 25) = 25 << 2 = 100 ✓
        recompose_from_theta(2, 19) = bit_reverse(19,5)=25, 25<<2 = 100 ✓
```

---

## Appendix B: Quick Reference Card

```
MATHEMATICAL PRIMITIVE (unchanged):
    n = 2^v2 × core
    theta_key = bit_reverse(core, bit_length(core))

RECONSTRUCTION (v1.2):
    recompose_from_core(v2, core) → core << v2
    recompose_from_theta(v2, theta_key) → bit_reverse(theta_key) << v2

RAY EMBEDDING (v1.2):
    x_k = X × (2 - 2^(1-k))     # Shell x-coordinate
    y = Y + m_c × x             # Ray y-coordinate
    
    Default slopes: m_1=-1, m_3=+1, m_5=0, m_7=+4/3

INVARIANTS:
    - theta_key is always odd (LSB=1) for n > 0
    - theta_key = 0 only for n = 0 (sentinel)
    - decode(encode(n)) = n for all n ≥ 0
    - core_bits = bit_length(theta_key)

CODEC FIELDS:
    - theta_key: uint64
    - core_shell: uint8 = bit_length(theta_key) - 1
    - v2: uint8

ZERO ENCODING:
    - MUST encode as (0, 0, 0)
    - SHOULD warn if decoded (0, x, y) with x≠0 or y≠0

VIEWS:
    - Theta chart: (v2, core, theta_key, shell, quadrant)
    - Ray chart: (x, y, slope)
    Both derived from same 2-adic structure.
```

---

## Appendix C: Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2025-11-28 | Initial implementation |
| 1.0 | 2025-11-28 | Added core_bits, security disclaimers, ML guidelines |
| 1.1 | 2025-11-28 | Unified shell terminology, zero handling, codec validation, implementation roadmap |
| 1.2 | 2025-11-28 | Split recompose functions, strict zero encoding, Ray Embedding (Part V) |

---

*End of Theta Toolkit Specification v1.2*
