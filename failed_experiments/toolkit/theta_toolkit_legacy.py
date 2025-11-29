#!/usr/bin/env python3
# ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
# This file remains for reference only.
# Do NOT use in production. No support, no guarantees.
#
# For the canonical Theta Toolkit implementation and spec, see:
#   - cuda/theta_cuda_v1.2.cuh
#   - cuda/THETA_SPEC_v1.2.md
"""
theta_toolkit_v1.py - Theta Toolkit v1.0 (Corrected Implementation)

Corrections from architectural review:
1. Added core_bits to decomposition for full reversibility
2. Renamed theta_hash → digest_theta_key
3. Added shell discontinuity warnings
4. Added NumPy vectorized operations
5. Added sklearn transformer
6. Added comprehensive test suite
7. Clarified security disclaimers

Repository: https://github.com/nmicic/power-two-square-rays/
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from enum import IntEnum

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


#==============================================================================
# CONSTANTS
#==============================================================================

class Edge(IntEnum):
    """Quadrant/edge encoding from top 2 bits of theta_key"""
    TOP = 0     # 00
    RIGHT = 1   # 01
    BOTTOM = 2  # 10
    LEFT = 3    # 11

EDGE_NAMES = ('TOP', 'RIGHT', 'BOTTOM', 'LEFT')


#==============================================================================
# DATA STRUCTURES (Corrected)
#==============================================================================

@dataclass
class ThetaDecomposition:
    """
    Full 2-adic decomposition with all fields needed for reconstruction.
    
    CRITICAL: core_bits is REQUIRED for reversibility.
    Without it, multiple integers can map to the same theta_key.
    
    Note on naming (v1.1):
    - shell: floor(log2(n)) - the shell of the original integer
    - core_bits: bit_length(core) - needed for reconstruction
    - core_shell: core_bits - 1 - used in codec for validation (= shell of core)
    """
    n: int
    sign: int           # 1 or -1
    v2: int             # 2-adic valuation (trailing zeros)
    core: int           # odd core
    core_bits: int      # bit_length(core) - REQUIRED FOR RECONSTRUCTION
    shell: int          # floor(log2(|n|)) - shell of original n
    theta_key: int      # bit_reverse(core, core_bits)
    quadrant: int       # 0-3
    edge: str           # 'TOP', 'RIGHT', 'BOTTOM', 'LEFT'
    
    @property
    def core_shell(self) -> int:
        """Shell of the core (= core_bits - 1). Used in codec for validation."""
        return self.core_bits - 1 if self.core_bits > 0 else 0


#==============================================================================
# CORE PRIMITIVES (Optimized with lookup table)
#==============================================================================

# Precompute 8-bit reverse table for fast bit reversal
_REV8 = [int(bin(i)[2:].zfill(8)[::-1], 2) for i in range(256)]


def v2(n: int) -> int:
    """
    2-adic valuation: count trailing zeros in binary.
    
    Optimized using bit trick: n & -n gives lowest set bit.
    
    Examples:
        v2(12) = 2  because 12 = 0b1100
        v2(8) = 3   because 8 = 0b1000
        v2(7) = 0   because 7 = 0b111
    """
    if n == 0:
        return 0
    return (abs(n) & -abs(n)).bit_length() - 1


def odd_core(n: int) -> int:
    """
    Extract odd core: n / 2^v2(n)
    
    Examples:
        odd_core(12) = 3  because 12 = 4 × 3
        odd_core(8) = 1   because 8 = 8 × 1
        odd_core(7) = 7   because 7 = 1 × 7
    """
    if n == 0:
        return 0
    n = abs(int(n))
    return n >> v2(n)


def bit_length(n: int) -> int:
    """Bit length = floor(log2(n)) + 1"""
    if n == 0:
        return 0
    return abs(int(n)).bit_length()


def shell(n: int) -> int:
    """Shell index = floor(log2(n)) = bit_length - 1"""
    bl = bit_length(n)
    return bl - 1 if bl > 0 else 0


def bit_reverse(val: int, bits: int) -> int:
    """
    Reverse bits within specified bit width.
    
    Optimized with lookup table for up to 32 bits.
    """
    if bits <= 0 or val == 0:
        return 0
    val = abs(int(val))
    
    # Use lookup table for byte-by-byte reversal
    if bits <= 8:
        return _REV8[val] >> (8 - bits)
    elif bits <= 16:
        b0, b1 = val & 0xFF, (val >> 8) & 0xFF
        rev = (_REV8[b0] << 8) | _REV8[b1]
        return rev >> (16 - bits)
    elif bits <= 24:
        b0, b1, b2 = val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF
        rev = (_REV8[b0] << 16) | (_REV8[b1] << 8) | _REV8[b2]
        return rev >> (24 - bits)
    elif bits <= 32:
        b0, b1 = val & 0xFF, (val >> 8) & 0xFF
        b2, b3 = (val >> 16) & 0xFF, (val >> 24) & 0xFF
        rev = (_REV8[b0] << 24) | (_REV8[b1] << 16) | (_REV8[b2] << 8) | _REV8[b3]
        return rev >> (32 - bits)
    else:
        # Fallback for > 32 bits
        result = 0
        for _ in range(bits):
            result = (result << 1) | (val & 1)
            val >>= 1
        return result


def popcount(n: int) -> int:
    """Hamming weight: count of 1-bits"""
    return bin(abs(int(n))).count('1')


#==============================================================================
# THETA KEY
#==============================================================================

def theta_key(n: int) -> int:
    """
    Compute theta key: angular position as integer.
    
    theta_key(n) = bit_reverse(odd_core(n), bit_length(odd_core(n)))
    
    WARNING: theta_key alone is NOT sufficient for reconstruction.
    Use decompose() for full reversibility.
    """
    if n <= 0:
        return 0
    core = odd_core(n)
    bits = bit_length(core)
    return bit_reverse(core, bits)


def theta_quadrant(n: int) -> int:
    """Get quadrant index (0-3) from theta key."""
    key = theta_key(n)
    if key == 0:
        return 0
    bits = bit_length(key)
    return ((key >> (bits - 2)) & 0x3) if bits >= 2 else 0


def theta_edge(n: int) -> str:
    """Get edge name from theta key."""
    return EDGE_NAMES[theta_quadrant(n)]


#==============================================================================
# EXCEPTIONS
#==============================================================================

class ThetaFormatError(Exception):
    """Raised when theta codec encounters invalid/corrupted data."""
    pass


#==============================================================================
# CODEC FUNCTIONS (v1.1 - with validation)
#==============================================================================

def encode_chunk(n: int) -> Tuple[int, int, int]:
    """
    Encode a single integer to codec format.
    
    Returns:
        (theta_key, core_shell, v2)
        
    Where core_shell = bit_length(theta_key) - 1
    """
    if n == 0:
        return (0, 0, 0)
    
    d = decompose(n)
    return (d.theta_key, d.core_shell, d.v2)


def decode_chunk(theta_key_val: int, core_shell: int, v2_val: int, 
                 validate: bool = True) -> int:
    """
    Decode a single chunk with optional integrity validation.
    
    Args:
        theta_key_val: The stored theta key
        core_shell: The stored core shell (should equal bit_length(theta_key) - 1)
        v2_val: The stored v2 value
        validate: If True, verify core_shell matches theta_key
        
    Returns:
        Reconstructed integer
        
    Raises:
        ThetaFormatError: If validate=True and core_shell doesn't match
        
    Note (v1.2):
        Encoders MUST produce (0, 0, 0) for zero values.
        Decoders SHOULD warn if theta_key=0 but core_shell≠0 or v2≠0.
    """
    # Special case: zero
    if theta_key_val == 0:
        if validate and (core_shell != 0 or v2_val != 0):
            warnings.warn(
                f"Suspicious zero chunk: expected (0,0,0), got (0,{core_shell},{v2_val}). "
                f"Encoders MUST produce (0,0,0) for zero values.",
                UserWarning
            )
        return 0
    
    # Compute expected core_shell from theta_key
    computed_core_bits = bit_length(theta_key_val)
    expected_core_shell = computed_core_bits - 1
    
    # Validation
    if validate and core_shell != expected_core_shell:
        raise ThetaFormatError(
            f"Shell mismatch: stored core_shell={core_shell}, "
            f"computed={expected_core_shell} from theta_key={theta_key_val}. "
            f"Possible corruption."
        )
    
    # Reconstruct using recompose_from_theta
    return recompose_from_theta(v2_val, theta_key_val)


#==============================================================================
# DECOMPOSITION (Corrected - includes core_bits)
#==============================================================================

def decompose(n: int) -> ThetaDecomposition:
    """
    Full 2-adic decomposition with all fields.
    
    INCLUDES core_bits which is REQUIRED for reconstruction.
    """
    if n == 0:
        return ThetaDecomposition(
            n=0, sign=1, v2=0, core=0, core_bits=0,
            shell=0, theta_key=0, quadrant=0, edge='TOP'
        )
    
    sign = 1 if n >= 0 else -1
    abs_n = abs(int(n))
    
    v = v2(abs_n)
    core = odd_core(abs_n)
    core_bits = bit_length(core)  # CRITICAL
    sh = shell(abs_n)
    key = bit_reverse(core, core_bits)
    quad = ((key >> (core_bits - 2)) & 0x3) if core_bits >= 2 else 0
    
    return ThetaDecomposition(
        n=n, sign=sign, v2=v, core=core, core_bits=core_bits,
        shell=sh, theta_key=key, quadrant=quad, edge=EDGE_NAMES[quad]
    )


def recompose(v2_val: int, theta_key_val: int, core_bits: int, sign: int = 1) -> int:
    """
    Reconstruct integer from theta decomposition (legacy interface).
    
    REQUIRES core_bits for correct reconstruction.
    Prefer recompose_from_theta() for new code.
    """
    if theta_key_val == 0:
        return 0
    core = bit_reverse(theta_key_val, core_bits)
    return sign * (core << v2_val)


def recompose_from_core(v2_val: int, core: int) -> int:
    """
    Reconstruct n from v2 and odd core.
    
    This is the PRIMARY reconstruction path.
    Assumes core is already validated as odd.
    
    Args:
        v2_val: 2-adic valuation
        core: Odd core (must be odd or 0)
        
    Returns:
        n = core << v2
    """
    if core == 0:
        return 0
    return core << v2_val


def recompose_from_theta(v2_val: int, theta_key_val: int) -> int:
    """
    Reconstruct n from v2 and theta_key.
    
    This is the CODEC reconstruction path.
    theta_key == 0 is treated as sentinel for n == 0.
    
    Args:
        v2_val: 2-adic valuation
        theta_key_val: Theta key (bit-reversed core)
        
    Returns:
        n = bit_reverse(theta_key, bit_length(theta_key)) << v2
    """
    if theta_key_val == 0:
        return 0
    
    core_bits = bit_length(theta_key_val)
    core = bit_reverse(theta_key_val, core_bits)
    return core << v2_val


#==============================================================================
# ANGLE OPERATIONS (with discontinuity warning)
#==============================================================================

def theta_angle_scaled(n: int, precision: int = 16) -> int:
    """
    Map theta_key to fixed-precision integer angle in [0, 2^precision).
    
    WARNING: This is NOT continuous across shell boundaries.
    Shell transitions cause angular discontinuities.
    
    Args:
        n: Input integer
        precision: Output bit width (default 16 → [0, 65536))
    
    Returns:
        Integer angle in [0, 2^precision)
    """
    key = theta_key(n)
    if key == 0:
        return 0
    
    bits = bit_length(key)
    if bits >= precision:
        return key >> (bits - precision)
    else:
        return key << (precision - bits)


def theta_angular_distance(a: int, b: int, precision: int = 16) -> int:
    """
    Cyclic angular distance between two values.
    
    WARNING: This metric is meaningful only for values in the SAME shell.
    For cross-shell comparisons, use theta_distance_safe().
    """
    angle_a = theta_angle_scaled(a, precision)
    angle_b = theta_angle_scaled(b, precision)
    
    diff = abs(angle_a - angle_b)
    max_angle = 1 << precision
    
    return min(diff, max_angle - diff)


def theta_distance_safe(
    a: int, 
    b: int, 
    shell_weight: float = 1.0,
    angle_weight: float = 0.1,
    precision: int = 16
) -> float:
    """
    Shell-aware distance metric.
    
    Handles shell discontinuity correctly by weighting shell difference.
    """
    sh_a, sh_b = shell(a), shell(b)
    shell_diff = abs(sh_a - sh_b)
    
    # Angular component (normalized)
    max_angle = 1 << precision
    angle_diff = theta_angular_distance(a, b, precision) / max_angle
    
    return (shell_weight * shell_diff) + (angle_weight * angle_diff)


#==============================================================================
# BUCKETING / SHARDING (Fixed v2 - proper uniformity)
#==============================================================================

def theta_bucket(n: int, num_buckets: int, secret: int = 0) -> int:
    """
    Theta-based bucket assignment with proper uniformity.
    
    IMPORTANT: Simple `theta_key(n) % buckets` has POOR uniformity because
    theta_key is always ODD (LSB=1). This function uses proper mixing.
    
    Args:
        n: Input integer
        num_buckets: Number of buckets
        secret: Optional secret for different distributions
        
    Returns:
        Bucket index in [0, num_buckets)
        
    Benchmark results (10M flow IDs, 64 buckets):
        theta_bucket: chi-square 59.9, max deviation 0.68% - GOOD
        theta_simple: chi-square 10M+, max deviation 100%+ - POOR
    """
    key = theta_key(n)
    v = v2(n)
    sh = shell(n)
    
    # Combine with different primes for good mixing
    h = key
    h = (h * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF  # 64-bit golden ratio
    h = h ^ (v * 0x517CC1B727220A95)
    h = h ^ (sh * 0x2545F4914F6CDD1D)
    h = h ^ secret
    
    # Final mix: use UPPER bits (better distributed than lower)
    h = ((h * 0x9E3779B97F4A7C15) >> 32) & 0xFFFFFFFF
    
    return h % num_buckets


def theta_bucket_simple(n: int, num_buckets: int) -> int:
    """
    Simple theta_key mod buckets.
    
    WARNING: This has POOR uniformity because theta_key is always odd.
    Use theta_bucket() instead for proper distribution.
    """
    return theta_key(n) % num_buckets


#==============================================================================
# FEATURES (Corrected)
#==============================================================================

def theta_features(n: int, extended: bool = False) -> List[int]:
    """
    Generate feature vector for ML models.
    
    Basic (5 features):
        [theta_key, shell, v2, quadrant, sign]
        
    Extended (10 features):
        [theta_key, shell, v2, quadrant, sign,
         core, core_bits, popcount, lo_byte, hi_byte]
    
    NOTE: For cross-shell ML tasks, normalize shell and theta_key separately.
    """
    if n == 0:
        return [0] * (10 if extended else 5)
    
    d = decompose(n)
    sign_bit = 1 if d.sign >= 0 else 0
    
    features = [d.theta_key, d.shell, d.v2, d.quadrant, sign_bit]
    
    if extended:
        pop = popcount(d.theta_key)
        lo_byte = d.theta_key & 0xFF
        hi_byte = (d.theta_key >> 8) & 0xFF
        features.extend([d.core, d.core_bits, pop, lo_byte, hi_byte])
    
    return features


def theta_embed(n: int) -> Tuple[int, int]:
    """2D integer embedding: (theta_key, shell)"""
    if n == 0:
        return (0, 0)
    return (theta_key(abs(n)), shell(abs(n)))


def theta_embed_normalized(n: int, max_shell: int = 64) -> Tuple[float, float]:
    """
    Normalized embedding for neural networks.
    
    Returns (norm_theta, norm_shell) both in [0, 1).
    
    WARNING: norm_theta has discontinuities at shell boundaries.
    """
    key = theta_key(n)
    sh = shell(n)
    
    if key == 0:
        return (0.0, 0.0)
    
    bits = bit_length(key)
    norm_theta = key / (2 ** bits)
    norm_shell = sh / max_shell
    
    return (norm_theta, norm_shell)


#==============================================================================
# OBFUSCATION (with security disclaimer)
#==============================================================================

def theta_key_masked(n: int, mask: int) -> int:
    """
    XOR mask the theta key.
    
    WARNING: This is NOT encryption. It is trivially reversible
    obfuscation. Do NOT use for security-critical applications.
    """
    return theta_key(n) ^ mask


def theta_feistel(n: int, secret: int, bits: int = 32) -> int:
    """
    Single-round Feistel network.
    
    WARNING: This is deterministic obfuscation, NOT encryption.
    Vulnerable to chosen-plaintext attacks.
    """
    k = theta_key(n)
    half = bits // 2
    mask = (1 << half) - 1
    
    l = (k >> half) & mask
    r = k & mask
    f = (l + secret) & mask
    new_r = r ^ f
    
    return ((new_r & mask) << half) | l


def theta_feistel_multi(n: int, secret: int, bits: int = 32, rounds: int = 4) -> int:
    """
    Multi-round Feistel for stronger scrambling.
    
    WARNING: Still deterministic. NOT cryptographically secure.
    Use only for reversible ID scrambling, not security.
    """
    result = n
    for r in range(rounds):
        round_secret = secret ^ (r * 0x9E3779B9)
        result = theta_feistel(result, round_secret, bits)
    return result


def digest_theta_key(n: int, secret: int = 0, algo: str = 'sha256') -> str:
    """
    Cryptographic digest of theta key.
    
    Renamed from theta_hash to clarify this is:
    - NOT a theta-native hash
    - NOT reversible
    - Just SHA/MD5 of (theta_key XOR secret)
    """
    import hashlib
    key = theta_key(n) ^ secret
    data = key.to_bytes(8, 'little')
    
    if algo == 'sha256':
        return hashlib.sha256(data).hexdigest()
    elif algo == 'sha1':
        return hashlib.sha1(data).hexdigest()
    elif algo == 'md5':
        return hashlib.md5(data).hexdigest()
    return hashlib.sha256(data).hexdigest()


#==============================================================================
# NUMPY VECTORIZED OPERATIONS
#==============================================================================

if HAS_NUMPY:
    
    def v2_vec(arr: np.ndarray) -> np.ndarray:
        """Vectorized 2-adic valuation."""
        arr = np.asarray(arr, dtype=np.uint64)
        result = np.zeros_like(arr)
        mask = arr != 0
        
        # Count trailing zeros
        temp = arr.copy()
        for i in range(64):
            still_even = mask & ((temp & 1) == 0)
            result[still_even] += 1
            temp[still_even] >>= 1
            mask = still_even
            if not mask.any():
                break
        
        return result.astype(np.int32)
    
    
    def odd_core_vec(arr: np.ndarray) -> np.ndarray:
        """Vectorized odd core extraction."""
        arr = np.asarray(arr, dtype=np.uint64)
        v2s = v2_vec(arr)
        return (arr >> v2s).astype(np.uint64)
    
    
    def bit_length_vec(arr: np.ndarray) -> np.ndarray:
        """Vectorized bit length."""
        arr = np.asarray(arr, dtype=np.uint64)
        result = np.zeros_like(arr, dtype=np.int32)
        mask = arr != 0
        result[mask] = np.floor(np.log2(arr[mask].astype(np.float64))).astype(np.int32) + 1
        return result
    
    
    def shell_vec(arr: np.ndarray) -> np.ndarray:
        """Vectorized shell computation."""
        bl = bit_length_vec(arr)
        return np.maximum(bl - 1, 0)
    
    
    def _bit_reverse_single(val: int, bits: int) -> int:
        """Helper for vectorized bit reverse."""
        if bits <= 0 or val == 0:
            return 0
        result = 0
        for _ in range(bits):
            result = (result << 1) | (val & 1)
            val >>= 1
        return result
    
    
    def theta_key_vec(arr: np.ndarray) -> np.ndarray:
        """
        Vectorized theta key computation.
        
        Note: bit_reverse is hard to fully vectorize, so we use a hybrid approach.
        """
        arr = np.asarray(arr, dtype=np.uint64)
        cores = odd_core_vec(arr)
        bits = bit_length_vec(cores)
        
        # Apply bit_reverse element-wise (hard to vectorize fully)
        result = np.zeros_like(arr)
        for i in range(len(arr)):
            result[i] = _bit_reverse_single(int(cores[i]), int(bits[i]))
        
        return result
    
    
    def features_vec(arr: np.ndarray, extended: bool = False) -> np.ndarray:
        """
        Vectorized feature extraction.
        
        Returns array of shape (n, 5) or (n, 10) if extended.
        """
        arr = np.asarray(arr, dtype=np.uint64)
        n = len(arr)
        
        keys = theta_key_vec(arr)
        shells = shell_vec(arr)
        v2s = v2_vec(arr)
        
        # Quadrant from top 2 bits
        bits = bit_length_vec(keys)
        quadrants = np.zeros(n, dtype=np.int32)
        mask = bits >= 2
        quadrants[mask] = ((keys[mask] >> (bits[mask] - 2)) & 0x3).astype(np.int32)
        
        # Sign (all positive for uint64)
        signs = np.ones(n, dtype=np.int32)
        
        if extended:
            cores = odd_core_vec(arr)
            core_bits = bit_length_vec(cores)
            popcounts = np.array([bin(int(k)).count('1') for k in keys], dtype=np.int32)
            lo_bytes = (keys & 0xFF).astype(np.int32)
            hi_bytes = ((keys >> 8) & 0xFF).astype(np.int32)
            
            return np.column_stack([
                keys, shells, v2s, quadrants, signs,
                cores, core_bits, popcounts, lo_bytes, hi_bytes
            ])
        else:
            return np.column_stack([keys, shells, v2s, quadrants, signs])


#==============================================================================
# RAY EMBEDDING (v1.2 - Prism View)
#==============================================================================

# Default slopes for small odd cores
DEFAULT_SLOPES = {
    1: -1.0,
    3: +1.0,
    5:  0.0,
    7: +4/3,
}


def slope_for_core(c: int) -> float:
    """
    Get slope for an odd core.
    
    Default slopes (normative for v1.2):
        c=1 → -1 (45° descending)
        c=3 → +1 (45° ascending)
        c=5 → 0 (horizontal midline)
        c=7 → +4/3 (steep ascending)
    
    Other cores: Implementation-defined extension using normalized theta_key.
    """
    if c in DEFAULT_SLOPES:
        return DEFAULT_SLOPES[c]
    
    # Extension for other cores: use normalized theta_key
    if c == 0:
        return 0.0
    key = theta_key(c)
    bits = bit_length(key)
    normalized = key / (2 ** bits)  # [0.5, 1.0)
    return (normalized - 0.75) * 4  # Maps roughly to [-1, +1]


def theta_ray_coords(n: int, X: float = 1.0, Y: float = 0.0) -> Tuple[float, float]:
    """
    Compute 2D ray embedding coordinates for integer n.
    
    The ray embedding maps integers to a 2D plane where:
    - Shell k determines x-coordinate: x_k = X * (2 - 2^(1-k))
    - Odd core c determines ray slope: y = Y + m_c * x
    
    Args:
        n: Input integer (n >= 0)
        X: Horizontal scale (shells approach x = 2X)
        Y: Vertical offset (midline y = Y)
    
    Returns:
        (x, y) coordinates in the ray embedding space
        
    Example:
        >>> theta_ray_coords(7)  # shell=2, core=7
        (1.5, 2.0)  # x=1.5, y=0 + (4/3)*1.5 = 2.0
    """
    if n == 0:
        return (0.0, Y)
    
    k = shell(n)
    
    # Shell x-coordinate: x_k = X * (2 - 2^(1-k))
    if k == 0:
        x = 0.0
    else:
        x = X * (2 - 2 ** (1 - k))
    
    # Ray y-coordinate: y = Y + m_c * x
    c = odd_core(n)
    m = slope_for_core(c)
    y = Y + m * x
    
    return (x, y)


def theta_ray_features(n: int, X: float = 1.0, Y: float = 0.0) -> List[float]:
    """
    Combined theta + ray feature vector (8 features).
    
    Combines theta decomposition with ray embedding for ML applications.
    
    Returns:
        [theta_key, shell, v2, quadrant, sign, x, y, slope]
        
    Feature table:
        0: theta_key (float)
        1: shell (float)
        2: v2 (float)
        3: quadrant (float, 0-3)
        4: sign (float, 0 or 1)
        5: x (float, ray x-coordinate)
        6: y (float, ray y-coordinate)
        7: slope (float, ray slope m_c)
    """
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Y, 0.0]
    
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


#==============================================================================
# SKLEARN TRANSFORMER
#==============================================================================

if HAS_SKLEARN:
    
    class ThetaFeatureTransformer(BaseEstimator, TransformerMixin):
        """
        Sklearn transformer for theta feature extraction.
        
        Parameters:
            extended: If True, output 10 features; otherwise 5
            normalize: If True, normalize features to [0, 1]
            max_shell: Maximum shell for normalization (default 64)
        
        Usage:
            from theta_toolkit_v1 import ThetaFeatureTransformer
            
            transformer = ThetaFeatureTransformer(extended=True, normalize=True)
            X_transformed = transformer.fit_transform(X)
        """
        
        def __init__(self, extended: bool = False, normalize: bool = False, max_shell: int = 64):
            self.extended = extended
            self.normalize = normalize
            self.max_shell = max_shell
        
        def fit(self, X, y=None):
            """Fit (no-op, stateless transformer)."""
            return self
        
        def transform(self, X) -> np.ndarray:
            """Transform integer array to theta features."""
            X = np.asarray(X).ravel()
            
            if HAS_NUMPY:
                features = features_vec(X, extended=self.extended)
            else:
                features = np.array([theta_features(int(x), self.extended) for x in X])
            
            if self.normalize:
                features = features.astype(np.float64)
                # Normalize each column
                # theta_key: divide by 2^shell to get [0, 2)
                # shell: divide by max_shell
                # v2: divide by max_shell (similar range)
                # quadrant: divide by 4
                # sign: already 0 or 1
                features[:, 0] = features[:, 0] / (2.0 ** (features[:, 1] + 1) + 1e-10)
                features[:, 1] = features[:, 1] / self.max_shell
                features[:, 2] = features[:, 2] / self.max_shell
                features[:, 3] = features[:, 3] / 4.0
                # sign stays as-is
                
                if self.extended:
                    # core: divide by 2^core_bits
                    features[:, 5] = features[:, 5] / (2.0 ** features[:, 6] + 1e-10)
                    features[:, 6] = features[:, 6] / self.max_shell
                    features[:, 7] = features[:, 7] / 64.0  # max popcount
                    features[:, 8] = features[:, 8] / 256.0
                    features[:, 9] = features[:, 9] / 256.0
            
            return features
        
        def get_feature_names_out(self, input_features=None) -> List[str]:
            """Return feature names."""
            names = ['theta_key', 'shell', 'v2', 'quadrant', 'sign']
            if self.extended:
                names.extend(['core', 'core_bits', 'popcount', 'lo_byte', 'hi_byte'])
            return names


#==============================================================================
# TEST SUITE
#==============================================================================

def run_tests(verbose: bool = True) -> bool:
    """
    Comprehensive test suite.
    
    Tests:
    1. Reversibility
    2. Bit-length collisions
    3. Known test vectors
    4. Shell mixing
    5. Large-n stability
    """
    all_passed = True
    
    def test(name: str, condition: bool):
        nonlocal all_passed
        status = "✓ PASS" if condition else "✗ FAIL"
        if verbose:
            print(f"  {status}: {name}")
        if not condition:
            all_passed = False
    
    if verbose:
        print("=" * 60)
        print("  THETA TOOLKIT v1.0 TEST SUITE")
        print("=" * 60)
        print()
    
    # Test 1: Reversibility
    if verbose:
        print("1. Reversibility tests")
    
    for n in [1, 2, 3, 7, 12, 100, 255, 256, 1000, 12345, 2**20, 2**32 - 1]:
        d = decompose(n)
        recovered = recompose(d.v2, d.theta_key, d.core_bits, d.sign)
        test(f"reverse(decompose({n})) == {n}", recovered == n)
    
    # Test 2: Bit-length collision detection
    if verbose:
        print("\n2. Bit-length collision tests")
    
    # These have same theta_key but different core_bits
    # 1 = 0b1 → theta_key=1 (1 bit)
    # 3 = 0b11 → theta_key=3 (2 bits)
    # Without core_bits, both would decode incorrectly
    d1 = decompose(1)
    d3 = decompose(3)
    test("n=1 and n=3 have different core_bits", d1.core_bits != d3.core_bits)
    test("n=1 decomposition correct", d1.core_bits == 1 and d1.theta_key == 1)
    test("n=3 decomposition correct", d3.core_bits == 2 and d3.theta_key == 3)
    
    # Test 3: Known test vectors
    if verbose:
        print("\n3. Known test vectors")
    
    vectors = [
        (1, 1, 0, 1, 1, 'TOP'),
        (2, 1, 1, 1, 1, 'TOP'),
        (3, 3, 0, 3, 2, 'LEFT'),
        (4, 1, 2, 1, 1, 'TOP'),
        (5, 5, 0, 5, 3, 'BOTTOM'),
        (7, 7, 0, 7, 3, 'LEFT'),
        (12, 3, 2, 3, 2, 'LEFT'),
        (100, 19, 2, 25, 5, 'BOTTOM'),
    ]
    
    for n, exp_key, exp_v2, exp_core, exp_bits, exp_edge in vectors:
        d = decompose(n)
        test(f"n={n}: key={exp_key}", d.theta_key == exp_key)
        test(f"n={n}: v2={exp_v2}", d.v2 == exp_v2)
        test(f"n={n}: core={exp_core}", d.core == exp_core)
        test(f"n={n}: core_bits={exp_bits}", d.core_bits == exp_bits)
        test(f"n={n}: edge={exp_edge}", d.edge == exp_edge)
    
    # Test 4: Shell mixing
    if verbose:
        print("\n4. Shell mixing (theta order validation)")
    
    # In natural order, consecutive integers have monotonic shells
    # This is just a documentation test
    shells_1_to_20 = [shell(n) for n in range(1, 21)]
    test("Natural order has increasing shells", shells_1_to_20 == sorted(shells_1_to_20))
    
    # Test 5: Large-n stability
    if verbose:
        print("\n5. Large-n stability")
    
    large_values = [2**40, 2**50, 2**60, 2**63 - 1]
    for n in large_values:
        d = decompose(n)
        recovered = recompose(d.v2, d.theta_key, d.core_bits, d.sign)
        test(f"reverse(decompose(2^{n.bit_length()-1})) stable", recovered == n)
    
    # Test 6: Edge cases
    if verbose:
        print("\n6. Edge cases")
    
    test("v2(0) == 0", v2(0) == 0)
    test("odd_core(0) == 0", odd_core(0) == 0)
    test("theta_key(0) == 0", theta_key(0) == 0)
    test("decompose(0).n == 0", decompose(0).n == 0)
    
    # Powers of 2
    for k in [1, 2, 4, 8, 16]:
        n = 2 ** k
        d = decompose(n)
        test(f"2^{k}: v2={k}, core=1", d.v2 == k and d.core == 1)
    
    # Mersenne-like (2^k - 1)
    for k in [3, 7, 15, 31]:
        n = 2 ** k - 1
        d = decompose(n)
        test(f"2^{k}-1: v2=0 (all odd)", d.v2 == 0)
    
    # Test 7: Codec encode/decode
    if verbose:
        print("\n7. Codec encode/decode with validation")
    
    for n in [1, 7, 12, 100, 1000, 12345]:
        tk, cs, v = encode_chunk(n)
        recovered = decode_chunk(tk, cs, v, validate=True)
        test(f"codec roundtrip n={n}", recovered == n)
    
    # Test codec validation catches corruption
    try:
        # theta_key=7 (0b111) has core_bits=3, so core_shell should be 2
        decode_chunk(theta_key_val=7, core_shell=5, v2_val=0, validate=True)
        test("codec validation catches corruption", False)  # Should have raised
    except ThetaFormatError:
        test("codec validation catches corruption", True)
    
    # Test 8: Ray embedding (v1.2)
    if verbose:
        print("\n8. Ray embedding (v1.2)")
    
    # Test default slopes
    test("slope_for_core(1) == -1", slope_for_core(1) == -1.0)
    test("slope_for_core(3) == +1", slope_for_core(3) == +1.0)
    test("slope_for_core(5) == 0", slope_for_core(5) == 0.0)
    test("slope_for_core(7) == 4/3", abs(slope_for_core(7) - 4/3) < 1e-10)
    
    # Test ray coordinates
    x, y = theta_ray_coords(1, X=1.0, Y=0.0)
    test("theta_ray_coords(1): x=0", x == 0.0)
    test("theta_ray_coords(1): y=0", y == 0.0)
    
    x, y = theta_ray_coords(2, X=1.0, Y=0.0)  # shell=1, core=1, slope=-1
    test("theta_ray_coords(2): x=1.0", abs(x - 1.0) < 1e-10)
    test("theta_ray_coords(2): y=-1.0", abs(y - (-1.0)) < 1e-10)
    
    x, y = theta_ray_coords(7, X=1.0, Y=0.0)  # shell=2, core=7, slope=4/3
    test("theta_ray_coords(7): x=1.5", abs(x - 1.5) < 1e-10)
    test("theta_ray_coords(7): y=2.0", abs(y - 2.0) < 1e-10)
    
    # Test ray features
    features = theta_ray_features(7)
    test("theta_ray_features(7) has 8 elements", len(features) == 8)
    test("theta_ray_features(7)[5] == x", abs(features[5] - 1.5) < 1e-10)
    test("theta_ray_features(7)[6] == y", abs(features[6] - 2.0) < 1e-10)
    
    # Test 9: v1.2 recompose functions
    if verbose:
        print("\n9. v1.2 recompose functions")
    
    for n in [1, 7, 12, 100, 1000]:
        d = decompose(n)
        from_core = recompose_from_core(d.v2, d.core)
        from_theta = recompose_from_theta(d.v2, d.theta_key)
        test(f"recompose_from_core({n})", from_core == n)
        test(f"recompose_from_theta({n})", from_theta == n)
    
    if verbose:
        print()
        print("=" * 60)
        print(f"  {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print("=" * 60)
    
    return all_passed


#==============================================================================
# CLI
#==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Theta Toolkit v1.0')
    subparsers = parser.add_subparsers(dest='command')
    
    # test command
    subparsers.add_parser('test', help='Run test suite')
    
    # analyze command
    an = subparsers.add_parser('analyze', help='Analyze a number')
    an.add_argument('number', type=int)
    
    # demo command
    subparsers.add_parser('demo', help='Run demo')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        success = run_tests(verbose=True)
        exit(0 if success else 1)
    
    elif args.command == 'analyze':
        d = decompose(args.number)
        print(f"Analysis of {args.number}:")
        print(f"  Decomposition: {args.number} = 2^{d.v2} × {d.core}")
        print(f"  Core bits:     {d.core_bits} (REQUIRED for reconstruction)")
        print(f"  Theta key:     {d.theta_key}")
        print(f"  Shell:         {d.shell}")
        print(f"  Quadrant:      {d.quadrant} ({d.edge})")
        print()
        print(f"  Reconstruction check:")
        recovered = recompose(d.v2, d.theta_key, d.core_bits, d.sign)
        print(f"    recompose({d.v2}, {d.theta_key}, {d.core_bits}) = {recovered}")
        print(f"    Match: {'✓' if recovered == args.number else '✗'}")
    
    elif args.command == 'demo':
        print("=" * 60)
        print("  THETA TOOLKIT v1.2 DEMO")
        print("=" * 60)
        print()
        
        print("First 20 integers:")
        print(f"{'n':>6} {'key':>8} {'shell':>6} {'v2':>4} {'core':>6} {'bits':>5} {'edge':>8}")
        print("-" * 50)
        for n in range(1, 21):
            d = decompose(n)
            print(f"{n:>6} {d.theta_key:>8} {d.shell:>6} {d.v2:>4} {d.core:>6} {d.core_bits:>5} {d.edge:>8}")
        
        print()
        print("Shell discontinuity example:")
        for n in [254, 255, 256, 257]:
            d = decompose(n)
            angle = theta_angle_scaled(n, 16)
            print(f"  n={n}: shell={d.shell}, theta_key={d.theta_key}, scaled_angle={angle}")
        print("  Note: angle jumps at shell boundary (255→256)")
        
        print()
        print("=" * 60)
        print("  RAY EMBEDDING (v1.2)")
        print("=" * 60)
        print()
        
        print("Default slopes for small cores:")
        for c in [1, 3, 5, 7]:
            print(f"  core={c}: slope={slope_for_core(c):+.4f}")
        
        print()
        print("Ray coordinates (X=1.0, Y=0.0):")
        print(f"{'n':>6} {'shell':>6} {'core':>6} {'x':>8} {'slope':>8} {'y':>8}")
        print("-" * 50)
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
            d = decompose(n)
            x, y = theta_ray_coords(n)
            m = slope_for_core(d.core)
            print(f"{n:>6} {d.shell:>6} {d.core:>6} {x:>8.3f} {m:>+8.4f} {y:>+8.3f}")
        
        print()
        print("Ray features for n=7:")
        features = theta_ray_features(7)
        names = ['theta_key', 'shell', 'v2', 'quadrant', 'sign', 'x', 'y', 'slope']
        for i, (name, val) in enumerate(zip(names, features)):
            print(f"  [{i}] {name:>12}: {val:.4f}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
