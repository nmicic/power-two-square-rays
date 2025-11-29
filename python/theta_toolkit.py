# Copyright (c) 2025
# Author: Nenad Micic <nenad@micic.be>
# License: MIT License
#
# This file contains the supported implementation of the Theta Toolkit
# (Python CPU reference version). See repository root README for details.
#
# Portions of this file include AI-assisted code generation
# (ChatGPT, Claude). All work reviewed and validated by the author.
#
# Theta Toolkit — CPU Reference Implementation
# Mirrors cuda/theta_cuda_v1.2.cuh
#
# Repository: https://github.com/nmicic/power-two-square-rays/
#
# DISCLAIMER:
#     This is an AI-assisted exploratory project.
#     Educational and experimental only - not peer-reviewed.
"""
theta_toolkit.py - CPU Reference Implementation for Theta Operations

Integer-native angular encoding via 2-adic decomposition.
All operations use only bitwise/integer ops - no floating point in core logic.

CORE PRINCIPLE:
    Every positive integer n has unique decomposition: n = 2^v2(n) × core(n)
    theta_key(n) = bit_reverse(odd_core(n))

USAGE:
    from theta_toolkit import v2, odd_core, theta_key, shell, theta_bucket

    n = 12
    print(f"v2={v2(n)}, core={odd_core(n)}, theta_key={theta_key(n)}")
"""

from dataclasses import dataclass
from typing import Tuple
from enum import IntEnum


#==============================================================================
# EDGE/QUADRANT CONSTANTS
#==============================================================================

class Edge(IntEnum):
    """Edge/quadrant encoding from top 2 bits of theta_key"""
    TOP = 0     # 00 binary
    RIGHT = 1   # 01 binary
    BOTTOM = 2  # 10 binary
    LEFT = 3    # 11 binary

EDGE_NAMES = ('TOP', 'RIGHT', 'BOTTOM', 'LEFT')


#==============================================================================
# CORE PRIMITIVES
#==============================================================================

def v2(n: int) -> int:
    """
    Count trailing zeros (2-adic valuation).

    v2(12) = 2 because 12 = 0b1100
    v2(8) = 3 because 8 = 0b1000
    v2(7) = 0 because 7 = 0b111
    """
    if n == 0:
        return 0
    n = abs(int(n))
    return (n & -n).bit_length() - 1


def odd_core(n: int) -> int:
    """
    Extract odd core: n >> v2(n)

    odd_core(12) = 3 because 12 = 4 × 3
    odd_core(8) = 1 because 8 = 8 × 1
    odd_core(7) = 7 because 7 = 1 × 7
    """
    if n == 0:
        return 0
    n = abs(int(n))
    return n >> v2(n)


def bit_length(n: int) -> int:
    """
    Bit length = floor(log2(n)) + 1

    bit_length(12) = 4
    bit_length(8) = 4
    bit_length(1) = 1
    """
    if n == 0:
        return 0
    return abs(int(n)).bit_length()


def shell(n: int) -> int:
    """
    Shell index = floor(log2(n)) = bit_length - 1

    shell(12) = 3 because 8 <= 12 < 16
    shell(8) = 3
    shell(1) = 0
    """
    bl = bit_length(n)
    return bl - 1 if bl > 0 else 0


def bit_reverse(val: int, bits: int) -> int:
    """
    Reverse bits within specified bit width.

    bit_reverse(0b1100, 4) = 0b0011
    bit_reverse(12, 4) = 3
    """
    if bits <= 0 or val == 0:
        return 0
    val = abs(int(val))
    result = 0
    for _ in range(bits):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result


#==============================================================================
# THETA KEY - The Core Angular Encoding
#==============================================================================

def theta_key(n: int) -> int:
    """
    Compute theta key: angular position as integer.

    theta_key(n) = bit_reverse(odd_core(n), bit_length(odd_core(n)))

    Top 2 bits encode edge: 00=TOP, 01=RIGHT, 10=BOTTOM, 11=LEFT

    WARNING: theta_key alone is NOT sufficient for reconstruction.
    Use decompose() for full reversibility.
    """
    if n <= 0:
        return 0
    core = odd_core(n)
    bits = bit_length(core)
    return bit_reverse(core, bits)


def theta_quadrant(n: int) -> int:
    """
    Get quadrant index (0-3) from theta key.

    Extracted from top 2 bits of theta_key.
    """
    key = theta_key(n)
    if key == 0:
        return 0
    bits = bit_length(key)
    return ((key >> (bits - 2)) & 0x3) if bits >= 2 else 0


def theta_edge(n: int) -> Edge:
    """
    Get edge enum from theta key.

    Returns Edge.TOP, Edge.RIGHT, Edge.BOTTOM, or Edge.LEFT
    """
    return Edge(theta_quadrant(n))


#==============================================================================
# BUCKETING / SHARDING
#==============================================================================

def theta_bucket(n: int, num_buckets: int, secret: int = 0) -> int:
    """
    Hash value to bucket with proper uniform distribution.

    IMPORTANT: Simple theta_key % buckets has POOR uniformity because
    theta_key is always ODD (LSB=1). This function uses proper mixing.

    Args:
        n: Input integer
        num_buckets: Number of buckets
        secret: Optional secret for different distributions

    Returns:
        Bucket index in [0, num_buckets)
    """
    key = theta_key(n)
    v = v2(n)
    sh = shell(n)

    # Combine with 64-bit arithmetic for better mixing
    h = key
    h = (h * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    h = h ^ ((v * 0x517CC1B727220A95) & 0xFFFFFFFFFFFFFFFF)
    h = h ^ ((sh * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF)
    h = h ^ secret

    # Final mix: upper bits are better distributed
    h = ((h * 0x9E3779B97F4A7C15) >> 32) & 0xFFFFFFFF

    return h % num_buckets


#==============================================================================
# DECOMPOSITION
#==============================================================================

@dataclass
class ThetaDecomposition:
    """
    Full 2-adic decomposition with all fields needed for reconstruction.

    CRITICAL: core_bits is REQUIRED for reversibility.
    Without it, multiple integers can map to the same theta_key.

    Fields:
        n: Original integer
        v2: 2-adic valuation (trailing zeros)
        core: Odd core
        core_bits: bit_length(core) - REQUIRED FOR RECONSTRUCTION
        shell: floor(log2(n)) - shell of original integer
        core_shell: core_bits - 1 - used in codec for validation
        theta_key: bit_reverse(core, core_bits)
        edge: Edge enum (TOP, RIGHT, BOTTOM, LEFT)
    """
    n: int
    v2: int
    core: int
    core_bits: int
    shell: int
    core_shell: int
    theta_key: int
    edge: Edge


def decompose(n: int) -> ThetaDecomposition:
    """
    Full 2-adic decomposition with all fields.

    INCLUDES core_bits which is REQUIRED for reconstruction.
    """
    if n <= 0:
        return ThetaDecomposition(
            n=0, v2=0, core=0, core_bits=0,
            shell=0, core_shell=0, theta_key=0, edge=Edge.TOP
        )

    v = v2(n)
    core = odd_core(n)
    core_b = bit_length(core)
    sh = shell(n)
    key = bit_reverse(core, core_b)
    quad = ((key >> (core_b - 2)) & 0x3) if core_b >= 2 else 0

    return ThetaDecomposition(
        n=n,
        v2=v,
        core=core,
        core_bits=core_b,
        shell=sh,
        core_shell=core_b - 1 if core_b > 0 else 0,
        theta_key=key,
        edge=Edge(quad)
    )


def recompose(v2_val: int, theta_key_val: int, core_bits: int) -> int:
    """
    Reconstruct integer from theta decomposition.

    REQUIRES core_bits for correct reconstruction.
    """
    if theta_key_val == 0:
        return 0
    core = bit_reverse(theta_key_val, core_bits)
    return core << v2_val


#==============================================================================
# RAY EMBEDDING (Minimal Core Version)
#==============================================================================

# Default slopes for small odd cores (normative for v1.2)
DEFAULT_SLOPES = {
    1: -1.0,   # 45° descending
    3: +1.0,   # 45° ascending
    5:  0.0,   # horizontal midline
    7: +4/3,   # steep ascending
}


def slope_for_core(c: int) -> float:
    """
    Get slope for an odd core.

    Default slopes (normative for v1.2):
        c=1 → -1 (45° descending)
        c=3 → +1 (45° ascending)
        c=5 → 0 (horizontal midline)
        c=7 → +4/3 (steep ascending)

    Other cores: Uses normalized theta_key for extension.
    """
    if c in DEFAULT_SLOPES:
        return DEFAULT_SLOPES[c]

    if c == 0:
        return 0.0

    # Extension for other cores
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
    """
    if n <= 0:
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


def theta_ray_features(n: int, X: float = 1.0, Y: float = 0.0) -> Tuple[float, ...]:
    """
    Combined theta + ray feature vector (8 features).

    Returns:
        (theta_key, shell, v2, quadrant, sign, x, y, slope)

    Feature indices:
        0: theta_key (float)
        1: shell (float)
        2: v2 (float)
        3: quadrant (float, 0-3)
        4: sign (float, 0 or 1)
        5: x (float, ray x-coordinate)
        6: y (float, ray y-coordinate)
        7: slope (float, ray slope m_c)
    """
    if n <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Y, 0.0)

    d = decompose(n)
    x, y = theta_ray_coords(n, X, Y)
    m = slope_for_core(d.core)

    return (
        float(d.theta_key),
        float(d.shell),
        float(d.v2),
        float(d.edge.value),
        1.0,  # sign (positive)
        x,
        y,
        m
    )


#==============================================================================
# MODULE EXPORTS
#==============================================================================

__all__ = [
    # Constants
    'Edge',
    'EDGE_NAMES',

    # Core primitives
    'v2',
    'odd_core',
    'bit_length',
    'shell',
    'bit_reverse',

    # Theta key
    'theta_key',
    'theta_quadrant',
    'theta_edge',

    # Bucketing
    'theta_bucket',

    # Decomposition
    'ThetaDecomposition',
    'decompose',
    'recompose',

    # Ray embedding
    'slope_for_core',
    'theta_ray_coords',
    'theta_ray_features',
]
