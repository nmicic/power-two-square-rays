#!/usr/bin/env python3
disclaimer = """

> **Warning / Disclaimer**  
>This is an AI-assisted exploratory visualization project.
>It is educational and experimental only—not peer-reviewed, not mathematical research, and not a new theorem.
>It does not serve as evidence for or against conjectures.
>This repository is math-art/visualization, not a research paper.
>Its purpose is educational exploration and aesthetic interest.
>Any visual patterns are artifacts of the construction.

"""

print(disclaimer)


"""
INTEGER-NATIVE COMPUTATION - Square Ray Operations Toolkit
==========================================================

Core Principle
--------------
Derive geometry from discrete constraints rather than discretizing continuous
geometry. All structure emerges from integer/bitwise operations - trigonometry
is never used in core logic, only in optional human-facing visualization.

The Model
---------
  * Discrete 2D grid, integers only
  * Every position labeled by positive integer n
  * Movement: UP, LEFT, DOWN, RIGHT (cardinal, one unit)
  * Operations: integers, bitwise ops, simple comparisons
  * NOT available: trigonometry, calculus, floating-point

This isn't a limitation - it's a design principle:
  * Eliminates redundant computation
  * Guarantees correctness by construction (bijection = proof)
  * Maps directly to GPU-efficient operations (CUDA intrinsics)
  * Forces optimal algorithms - no "verify what we already know"

Power-of-Two Square Shells
--------------------------
For a fixed exponent `exp`, we consider integers n in [1, 2**exp).

Each n is decomposed into its 2-adic form:
    n = core * 2**v2

  * v2 = v2(n) is the 2-adic valuation (shell index)
  * core is the odd factor (the "odd core", ray label)

All numbers with same odd core lie on the same ODD-CORE RAY across shells.

The Bijection
-------------
We define:
    core  <->  theta_key in [0, 2**(exp-1) - 1]

implemented as:
    odd_core_to_key(core, exp)
    key_to_odd_core(key, exp)

The `theta_key` is a perimeter coordinate: stepping key -> key+1 means
walking one step clockwise around the boundary.

THE BIJECTION IS THE PROOF: If you can convert core↔key and walk without
gaps or collisions, uniqueness is GUARANTEED. No overlap checking needed.

Algorithmic Constraints
-----------------------
All *core* algorithms must obey:

  * NO trigonometry or floating-point in core logic
    - Trig ONLY in [VISUALIZATION] helpers, clearly separated

  * Operations must be O(1) per step:
    - neighbor lookup, shell walks, ray walks, prime-only walks

  * NO redundant verification:
    - Don't check what the bijection guarantees
    - Don't compute angles to compare positions

  * Streaming-friendly memory usage

The scaling rule C(2*n) = 2*C(n) is preserved by construction.

Interpretation
--------------
  * Odd-core rays, gaps, deserts = structural artifacts of embedding
  * NO claims about number theory or prime structure
  * This is exploratory/educational ONLY

Evaluation Style
----------------
  * No fanfare, hype, or grandiose claims
  * Realistic, grounded evaluation only
  * If something is a tautology of construction, say so
  * "Interesting to look at" ≠ "mathematically significant"

==============================================================================
Module Organization:
  [CORE]              - Integer-only operations, GPU-portable
  [CORE/PRIME]        - Prime-related core operations  
  [NAVIGATION]        - Walking/traversal operations
  [NAVIGATION/PRIME]  - Prime-filtered walking
  [ANALYSIS]          - Statistical analysis (valid metrics only)
  [VISUALIZATION]     - Uses trig/floats - human display only
  [CLI]               - Command-line interface
==============================================================================

Author: Integer-native computation tools for power-two-square-rays
License: MIT
"""



from fractions import Fraction
from typing import Iterator, Tuple, Optional, Dict, List, Set
import math  # ONLY for [VISUALIZATION] helpers - not used in [CORE]
import sys
import time
import struct
import csv


# =============================================================================
# [CORE] FUNDAMENTAL BITWISE OPERATIONS - INTEGER-NATIVE
# =============================================================================
# These functions use ONLY:
#   - Integer arithmetic
#   - Bitwise operations (<<, >>, &, ^, |)
#   - Simple comparisons
# NO trigonometry, NO floating-point in the core logic.
# =============================================================================

def key_to_odd_core(k: int, exp: int) -> int:
    """
    [CORE] Convert theta-key to odd core using pure bitwise operations.
    
    The theta-key encodes a normalized binary fraction. We recover the
    original odd number by finding where the significant bits are.
    
    An ant can compute this with only bit shifts and comparisons.
    
    Args:
        k: Theta-key in range [0, 2^(exp-1))
        exp: Exponent defining the range
        
    Returns:
        Odd number corresponding to this key
        
    Example:
        >>> key_to_odd_core(0, 12)
        1
        >>> key_to_odd_core(1, 12)
        2049
    """
    if k == 0:
        return 1
    
    # Find trailing zeros (tells us how much padding was added)
    trailing = (k & -k).bit_length() - 1
    
    # Original fractional part length
    frac_len = exp - 1 - trailing
    
    # Remove padding zeros
    f = k >> trailing
    
    # Reconstruct: odd = 2^frac_len + f (i.e., 1.f in binary)
    return (1 << frac_len) + f


def key_to_odd_core_ants(k: int, exp: int) -> int:
    """
    [CORE] Ant-perspective alias for key_to_odd_core.
    
    Given a perimeter position (key), returns which ray the ant is on.
    Think of it as: "I'm at step k on the square boundary - what ray am I?"
    """
    return key_to_odd_core(k, exp)


def odd_core_to_key(a: int, exp: int) -> int:
    """
    [CORE] Convert odd core to theta-key using pure bitwise operations.
    
    An ant on ray 'a' needs to know its angular position (key) on the boundary.
    This is computed with only bit-length and shifts.
    
    Args:
        a: Odd number
        exp: Exponent defining normalization
        
    Returns:
        Theta-key for this odd number
        
    Example:
        >>> odd_core_to_key(1, 12)
        0
        >>> odd_core_to_key(3, 12)
        1024
    """
    if a == 1:
        return 0
    
    k = a.bit_length()
    f = a ^ (1 << (k - 1))  # Remove leading 1 bit
    return f << (exp - k)   # Normalize to (exp-1) bits


def odd_core_to_key_ants(a: int, exp: int) -> int:
    """
    [CORE] Ant-perspective alias for odd_core_to_key.
    
    Given a ray label (odd core), returns its perimeter position (theta_key).
    Think of it as: "I'm on ray a - which boundary step is this?"
    """
    return odd_core_to_key(a, exp)


def get_odd_core(n: int) -> Tuple[int, int]:
    """
    [CORE] Extract odd core and 2-adic valuation from integer n.
    
    Every positive integer n can be written as n = 2^v2 * a
    where a is odd (the "odd core") and v2 is the 2-adic valuation.
    
    For an ant: v2 tells which shell you're on, core tells which ray.
    
    Args:
        n: Positive integer
        
    Returns:
        (odd_core, v2) tuple
        
    Example:
        >>> get_odd_core(12)
        (3, 2)
        >>> get_odd_core(7)
        (7, 0)
    """
    if n == 0:
        return (0, 0)
    
    v2 = (n & -n).bit_length() - 1
    return (n >> v2, v2)


def get_odd_core_ants(n: int) -> Tuple[int, int]:
    """
    [CORE] Ant-perspective: decompose position n into (ray, shell).
    
    Returns:
        (ray_label, shell_distance) - where the ant is in the grid
    """
    return get_odd_core(n)


def v2_ants(n: int) -> int:
    """
    [CORE] Ant-perspective: which shell is position n on?
    
    Shell 0 = innermost (odd numbers)
    Shell k = distance 2^k from origin
    """
    if n == 0:
        return 0
    return (n & -n).bit_length() - 1


def ray_label_ants(n: int) -> int:
    """
    [CORE] Ant-perspective: which ray is position n on?
    
    All positions on the same ray have the same odd core.
    """
    core, _ = get_odd_core(n)
    return core


def popcount(n: int) -> int:
    """
    [CORE] Count the number of 1-bits in integer n.
    
    CUDA equivalent: __popc() or __popcll()
    
    Args:
        n: Non-negative integer
        
    Returns:
        Number of set bits
        
    Example:
        >>> popcount(7)   # 0b111
        3
        >>> popcount(12)  # 0b1100
        2
    """
    return bin(n).count('1')


def hamming_int(a: int, b: int) -> int:
    """
    [CORE] Hamming distance between two integers (XOR popcount).
    
    Counts how many bit positions differ between a and b.
    CUDA: __popc(a ^ b)
    
    Args:
        a, b: Non-negative integers
        
    Returns:
        Number of differing bits
        
    Example:
        >>> hamming_int(7, 12)  # 0b0111 vs 0b1100 -> differ in 4 bits
        4
    """
    return popcount(a ^ b)


# =============================================================================
# THETA-ORDER METRICS (CUDA-COMPATIBLE)
# =============================================================================
# 
# When analyzing primes in THETA ORDER (not natural order), use these metrics:
#
# VALID METRICS:
#   ┌──────────────────┬────────────────────────────────────┬─────────────┐
#   │ Metric           │ Formula                            │ CUDA Cost   │
#   ├──────────────────┼────────────────────────────────────┼─────────────┤
#   │ theta_key        │ odd_core_to_key(core, exp)         │ O(1) __clz  │
#   │ delta_theta_key  │ key2 - key1 (angular gap)          │ O(1) sub    │
#   │ hamming_primes   │ popcount(p1 ^ p2)                  │ O(1) __popc │
#   │ hamming_keys     │ popcount(key1 ^ key2)              │ O(1) __popc │
#   └──────────────────┴────────────────────────────────────┴─────────────┘
#
# INVALID METRICS (do NOT use for theta-order analysis):
#   - delta_prime (p2 - p1): Natural-order thinking, meaningless in theta order
#   - numeric_gap: Same problem - primes from different shells interleave
#   - angle overlap checking: Tautology - bijection guarantees no overlap
#
# WHY: In theta order, a small prime (7) and large prime (4099) may be
# angularly adjacent. Their numeric difference tells us nothing useful.
# =============================================================================


# =============================================================================
# [CORE] THETA-SORTED GENERATION - INTEGER-NATIVE
# =============================================================================
# Generation is done by counting through keys (perimeter steps).
# No sorting required - O(1) per element.
# =============================================================================

def generate_theta_sorted(exp: int) -> Iterator[int]:
    """
    [CORE] Generate all integers 1 to 2^exp - 1 in theta-sorted (angular) order.
    
    NO SORTING REQUIRED - direct O(1)-per-element generation by counting
    through theta-keys. An ant simply walks the perimeter.
    
    Args:
        exp: Exponent (generates integers up to 2^exp - 1)
        
    Yields:
        Integers in clockwise angular order
        
    Example:
        >>> list(generate_theta_sorted(4))[:10]
        [1, 2, 4, 8, 9, 5, 10, 11, 3, 6]
    """
    max_n = 1 << exp
    num_keys = 1 << (exp - 1)
    
    for key in range(num_keys):
        core = key_to_odd_core(key, exp)
        n = core
        while n < max_n:
            yield n
            n <<= 1


def generate_theta_sorted_ants(exp: int) -> Iterator[int]:
    """
    [CORE] Alias: walk the entire perimeter, visiting all positions.
    
    The ant walks clockwise around the boundary, at each angular position
    visiting all shell levels from innermost outward.
    """
    return generate_theta_sorted(exp)


def generate_theta_sorted_with_info(exp: int) -> Iterator[Tuple[int, int, int, int]]:
    """
    [CORE] Generate all integers with full metadata.
    
    Yields:
        Tuples of (n, odd_core, v2, theta_key)
    """
    max_n = 1 << exp
    num_keys = 1 << (exp - 1)
    
    for theta_key in range(num_keys):
        odd_core = key_to_odd_core(theta_key, exp)
        v2 = 0
        n = odd_core
        while n < max_n:
            yield (n, odd_core, v2, theta_key)
            n <<= 1
            v2 += 1


def generate_with_info_ants(exp: int) -> Iterator[Tuple[int, int, int, int]]:
    """
    [CORE] Alias: walk with full position awareness.
    
    At each position the ant knows:
      - n: the position label
      - ray: which ray it's on
      - shell: which shell (distance from origin)
      - step: which perimeter step (angular position)
    
    Yields:
        (position, ray, shell, perimeter_step)
    """
    return generate_theta_sorted_with_info(exp)


# =============================================================================
# [CORE] O(1) THETA-NEIGHBOR LOOKUP - INTEGER-NATIVE
# =============================================================================
# Finding angular neighbors requires only key arithmetic and bit shifts.
# =============================================================================

def theta_neighbors(n: int, exp: int) -> Tuple[Optional[int], Optional[int]]:
    """
    [CORE] Find theta-neighbors in O(1) using pure bitwise operations.
    
    Returns integers on the adjacent rays at the same shell level as n.
    An ant can find its angular neighbors without computing angles.
    
    Args:
        n: Integer to find neighbors for
        exp: Exponent defining the range
        
    Returns:
        (prev_ray_n, next_ray_n) - integers on adjacent rays, or None if out of range
        
    Example:
        >>> theta_neighbors(12345, 24)
        (12641279, 12641281)
    """
    core, v2 = get_odd_core(n)
    key = odd_core_to_key(core, exp)
    
    max_key = (1 << (exp - 1)) - 1
    max_n = 1 << exp
    
    # Previous ray (counter-clockwise)
    prev_key = key - 1 if key > 0 else max_key
    prev_core = key_to_odd_core(prev_key, exp)
    prev_n = prev_core << v2
    prev_n = prev_n if prev_n < max_n else None
    
    # Next ray (clockwise)
    next_key = (key + 1) if key < max_key else 0
    next_core = key_to_odd_core(next_key, exp)
    next_n = next_core << v2
    next_n = next_n if next_n < max_n else None
    
    return (prev_n, next_n)


def theta_neighbors_ants(n: int, exp: int) -> Tuple[Optional[int], Optional[int]]:
    """
    [CORE] Alias: find neighboring positions on adjacent rays.
    
    Returns (left_neighbor, right_neighbor) at the same shell distance.
    The ant can step left or right along the current shell.
    """
    return theta_neighbors(n, exp)


def theta_neighbors_all_shells(n: int, exp: int) -> Dict:
    """
    [CORE] Get neighbors for all shells (dyadic multiples) of n's ray.
    
    Args:
        n: Integer
        exp: Exponent
        
    Returns:
        Dictionary with neighbor information for each shell
    """
    core, v2 = get_odd_core(n)
    key = odd_core_to_key(core, exp)
    max_key = (1 << (exp - 1)) - 1
    max_n = 1 << exp
    
    prev_key = (key - 1) % (max_key + 1)
    next_key = (key + 1) % (max_key + 1)
    
    prev_core = key_to_odd_core(prev_key, exp)
    next_core = key_to_odd_core(next_key, exp)
    
    result = {
        'n': n,
        'core': core,
        'v2': v2,
        'key': key,
        'prev_core': prev_core,
        'next_core': next_core,
        'shells': []
    }
    
    curr = core
    shell = 0
    while curr < max_n:
        prev_at_shell = prev_core << shell if (prev_core << shell) < max_n else None
        next_at_shell = next_core << shell if (next_core << shell) < max_n else None
        
        result['shells'].append({
            'shell': shell,
            'value': curr,
            'prev': prev_at_shell,
            'next': next_at_shell
        })
        
        curr <<= 1
        shell += 1
    
    return result


# =============================================================================
# [CORE] DIRECTION-TO-RAY BRACKETING - INTEGER-NATIVE (No trigonometry!)
# =============================================================================
# Uses only rational arithmetic (Fraction) and comparisons.
# =============================================================================

def direction_to_t(dx: int, dy: int) -> Fraction:
    """
    [CORE] Convert direction vector to perimeter parameter t in [0, 1).
    
    PURE RATIONAL ARITHMETIC - NO TRIGONOMETRY!
    
    The ant doesn't know angles, but it knows edges:
    - t=0:    corner (1, 1)   "northeast"
    - t=0.25: corner (-1, 1)  "northwest"
    - t=0.50: corner (-1, -1) "southwest"
    - t=0.75: corner (1, -1)  "southeast"
    
    Args:
        dx, dy: Direction vector (integers, not both zero)
        
    Returns:
        Perimeter parameter t as exact Fraction in [0, 1)
        
    Raises:
        ValueError: If both dx and dy are zero
    """
    if dx == 0 and dy == 0:
        raise ValueError("Direction cannot be zero vector")
    
    abs_dx, abs_dy = abs(dx), abs(dy)
    
    if dy > 0 and abs_dy >= abs_dx:
        # TOP edge: y = 1, x from 1 to -1
        x = Fraction(dx, dy)
        return (1 - x) / 8
        
    elif dx < 0 and abs_dx >= abs_dy:
        # LEFT edge: x = -1, y from 1 to -1
        y = Fraction(-dy, dx)
        return Fraction(1, 4) + (1 - y) / 8
        
    elif dy < 0 and abs_dy >= abs_dx:
        # BOTTOM edge: y = -1, x from -1 to 1
        x = Fraction(-dx, dy)
        return Fraction(1, 2) + (x + 1) / 8
        
    else:  # dx > 0 and abs_dx >= abs_dy
        # RIGHT edge: x = 1, y from -1 to 1
        y = Fraction(dy, dx)
        return Fraction(3, 4) + (y + 1) / 8


def direction_to_t_ants(dx: int, dy: int) -> Fraction:
    """
    [CORE] Alias: convert a direction to perimeter position.
    
    The ant faces direction (dx, dy) and wants to know where on the 
    square boundary this direction intersects.
    """
    return direction_to_t(dx, dy)


def find_bracketing_rays(dx: int, dy: int, exp: int) -> Tuple[int, int]:
    """
    [CORE] Find the two odd cores that bracket the direction (dx, dy).
    
    NO TRIGONOMETRY - only comparisons, one division, and bitwise ops!
    
    An ant looking in direction (dx, dy) can find which rays bracket that view.
    
    Args:
        dx, dy: Direction vector as integers
        exp: Grid resolution exponent
        
    Returns:
        (lower_core, upper_core) - the bracketing odd cores
        
    Example:
        >>> find_bracketing_rays(1, 1, 24)  # 45 degrees
        (1, 8388609)
        >>> find_bracketing_rays(3, 4, 24)  # ~53 degrees
        (33, 8650753)
    """
    t = direction_to_t(dx, dy)
    num_keys = 1 << (exp - 1)
    
    key_lo = int(t * num_keys)
    key_hi = (key_lo + 1) % num_keys
    
    return (key_to_odd_core(key_lo, exp), key_to_odd_core(key_hi, exp))


def find_bracketing_rays_ants(dx: int, dy: int, exp: int) -> Tuple[int, int]:
    """
    [CORE] Alias: which rays bracket my line of sight?
    
    Returns (ray_left, ray_right) for the given facing direction.
    """
    return find_bracketing_rays(dx, dy, exp)


def find_bracketing_rays_detailed(dx: int, dy: int, exp: int) -> Dict:
    """
    [CORE] Find bracketing rays with full details.
    
    Args:
        dx, dy: Direction vector
        exp: Grid resolution
        
    Returns:
        Dictionary with full bracketing information
    """
    t = direction_to_t(dx, dy)
    num_keys = 1 << (exp - 1)
    
    scaled = t * num_keys
    key_lo = int(scaled)
    key_hi = (key_lo + 1) % num_keys
    
    core_lo = key_to_odd_core(key_lo, exp)
    core_hi = key_to_odd_core(key_hi, exp)
    
    return {
        'direction': (dx, dy),
        't': float(t),
        't_exact': t,
        'key_lower': key_lo,
        'key_upper': key_hi,
        'core_lower': core_lo,
        'core_upper': core_hi,
    }


# =============================================================================
# [CORE] UTILITY OPERATIONS - INTEGER-NATIVE
# =============================================================================

def hamming_distance(a: int, b: int) -> int:
    """[CORE] Compute Hamming distance between two integers."""
    return bin(a ^ b).count('1')


def hamming_distance_ants(pos1: int, pos2: int) -> int:
    """
    [CORE] Alias: bit-difference between two positions.
    
    How many bits differ between position labels?
    """
    return hamming_distance(pos1, pos2)


def ray_members(odd_core: int, exp: int) -> List[int]:
    """[CORE] Get all integers on a ray (all dyadic multiples of odd_core)."""
    max_n = 1 << exp
    result = []
    n = odd_core
    while n < max_n:
        result.append(n)
        n <<= 1
    return result


def ray_members_ants(ray: int, exp: int) -> List[int]:
    """
    [CORE] Alias: all positions along a given ray.
    
    Returns positions from innermost shell outward on the specified ray.
    """
    return ray_members(ray, exp)


def integers_in_shell(shell: int, exp: int) -> Iterator[int]:
    """
    [CORE] Generate all integers in a given shell (distance 2^shell from origin).
    
    Shell k contains integers in range [2^k, 2^(k+1)).
    """
    start = 1 << shell
    end = min(1 << (shell + 1), 1 << exp)
    for n in range(start, end):
        yield n


def shell_positions_ants(shell: int, exp: int) -> Iterator[int]:
    """
    [CORE] Alias: all positions on a given shell.
    
    Walk around a single shell (constant distance from origin).
    """
    return integers_in_shell(shell, exp)


def angular_distance_keys(key1: int, key2: int, exp: int) -> int:
    """
    [CORE] Compute angular distance between two rays (in key units).
    
    Returns the minimum of clockwise and counter-clockwise distances.
    An ant measures angular separation by counting perimeter steps.
    """
    num_keys = 1 << (exp - 1)
    diff = abs(key1 - key2)
    return min(diff, num_keys - diff)


def angular_distance_ants(pos1: int, pos2: int, exp: int) -> int:
    """
    [CORE] Alias: angular separation between two positions.
    
    Returns minimum perimeter steps between the rays of pos1 and pos2.
    """
    core1, _ = get_odd_core(pos1)
    core2, _ = get_odd_core(pos2)
    key1 = odd_core_to_key(core1, exp)
    key2 = odd_core_to_key(core2, exp)
    return angular_distance_keys(key1, key2, exp)


# =============================================================================
# [CORE/PRIME] PRIMALITY TESTING - INTEGER-NATIVE
# =============================================================================
# Miller-Rabin uses only integer modular arithmetic.
# =============================================================================

def _miller_rabin_witness(n: int, a: int, d: int, r: int) -> bool:
    """[CORE/PRIME] Test if 'a' is a witness for compositeness of 'n'."""
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False  # Probably prime
    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return False  # Probably prime
    return True  # Composite


def is_prime(n: int) -> bool:
    """
    [CORE/PRIME] Deterministic primality test for n < 2^64.
    
    Uses Miller-Rabin with witnesses that guarantee correctness
    for all integers below 2^64.
    
    Pure integer modular arithmetic - integer-native.
    
    Args:
        n: Integer to test
        
    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witnesses that work for all n < 2^64
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for a in witnesses:
        if a >= n:
            continue
        if _miller_rabin_witness(n, a, d, r):
            return False  # Composite
    
    return True  # Prime


def is_prime_ants(n: int) -> bool:
    """
    [CORE/PRIME] Alias: is position n a prime?
    
    The ant can test any position for primality using integer operations.
    """
    return is_prime(n)


# =============================================================================
# [NAVIGATION] WALKING OPERATIONS - INTEGER-NATIVE
# =============================================================================
# Ants navigate by counting keys (perimeter steps).
# All operations are O(1) per step.
# =============================================================================

def walk_theta(
    start_n: int,
    exp: int,
    count: int,
    direction: str = 'cw',
    shell_min: Optional[int] = None,
    shell_max: Optional[int] = None
) -> Iterator[int]:
    """
    [NAVIGATION] Walk through theta-sorted integers starting from a given number.
    
    This allows exploring a narrow angular region without generating
    all integers - perfect for an ant zooming into specific areas.
    
    Args:
        start_n: Starting integer (determines starting ray)
        exp: Exponent defining the range
        count: Number of integers to yield
        direction: 'cw' for clockwise (increasing theta), 'ccw' for counter-clockwise
        shell_min: Minimum shell to include (None = 0)
        shell_max: Maximum shell to include (None = exp-1)
        
    Yields:
        Integers in theta order starting from start_n's ray
        
    Example:
        >>> list(walk_theta(1000, exp=16, count=20, direction='cw'))
        [1000, 2000, 4000, 8000, 16000, 32000, ...]
    """
    core, v2 = get_odd_core(start_n)
    key = odd_core_to_key(core, exp)
    
    max_key = (1 << (exp - 1)) - 1
    max_n = 1 << exp
    
    shell_lo = shell_min if shell_min is not None else 0
    shell_hi = shell_max if shell_max is not None else exp - 1
    
    step = 1 if direction == 'cw' else -1
    yielded = 0
    current_key = key
    
    while yielded < count:
        current_core = key_to_odd_core(current_key, exp)
        
        # Yield all integers on this ray within shell bounds
        for shell in range(shell_lo, shell_hi + 1):
            n = current_core << shell
            if n >= max_n:
                break
            if n >= (1 << shell_lo):  # Ensure we're at minimum shell
                yield n
                yielded += 1
                if yielded >= count:
                    return
        
        # Move to next ray
        current_key = (current_key + step) % (max_key + 1)
        
        # Safety: if we've wrapped around completely, stop
        if current_key == key:
            return


def walk_theta_ants(
    start_pos: int,
    exp: int,
    steps: int,
    direction: str = 'cw',
    shell_min: Optional[int] = None,
    shell_max: Optional[int] = None
) -> Iterator[int]:
    """
    [NAVIGATION] Alias: walk around the perimeter.
    
    Starting from start_pos, the ant walks clockwise or counter-clockwise,
    visiting positions on each ray within the shell range.
    
    Args:
        start_pos: Starting position
        exp: Grid size exponent
        steps: Number of positions to visit
        direction: 'cw' (right/clockwise) or 'ccw' (left/counter-clockwise)
        shell_min: Innermost shell to visit
        shell_max: Outermost shell to visit
    """
    return walk_theta(start_pos, exp, steps, direction, shell_min, shell_max)


def walk_theta_at_shell(
    start_n: int,
    exp: int,
    count: int,
    direction: str = 'cw',
    shell: Optional[int] = None
) -> Iterator[int]:
    """
    [NAVIGATION] Walk through integers at a specific shell level.
    
    This yields exactly one integer per ray at the specified shell,
    allowing exploration of a "ring" at constant distance from origin.
    
    Perfect for an ant walking a single shell.
    
    Args:
        start_n: Starting integer (determines starting ray)
        exp: Exponent defining the range
        count: Number of integers to yield
        direction: 'cw' for clockwise, 'ccw' for counter-clockwise
        shell: Shell level (if None, uses start_n's shell)
        
    Yields:
        Integers at the specified shell in theta order
    """
    core, v2 = get_odd_core(start_n)
    key = odd_core_to_key(core, exp)
    
    target_shell = shell if shell is not None else v2
    max_key = (1 << (exp - 1)) - 1
    max_n = 1 << exp
    
    step = 1 if direction == 'cw' else -1
    yielded = 0
    current_key = key
    start_key = key
    
    while yielded < count:
        current_core = key_to_odd_core(current_key, exp)
        n = current_core << target_shell
        
        if n < max_n:
            yield n
            yielded += 1
        
        # Move to next ray
        current_key = (current_key + step) % (max_key + 1)
        
        # Safety: if we've wrapped around completely, stop
        if current_key == start_key and yielded > 0:
            return


def walk_shell_ants(
    start_pos: int,
    exp: int,
    steps: int,
    direction: str = 'cw',
    shell: Optional[int] = None
) -> Iterator[int]:
    """
    [NAVIGATION] Alias: walk around a single shell.
    
    Stay at constant distance from origin, visiting one position per ray.
    """
    return walk_theta_at_shell(start_pos, exp, steps, direction, shell)


def walk_rays(
    start_n: int,
    exp: int,
    ray_count: int,
    direction: str = 'cw'
) -> Iterator[Tuple[int, List[int]]]:
    """
    [NAVIGATION] Walk through adjacent rays, yielding all integers on each ray.
    
    Useful for examining the full structure of nearby rays.
    
    Args:
        start_n: Starting integer (determines starting ray)
        exp: Exponent defining the range
        ray_count: Number of rays to examine
        direction: 'cw' for clockwise, 'ccw' for counter-clockwise
        
    Yields:
        Tuples of (odd_core, [list of integers on that ray])
    """
    core, _ = get_odd_core(start_n)
    key = odd_core_to_key(core, exp)
    
    max_key = (1 << (exp - 1)) - 1
    max_n = 1 << exp
    
    step = 1 if direction == 'cw' else -1
    
    for _ in range(ray_count):
        current_core = key_to_odd_core(key, exp)
        members = ray_members(current_core, exp)
        yield (current_core, members)
        
        key = (key + step) % (max_key + 1)


def walk_rays_ants(
    start_pos: int,
    exp: int,
    num_rays: int,
    direction: str = 'cw'
) -> Iterator[Tuple[int, List[int]]]:
    """
    [NAVIGATION] Alias: examine consecutive rays.
    
    For each ray, yields (ray_label, positions_on_ray).
    """
    return walk_rays(start_pos, exp, num_rays, direction)


def walk_ray_ants(ray: int, exp: int) -> Iterator[int]:
    """
    [NAVIGATION] Alias: walk outward along a single ray.
    
    Yields positions from innermost to outermost shell on the given ray.
    """
    max_n = 1 << exp
    n = ray
    while n < max_n:
        yield n
        n <<= 1


def zoom_to_region(
    center_n: int,
    exp: int,
    half_width_rays: int = 10,
    shell_min: Optional[int] = None,
    shell_max: Optional[int] = None
) -> Iterator[int]:
    """
    [NAVIGATION] Generate integers in a narrow angular wedge around center_n.
    
    Perfect for zooming into a local region without global generation.
    
    Args:
        center_n: Center of the region
        exp: Exponent
        half_width_rays: Number of rays in each direction from center
        shell_min: Minimum shell to include
        shell_max: Maximum shell to include
        
    Yields:
        Integers in the angular wedge
    """
    core, v2 = get_odd_core(center_n)
    center_key = odd_core_to_key(core, exp)
    
    max_key = (1 << (exp - 1)) - 1
    max_n = 1 << exp
    
    shell_lo = shell_min if shell_min is not None else 0
    shell_hi = shell_max if shell_max is not None else exp - 1
    
    for offset in range(-half_width_rays, half_width_rays + 1):
        key = (center_key + offset) % (max_key + 1)
        current_core = key_to_odd_core(key, exp)
        
        for shell in range(shell_lo, shell_hi + 1):
            n = current_core << shell
            if n >= max_n:
                break
            yield n


def zoom_region_ants(
    center_pos: int,
    exp: int,
    half_width: int = 10,
    shell_min: Optional[int] = None,
    shell_max: Optional[int] = None
) -> Iterator[int]:
    """
    [NAVIGATION] Alias: explore a local angular neighborhood.
    
    The ant examines positions in a wedge around its current location.
    """
    return zoom_to_region(center_pos, exp, half_width, shell_min, shell_max)


# =============================================================================
# [NAVIGATION/PRIME] PRIME-FILTERED WALKING - INTEGER-NATIVE
# =============================================================================
# Walk through only prime positions (skipping composites).
# =============================================================================

def walk_primes(
    start_n: int,
    exp: int,
    count: int,
    direction: str = 'cw',
    shell_min: Optional[int] = None,
    shell_max: Optional[int] = None
) -> Iterator[Tuple[int, int]]:
    """
    [NAVIGATION/PRIME] Walk through ONLY primes in theta order.
    
    Highly efficient for prime-focused analysis - yields only primes
    without returning all the composite numbers in between.
    
    Args:
        start_n: Starting integer (determines starting ray)
        exp: Exponent defining the range
        count: Number of primes to yield
        direction: 'cw' for clockwise, 'ccw' for counter-clockwise
        shell_min: Minimum shell to include
        shell_max: Maximum shell to include
        
    Yields:
        Tuples of (prime, steps_from_last_prime)
    """
    primes_found = 0
    steps_since_last = 0
    
    for n in walk_theta(start_n, exp, count * 1000, direction, shell_min, shell_max):
        steps_since_last += 1
        if is_prime(n):
            yield (n, steps_since_last)
            primes_found += 1
            steps_since_last = 0
            if primes_found >= count:
                return


def walk_primes_ants(
    start_pos: int,
    exp: int,
    prime_count: int,
    direction: str = 'cw',
    shell_min: Optional[int] = None,
    shell_max: Optional[int] = None
) -> Iterator[Tuple[int, int]]:
    """
    [NAVIGATION/PRIME] Alias: walk through prime positions only.
    
    Yields (prime_position, steps_since_last_prime).
    """
    return walk_primes(start_pos, exp, prime_count, direction, shell_min, shell_max)


def walk_primes_at_shell(
    start_n: int,
    exp: int,
    count: int,
    direction: str = 'cw',
    shell: Optional[int] = None
) -> Iterator[Tuple[int, int]]:
    """
    [NAVIGATION/PRIME] Walk through ONLY primes at a specific shell level.
    
    Perfect for analyzing prime distribution at constant distance from origin.
    
    Args:
        start_n: Starting integer (determines starting ray)
        exp: Exponent defining the range
        count: Number of primes to yield
        direction: 'cw' for clockwise, 'ccw' for counter-clockwise
        shell: Shell level (if None, uses start_n's shell)
        
    Yields:
        Tuples of (prime, angular_gap_from_last_prime)
    """
    core, v2 = get_odd_core(start_n)
    target_shell = shell if shell is not None else v2
    
    primes_found = 0
    last_key = None
    num_keys = 1 << (exp - 1)
    
    for n in walk_theta_at_shell(start_n, exp, count * 1000, direction, target_shell):
        if is_prime(n):
            current_core, _ = get_odd_core(n)
            current_key = odd_core_to_key(current_core, exp)
            
            if last_key is not None:
                # Compute angular gap
                if direction == 'cw':
                    gap = (current_key - last_key) % num_keys
                else:
                    gap = (last_key - current_key) % num_keys
            else:
                gap = 0
            
            yield (n, gap)
            last_key = current_key
            primes_found += 1
            if primes_found >= count:
                return


def walk_shell_primes_ants(
    start_pos: int,
    exp: int,
    prime_count: int,
    direction: str = 'cw',
    shell: Optional[int] = None
) -> Iterator[Tuple[int, int]]:
    """
    [NAVIGATION/PRIME] Alias: find primes on a single shell.
    
    Yields (prime_position, angular_gap_from_last).
    """
    return walk_primes_at_shell(start_pos, exp, prime_count, direction, shell)


def find_next_prime_on_ray(
    n: int,
    exp: int,
    max_shells: Optional[int] = None
) -> Optional[int]:
    """
    [NAVIGATION/PRIME] Find the next prime on the same ray (same odd core, higher shell).
    
    Args:
        n: Starting integer
        exp: Exponent
        max_shells: Maximum shells to search (None = search all)
        
    Returns:
        Next prime on the ray, or None if not found
    """
    core, v2 = get_odd_core(n)
    max_n = 1 << exp
    
    max_v2 = max_shells + v2 if max_shells is not None else exp - 1
    
    # Start from next shell
    current_shell = v2 + 1
    
    while current_shell <= max_v2:
        candidate = core << current_shell
        if candidate >= max_n:
            return None
        if is_prime(candidate):
            return candidate
        current_shell += 1
    
    return None


def find_next_ray_prime_ants(pos: int, exp: int, max_shells: Optional[int] = None) -> Optional[int]:
    """
    [NAVIGATION/PRIME] Alias: find next prime outward on my ray.
    """
    return find_next_prime_on_ray(pos, exp, max_shells)


def find_prev_prime_on_ray(
    n: int,
    exp: int
) -> Optional[int]:
    """
    [NAVIGATION/PRIME] Find the previous prime on the same ray (same odd core, lower shell).
    
    Args:
        n: Starting integer
        exp: Exponent
        
    Returns:
        Previous prime on the ray, or None if not found
    """
    core, v2 = get_odd_core(n)
    
    if v2 == 0:
        return None
    
    # Walk down the ray
    for shell in range(v2 - 1, -1, -1):
        candidate = core << shell
        if is_prime(candidate):
            return candidate
    
    return None


def find_prev_ray_prime_ants(pos: int, exp: int) -> Optional[int]:
    """
    [NAVIGATION/PRIME] Alias: find previous prime inward on my ray.
    """
    return find_prev_prime_on_ray(pos, exp)


def find_prime_neighbor(
    n: int,
    exp: int,
    direction: str = 'cw',
    max_steps: int = 100000,
    same_shell: bool = False
) -> Optional[Tuple[int, int]]:
    """
    [NAVIGATION/PRIME] Find the nearest prime in the specified angular direction.
    
    Args:
        n: Starting integer
        exp: Exponent defining the range
        direction: 'cw' for clockwise, 'ccw' for counter-clockwise
        max_steps: Maximum integers to check before giving up
        same_shell: If True, only look at same shell level as n
        
    Returns:
        (prime, steps) tuple, or None if not found within max_steps
    """
    core, v2 = get_odd_core(n)
    
    if same_shell:
        iterator = walk_theta_at_shell(n, exp, max_steps, direction, shell=v2)
    else:
        iterator = walk_theta(n, exp, max_steps, direction)
    
    steps = 0
    for candidate in iterator:
        if candidate != n:  # Skip starting point
            steps += 1
            if is_prime(candidate):
                return (candidate, steps)
    
    return None


def find_angular_prime_ants(
    pos: int,
    exp: int,
    direction: str = 'cw',
    max_steps: int = 100000,
    same_shell: bool = False
) -> Optional[Tuple[int, int]]:
    """
    [NAVIGATION/PRIME] Alias: find nearest prime in angular direction.
    
    Returns (prime_position, steps_taken) or None.
    """
    return find_prime_neighbor(pos, exp, direction, max_steps, same_shell)


def bidirectional_prime_search(
    n: int,
    exp: int,
    max_distance: int = 10000,
    same_shell: bool = False
) -> dict:
    """
    [NAVIGATION/PRIME] Search for primes in both directions simultaneously.
    
    Finds the nearest primes both clockwise and counter-clockwise.
    
    Args:
        n: Center integer
        exp: Exponent
        max_distance: Maximum steps to search in each direction
        same_shell: If True, only search at same shell level
        
    Returns:
        Dictionary with primes found in both directions
    """
    core, v2 = get_odd_core(n)
    
    cw_result = None
    ccw_result = None
    
    # Search clockwise
    if same_shell:
        iterator_cw = walk_primes_at_shell(n, exp, count=1, direction='cw', shell=v2)
    else:
        iterator_cw = walk_primes(n, exp, count=1, direction='cw')
    
    for prime, steps in iterator_cw:
        if prime != n and steps <= max_distance:
            cw_result = (prime, steps)
        break
    
    # Search counter-clockwise
    if same_shell:
        iterator_ccw = walk_primes_at_shell(n, exp, count=1, direction='ccw', shell=v2)
    else:
        iterator_ccw = walk_primes(n, exp, count=1, direction='ccw')
    
    for prime, steps in iterator_ccw:
        if prime != n and steps <= max_distance:
            ccw_result = (prime, steps)
        break
    
    return {
        'n': n,
        'is_prime': is_prime(n),
        'cw_prime': cw_result[0] if cw_result else None,
        'cw_distance': cw_result[1] if cw_result else None,
        'ccw_prime': ccw_result[0] if ccw_result else None,
        'ccw_distance': ccw_result[1] if ccw_result else None,
        'nearest_prime': (cw_result if cw_result and 
                         (not ccw_result or cw_result[1] <= ccw_result[1]) 
                         else ccw_result),
    }


def find_primes_both_ways_ants(
    pos: int,
    exp: int,
    max_distance: int = 10000,
    same_shell: bool = False
) -> dict:
    """
    [NAVIGATION/PRIME] Alias: find nearest primes left and right.
    """
    return bidirectional_prime_search(pos, exp, max_distance, same_shell)


# =============================================================================
# [ANALYSIS] STATISTICAL AND DISTRIBUTION ANALYSIS - INTEGER-NATIVE
# =============================================================================
# All analysis uses integer counting and basic arithmetic.
# =============================================================================

def angular_relationship(n1: int, n2: int, exp: int) -> dict:
    """
    [ANALYSIS] Analyze the angular relationship between two integers.
    
    Args:
        n1, n2: Integers to compare
        exp: Exponent defining the space
        
    Returns:
        Dictionary with relationship metrics
    """
    core1, v2_1 = get_odd_core(n1)
    core2, v2_2 = get_odd_core(n2)
    
    key1 = odd_core_to_key(core1, exp)
    key2 = odd_core_to_key(core2, exp)
    
    num_keys = 1 << (exp - 1)
    
    # Angular distance (in keys)
    diff_cw = (key2 - key1) % num_keys
    diff_ccw = (key1 - key2) % num_keys
    
    return {
        'n1': n1,
        'n2': n2,
        'core1': core1,
        'core2': core2,
        'v2_1': v2_1,
        'v2_2': v2_2,
        'key1': key1,
        'key2': key2,
        'same_ray': core1 == core2,
        'angular_distance_cw': diff_cw,
        'angular_distance_ccw': diff_ccw,
        'min_angular_distance': min(diff_cw, diff_ccw),
        'shell_difference': abs(v2_1 - v2_2),
        'hamming_distance': hamming_distance(n1, n2),
    }


def angular_relationship_ants(pos1: int, pos2: int, exp: int) -> dict:
    """
    [ANALYSIS] Alias: how are two positions related?
    """
    return angular_relationship(pos1, pos2, exp)


def analyze_ray_primes(odd_core: int, exp: int) -> dict:
    """
    [ANALYSIS] Analyze primes on a single ray.
    
    Args:
        odd_core: The odd core defining the ray
        exp: Exponent
        
    Returns:
        Analysis dictionary
    """
    members = ray_members(odd_core, exp)
    primes = [n for n in members if is_prime(n)]
    
    return {
        'odd_core': odd_core,
        'core_is_prime': is_prime(odd_core),
        'ray_length': len(members),
        'prime_count': len(primes),
        'prime_density': len(primes) / len(members) if members else 0,
        'primes': primes,
        'members': members,
    }


def analyze_ray_ants(ray: int, exp: int) -> dict:
    """
    [ANALYSIS] Alias: analyze a single ray.
    """
    return analyze_ray_primes(ray, exp)


def prime_walk_statistics(
    start_n: int,
    exp: int,
    num_primes: int,
    direction: str = 'cw',
    shell: Optional[int] = None
) -> dict:
    """
    [ANALYSIS] Gather comprehensive statistics about a prime walk.
    
    Args:
        start_n: Starting integer
        exp: Exponent
        num_primes: Number of primes to collect
        direction: Direction to walk
        shell: Optional shell restriction
        
    Returns:
        Dictionary with comprehensive statistics
    """
    primes = []
    gaps = []
    
    if shell is not None:
        for prime, gap in walk_primes_at_shell(start_n, exp, num_primes, direction, shell):
            primes.append(prime)
            if gap > 0:  # Skip first prime (gap=0)
                gaps.append(gap)
    else:
        for prime, gap in walk_primes(start_n, exp, num_primes, direction):
            primes.append(prime)
            if gap > 0:
                gaps.append(gap)
    
    if not gaps:
        return {'error': 'Not enough primes found'}
    
    # Compute statistics (integer-safe)
    avg_gap = sum(gaps) / len(gaps)
    min_gap = min(gaps)
    max_gap = max(gaps)
    
    # Gap distribution
    gap_dist = {}
    for g in gaps:
        gap_dist[g] = gap_dist.get(g, 0) + 1
    
    # Median gap
    sorted_gaps = sorted(gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2]
    
    # Variance and std dev
    variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
    std_dev = variance ** 0.5
    
    return {
        'num_primes': len(primes),
        'primes': primes,
        'gaps': gaps,
        'min_gap': min_gap,
        'max_gap': max_gap,
        'avg_gap': avg_gap,
        'median_gap': median_gap,
        'std_dev': std_dev,
        'variance': variance,
        'gap_distribution': dict(sorted(gap_dist.items())),
        'start_prime': primes[0] if primes else None,
        'end_prime': primes[-1] if primes else None,
    }


def prime_theta_gap_analysis(
    start_n: int,
    exp: int,
    num_primes: int = 100,
    direction: str = 'cw'
) -> dict:
    """
    [ANALYSIS] Analyze angular gaps (delta_theta_key) between consecutive primes.
    
    NOTE: "gap" here means ANGULAR gap (position difference in theta order),
    NOT value difference. For full theta-order metrics including hamming 
    distances, use prime_theta_metrics() instead.
    
    Args:
        start_n: Starting integer
        exp: Exponent
        num_primes: Number of primes to find
        direction: Direction to walk
        
    Returns:
        Angular gap analysis dictionary
    """
    primes_found = []
    positions = []
    
    # Walk until we find enough primes
    max_walk = num_primes * 1000  # Reasonable upper bound
    
    for i, n in enumerate(walk_theta(start_n, exp, max_walk, direction)):
        if is_prime(n):
            primes_found.append(n)
            positions.append(i)
            if len(primes_found) >= num_primes:
                break
    
    if len(primes_found) < 2:
        return {'error': 'Not enough primes found'}
    
    # Compute gaps (in position, not value)
    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    
    return {
        'num_primes': len(primes_found),
        'primes': primes_found,
        'positions': positions,
        'delta_theta_keys': gaps,  # Renamed from 'gaps' for clarity
        'min_delta_theta_key': min(gaps),
        'max_delta_theta_key': max(gaps),
        'avg_delta_theta_key': sum(gaps) / len(gaps),
        'total_positions_checked': positions[-1] + 1,
    }


# Keep old name as alias for backward compatibility
def prime_gap_analysis(start_n: int, exp: int, num_primes: int = 100, direction: str = 'cw') -> dict:
    """[ANALYSIS] Deprecated alias for prime_theta_gap_analysis()."""
    return prime_theta_gap_analysis(start_n, exp, num_primes, direction)


def prime_theta_metrics(
    start_n: int,
    exp: int,
    num_primes: int = 100,
    direction: str = 'cw'
) -> dict:
    """
    [ANALYSIS] Compute CUDA-compatible theta-order metrics for consecutive primes.
    
    This analyzes primes sorted by THETA_KEY (angular position), not by value.
    All metrics are O(1) bitwise operations suitable for GPU implementation.
    
    METRICS COMPUTED:
      delta_theta_key      - Angular gap to next prime (key2 - key1)
      hamming_primes - Bit difference between prime values: popcount(p1 ^ p2)
      hamming_keys   - Bit difference between keys: popcount(key1 ^ key2)
    
    NOT COMPUTED (invalid for theta-order):
      delta_prime    - Value difference (p2 - p1) is meaningless here
    
    Args:
        start_n: Starting position
        exp: Exponent defining range [1, 2^exp)
        num_primes: Number of primes to analyze
        direction: 'cw' (clockwise) or 'ccw'
        
    Returns:
        Dictionary with metrics and statistics
    """
    num_keys = 1 << (exp - 1)
    key_bits = exp - 1
    
    # Collect primes with their keys
    primes_data = []  # list of (prime, key, position)
    
    for pos, n in enumerate(walk_theta(start_n, exp, num_primes * 1000, direction)):
        if is_prime(n):
            core, _ = get_odd_core(n)
            key = odd_core_to_key(core, exp)
            primes_data.append((n, key, pos))
            if len(primes_data) >= num_primes:
                break
    
    if len(primes_data) < 2:
        return {'error': 'Not enough primes found'}
    
    # Compute metrics for consecutive pairs
    delta_theta_keys = []
    hamming_primes_list = []
    hamming_keys_list = []
    
    for i in range(len(primes_data) - 1):
        p1, key1, _ = primes_data[i]
        p2, key2, _ = primes_data[i + 1]
        
        # Delta key (angular gap)
        if direction == 'cw':
            dk = (key2 - key1) % num_keys
        else:
            dk = (key1 - key2) % num_keys
        delta_theta_keys.append(dk)
        
        # Hamming distances
        hamming_primes_list.append(hamming_int(p1, p2))
        hamming_keys_list.append(hamming_int(key1, key2))
    
    # Statistics helper
    def stats(lst):
        if not lst:
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        sorted_lst = sorted(lst)
        return {
            'min': min(lst),
            'max': max(lst),
            'avg': sum(lst) / len(lst),
            'median': sorted_lst[len(sorted_lst) // 2]
        }
    
    return {
        'num_primes': len(primes_data),
        'exp': exp,
        'num_keys': num_keys,
        'key_bits': key_bits,
        
        # Raw data (first 20 for inspection)
        'primes': [p for p, _, _ in primes_data[:20]],
        'keys': [k for _, k, _ in primes_data[:20]],
        'delta_theta_keys': delta_theta_keys[:20],
        'hamming_primes': hamming_primes_list[:20],
        'hamming_keys': hamming_keys_list[:20],
        
        # Full data for analysis
        'all_delta_theta_keys': delta_theta_keys,
        'all_hamming_primes': hamming_primes_list,
        'all_hamming_keys': hamming_keys_list,
        
        # Statistics
        'delta_theta_key_stats': stats(delta_theta_keys),
        'hamming_primes_stats': stats(hamming_primes_list),
        'hamming_keys_stats': stats(hamming_keys_list),
        
        # Expected values for comparison
        'expected_delta_theta_key': num_keys / len(primes_data),  # If primes were uniform
    }


def prime_theta_metrics_ants(
    start_pos: int,
    exp: int,
    prime_count: int = 100,
    direction: str = 'cw'
) -> dict:
    """
    [ANALYSIS] Alias: compute theta-order metrics for primes.
    
    Returns metrics suitable for CUDA implementation:
    - delta_theta_key (angular gap)
    - hamming_primes (bit difference of values)
    - hamming_keys (bit difference of positions)
    """
    return prime_theta_metrics(start_pos, exp, prime_count, direction)


def find_prime_desert_angular(
    exp: int,
    min_gap: int,
    start_n: int = 1,
    direction: str = 'cw',
    max_search: int = 1000000
) -> Optional[dict]:
    """
    [ANALYSIS] Search for a "prime desert" - a region with no primes.
    
    A "desert" is a large angular gap (delta_theta_key) between consecutive primes.
    
    NOTE: We measure angular gap (delta_theta_key), NOT numeric difference.
    In theta order, adjacent primes may have vastly different values.
    
    Args:
        exp: Exponent
        min_gap: Minimum angular gap (delta_theta_key) to qualify as a desert
        start_n: Starting position
        direction: Search direction
        max_search: Maximum positions to check
        
    Returns:
        Desert information dict with theta-order metrics, or None if not found
    """
    last_prime = None
    last_prime_key = None
    last_prime_pos = 0
    
    for pos, n in enumerate(walk_theta(start_n, exp, max_search, direction)):
        if is_prime(n):
            core, _ = get_odd_core(n)
            current_key = odd_core_to_key(core, exp)
            
            if last_prime is not None:
                gap = pos - last_prime_pos
                if gap >= min_gap:
                    return {
                        'start_prime': last_prime,
                        'end_prime': n,
                        'start_key': last_prime_key,
                        'end_key': current_key,
                        'start_position': last_prime_pos,
                        'end_position': pos,
                        'delta_theta_key': gap,
                        'hamming_primes': hamming_int(last_prime, n),
                        'hamming_keys': hamming_int(last_prime_key, current_key),
                    }
            last_prime = n
            last_prime_key = current_key
            last_prime_pos = pos
    
    return None


def find_prime_desert_ants(
    exp: int,
    min_gap: int,
    start_pos: int = 1,
    direction: str = 'cw',
    max_search: int = 1000000
) -> Optional[dict]:
    """
    [ANALYSIS] Alias: find a large prime-free region.
    """
    return find_prime_desert_angular(exp, min_gap, start_pos, direction, max_search)


def analyze_prime_distribution(
    start_n: int,
    exp: int,
    num_rays: int,
    direction: str = 'cw'
) -> dict:
    """
    [ANALYSIS] Analyze prime distribution across multiple adjacent rays.
    
    Args:
        start_n: Starting integer (determines starting ray)
        exp: Exponent
        num_rays: Number of rays to analyze
        direction: Direction to walk
        
    Returns:
        Analysis dictionary with statistics
    """
    ray_analyses = []
    total_integers = 0
    total_primes = 0
    core_primes = 0
    
    for odd_core, members in walk_rays(start_n, exp, num_rays, direction):
        analysis = analyze_ray_primes(odd_core, exp)
        ray_analyses.append(analysis)
        
        total_integers += analysis['ray_length']
        total_primes += analysis['prime_count']
        if analysis['core_is_prime']:
            core_primes += 1
    
    # Find rays with most/least primes
    sorted_by_primes = sorted(ray_analyses, key=lambda x: x['prime_count'], reverse=True)
    
    return {
        'num_rays': num_rays,
        'total_integers': total_integers,
        'total_primes': total_primes,
        'overall_density': total_primes / total_integers if total_integers else 0,
        'core_primes': core_primes,
        'core_prime_density': core_primes / num_rays if num_rays else 0,
        'max_primes_ray': sorted_by_primes[0] if sorted_by_primes else None,
        'min_primes_ray': sorted_by_primes[-1] if sorted_by_primes else None,
        'ray_analyses': ray_analyses,
    }


def find_angular_gaps(
    exp: int,
    shell: int,
    threshold_percentile: float = 90.0
) -> List[Dict]:
    """
    [ANALYSIS] Find angular regions with sparse integer coverage at a specific shell.
    
    Args:
        exp: Exponent
        shell: Shell level to analyze
        threshold_percentile: Consider gaps above this percentile
        
    Returns:
        List of gap dictionaries with angular positions
    """
    # Collect all integers at this shell with their keys
    shell_integers = []
    max_n = 1 << exp
    num_keys = 1 << (exp - 1)
    
    for key in range(num_keys):
        core = key_to_odd_core(key, exp)
        n = core << shell
        if n < max_n:
            shell_integers.append((key, n, core))
    
    if len(shell_integers) < 2:
        return []
    
    # Sort by key (already in angular order)
    shell_integers.sort()
    
    # Compute angular gaps between consecutive rays
    gaps = []
    for i in range(len(shell_integers)):
        next_i = (i + 1) % len(shell_integers)
        key1 = shell_integers[i][0]
        key2 = shell_integers[next_i][0]
        
        # Angular distance
        if key2 > key1:
            gap = key2 - key1
        else:
            gap = (num_keys - key1) + key2
        
        gaps.append({
            'start_key': key1,
            'end_key': key2,
            'gap_size': gap,
            'start_n': shell_integers[i][1],
            'end_n': shell_integers[next_i][1],
        })
    
    # Find threshold
    gap_sizes = [g['gap_size'] for g in gaps]
    gap_sizes_sorted = sorted(gap_sizes)
    threshold_idx = int(len(gap_sizes_sorted) * threshold_percentile / 100)
    threshold = gap_sizes_sorted[threshold_idx] if threshold_idx < len(gap_sizes_sorted) else gap_sizes_sorted[-1]
    
    # Return gaps above threshold
    large_gaps = [g for g in gaps if g['gap_size'] >= threshold]
    large_gaps.sort(key=lambda x: x['gap_size'], reverse=True)
    
    return large_gaps


def prime_richness_comparison(
    n1: int,
    n2: int,
    exp: int,
    window_size: int = 100
) -> Dict:
    """
    [ANALYSIS] Compare prime richness in angular neighborhoods of two numbers.
    
    Args:
        n1, n2: Numbers to compare
        exp: Exponent
        window_size: Size of angular window around each
        
    Returns:
        Comparison dictionary
    """
    # Count primes around n1
    primes_n1 = 0
    for n in zoom_to_region(n1, exp, half_width_rays=window_size//2):
        if is_prime(n):
            primes_n1 += 1
    
    # Count primes around n2
    primes_n2 = 0
    for n in zoom_to_region(n2, exp, half_width_rays=window_size//2):
        if is_prime(n):
            primes_n2 += 1
    
    return {
        'n1': n1,
        'n2': n2,
        'window_size': window_size,
        'n1_prime_count': primes_n1,
        'n2_prime_count': primes_n2,
        'difference': primes_n2 - primes_n1,
        'ratio': primes_n2 / primes_n1 if primes_n1 > 0 else float('inf'),
        'richer_region': 'n1' if primes_n1 > primes_n2 else 'n2',
    }


def prime_density_by_ray(
    exp: int,
    sample_rays: int = 1000,
    seed: Optional[int] = None
) -> dict:
    """
    [ANALYSIS] Analyze prime distribution across rays comprehensively.
    
    Args:
        exp: Exponent
        sample_rays: Number of rays to sample
        seed: Random seed for reproducibility
        
    Returns:
        Comprehensive analysis dictionary with statistics
    """
    import random
    if seed is not None:
        random.seed(seed)
    
    max_key = (1 << (exp - 1)) - 1
    
    densities = []
    prime_counts = []
    ray_lengths = []
    core_prime_flags = []
    
    for _ in range(sample_rays):
        key = random.randint(0, max_key)
        odd_core = key_to_odd_core(key, exp)
        analysis = analyze_ray_primes(odd_core, exp)
        
        densities.append(analysis['prime_density'])
        prime_counts.append(analysis['prime_count'])
        ray_lengths.append(analysis['ray_length'])
        core_prime_flags.append(analysis['core_is_prime'])
    
    # Statistical measures
    avg_density = sum(densities) / len(densities)
    avg_prime_count = sum(prime_counts) / len(prime_counts)
    avg_ray_length = sum(ray_lengths) / len(ray_lengths)
    core_prime_ratio = sum(core_prime_flags) / len(core_prime_flags)
    
    # Find variance
    variance = sum((d - avg_density) ** 2 for d in densities) / len(densities)
    std_dev = variance ** 0.5
    
    return {
        'sample_size': sample_rays,
        'avg_density': avg_density,
        'std_deviation': std_dev,
        'variance': variance,
        'avg_prime_count': avg_prime_count,
        'avg_ray_length': avg_ray_length,
        'core_prime_ratio': core_prime_ratio,
    }


def angular_density_profile(
    exp: int,
    shell: int,
    num_buckets: int = 100
) -> Dict:
    """
    [ANALYSIS] Create an angular density profile showing integer distribution.
    
    Args:
        exp: Exponent
        shell: Shell level
        num_buckets: Number of angular buckets
        
    Returns:
        Density profile dictionary
    """
    num_keys = 1 << (exp - 1)
    bucket_size = num_keys // num_buckets
    
    buckets = [0] * num_buckets
    max_n = 1 << exp
    
    for key in range(num_keys):
        core = key_to_odd_core(key, exp)
        n = core << shell
        if n < max_n:
            bucket_idx = min(key // bucket_size, num_buckets - 1)
            buckets[bucket_idx] += 1
    
    return {
        'shell': shell,
        'num_buckets': num_buckets,
        'bucket_size': bucket_size,
        'buckets': buckets,
        'total_integers': sum(buckets),
        'max_bucket': max(buckets),
        'min_bucket': min(buckets),
    }


# =============================================================================
# [VISUALIZATION] HELPERS THAT USE TRIGONOMETRY
# =============================================================================
# WARNING: These functions use floating-point math and trigonometry.
# They are for HUMAN VISUALIZATION only, NOT for core ant algorithms.
# An ant cannot compute these functions!
# =============================================================================

def compute_angle_deg(n: int) -> float:
    """
    [VISUALIZATION] Compute angle for integer n in degrees.
    
    WARNING: Uses trigonometry (atan2) - for human display only, NOT integer-native!
    
    Args:
        n: Integer position
        
    Returns:
        Angle in degrees [0, 360)
    """
    core, _ = get_odd_core(n)
    
    if core == 0:
        return 0.0
    
    k = core.bit_length() - 1
    shell_size = 1 << k
    t = (core - shell_size) / shell_size if shell_size > 0 else 0
    
    # Map t to (x, y) on square
    if t < 0.25:
        x, y = 1 - 8*t, 1
    elif t < 0.5:
        x, y = -1, 1 - 8*(t - 0.25)
    elif t < 0.75:
        x, y = -1 + 8*(t - 0.5), -1
    else:
        x, y = 1, -1 + 8*(t - 0.75)
    
    # THIS IS THE TRIGONOMETRY - not allowed in core!
    angle = math.atan2(y, x)
    return math.degrees(angle) % 360


def compute_angle_deg_visualization(n: int) -> float:
    """
    [VISUALIZATION] Explicit alias marking this as visualization-only.
    
    WARNING: Uses trigonometry - NOT for ant algorithms!
    """
    return compute_angle_deg(n)


def core_to_xy_visualization(core: int) -> Tuple[float, float]:
    """
    [VISUALIZATION] Convert odd core to (x, y) coordinates on unit square.
    
    WARNING: Returns floating-point coordinates for human display only.
    """
    if core == 1:
        return (1.0, 1.0)
    
    k = core.bit_length() - 1
    shell_size = 1 << k
    t = (core - shell_size) / shell_size
    
    if t < 0.25:
        return (1 - 8*t, 1.0)
    elif t < 0.5:
        return (-1.0, 1 - 8*(t - 0.25))
    elif t < 0.75:
        return (-1 + 8*(t - 0.5), -1.0)
    else:
        return (1.0, -1 + 8*(t - 0.75))


# =============================================================================
# [CLI] COMMAND-LINE INTERFACE AND DEMO
# =============================================================================

def demo():
    """Run demonstration of all operations."""
    exp = 24
    
    print("=" * 70)
    print("INTEGER-NATIVE COMPUTATION: Square Ray Operations Toolkit")
    print("=" * 70)
    print(f"\nGrid: exp={exp} ({1 << (exp-1):,} odd-core rays)")
    print("\nAll CORE operations use NO TRIGONOMETRY - only:")
    print("  - Integer comparisons")
    print("  - Rational arithmetic (Fraction)")
    print("  - Bitwise operations")
    print("\n[VISUALIZATION] helpers use trig for human display only.")
    
    # Demo 1: Generation
    print("\n" + "-" * 70)
    print("1. THETA-SORTED GENERATION (first 15) [CORE]")
    print("-" * 70)
    
    for i, n in enumerate(generate_theta_sorted(exp)):
        if i >= 15:
            break
        core, v2 = get_odd_core(n)
        angle = compute_angle_deg(n)  # [VISUALIZATION] only
        print(f"  {i:3d}: n={n:>10,}  ray={core:>10,}  shell={v2}  angle={angle:.4f}deg")
    
    # Demo 2: Neighbor lookup
    print("\n" + "-" * 70)
    print("2. O(1) THETA-NEIGHBOR LOOKUP [CORE]")
    print("-" * 70)
    
    for test_n in [12345, 1000000, 7654321]:
        prev_n, next_n = theta_neighbors(test_n, exp)
        core, v2 = get_odd_core(test_n)
        
        print(f"\n  n = {test_n:,} (ray={core:,}, shell={v2})")
        print(f"    [VIZ] angle = {compute_angle_deg(test_n):.6f}deg")
        if prev_n:
            print(f"    Prev (CCW): {prev_n:,}")
        if next_n:
            print(f"    Next (CW): {next_n:,}")
    
    # Demo 3: Direction bracketing
    print("\n" + "-" * 70)
    print("3. DIRECTION-TO-RAY BRACKETING [CORE - No Trig!]")
    print("-" * 70)
    
    directions = [
        (1, 1, "northeast"),
        (1, 2, "~63deg"),
        (3, 4, "~53deg"),
        (-1, 1, "northwest"),
        (17, 31, "~61deg"),
    ]
    
    for dx, dy, expected in directions:
        core_lo, core_hi = find_bracketing_rays(dx, dy, exp)
        t = direction_to_t(dx, dy)
        
        print(f"\n  Direction ({dx}, {dy}) ~ {expected}")
        print(f"    Perimeter t = {float(t):.6f} (exact: {t})")
        print(f"    Bracketing rays: [{core_lo:,}, {core_hi:,}]")
    
    # Demo 4: Angular relationship
    print("\n" + "-" * 70)
    print("4. ANGULAR RELATIONSHIP ANALYSIS [CORE]")
    print("-" * 70)
    
    n1, n2 = 1000, 2000
    rel = angular_relationship(n1, n2, exp)
    print(f"\n  Comparing {n1} and {n2}:")
    print(f"    Same ray: {rel['same_ray']}")
    print(f"    Angular distance (CW): {rel['angular_distance_cw']}")
    print(f"    Hamming distance: {rel['hamming_distance']}")
    print(f"    Shell difference: {rel['shell_difference']}")
    
    # Demo 5: Prime analysis
    print("\n" + "-" * 70)
    print("5. PRIME RAY ANALYSIS [CORE/PRIME]")
    print("-" * 70)
    
    print(f"\n  Analyzing prime distribution across 100 random rays...")
    analysis = prime_density_by_ray(exp, sample_rays=100, seed=42)
    print(f"    Average prime density: {analysis['avg_density']:.4f}")
    print(f"    Std deviation: {analysis['std_deviation']:.4f}")
    print(f"    Core prime ratio: {analysis['core_prime_ratio']:.4f}")
    
    # Demo 6: Convenience aliases (_ants suffix - legacy naming)
    print("\n" + "-" * 70)
    print("6. CONVENIENCE ALIASES (_ants suffix)")
    print("-" * 70)
    
    test_pos = 12345
    ray, shell = get_odd_core_ants(test_pos)
    key = odd_core_to_key_ants(ray, exp)
    
    print(f"\n  Position {test_pos}:")
    print(f"    ray_label_ants() = {ray_label_ants(test_pos)}")
    print(f"    v2_ants() = {v2_ants(test_pos)}")
    print(f"    get_odd_core_ants() = (ray={ray}, shell={shell})")
    print(f"    odd_core_to_key_ants(ray) = {key}")
    
    # Walking
    print(f"\n  First 5 positions from walk_theta_ants({test_pos}, ...):")
    for i, n in enumerate(walk_theta_ants(test_pos, exp, 5, 'cw')):
        print(f"    {i}: {n}")
    
    # Demo 7: Theta-order metrics (NEW)
    print("\n" + "-" * 70)
    print("7. THETA-ORDER PRIME METRICS [ANALYSIS] - CUDA COMPATIBLE")
    print("-" * 70)
    
    print(f"\n  Computing metrics for 20 primes starting from 1...")
    metrics = prime_theta_metrics(1, exp, num_primes=20)
    
    print(f"\n  VALID metrics (use these for theta-order analysis):")
    print(f"    delta_theta_key stats:      min={metrics['delta_theta_key_stats']['min']}, "
          f"max={metrics['delta_theta_key_stats']['max']}, "
          f"avg={metrics['delta_theta_key_stats']['avg']:.1f}")
    print(f"    hamming_primes stats: min={metrics['hamming_primes_stats']['min']}, "
          f"max={metrics['hamming_primes_stats']['max']}, "
          f"avg={metrics['hamming_primes_stats']['avg']:.1f}")
    print(f"    hamming_keys stats:   min={metrics['hamming_keys_stats']['min']}, "
          f"max={metrics['hamming_keys_stats']['max']}, "
          f"avg={metrics['hamming_keys_stats']['avg']:.1f}")
    
    print(f"\n  INVALID metrics (NOT computed - meaningless in theta order):")
    print(f"    delta_prime (p2 - p1): Adjacent primes may be millions apart")
    print(f"    numeric_gap: Same problem - comparing values not positions")
    
    print("\n" + "=" * 70)
    print("Demo complete. All [CORE] operations used integers only.")
    print("=" * 70)


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integer-Native Computation: Square Ray Operations Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python square_ants_ops.py --demo
  python square_ants_ops.py --neighbors 12345 --exp 24
  python square_ants_ops.py --walk 1000 --exp 20 --count 50
  python square_ants_ops.py --walk-primes 1 --exp 24 --prime-count 100
  python square_ants_ops.py --theta-metrics 1 --prime-count 100 --exp 24
        """
    )
    
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--exp', type=int, default=24, help='Exponent (default: 24)')
    parser.add_argument('--neighbors', type=int, help='Find theta-neighbors of N')
    parser.add_argument('--walk', type=int, help='Walk theta from N')
    parser.add_argument('--walk-primes', type=int, help='Walk through primes from N')
    parser.add_argument('--count', type=int, default=20, help='Number of items (default: 20)')
    parser.add_argument('--prime-count', type=int, default=50, help='Number of primes (default: 50)')
    parser.add_argument('--direction', choices=['cw', 'ccw'], default='cw', help='Direction')
    parser.add_argument('--shell', type=int, help='Restrict to specific shell')
    parser.add_argument('--analyze-ray', type=int, help='Analyze primes on ray with odd core N')
    parser.add_argument('--bracket', nargs=2, type=int, metavar=('DX', 'DY'),
                        help='Find rays bracketing direction (DX, DY)')
    parser.add_argument('--theta-metrics', type=int, metavar='N',
                        help='Compute CUDA-compatible theta-order metrics for primes starting from N')
    
    args = parser.parse_args()
    
    if args.demo or len(sys.argv) == 1:
        demo()
    elif args.neighbors is not None:
        n = args.neighbors
        exp = args.exp
        prev_n, next_n = theta_neighbors(n, exp)
        core, v2 = get_odd_core(n)
        
        print(f"n = {n:,}")
        print(f"  ray = {core:,}")
        print(f"  shell = {v2}")
        print(f"  key = {odd_core_to_key(core, exp)}")
        print(f"  [VIZ] angle = {compute_angle_deg(n):.6f}deg")
        print(f"\nNeighbors at shell {v2}:")
        print(f"  CCW: {prev_n:,}" if prev_n else "  CCW: None (out of range)")
        print(f"  CW:  {next_n:,}" if next_n else "  CW:  None (out of range)")
    elif args.walk is not None:
        print(f"Walking theta from {args.walk} (exp={args.exp}, {args.direction}):")
        for i, n in enumerate(walk_theta(args.walk, args.exp, args.count, args.direction)):
            core, v2 = get_odd_core(n)
            print(f"  {i:3d}: {n:>12,}  ray={core:>10,}  shell={v2}")
    elif args.walk_primes is not None:
        print(f"Walking primes from {args.walk_primes} (exp={args.exp}, {args.direction}):")
        print(f"\n{'#':>4} {'Prime':>14} {'Gap':>8}")
        print("-" * 30)
        for i, (prime, gap) in enumerate(walk_primes(args.walk_primes, args.exp, 
                                                       args.prime_count, args.direction)):
            print(f"{i+1:>4} {prime:>14,} {gap:>8}")
    elif args.analyze_ray is not None:
        analysis = analyze_ray_primes(args.analyze_ray, args.exp)
        print(f"Ray analysis for odd core {args.analyze_ray}:")
        print(f"  Core is prime: {analysis['core_is_prime']}")
        print(f"  Ray length: {analysis['ray_length']}")
        print(f"  Prime count: {analysis['prime_count']}")
        print(f"  Prime density: {analysis['prime_density']:.4f}")
        print(f"  Primes: {analysis['primes']}")
    elif args.bracket is not None:
        dx, dy = args.bracket
        core_lo, core_hi = find_bracketing_rays(dx, dy, args.exp)
        t = direction_to_t(dx, dy)
        print(f"Direction ({dx}, {dy}):")
        print(f"  Perimeter t = {float(t):.6f}")
        print(f"  Bracketing rays: [{core_lo:,}, {core_hi:,}]")
        print(f"  [VIZ] Angles: [{compute_angle_deg(core_lo):.4f}deg, {compute_angle_deg(core_hi):.4f}deg]")
    elif args.theta_metrics is not None:
        metrics = prime_theta_metrics(args.theta_metrics, args.exp, args.prime_count, args.direction)
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
        else:
            print(f"\n{'='*60}")
            print(f"THETA-ORDER PRIME METRICS")
            print(f"{'='*60}")
            print(f"exp={args.exp}, {metrics['num_primes']} primes, direction={args.direction}")
            print(f"Key space: {metrics['num_keys']:,} keys ({metrics['key_bits']} bits)")
            
            print(f"\nDelta Key (angular gap) statistics:")
            s = metrics['delta_theta_key_stats']
            print(f"  Min: {s['min']:,}  Max: {s['max']:,}  Avg: {s['avg']:.2f}  Median: {s['median']}")
            print(f"  Expected (uniform): {metrics['expected_delta_theta_key']:.2f}")
            
            print(f"\nHamming(primes) - bit difference between prime values:")
            s = metrics['hamming_primes_stats']
            print(f"  Min: {s['min']}  Max: {s['max']}  Avg: {s['avg']:.2f}  Median: {s['median']}")
            
            print(f"\nHamming(keys) - bit difference between angular positions:")
            s = metrics['hamming_keys_stats']
            print(f"  Min: {s['min']}  Max: {s['max']}  Avg: {s['avg']:.2f}  Median: {s['median']}")
            
            print(f"\nFirst 10 primes in theta order:")
            print(f"{'Prime':>12} {'Key':>10} {'ΔKey':>8} {'H(p)':>6} {'H(k)':>6}")
            print("-" * 50)
            for i in range(min(10, len(metrics['primes']))):
                p = metrics['primes'][i]
                k = metrics['keys'][i]
                dk = metrics['delta_theta_keys'][i] if i < len(metrics['delta_theta_keys']) else '-'
                hp = metrics['hamming_primes'][i] if i < len(metrics['hamming_primes']) else '-'
                hk = metrics['hamming_keys'][i] if i < len(metrics['hamming_keys']) else '-'
                dk_str = f"{dk:>8}" if isinstance(dk, int) else f"{dk:>8}"
                hp_str = f"{hp:>6}" if isinstance(hp, int) else f"{hp:>6}"
                hk_str = f"{hk:>6}" if isinstance(hk, int) else f"{hk:>6}"
                print(f"{p:>12,} {k:>10,} {dk_str} {hp_str} {hk_str}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
