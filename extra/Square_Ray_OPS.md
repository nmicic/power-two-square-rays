# Square Ray Operations - Unified Documentation

**Version 3.0** — Unified Edition covering both modules

---

## Disclaimer

> **Disclaimer**  
>This is an AI-assisted exploratory visualization project.
>It is educational and experimental only—not peer-reviewed, not mathematical research, and not a new theorem.
>It does not serve as evidence for or against conjectures.
>This repository is math-art/visualization, not a research paper.
>Its purpose is educational exploration and aesthetic interest.
>Any visual patterns are artifacts of the construction.

---

## Table of Contents

1. [Overview](#overview)
2. [Which Module to Use](#which-module-to-use)
3. [ROBO-ANT WORLD META-SPEC](#robo-ant-world-meta-spec)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Core Concepts](#core-concepts)
7. [API Reference — Shared Core](#api-reference--shared-core)
8. [API Reference — Full Toolkit (square_rays_ops.py)](#api-reference--full-toolkit-square_rays_opspy)
9. [API Reference — Ant Perspective (square_ants_ops.py)](#api-reference--ant-perspective-square_ants_opspy)
10. [Command Line Usage](#command-line-usage)
11. [Use Cases and Examples](#use-cases-and-examples)
12. [Performance](#performance)
13. [Mathematical Background](#mathematical-background)

---

## Overview

This toolkit provides efficient algorithms for navigating the Power-of-Two Square Rays construction. All core operations use **NO TRIGONOMETRY** — only integer comparisons, rational arithmetic, and bitwise operations.

The project provides **two modules** with different focus areas:

| Module | Purpose | Functions | Focus |
|--------|---------|-----------|-------|
| `square_rays_ops.py` | Full research toolkit | 72 | Complete feature set, exports, benchmarks |
| `square_ants_ops.py` | Educational edition | 70 | Clean philosophy, streamlined, ant perspective |

### Key Features (Both Modules)

**Core Operations:**
- Theta-sorted generation (O(1) per element, no sorting required)
- O(1) theta-neighbor lookup
- Angle-to-ray bracketing without trigonometry
- Key to odd core conversion (pure bitwise operations)

**Prime Walking:**
- Walk through ONLY primes in theta order (skip composites)
- Shell-specific prime walking
- Bidirectional prime search
- Ray-specific prime operations

### Additional Features (square_rays_ops.py only)

- Binary pattern search and analysis
- Export to binary/CSV formats
- Shell statistics and comparison
- Prime gap analysis and desert finding
- Ray arithmetic operations
- Comprehensive benchmarking

### Ant-Perspective Features (square_ants_ops.py only)

- 22 `_ants` suffix functions with ant-centric naming
- Clear `[SECTION]` tags on all functions
- Explicit separation of `[CORE]` vs `[VISUALIZATION]`
- Docstrings written from the robo-ant perspective

---

## Which Module to Use

**Use `square_rays_ops.py` when:**
- You need the complete feature set
- Exporting data to CSV/binary
- Running benchmarks
- Advanced prime analysis (gap analysis, deserts, sequences)
- Binary pattern matching
- Shell statistics and comparison

**Use `square_ants_ops.py` when:**
- Teaching or learning the construction
- You want clean philosophical grounding
- Building on the ant-perspective model
- You need explicit core vs visualization separation
- You prefer streamlined code without extra features

---

## ROBO-ANT WORLD META-SPEC

This section defines the philosophical framework used in `square_ants_ops.py`.

### Perspective

We work in the "robo-ant" model:

- The world is a discrete grid (2D square by default)
- Every state is labeled by a positive integer n
- Ants are born at a common origin ("home") and can only move one unit in the four cardinal directions: UP, LEFT, DOWN, RIGHT
- Ants do NOT know trigonometry, calculus, or floating-point geometry — they only know integers, bitwise operations, and simple comparisons

### Power-of-Two Square Shells

For a fixed exponent `exp`, we consider integers n in [1, 2^exp).

Each n is decomposed into its 2-adic form:

```
n = core * 2^v2
```

Where:
- `v2 = v2(n)` is the 2-adic valuation: the largest k such that 2^k divides n. This is the "shell index" — which power-of-two square the ant stands on.
- `core` is the odd factor of n (the "odd core"). This labels the "ray" — all numbers with the same odd core lie on the same straight-looking grid line across shells.

We define a bijection:

```
core  <->  key in [0, 2^(exp-1) - 1]
```

The `key` is a combinatorial perimeter coordinate on the square: stepping key → key+1 means walking one step clockwise around the boundary.

### Algorithmic Constraints

All *core* algorithms must obey:

1. **NO trigonometry or floating-point** in core logic
   - No sin, cos, tan, atan2, floats for placement, ordering, or neighbor search
   - Trig allowed ONLY in visualization helpers marked with `[VISUALIZATION]`

2. **Shells and rays defined purely from integer n**
   - `v2(n)` = shell index
   - `core(n)` = ray label (odd core)
   - `key(core)` = angular ordering along square perimeter

3. **O(1) operations per step** in terms of exp

4. **Streaming-friendly memory usage**
   - No huge precomputed angle tables
   - Conceptually fits in an ant's tiny memory

### Evaluation Style

When assessing results, significance, or novelty:
- No fanfare, no hype, no grandiose claims
- Realistic, grounded evaluation only
- "Interesting to look at" is NOT "mathematically significant"
- If something is just a tautology of the construction, say so

### Module Organization Tags

Functions in `square_ants_ops.py` are marked with section tags:

| Tag | Description |
|-----|-------------|
| `[CORE]` | Integer-only operations, ant-safe |
| `[CORE/PRIME]` | Prime-related core operations |
| `[NAVIGATION]` | Walking/traversal operations |
| `[NAVIGATION/PRIME]` | Prime-filtered walking |
| `[ANALYSIS]` | Statistical and distribution analysis |
| `[VISUALIZATION]` | Uses trig/floats — human display only |
| `[CLI]` | Command-line interface |

---

## Installation

No external dependencies required — uses only Python standard library.

```bash
# Clone the repository
git clone https://github.com/nmicic/power-two-square-rays.git
cd power-two-square-rays

# Run directly
python square_rays_ops.py      # Full toolkit
python square_ants_ops.py      # Educational edition

# Or import as library
from square_rays_ops import walk_primes, theta_neighbors, generate_theta_sorted
from square_ants_ops import walk_primes_ants, theta_neighbors_ants
```

**Requirements:** Python 3.6 or later

---

## Quick Start

### Command Line Examples

```bash
# Run demonstration
python square_rays_ops.py --demo
python square_ants_ops.py --demo

# Walk through 100 primes
python square_rays_ops.py --walk-primes 1 --prime-count 100 --exp 24

# Find nearest primes to a number
python square_rays_ops.py --bidirectional-prime 1000000 --exp 24

# Get prime statistics
python square_rays_ops.py --walk-primes 1 --prime-count 1000 --prime-stats --exp 24

# Export data to CSV (square_rays_ops.py only)
python square_rays_ops.py --walk 1 --count 10000 --export-csv output.csv --exp 24
```

### Python Library Examples

```python
from square_rays_ops import (
    walk_primes, theta_neighbors, generate_theta_sorted,
    find_bracketing_rays, prime_walk_statistics
)

# Walk through primes only
for prime, gap in walk_primes(1, exp=24, count=100):
    print(f"Prime: {prime}, gap: {gap}")

# Find neighbors
prev_n, next_n = theta_neighbors(12345, exp=24)

# Generate in theta order
for n in generate_theta_sorted(exp=20):
    process(n)

# Get prime statistics
stats = prime_walk_statistics(1, exp=24, num_primes=1000)
print(f"Average gap: {stats['avg_gap']:.2f}")
```

### Ant-Perspective Examples

```python
from square_ants_ops import (
    get_odd_core_ants, walk_theta_ants, ray_label_ants, v2_ants
)

# Decompose position into (ray, shell)
position = 12345
ray, shell = get_odd_core_ants(position)
print(f"Position {position}: ray={ray}, shell={shell}")

# Alternative single-value queries
print(f"Ray: {ray_label_ants(position)}")
print(f"Shell: {v2_ants(position)}")

# Walk clockwise from a position
for n in walk_theta_ants(1000, exp=24, count=20, direction='cw'):
    print(n)
```

---

## Core Concepts

### The Construction

Every positive integer n can be written as:

```
n = 2^v2 * a
```

Where:
- `a` is odd (the "odd core")
- `v2` is the 2-adic valuation (trailing zeros in binary)

This creates a natural angular ordering of integers.

### Rays and Shells

**Ray:** All integers with the same odd core (e.g., 3, 6, 12, 24, 48...)

**Shell:** All integers with the same v2 (same distance from origin)

**Theta-key:** Normalized representation that determines angular position

### Example

```
Integer: 12
Binary: 1100
Factorization: 12 = 3 × 2²
Odd core: 3
v2 (shell): 2
Theta-key: odd_core_to_key(3, exp)
```

---

## API Reference — Shared Core

These functions exist in **both modules** with identical behavior.

### Core Bitwise Operations

#### `get_odd_core(n: int) -> Tuple[int, int]`

Extract odd core and 2-adic valuation.

```python
core, v2 = get_odd_core(12)  # Returns (3, 2)
core, v2 = get_odd_core(7)   # Returns (7, 0)
```

#### `odd_core_to_key(a: int, exp: int) -> int`

Convert odd core to theta-key.

```python
key = odd_core_to_key(3, exp=12)  # Returns 1024
key = odd_core_to_key(1, exp=12)  # Returns 0
```

#### `key_to_odd_core(k: int, exp: int) -> int`

Convert theta-key back to odd core.

```python
core = key_to_odd_core(0, exp=12)     # Returns 1
core = key_to_odd_core(1024, exp=12)  # Returns 3
```

### Generation

#### `generate_theta_sorted(exp: int) -> Iterator[int]`

Generate all integers 1 to 2^exp - 1 in theta-sorted order. No sorting required.

```python
for n in generate_theta_sorted(exp=16):
    process(n)

# First few: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, ...
```

#### `generate_theta_sorted_with_info(exp: int) -> Iterator[Tuple]`

Generate with full metadata.

```python
for n, odd_core, v2, theta_key in generate_theta_sorted_with_info(exp=16):
    print(f"n={n}, core={odd_core}, shell={v2}, key={theta_key}")
```

### Neighbor Operations

#### `theta_neighbors(n: int, exp: int) -> Tuple[Optional[int], Optional[int]]`

Find integers on adjacent rays at same shell level. O(1) complexity.

```python
prev_n, next_n = theta_neighbors(12345, exp=24)
# Returns integers on neighboring rays at same distance from origin
```

### Direction Operations

#### `direction_to_t(dx: int, dy: int) -> Fraction`

Convert direction vector to perimeter parameter t in [0, 1). Pure rational arithmetic.

```python
from fractions import Fraction
t = direction_to_t(1, 1)    # 45 degrees -> Fraction(0, 1)
t = direction_to_t(0, 1)    # 90 degrees -> Fraction(1, 8)
```

#### `find_bracketing_rays(dx: int, dy: int, exp: int) -> Tuple[int, int]`

Find two odd cores that bracket the direction (dx, dy). No trigonometry.

```python
core_lo, core_hi = find_bracketing_rays(3, 4, exp=24)
# Finds rays that surround the angle of direction (3, 4)
```

### Walk Operations

#### `walk_theta(start_n, exp, count, direction='cw', shell_min=None, shell_max=None)`

Walk through theta-sorted integers from a starting point.

```python
for n in walk_theta(1000, exp=24, count=100, direction='cw'):
    process(n)
```

Parameters:
- `start_n`: Starting integer
- `exp`: Exponent
- `count`: Number of integers to yield
- `direction`: 'cw' (clockwise) or 'ccw' (counter-clockwise)
- `shell_min`, `shell_max`: Optional shell bounds

#### `walk_theta_at_shell(start_n, exp, count, direction='cw', shell=None)`

Walk at specific shell level only.

```python
for n in walk_theta_at_shell(1000, exp=24, count=50, shell=10):
    process(n)
```

#### `zoom_to_region(center_n, exp, half_width_rays, shell_min=None, shell_max=None)`

Zoom into angular region around a number.

```python
for n in zoom_to_region(12345678, exp=32, half_width_rays=50):
    process(n)
```

### Prime Walking

#### `walk_primes(start_n, exp, count, direction='cw', shell_min=None, shell_max=None)`

Walk through ONLY primes in theta order.

```python
for prime, gap in walk_primes(1, exp=32, count=1000):
    print(f"Prime: {prime}, gap from previous: {gap} steps")
```

Returns: Iterator of (prime, steps_from_last_prime)

#### `walk_primes_at_shell(start_n, exp, count, direction='cw', shell=None)`

Walk through only primes at specific shell level.

```python
for prime, angular_gap in walk_primes_at_shell(1, exp=24, count=100, shell=15):
    process(prime)
```

#### `bidirectional_prime_search(n, exp, max_distance=10000, same_shell=False)`

Find nearest primes in both directions.

```python
result = bidirectional_prime_search(1000000, exp=24)
print(f"Nearest CW: {result['cw_prime']}")
print(f"Nearest CCW: {result['ccw_prime']}")
```

### Prime Testing

#### `is_prime(n: int) -> bool`

Miller-Rabin primality test. Deterministic for n < 3,317,044,064,679,887,385,961,981.

```python
is_prime(17)       # True
is_prime(1000003)  # True
is_prime(1000000)  # False
```

### Analysis

#### `angular_relationship(n1: int, n2: int, exp: int) -> Dict`

Analyze angular relationship between two integers.

```python
rel = angular_relationship(1000, 2000, exp=24)
print(f"Same ray: {rel['same_ray']}")
print(f"Angular distance: {rel['angular_distance_cw']}")
```

#### `analyze_ray_primes(odd_core: int, exp: int) -> Dict`

Analyze prime distribution along a single ray.

```python
analysis = analyze_ray_primes(3, exp=24)
print(f"Primes on ray 3: {analysis['primes']}")
```

#### `prime_walk_statistics(start_n, exp, num_primes, direction='cw', shell=None)`

Compute statistics on prime gaps.

```python
stats = prime_walk_statistics(1, exp=24, num_primes=1000)
print(f"Average gap: {stats['avg_gap']:.2f}")
print(f"Max gap: {stats['max_gap']}")
```

---

## API Reference — Full Toolkit (square_rays_ops.py)

These functions are **only available in `square_rays_ops.py`**.

### Binary Pattern Analysis

#### `find_by_binary_pattern(pattern: str, exp: int, max_results=1000) -> List[int]`

Find integers matching a binary pattern. Use `?` for wildcard bits.

```python
matches = find_by_binary_pattern("1???1", exp=20)
# Finds all integers with binary form 1???1
```

#### `find_by_popcount(popcount: int, exp: int) -> Iterator[int]`

Find integers with specific population count (number of 1-bits).

```python
for n in find_by_popcount(5, exp=16):
    # All integers up to 2^16 with exactly 5 bits set
    process(n)
```

#### `analyze_binary_patterns(n: int) -> Dict`

Comprehensive binary pattern analysis.

```python
analysis = analyze_binary_patterns(12345)
print(f"Popcount: {analysis['popcount']}")
print(f"Leading zeros: {analysis['leading_zeros']}")
```

### Export Functions

#### `walk_to_csv(start_n, exp, count, filename, columns=None)`

Export walk data to CSV file.

```python
walk_to_csv(
    start_n=1,
    exp=24,
    count=100000,
    filename="output.csv",
    columns=['n', 'core', 'v2', 'key', 'is_prime', 'popcount']
)
```

#### `walk_to_binary(start_n, exp, count, filename)`

Export walk data to binary format.

```python
walk_to_binary(1, exp=24, count=100000, filename="output.bin")
```

### Shell Analysis

#### `shell_statistics(exp: int) -> Dict`

Compute statistics for all shells.

```python
stats = shell_statistics(exp=20)
for shell, data in stats['shells'].items():
    print(f"Shell {shell}: {data['count']} integers")
```

#### `compare_shells(shell1: int, shell2: int, exp: int) -> Dict`

Compare two shells.

```python
comparison = compare_shells(5, 10, exp=24)
print(f"Size ratio: {comparison['size_ratio']}")
```

### Advanced Prime Analysis

#### `analyze_prime_angular_gaps(start_n, exp, direction, count, same_shell=False)`

Detailed analysis of gaps between consecutive primes.

```python
gaps = analyze_prime_angular_gaps(1, exp=24, 'cw', 1000)
print(f"Max angular gap: {gaps['max_angular_gap']}")
```

#### `find_prime_desert_angular(exp, min_gap, start_n=1, direction='cw')`

Find regions with no primes (deserts).

```python
desert = find_prime_desert_angular(exp=24, min_gap=1000)
if desert:
    print(f"Desert from {desert['start']} to {desert['end']}")
```

#### `find_ray_prime_sequence(odd_core, exp, direction, count)`

Find sequence of primes starting from a ray.

```python
primes = find_ray_prime_sequence(3, exp=24, 'cw', count=50)
```

### Sampling and Search

#### `sample_uniform_angular(exp: int, num_samples: int) -> List[int]`

Sample integers uniformly distributed by angle.

```python
samples = sample_uniform_angular(exp=24, num_samples=1000)
```

#### `find_similar_integers(n: int, exp: int, max_hamming=3) -> List[int]`

Find integers within Hamming distance.

```python
similar = find_similar_integers(12345, exp=24, max_hamming=2)
```

### Benchmarking

#### `benchmark(exp: int, iterations: int = 10000)`

Run performance benchmarks.

```python
benchmark(exp=24, iterations=50000)
```

---

## API Reference — Ant Perspective (square_ants_ops.py)

These `_ants` suffix functions are **only available in `square_ants_ops.py`**. They provide ant-centric naming and docstrings.

### Core Ant Functions

| Function | Description |
|----------|-------------|
| `get_odd_core_ants(n)` | Decompose position into (ray, shell) |
| `v2_ants(n)` | Which shell is position n on? |
| `ray_label_ants(n)` | Which ray is position n on? |
| `key_to_odd_core_ants(k, exp)` | "I'm at perimeter step k, what ray?" |
| `odd_core_to_key_ants(a, exp)` | "I'm on ray a, what perimeter step?" |

### Navigation Ant Functions

| Function | Description |
|----------|-------------|
| `generate_theta_sorted_ants(exp)` | Walk entire perimeter |
| `generate_with_info_ants(exp)` | Walk with full awareness |
| `theta_neighbors_ants(n, exp)` | Find left/right neighbors |
| `walk_theta_ants(start, exp, count, dir)` | Walk around perimeter |
| `walk_shell_ants(shell, exp, count)` | Walk single shell |
| `walk_rays_ants(start_core, exp, count)` | Examine consecutive rays |
| `walk_ray_ants(core, exp)` | Walk outward on single ray |
| `zoom_region_ants(center, exp, width)` | Explore local neighborhood |

### Prime Navigation Ant Functions

| Function | Description |
|----------|-------------|
| `walk_primes_ants(start, exp, count, dir)` | Walk through primes only |
| `walk_shell_primes_ants(shell, exp, count)` | Primes on single shell |
| `find_next_ray_prime_ants(n, exp)` | Next prime outward on ray |
| `find_prev_ray_prime_ants(n, exp)` | Previous prime inward |
| `find_angular_prime_ants(n, exp, dir)` | Nearest prime in direction |
| `find_primes_both_ways_ants(n, exp)` | Nearest primes left/right |

### Analysis Ant Functions

| Function | Description |
|----------|-------------|
| `angular_relationship_ants(n1, n2, exp)` | How are two positions related? |
| `angular_distance_ants(n1, n2, exp)` | Angular separation |
| `analyze_ray_ants(core, exp)` | Analyze single ray |
| `find_prime_desert_ants(exp, min_gap)` | Find prime-free region |

### Example: Ant-Style Code

```python
from square_ants_ops import (
    get_odd_core_ants, walk_theta_ants, 
    find_primes_both_ways_ants, ray_label_ants
)

# An ant at position 12345 wants to know where it is
position = 12345
ray, shell = get_odd_core_ants(position)
print(f"I'm on ray {ray} at shell {shell}")

# The ant looks for the nearest primes
primes = find_primes_both_ways_ants(position, exp=24)
print(f"Nearest prime to my left: {primes['ccw_prime']}")
print(f"Nearest prime to my right: {primes['cw_prime']}")

# The ant walks clockwise, noting each ray
print("Walking clockwise...")
for n in walk_theta_ants(position, exp=24, count=10, direction='cw'):
    print(f"  Position {n} is on ray {ray_label_ants(n)}")
```

---

## Command Line Usage

### Common Options (Both Modules)

```
--demo              Run demonstration
--exp N             Exponent (default: 24)
--neighbors N       Find theta-neighbors of N
--walk N            Walk theta from N
--walk-primes N     Walk through primes from N
--count N           Number of items (default: 20)
--prime-count N     Number of primes (default: 50)
--direction DIR     Direction: 'cw' or 'ccw'
--shell N           Restrict to specific shell
--analyze-ray N     Analyze primes on ray with odd core N
--bracket DX DY     Find rays bracketing direction (DX, DY)
```

### Additional Options (square_rays_ops.py only)

```
--bidirectional-prime N    Find nearest primes in both directions
--prime-stats              Show prime gap statistics
--export-csv FILE          Export to CSV
--export-binary FILE       Export to binary
--find-desert GAP          Find prime desert with minimum gap
--prime-gaps-analysis N    Analyze prime gaps from N
--next-prime-ray N         Find next prime on same ray
--prime-sequence N         Find prime sequence from ray N
```

### Example Sessions

```bash
# Basic demo
python square_rays_ops.py --demo

# Find neighbors of a large number
python square_rays_ops.py --neighbors 12345678 --exp 28

# Walk primes with statistics
python square_rays_ops.py --walk-primes 1 --prime-count 1000 --prime-stats --exp 24

# Export first million integers
python square_rays_ops.py --walk 1 --count 1000000 --export-csv data.csv --exp 24

# Find a prime desert
python square_rays_ops.py --find-desert 500 --exp 24
```

---

## Use Cases and Examples

### Use Case 1: Analyzing 32-bit Space

```python
from square_rays_ops import prime_walk_statistics

stats = prime_walk_statistics(1, exp=32, num_primes=100000)

print(f"32-bit Space Analysis:")
print(f"  Average gap: {stats['avg_gap']:.2f} integers")
print(f"  Median gap: {stats['median_gap']}")
print(f"  Max gap: {stats['max_gap']}")
```

### Use Case 2: Shell Comparison

```python
from square_rays_ops import walk_primes_at_shell

for shell in [15, 20, 25, 30]:
    primes = list(walk_primes_at_shell(1, exp=32, count=1000, shell=shell))
    avg_gap = sum(gap for _, gap in primes) / len(primes)
    print(f"Shell {shell}: avg gap = {avg_gap:.2f}")
```

### Use Case 3: Local Prime Neighborhood

```python
from square_rays_ops import bidirectional_prime_search

for n in [1_000_000, 10_000_000, 100_000_000]:
    result = bidirectional_prime_search(n, exp=32)
    print(f"{n:,}: nearest prime {result['nearest_prime'][0]:,}")
```

### Use Case 4: Ant Exploration

```python
from square_ants_ops import (
    get_odd_core_ants, walk_theta_ants, 
    find_primes_both_ways_ants
)

# Start at position 1000
pos = 1000
ray, shell = get_odd_core_ants(pos)

# Walk and count primes encountered
prime_count = 0
for n in walk_theta_ants(pos, exp=24, count=1000, direction='cw'):
    result = find_primes_both_ways_ants(n, exp=24)
    if result['is_prime']:
        prime_count += 1

print(f"Found {prime_count} primes in 1000 steps")
```

### Use Case 5: Export for External Analysis

```python
from square_rays_ops import walk_to_csv

walk_to_csv(
    start_n=1,
    exp=28,
    count=1_000_000,
    filename="primes_theta_order.csv",
    columns=['n', 'core', 'v2', 'key', 'is_prime', 'popcount']
)
```

---

## Performance

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `generate_theta_sorted` | O(1) per element | O(1) |
| `theta_neighbors` | O(1) | O(1) |
| `find_bracketing_rays` | O(1) | O(1) |
| `walk_primes` | O(1) per prime | O(1) |
| `is_prime` | O(log n) | O(1) |

### Benchmarks

On typical modern CPU (exp=24, ~16.7M integers):
- Generation: 5-10M integers/sec
- Neighbor lookup: 2-5M lookups/sec
- Angle bracketing: 1-2M lookups/sec
- Prime testing: 500K-1M tests/sec

### Prime Walking Efficiency

| Exp | Total Integers | Approx Primes | Speedup vs testing all |
|-----|----------------|---------------|------------------------|
| 20 | 1,048,576 | ~85,000 | 12× |
| 24 | 16,777,216 | ~1,200,000 | 14× |
| 28 | 268,435,456 | ~16,000,000 | 17× |
| 32 | 4,294,967,296 | ~203,000,000 | 21× |

### Memory Usage

All functions use constant memory:
- Iterators yield one element at a time
- No caching or global state
- Statistics collection is only operation that builds lists

---

## Mathematical Background

### The Odd Core Decomposition

Every positive integer n has unique factorization:

```
n = 2^v2 × a
```

Where a is odd. This creates natural hierarchy:
- **Rays:** All integers with same odd core
- **Shells:** All integers with same v2
- **Angular ordering:** Determined by normalized odd core

### Why No Trigonometry?

The square perimeter can be parameterized using pure rational arithmetic. For direction (dx, dy), we:

1. Determine which edge it intersects
2. Compute intersection using rational arithmetic only
3. Map to perimeter parameter t in [0, 1)

This t directly computes the theta-key without ever computing actual angles.

### Theta-Key Construction

For odd core a:
1. Write a in binary: a = 1.f (implicit leading 1)
2. Normalize fractional part f to (exp-1) bits
3. Key k = f × 2^(exp-1-bitlength(a))

This creates bijection between odd cores and angular positions.

### Prime Distribution

Primes in this construction follow interesting patterns:
- Density decreases with shell (larger numbers)
- Some angular regions have higher prime density
- Most rays have at most one prime (the odd core if prime)
- Exception: Ray with odd core 1 (powers of 2) has one prime: 2

---

## Tips and Best Practices

### When to Use Each Function

**Use `generate_theta_sorted` when:**
- You need ALL integers up to some bound
- Processing order matters
- Doing global analysis

**Use `walk_theta` when:**
- Exploring specific region
- Only need subset of integers
- Zooming into particular area

**Use `walk_primes` when:**
- Analyzing primes specifically
- Don't want to test composites
- Need efficient prime-focused analysis

### Memory Management

All walk/iteration functions use iterators (lazy evaluation):

```python
# Good - constant memory
for n in walk_theta(1, exp=32, count=1_000_000):
    process(n)

# Bad - would require huge memory
data = list(walk_theta(1, exp=32, count=100_000_000))
```

### Efficient Analysis

```python
# Don't do this - tests 4 billion integers
for n in generate_theta_sorted(32):
    if is_prime(n):
        analyze(n)

# Do this - only walks through ~200M primes
for prime, gap in walk_primes(1, exp=32, count=1000000):
    analyze(prime)
```

---

## Version History

**v3.0 — Unified Edition**
- Unified documentation covering both modules
- Clear guidance on module selection
- Complete API reference for both variants

**v2.0 — Enhanced Edition (square_rays_ops.py)**
- 30+ new analysis functions
- 6 prime walking functions
- Full CLI integration
- Export capabilities

**v2.0 — Ant Edition (square_ants_ops.py)**
- ROBO-ANT META-SPEC framework
- 22 ant-perspective functions
- Clear section tagging
- Streamlined codebase

**v1.0 — Original Release**
- Core bitwise operations
- Theta-sorted generation
- O(1) neighbor lookup
- Basic walk operations

---

## License

MIT License

---

## Repository

**GitHub:** https://github.com/nmicic/power-two-square-rays/

For questions, bug reports, or contributions, please refer to the repository.
