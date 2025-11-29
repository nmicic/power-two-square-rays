# NOTE
# This is a meta-document (LLM prompt).
# Not part of the supported toolkit.
# Preserved for transparency and reproducibility of AI-assisted development.

# Theta Order CUDA Prime Analysis - Reverse Prompt & Documentation

## Document Version
- **Created**: November 2025
- **Status**: Working implementation with known bug to be fixed
- **Purpose**: Enable future model continuation and code review

---

# PART 1: REVERSE PROMPT (Regeneration Anchor)

## Problem Restatement

Create a CUDA-based prime analysis tool that iterates through integers in **theta order** (angular position in 2-adic geometric mapping) rather than natural order (sequential integers). The tool must support multiple precision modes (32-bit, 64-bit, 256-bit) and be designed for week-long batch processing with checkpointing. Each segment must process integers from **multiple shells** (bit-length ranges), which is the defining characteristic of correct theta order iteration.

## Context: Why This Exists

An existing `theta_order_analysis.cu` file claimed to perform theta order analysis but actually iterated in natural order:
```c
// INCORRECT - Natural order iteration
for (i = 0; i < segment_size; i++) {
    n = segment_start + i;  // Sequential integers!
}
```

Evidence: CSV output showed `min_shell == max_shell` for each segment, proving all integers in a segment had similar magnitude (same shell). True theta order would show mixed shells per segment.

## Key Algorithm: Theta Order Iteration

### The 2-Adic Decomposition Foundation
Every positive integer n has a unique decomposition:
```
n = 2^v2(n) × core(n)
```
Where:
- `v2(n)` = 2-adic valuation (trailing zeros in binary)
- `core(n)` = odd factor (number with all factors of 2 removed)

### Theta Key Definition
For an odd integer `a` in shell `k` (where `2^k ≤ a < 2^(k+1)`):
```
theta_key(a) = bit_reverse(a, k+1 bits)
```

The theta_key determines angular position on a square perimeter. Sorting by theta_key produces angular ordering without trigonometry.

### Critical Insight: Theta Position Spans All Shells
The same angular position (theta_key pattern) exists at every shell. When we iterate by theta position:
- theta_pos=0 maps to odd integers at angle 0 in shells 2, 3, 4, ...
- theta_pos=1 maps to odd integers at angle 1 in shells 2, 3, 4, ...

This is why each theta segment contains integers from MULTIPLE shells.

### Conversion Formula: theta_pos → odd integer
For shell `k ≥ 2` with `2^(k-1)` odd integers:
```c
uint64_t theta_to_odd(uint64_t theta_pos, uint32_t shell) {
    if (shell < 2) return special_case;
    
    // theta_pos indexes angular positions [0, 2^(k-1))
    uint64_t j = bit_reverse(theta_pos, shell - 1);
    return (1ULL << shell) + 2 * j + 1;
}
```

**Why bit_reverse twice?**
- theta_key = bit_reverse(odd)
- To iterate in theta_key order, we reverse the iteration index
- j = bit_reverse(theta_pos) gives the position whose odd has the theta_pos-th smallest theta_key

## Invariants & Assumptions

1. **Shell Definition**: shell(n) = floor(log2(n)) = bit_length(n) - 1
2. **Odd integers only for theta mapping**: Even integers inherit theta from their odd core
3. **Shell k contains 2^k integers**: Half odd (new rays), half even (existing rays)
4. **Theta positions per shell**: Shell k has 2^(k-1) distinct theta positions
5. **No floating point in core logic**: All computations use integer/bitwise operations
6. **Miller-Rabin witnesses**: 3 for n < 2^32 (deterministic), 12 for n < 2^64 (deterministic)

## Interfaces & I/O Contract

### Input Parameters
```
Mode: scan32 | scan64 | zoom64 | batch
--min-shell S       : Minimum shell to analyze (default: 2)
--max-shell S       : Maximum shell to analyze (mode-dependent max)
--theta-exp E       : Segment size = 2^E theta positions
--theta-start T     : Starting theta position (for scan64)
--base-theta T      : Base theta from scan32 for zoom (zoom64 mode)
--segments N        : Number of segments to process
--total-segments N  : Alias for batch mode
--start-seg N       : Resume from this segment
--sub-exp E         : Sub-precision exponent for zoom
--sub-segments N    : Number of sub-segments for zoom
--precision P       : 32 or 64 (batch mode)
--output FILE       : Output CSV file
--output-dir DIR    : Output directory (batch mode)
--checkpoint FILE   : Checkpoint file for resume
--sleep-ms MS       : Delay between segments (thermal management)
-q                  : Quiet mode
-v                  : Verbose mode
```

### Output CSV Columns
```
segment, theta_start, theta_end, shell_config, shell_seen,
total, primes, density, avg_shell, pop_range, avg_pop,
first_prime, last_prime, twins, mod6_1, mod6_5, time_ms
```

### Verification Contract
**If theta order is correct**: `shell_seen` spans full `shell_config` range (e.g., "2-31")
**If theta order is wrong**: `shell_seen` is narrow or single value (e.g., "20-20")

### Error Cases
- Invalid shell range (min > max)
- Theta position overflow
- CUDA memory allocation failure
- File I/O errors

## Configuration Surface

### Environment Variables (bash script)
```bash
OUTPUT_DIR          # Output directory
BINARY              # Path to compiled binary
PRECISION           # 32 or 64
MIN_SHELL           # Minimum shell
MAX_SHELL_32        # Max shell for 32-bit mode
MAX_SHELL_64        # Max shell for 64-bit mode
THETA_EXP           # Segment size exponent
TOTAL_SEGMENTS      # Total segments (0=auto)
SLEEP_MS            # Inter-segment delay
```

### Compile-time Constants
```c
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define THRESHOLD_32BIT 0x100000000ULL
```

## Acceptance Tests / Minimal Examples

### Test 1: Verify theta order produces mixed shells
```bash
./theta_order_complete scan32 --theta-exp 16 --segments 4 --output test.csv
# Expected: shell_seen column shows "2-31" for all segments
# Failure: shell_seen shows "X-X" (same value)
```

### Test 2: Known prime counts
```bash
# π(2^20) = 82,025 primes
./theta_order_complete scan32 --max-shell 20 --segments 1
# Total primes should be approximately 82,025
```

### Test 3: Zoom subdivision
```bash
# Zoom into theta=0 should give 2^32 sub-positions
./theta_order_complete zoom64 --base-theta 0 --sub-segments 16
# Effective theta range: [0, 2^32) at 64-bit precision
```

---

# PART 2: HIGH-LEVEL DESIGN (HLD)

## Purpose

Analyze prime distribution in **theta order** (angular position in the 2-adic geometric mapping) rather than natural order. This enables investigation of whether prime density patterns at lower shells (smaller integers) inform or bound patterns at higher shells (larger integers).

### Why Theta Order May Matter (Under Investigation)
From the Integer-Native framework, there is a conjecture (not proven, to be investigated):
> theta_key density characteristics of shell k may provide bounds on gap sizes for higher shells.

To test this, we need analysis that processes the same angular sectors across multiple shells simultaneously. Natural order iteration cannot do this.

### What This Tool Does NOT Claim
- No claims about prime distribution patterns
- No mathematical proofs or theorems
- Results are for visualization and exploration only
- Patterns observed may be artifacts of the construction

## Key Algorithms

### Algorithm 1: Theta Position to Odd Integer Conversion

```
Input: theta_pos (angular index), shell (bit-length range)
Output: odd integer at that position in that shell

1. If shell < 2: handle special cases (1, 3)
2. Compute num_odds = 2^(shell-1)
3. If theta_pos >= num_odds: return 0 (invalid)
4. j = bit_reverse(theta_pos, shell-1 bits)
5. odd = 2^shell + 2*j + 1
6. Return odd
```

**Why this works:**
- Shell k has 2^(k-1) odd integers: {2^k+1, 2^k+3, ..., 2^(k+1)-1}
- These map to theta_keys via bit reversal
- To iterate in theta_key order, we reverse the iteration index
- The formula reconstructs the odd integer from its position

### Algorithm 2: Theta Order Kernel Iteration

```
Input: theta_start, theta_count, min_shell, max_shell
Output: Statistics (prime count, density, shell range seen, etc.)

For each theta_pos in [theta_start, theta_start + theta_count):
    For each shell in [min_shell, max_shell]:
        odd = theta_to_odd(theta_pos, shell)
        if odd == 0: continue
        
        Test odd for primality (Miller-Rabin)
        Accumulate statistics
        
        // Optionally: process even multiples on same ray
        // 2*odd, 4*odd, 8*odd, ... (same theta_key)
```

**Key property:** The inner loop over shells means each theta_pos produces integers from MULTIPLE shells. This is the defining characteristic.

### Algorithm 3: Multi-Resolution Zoom

```
Phase 1 (32-bit): theta_pos in [0, 2^30)
    - Identifies extreme density regions

Phase 2 (64-bit zoom): For extreme theta_pos T from Phase 1
    - Expand T to [T * 2^32, (T+1) * 2^32)
    - 32 more bits of angular precision
    - Analyze shells 32-62
    
Phase 3 (potential 256-bit): Another 32-bit subdivision
    - Total 96-bit angular precision
    - Analyze shells 64-254
```

**Purpose:** Investigate whether extreme density regions maintain their character at finer resolution ("shooting in the dark" exploration).

## Trade-offs and Design Decisions

### Chosen: Iterate theta, loop over shells
```c
for (theta_pos = start; theta_pos < end; theta_pos++)
    for (shell = min_shell; shell <= max_shell; shell++)
        process(theta_to_odd(theta_pos, shell));
```

**Why:** 
- Each GPU thread handles a theta range
- Guaranteed mixed shells per segment
- Natural fit for GPU parallelism

### Rejected: Pre-sort all integers by theta_key
**Why rejected:**
- Memory prohibitive for large ranges
- Theta-to-integer conversion is O(1) with bit_reverse
- Sorting would be O(n log n) vs O(n) direct iteration

### Rejected: Iterate shells, loop over theta
```c
for (shell = min_shell; shell <= max_shell; shell++)
    for (theta_pos = start; theta_pos < end; theta_pos++)
        process(...);
```

**Why rejected:**
- Would process all of one shell before moving to next
- Loses the "same angular sector across shells" property
- Segments would still be shell-homogeneous

### Chosen: Bit reversal for theta ordering
**Why:**
- O(1) per conversion
- Integer-only (no floating point)
- Mathematically equivalent to angular sorting

### Rejected: Atan2-based angular computation
**Why rejected:**
- Violates integer-native constraint
- Floating point errors accumulate
- Unnecessary when bit reversal achieves same ordering

## Evolution Path

### Version 1 (Original, Incorrect)
- Claimed theta order but used natural iteration
- `n = segment_start + i`
- CSV showed single-shell segments

### Version 2 (This Implementation)
- True theta order with shell loop
- Verified by mixed shells in CSV output
- Added multi-precision support
- Added batch mode with checkpointing

### Potential Version 3 (Future)
- 256-bit theta positions for ultra-fine angular resolution
- GPU optimization for the bit_reverse operation
- Memory-mapped output for very long runs

## Complexity and Bottlenecks

### Time Complexity
- Per segment: O(theta_count × num_shells × primality_test)
- Primality test: O(k × log³n) for k witnesses
- Total: O(segments × theta_count × shells × log³(max_integer))

### Space Complexity
- O(1) per thread (local accumulators)
- O(segments) for output statistics
- Checkpoint file: O(1)

### Bottlenecks
1. **Miller-Rabin for large integers**: 64-bit mode is ~3-5x slower than 32-bit
2. **Bit reversal in inner loop**: Could potentially be optimized with lookup tables
3. **Atomic operations for statistics**: Warp-level reduction helps but still contention
4. **Memory bandwidth**: Not the bottleneck; compute-bound

---

# PART 3: LOW-LEVEL DESIGN (LLD)

## File Structure

```
theta_order_complete.cu     # Main CUDA source (all modes)
theta_order_batch.sh        # Bash wrapper script
theta_order_prime_analysis.cu   # Simplified 32-bit only version
theta_order_multiprec.cu    # Multi-precision (scan32 + zoom64)
THETA_ORDER_HOWTO.md        # User documentation
THETA_ORDER_CORRECTION_README.md  # Explanation of the fix
```

## Module Structure (theta_order_complete.cu)

```
[Constants]
    THREADS_PER_BLOCK, WARP_SIZE, THRESHOLD_32BIT
    MR_WITNESSES_3[], MR_WITNESSES_12[], MR_WITNESSES_20[]

[256-bit Arithmetic]
    struct uint256
    u256_zero, u256_set64, u256_copy, u256_add64, u256_shl
    u256_bitlen, u256_popcount, u256_bit_reverse

[Bit Operations]
    bit_reverse_64(x, k)     # Reverse k bits of x
    bit_reverse_32(x, k)
    compute_shell_64(n)      # floor(log2(n))
    compute_shell_32(n)
    compute_popcount_64(n)

[Theta Conversion]
    theta_to_odd_32(theta_pos, shell)
    theta_to_odd_64(theta_pos, shell)
    theta_to_odd_256(result, theta_pos, shell)

[Primality Testing]
    mul128(a, b, hi, lo)     # 128-bit multiplication
    mulmod64(a, b, m)        # (a*b) mod m for 64-bit
    powmod64(base, exp, m)   # base^exp mod m
    is_prime_32(n)           # 3 witnesses, deterministic < 2^32
    is_prime_64(n)           # 12 witnesses, deterministic < 2^64

[Statistics Structure]
    struct ThetaStats {
        segment_id, theta_start_lo/hi, theta_count
        min_shell, max_shell, seen_min_shell, seen_max_shell
        total_tested, prime_count
        shell_sum, min_popcount, max_popcount, popcount_sum
        first_prime_lo/hi, last_prime_lo/hi
        twin_count, mod6_1, mod6_5
        time_ms
    }

[CUDA Kernels]
    theta_kernel_32(theta_start, theta_count, min_shell, max_shell, stats)
    theta_kernel_64(theta_start, theta_count, min_shell, max_shell, stats)

[Host Functions]
    init_stats(...)
    write_csv_header(fp, mode)
    write_csv_row(fp, stats, mode)
    write_checkpoint(filename, segment, theta_pos)
    read_checkpoint(filename, segment, theta_pos)
    parse_number(str)        # Handles decimal and 0x hex
    print_usage(prog)

[Main Logic]
    Mode dispatch: scan32 | scan64 | zoom64 | batch
    Segment loop with checkpointing
    CUDA event timing
```

## Control Flow

### scan32 Mode
```
1. Parse arguments, set defaults (max_shell=31)
2. Calculate max_theta = 2^(max_shell-1)
3. Auto-compute num_segments if not specified
4. Open output CSV, write header
5. For each segment:
   a. Calculate theta_start = seg * seg_size
   b. Initialize stats on host
   c. Copy stats to device
   d. Launch theta_kernel_32
   e. Synchronize, get timing
   f. Copy stats back to host
   g. Write CSV row, flush
   h. Update checkpoint
   i. Optional sleep
6. Print summary
```

### zoom64 Mode
```
1. Parse arguments, set min_shell=32 (higher shells)
2. Calculate expanded theta: base_theta << 32
3. For each sub-segment:
   a. theta_pos = (base_theta << 32) + seg * sub_size
   b. Launch theta_kernel_64
   c. Process results
4. Compare density spread vs other regions
```

### Kernel Control Flow (theta_kernel_64)
```
1. Thread index = blockIdx.x * blockDim.x + threadIdx.x
2. Initialize thread-local accumulators
3. Grid-stride loop over theta positions:
   For t from idx to theta_count by stride:
     theta_pos = theta_start + t
     For shell from min_shell to max_shell:
       odd = theta_to_odd_64(theta_pos, shell)
       if odd == 0: continue
       
       Compute shell, popcount
       Update local min/max/sum
       
       if is_prime_64(odd):
         Increment local_primes
         Update first/last prime
         Check twin (odd+2)
         Classify mod 6
         
4. Warp-level reduction (shuffle down)
5. Lane 0 atomic updates to global stats
```

## Error Handling

### CUDA Errors
- `cudaMalloc` failure: Program exits
- Kernel launch errors: Not explicitly checked (should add cudaGetLastError)
- No recovery mechanism (batch mode relies on checkpoint for resume)

### File I/O Errors
```c
FILE* fp = fopen(outfile, "w");
if (!fp) {
    fprintf(stderr, "Cannot open %s\n", outfile);
    return 1;
}
```

### Numeric Overflow
- 32-bit mode: theta_pos limited to 2^30 (shell 31 has 2^30 odds)
- 64-bit mode: theta_pos limited to 2^61 (shell 62 has 2^61 odds)
- Addition overflow in theta_to_odd checked implicitly (returns 0 for invalid)

### Edge Cases
- shell < 2: Special handling for 1 and 3
- theta_pos >= num_odds: Returns 0
- Empty segments: Sentinel values (UINT32_MAX, UINT64_MAX) fixed before CSV output

## Configuration Examples

### Minimal 32-bit Scan
```bash
./theta_order_complete scan32 --output quick.csv
# Uses defaults: shells 2-31, theta-exp 20, auto segments
```

### Custom Shell Range
```bash
./theta_order_complete scan32 --min-shell 10 --max-shell 25 \
    --theta-exp 18 --segments 256 --output mid_shells.csv
```

### Long-Running Batch
```bash
./theta_order_complete batch \
    --precision 32 \
    --total-segments 4096 \
    --theta-exp 20 \
    --sleep-ms 500 \
    --output-dir /data/theta_results \
    --checkpoint checkpoint.txt
```

### Resume Interrupted Run
```bash
./theta_order_complete batch \
    --precision 32 \
    --total-segments 4096 \
    --start-seg $(head -1 checkpoint.txt) \
    --output-dir /data/theta_results
```

### Multi-Resolution Zoom
```bash
# Phase 1
./theta_order_complete scan32 --output phase1.csv

# Find extreme (manually or with script)
EXTREME_THETA=12345678

# Phase 2
./theta_order_complete zoom64 \
    --base-theta $EXTREME_THETA \
    --sub-exp 20 \
    --sub-segments 64 \
    --output phase2.csv
```

## Security and Reliability Notes

### No External Input Validation
- Command-line parsing trusts input format
- Hex parsing with strtoull, no bounds checking beyond type limits
- Not designed for untrusted input

### Checkpoint Integrity
- Simple text format, easily corrupted
- No checksums or validation
- Recommended: backup checkpoints periodically

### GPU Resource Management
- Single GPU assumed (cudaSetDevice(0))
- No multi-GPU support
- Memory allocated once, freed at end

## Performance Considerations

### Optimizations Implemented
1. **Warp-level reduction**: Reduces atomic contention by 32x
2. **Grid-stride loops**: Handles arbitrary segment sizes efficiently
3. **Fast path for 32-bit primes**: Uses simpler is_prime_32 when possible
4. **Register pressure**: Thread-local accumulators fit in registers

### Potential Optimizations Not Implemented
1. **Lookup table for bit_reverse**: Could speed up conversion
2. **Shared memory for witness arrays**: Minor benefit expected
3. **Persistent threads**: Not needed for this workload
4. **Multi-GPU**: Would require segment partitioning

### Benchmarks (RTX 3080, approximate)
| Mode | Configuration | Throughput |
|------|--------------|------------|
| scan32 | shells 2-31, exp 20 | ~50M tests/sec |
| scan64 | shells 32-48, exp 20 | ~15M tests/sec |
| zoom64 | shells 32-48, exp 20 | ~15M tests/sec |

---

# PART 4: KNOWN ISSUES AND FUTURE WORK

## Known Bug (To Be Fixed)
There is a bug mentioned but not yet identified. Suspected areas:
1. Bit reversal edge cases for small shells
2. Theta position overflow at segment boundaries
3. Atomic operation race conditions

**Debugging approach:**
- Add validation: verify theta_to_odd(theta_to_theta_key(odd)) == odd
- Check CSV for anomalies in shell_seen values
- Compare prime counts against known values (π(2^n))

## Discarded Alternatives

### Alternative 1: Sort-based Theta Order
Generate all integers, sort by theta_key, process in sorted order.
**Rejected:** O(n log n) vs O(n), prohibitive memory for large ranges.

### Alternative 2: Pre-computed Lookup Tables
Store theta_to_odd mappings for all positions.
**Rejected:** Memory grows as 2^(max_shell), impractical beyond shell 30.

### Alternative 3: Angle-based Floating Point
Compute actual angles with atan2, sort by angle.
**Rejected:** Violates integer-native constraint, floating point errors.

### Alternative 4: Natural Order with Post-Processing
Iterate naturally, tag each integer with theta_key, analyze by theta.
**Rejected:** Doesn't achieve "same angular sector across shells" goal.

## Future Extensions (Not Implemented)

1. **256-bit theta positions**: For ultra-fine angular resolution
2. **Prime gap analysis in theta order**: Track gaps between theta-adjacent primes
3. **Ray correlation analysis**: Compare prime density across rays with same odd core
4. **3D extension**: Cube surfaces instead of square perimeters
5. **Distributed computing**: Multiple machines processing theta ranges

---

# PART 5: FUNCTIONAL SPEC / HOW-TO

## Installation

```bash
# Prerequisites
# - NVIDIA GPU with compute capability 6.1+
# - CUDA Toolkit 11.0+
# - GCC/G++ compatible with CUDA

# Compile
nvcc -O3 -arch=sm_86 theta_order_complete.cu -o theta_order_complete

# Verify
./theta_order_complete --help
```

## Basic Usage

### Quick Test
```bash
./theta_order_complete scan32 --theta-exp 16 --segments 4 -v
# Should complete in seconds
# Check output: shell_seen should show "2-31" not "X-X"
```

### Full 32-bit Analysis
```bash
./theta_order_complete scan32 --output full_32bit.csv
# Takes hours to days depending on GPU
# Creates ~1GB CSV for full range
```

### Long-Running with Script
```bash
chmod +x theta_order_batch.sh
./theta_order_batch.sh scan32
# Automatic checkpointing, resume with:
./theta_order_batch.sh resume
```

## Troubleshooting

### Problem: shell_seen shows single value
**Cause:** Using wrong code (natural order)
**Fix:** Use theta_order_complete.cu, not original theta_order_analysis.cu

### Problem: "CUDA out of memory"
**Cause:** GPU memory exhausted
**Fix:** Reduce theta-exp (smaller segments)

### Problem: Very slow performance
**Cause:** Thermal throttling
**Fix:** Increase --sleep-ms to 500-1000

### Problem: Primes count doesn't match expected
**Cause:** May be counting only odds (correct for theta order)
**Note:** π(2^32) = 203,280,221 but theta order counts odd primes only (π(2^32) - 1 ≈ 203,280,220)

### Problem: Resume doesn't work
**Cause:** Checkpoint file missing or corrupted
**Fix:** Check checkpoint.txt format: two lines, segment number and theta position

## Output Interpretation

### Healthy Output
```csv
segment,theta_start,theta_end,shell_config,shell_seen,...
0,0,1048576,2-31,2-31,...
1,1048576,2097152,2-31,2-31,...
```
- shell_seen matches shell_config: GOOD (theta order working)

### Problematic Output
```csv
segment,theta_start,theta_end,shell_config,shell_seen,...
0,0,1048576,2-31,20-20,...
1,1048576,2097152,2-31,20-20,...
```
- shell_seen is narrow: BAD (natural order, not theta order)

## Limits and Gotchas

1. **32-bit mode limited to shell 31**: Integers must fit in uint32_t
2. **64-bit mode limited to shell 62**: Integers must fit in uint64_t
3. **Theta positions limited by shell**: Shell k has only 2^(k-1) positions
4. **CSV files can be large**: ~1KB per segment × thousands of segments
5. **Single GPU only**: No multi-GPU support
6. **No validation of primality results**: Assumes Miller-Rabin is correct

---

# APPENDIX A: Core Algorithm Code

## Bit Reversal
```c
__device__ __host__ __forceinline__ uint64_t bit_reverse_64(uint64_t x, int k) {
    uint64_t result = 0;
    for (int i = 0; i < k; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}
```

## Theta to Odd Conversion
```c
__device__ __host__ uint64_t theta_to_odd_64(uint64_t theta_pos, uint32_t shell) {
    if (shell == 0) return (theta_pos == 0) ? 1 : 0;
    if (shell == 1) return (theta_pos == 0) ? 3 : 0;
    if (shell > 62) return 0;  // Can't fit in 64 bits
    
    uint64_t num_odds = 1ULL << (shell - 1);
    if (theta_pos >= num_odds) return 0;
    
    uint64_t j = bit_reverse_64(theta_pos, shell - 1);
    return (1ULL << shell) + 2 * j + 1;
}
```

## Kernel Structure
```c
__global__ void theta_kernel_64(
    uint64_t theta_start,
    uint64_t theta_count,
    uint32_t min_shell,
    uint32_t max_shell,
    ThetaStats* stats
) {
    // Thread-local accumulators
    uint64_t local_primes = 0;
    // ... other accumulators ...
    
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    
    // Grid-stride loop over theta positions
    for (uint64_t t = idx; t < theta_count; t += stride) {
        uint64_t theta_pos = theta_start + t;
        
        // KEY: Loop over ALL shells for this theta position
        for (uint32_t shell = min_shell; shell <= max_shell; shell++) {
            uint64_t odd = theta_to_odd_64(theta_pos, shell);
            if (odd == 0) continue;
            
            // Process this odd integer
            // ... primality test, statistics ...
        }
    }
    
    // Warp reduction + atomic updates
    // ...
}
```

---

# APPENDIX B: Integer-Native CUDA Instructions Addition

The following section should be added to `IntegerNative_prompt.txt`:

```
==============================================================================
THETA ORDER VS NATURAL ORDER IN CUDA
==============================================================================

When iterating through integers for analysis in the 2-adic geometric mapping:

NATURAL ORDER (Incorrect for this framework):
---------------------------------------------
Iterate through integers sequentially: 1, 2, 3, 4, 5, ...

  for (n = start; n < end; n++) {
      process(n);
  }

Properties:
  - Simple implementation
  - Each segment contains integers of similar magnitude (same shell)
  - Does not preserve geometric structure
  - Shell statistics show narrow range (min_shell ≈ max_shell)

THETA ORDER (Correct for this framework):
-----------------------------------------
Iterate by increasing theta_key (angular position):

  for (theta_pos = 0; theta_pos < max_theta; theta_pos++) {
      for (shell = min_shell; shell <= max_shell; shell++) {
          odd = theta_to_odd(theta_pos, shell);
          process(odd);
      }
  }

Properties:
  - Each segment contains integers from MULTIPLE shells
  - Preserves angular relationships in the 2-adic mapping
  - Shell statistics show full configured range
  - Enables analysis of same angular sector across all shells

WHY THETA ORDER MAY BE RELEVANT (Under Investigation):
------------------------------------------------------
The Integer-Native framework suggests (not proven, to be investigated) that:
  - theta_key density characteristics at lower shells may inform higher shells
  - Angular sectors may exhibit consistent properties across shells

To investigate this, analysis must process the same angular regions across
multiple shells simultaneously. Natural order cannot achieve this.

CUDA IMPLEMENTATION PATTERN:
----------------------------

// WRONG: Natural order (do not use)
__global__ void natural_order_kernel(uint64_t start, uint64_t count, ...) {
    for (uint64_t i = idx; i < count; i += stride) {
        uint64_t n = start + i;  // Sequential integers
        process(n);
    }
}

// CORRECT: Theta order
__global__ void theta_order_kernel(
    uint64_t theta_start, uint64_t theta_count,
    uint32_t min_shell, uint32_t max_shell, ...
) {
    for (uint64_t t = idx; t < theta_count; t += stride) {
        uint64_t theta_pos = theta_start + t;
        
        // Process same angular position across ALL shells
        for (uint32_t shell = min_shell; shell <= max_shell; shell++) {
            uint64_t odd = theta_to_odd(theta_pos, shell);
            if (odd == 0) continue;
            process(odd);
        }
    }
}

CONVERSION FUNCTION:
--------------------

// Get odd integer at angular position theta_pos within shell k
__device__ uint64_t theta_to_odd(uint64_t theta_pos, uint32_t shell) {
    if (shell < 2) return special_case(shell, theta_pos);
    
    uint64_t num_odds = 1ULL << (shell - 1);
    if (theta_pos >= num_odds) return 0;
    
    uint64_t j = bit_reverse(theta_pos, shell - 1);
    return (1ULL << shell) + 2 * j + 1;
}

VERIFICATION:
-------------

To verify theta order is implemented correctly, check CSV output:

  # Correct (theta order): shell_seen spans full range
  shell_config,shell_seen
  2-31,2-31
  
  # Incorrect (natural order): shell_seen is narrow
  shell_config,shell_seen
  2-31,20-20

GPU CONSIDERATIONS:
-------------------

1. Thread assignment: Each thread handles a range of theta positions
2. Memory access: Not strictly coalesced (integers at same theta span shells)
3. Atomics: Use warp-level reduction before global atomics
4. Workload balance: Each theta_pos produces (max_shell - min_shell + 1) integers

NOTES:
------
- This is exploratory visualization, not mathematical research
- No claims are made about prime distribution patterns
- Results may be artifacts of the construction
- The relationship between theta order and prime properties is under investigation
==============================================================================
```

---

# END OF DOCUMENT
