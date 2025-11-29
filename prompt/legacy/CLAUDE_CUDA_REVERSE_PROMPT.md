# NOTE
# This is a meta-document (LLM prompt).
# Not part of the supported toolkit.
# Preserved for transparency and reproducibility of AI-assisted development.

# ARCHIVE NOTE
# This prompt documents the previous CUDA prime-analysis tool used
# in early versions of the project. The tool is deprecated but the
# prompt is preserved as reference for AI-based regeneration.

---

CLAUDE_CUDA_REVERSE_PROMPT.md

Theta Order CUDA Prime Analysis — Claude-Oriented Reverse Prompt

Goal for Claude:
Work as a CUDA engineer on this codebase.
Maintain the existing design, invariants, and iteration logic.
Focus on code, not new math. Do not redefine the number system or theta order; only improve, extend, or debug the implementation.

⸻

PART 1: REVERSE PROMPT (Regeneration Anchor)

Problem Restatement (Programming View)

Implement and maintain a CUDA tool that:
	1.	Iterates integers in theta order (by angular position from a 2-adic mapping), not natural numeric order.
	2.	Supports multiple integer sizes: 32-bit, 64-bit (later possibly 256-bit via software big-int).
	3.	Runs in long batches with checkpointing, CSV output, and restart capability.
	4.	Ensures each theta segment contains integers from multiple shells (shell = bit-length-1) as a correctness check for theta order.

Core condition to enforce in code:
	•	Theta order segments must include integers from multiple shells.
	•	If a segment only sees a single shell, the code has silently reverted to natural order.

⸻

The Original Bug (for Context)

Original wrong pattern (natural order):

// WRONG - Natural order
for (i = 0; i < segment_size; i++) {
    n = segment_start + i;  // Sequential integers: 1, 2, 3, ...
}

Symptom in CSV:
	•	shell_seen = X-X (single shell) → this is wrong for theta order.
	•	True theta order should yield shell_seen = min-max with multiple shell values.

⸻

PART 2: CORE INTEGER & THETA OPERATIONS (CODE, NOT MATH)

2.1 2-Adic Decomposition — as Functions

We treat these as primitive building blocks:

// Count trailing zeros
__device__ __host__ __forceinline__
uint32_t v2_u32(uint32_t n);   // or use __ffs / builtin_ctz

__device__ __host__ __forceinline__
uint32_t v2_u64(uint64_t n);

// Odd core: n >> v2(n)
__device__ __host__ __forceinline__
uint32_t core_u32(uint32_t n);

__device__ __host__ __forceinline__
uint64_t core_u64(uint64_t n);

// Shell: bit_length(n) - 1
__device__ __host__ __forceinline__
uint32_t shell_u32(uint32_t n);

__device__ __host__ __forceinline__
uint32_t shell_u64(uint64_t n);

Invariants (do not change):
	•	n = (1u << v2(n)) * core(n)
	•	core(n) is always odd.
	•	shell(n) = floor(log2(n)) implemented via bit_length-1.

⸻

2.2 Bit Reversal — Performance-Critical

We use fast bit reversal for theta keys.

2.2.1 Generic O(log k) bit reversal

__device__ __host__ __forceinline__ 
uint64_t bit_reverse_64_fast(uint64_t x, int k) {
    x = ((x & 0x5555555555555555ULL) << 1)  | ((x >> 1)  & 0x5555555555555555ULL);
    x = ((x & 0x3333333333333333ULL) << 2)  | ((x >> 2)  & 0x3333333333333333ULL);
    x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4)  | ((x >> 4)  & 0x0F0F0F0F0F0F0F0FULL);
    x = ((x & 0x00FF00FF00FF00FFULL) << 8)  | ((x >> 8)  & 0x00FF00FF00FF00FFULL);
    x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
    x = (x << 32) | (x >> 32);
    return x >> (64 - k);
}

__device__ __host__ __forceinline__ 
uint32_t bit_reverse_32_fast(uint32_t x, int k) {
    x = ((x & 0x55555555u) << 1)  | ((x >> 1)  & 0x55555555u);
    x = ((x & 0x33333333u) << 2)  | ((x >> 2)  & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4)  | ((x >> 4)  & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8)  | ((x >> 8)  & 0x00FF00FFu);
    x = (x << 16) | (x >> 16);
    return x >> (32 - k);
}

2.2.2 CUDA intrinsic versions (preferred on device)

#ifdef __CUDA_ARCH__
__device__ __forceinline__ 
uint32_t bit_reverse_32(uint32_t x, int k) {
    return __brev(x) >> (32 - k);
}

__device__ __forceinline__ 
uint64_t bit_reverse_64(uint64_t x, int k) {
    return __brevll(x) >> (64 - k);
}
#endif

Constraint for Claude:
When optimizing or refactoring, you must ensure that all bit reversal functions still match a simple reference implementation. Do not change semantics.

⸻

2.3 Theta Mapping — theta_pos -> odd integer

The core function:

// theta_pos: angular index for given shell (0..2^(shell-1)-1)
// shell    : bit-length-1, >= 2
// Returns: odd integer in that shell at that theta position, or 0 on invalid input.
__device__ __host__ __forceinline__
uint64_t theta_to_odd(uint64_t theta_pos, uint32_t shell) {
    if (shell < 2) {
        // Shells 0 and 1 can be treated as special or skipped.
        return 0ULL;
    }

    uint64_t num_odds = 1ULL << (shell - 1);
    if (theta_pos >= num_odds) {
        return 0ULL;  // invalid for this shell
    }

    // Reverse (shell-1)-bit theta_pos
    uint64_t j = bit_reverse_64(theta_pos, shell - 1);

    // Map j to the odd value in [2^shell, 2^(shell+1))
    return (1ULL << shell) + 2ULL * j + 1ULL;
}

Invariants:
	•	theta_pos is shell-local, not global.
	•	For fixed shell, this is a bijection:
	•	theta_pos ∈ [0, 2^(shell-1)) ↔ odd integers in that shell.
	•	Same theta_pos across different shells → different integers, but “same angular position” in a conceptual sense.

Claude must not change this mapping without explicitly being asked to redesign it.

⸻

PART 3: HIGH-LEVEL DESIGN (HLD)

3.1 Purpose of the Program

Implement a CUDA program that:
	•	Iterates over a range of theta positions and range of shells.
	•	For each valid (theta_pos, shell), computes the corresponding odd integer.
	•	Runs primality tests and other integer observations (popcount, etc.).
	•	Aggregates per-segment statistics into CSV rows.
	•	Supports long scans with restart/resume via checkpoint files.

3.2 Core Algorithmic Pattern

Theta order iteration:

for (theta_pos = theta_start; theta_pos < theta_start + theta_count; ++theta_pos) {
    for (shell = min_shell; shell <= max_shell; ++shell) {
        if (theta_pos >= (1ULL << (shell - 1))) continue;

        uint64_t odd = theta_to_odd(theta_pos, shell);
        if (!odd) continue;

        // Observe odd: primality, popcount, etc.
    }
}

Key property:
Given a fixed segment [theta_start, theta_start + theta_count), the observed integers must come from multiple shells, not a single shell.

⸻

PART 4: LOW-LEVEL DESIGN (LLD)

4.1 Files & Modules

theta_order_complete.cu      // Main CUDA program
bit_reverse_optimized.cuh    // Bit reversal utilities
theta_order_batch.sh         // Batch wrapper script

4.2 Suggested Module Structure

[Bit Reversal]
  bit_reverse_32_fast()
  bit_reverse_64_fast()
  bit_reverse_32()/bit_reverse_64() using __brev/__brevll when available

[Theta Mapping]
  theta_to_odd_32()
  theta_to_odd_64()

[Primality]
  bool is_prime_32(uint32_t n);   // Miller-Rabin with fixed witnesses
  bool is_prime_64(uint64_t n);   // Miller-Rabin with fixed witnesses

[Statistics]
  struct ThetaStats {
      uint64_t total;
      uint64_t primes;
      uint64_t popcount_sum;
      uint64_t shell_min;
      uint64_t shell_max;
      // more fields as needed
  };

[Kernels]
  __global__ void theta_kernel_32(...);
  __global__ void theta_kernel_64(...);

[Host Utilities]
  parse_args(...)
  init_stats(...)
  write_csv_row(...)
  save_checkpoint(...)
  load_checkpoint(...)
  main()


⸻

4.3 Kernel Contract

Example for 64-bit kernel:

__global__ void theta_kernel_64(
    uint64_t theta_start,
    uint64_t theta_count,
    uint32_t min_shell,
    uint32_t max_shell,
    ThetaStats* stats_out  // device array, one per block or per grid
) {
    uint64_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;

    // Local accumulators
    uint64_t local_total   = 0;
    uint64_t local_primes  = 0;
    uint64_t local_pop_sum = 0;
    uint32_t local_shell_min = 0xFFFFFFFFu;
    uint32_t local_shell_max = 0;

    for (uint64_t t = idx; t < theta_count; t += stride) {
        uint64_t theta_pos = theta_start + t;

        for (uint32_t shell = min_shell; shell <= max_shell; ++shell) {
            if (theta_pos >= (1ULL << (shell - 1))) continue;

            uint64_t odd = theta_to_odd(theta_pos, shell);
            if (!odd) continue;

            local_total++;
            uint32_t sh = shell;
            if (sh < local_shell_min) local_shell_min = sh;
            if (sh > local_shell_max) local_shell_max = sh;

            bool prime = is_prime_64(odd);
            if (prime) {
                local_primes++;
            }

            uint32_t pop = __popcll(odd);
            local_pop_sum += pop;
        }
    }

    // Reduce within block and store into stats_out[blockIdx.x]
    // (warp reduction + shared memory + atomics or equivalent)
}

Claude’s job when editing this:
	•	Maintain the theta-to-odd iteration structure.
	•	Optimize warp divergence if possible.
	•	Ensure race-free reduction and accurate statistics.

⸻

PART 5: CLI, MODES, AND CSV

5.1 CLI Parameters (contract)

Program is expected to support flags like:

Mode:        scan32 | scan64 | zoom64 | batch
--min-shell S
--max-shell S
--theta-exp E        # segment size = 2^E theta positions
--theta-start T
--segments N
--output FILE
--checkpoint FILE
--sleep-ms MS
--precision 32|64    # where applicable

Claude can:
	•	refactor argument parsing
	•	add safety checks
	•	improve help messages
	•	but must keep core semantics compatible.

⸻

5.2 CSV Output Schema

Each segment produces a summary row like:

segment, theta_start, theta_end, shell_config, shell_seen,
total, primes, density, avg_shell, pop_range, avg_pop,
first_prime, last_prime, twins, mod6_1, mod6_5, time_ms

Claude can extend or reorganize columns, but should:
	•	keep shell_config and shell_seen semantics
	•	keep enough info to validate “mixed shells per segment”

⸻

PART 6: ACCEPTANCE & VALIDATION TESTS

6.1 Theta Order Validation

Compile:

nvcc -O3 -arch=sm_86 theta_order_complete.cu -o theta_order_complete

Run:

./theta_order_complete scan32 --theta-exp 16 --segments 4 --output test.csv

Check test.csv:
	•	Good: shell_seen = 2-31 (or similar, meaning a range of shells)
	•	Bad: shell_seen = 20-20 (single shell: indicates natural-order-like behavior)

6.2 Bit Reversal Consistency

Implement a reference test:

void verify_bit_reverse() {
    for (int k = 1; k <= 31; ++k) {
        uint32_t limit = 1u << k;
        for (uint32_t x = 0; x < limit; ++x) {
            uint32_t ref = bit_reverse_loop(x, k);
            uint32_t fast = bit_reverse_32_fast(x, k);
            if (ref != fast) {
                printf("Mismatch 32: k=%d x=%u ref=%u fast=%u\n", k, x, ref, fast);
                return;
            }
        }
    }
    // similarly for 64-bit
}

Claude must ensure any changes preserve this test.

⸻

PART 7: HOW CLAUDE SHOULD INTERACT WITH THIS CODEBASE
	•	Treat all number-theoretic and geometric definitions as fixed API.
	•	Do not redesign theta mapping, shells, or 2-adic decomposition unless explicitly asked.
	•	Focus on:
	•	correctness,
	•	performance,
	•	code organization,
	•	CUDA best practices,
	•	testing, logging, restartability.

When you (Claude) respond, use:

[OBSERVATION]
Type: (Error | Simplification | Limitation | Optimization | Refactor)
What: ...
Why: ...
Confidence: (high/medium/low)

And provide concrete code (functions, kernels, structs, CLI handling), not abstract math.

⸻
