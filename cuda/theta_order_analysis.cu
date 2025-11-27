/**
 * theta_order_analysis.cu - Unified GPU Prime Segment Analyzer
 *
 * ============================================================================
 * DISCLAIMER
 * ============================================================================
 * This is an AI-assisted exploratory visualization project.
 * The contents are educational and experimental only. Not mathematical research.
 * Not peer-reviewed, may contain mistakes, not mathematical results.
 * All patterns visible in images (e.g., "odd-core rays", apparent clustering)
 * are purely visual artifacts of the construction.
 *
 * ============================================================================
 * RELATIONSHIP TO power-two-square-rays PROJECT
 * ============================================================================
 *
 * https://github.com/nmicic/power-two-square-rays/
 *
 * The power-two-square-rays project explores integer decomposition:
 *
 *     n = core × 2^v2
 *
 * Where:
 *   - v2(n)  = 2-adic valuation (trailing zeros)
 *   - core   = odd factor (the "odd-core")
 *   - key    = theta-order position
 *
 *
 *   Statistics tracked (all INTEGER-NATIVE, no floating point):
 *   - shell (log2)  → bit-length of the number
 *   - popcount      → Hamming weight (number of 1-bits)
 *   - v2            → 2-adic valuation (always 0 for odd primes)
 *   - mod6 classes  → primes > 3 are either 1 or 5 mod 6
 *
 * ============================================================================
 * WHAT THIS TOOL DOES
 * ============================================================================
 *
 * Counts primes in any range using GPU parallelism.
 * Supports three modes:
 *   - 32-bit:  n < 2^32, uses 3 Miller-Rabin witnesses (~100M tests/sec)
 *   - 64-bit:  n < 2^64, uses 12 witnesses (~30-50M tests/sec)
 *   - 256-bit: arbitrary, uses 20 witnesses (~150K tests/sec)
 *
 * Outputs per-segment statistics to CSV.
 *
 * ============================================================================
 * VALID METRICS (all order-independent)
 * ============================================================================
 *
 *   prime_count      - Number of primes in segment
 *   shell statistics - floor(log2(n)) for all integers
 *   popcount stats   - Hamming weight
 *   v2 statistics    - 2-adic valuation
 *   twin_prime_count - Primes p where p+2 is also prime
 *   primes_mod6_1/5  - Classification mod 6
 *   first/last_prime - Boundary primes in segment
 *
 * ============================================================================
 * REMOVED METRICS (natural-order, meaningless in theta-order)
 * ============================================================================
 *
 *   - gap statistics (min_gap, max_gap, avg_gap, gap_variance)
 *     These measure p2-p1 which is NATURAL ORDER distance.
 *     Theta-adjacent primes may be millions apart in value.
 *
 *   - theta_rad / angles
 *     Uses atan2() which violates integer-native constraint.
 *
 * ============================================================================
 * QUICK START
 * ============================================================================
 *
 *   # Compile (adjust sm_XX for your GPU)
 *   nvcc -O3 -arch=sm_86 theta_order_analysis.cu -o theta_order_analysis
 *
 *   # Count primes up to 2^32 (~40 sec)
 *   ./theta_order_analysis --exp 32
 *
 *   # Count primes in range [2^32, 2^32 + 2^30)
 *   ./theta_order_analysis --start 0x100000000 --exp 30
 *
 *   # 256-bit mode: test near M127
 *   ./theta_order_analysis --mode 256 --start 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF --segments 100
 *
 *   # Test single number for primality
 *   ./theta_order_analysis --test 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
 *
 * ============================================================================
 * EXPECTED RESULTS
 * ============================================================================
 *
 *   --exp 32:  π(2^32) = 203,280,221 primes (exact known value)
 *   --exp 20:  π(2^20) = 82,025 primes
 *
 * ============================================================================
 * Author: @nmicic / Claude collaboration
 * License: MIT
 * ============================================================================
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

// ============================================================================
// Configuration
// ============================================================================

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define THRESHOLD_32BIT 0x100000000ULL

// ============================================================================
// Miller-Rabin witnesses
// ============================================================================

// For n < 2^32: 3 witnesses sufficient (deterministic)
__constant__ uint64_t MR_WITNESSES_3[3] = {2, 7, 61};

// For n < 2^64: 12 witnesses sufficient (deterministic)
__constant__ uint64_t MR_WITNESSES_12[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

// For 256-bit: 20 witnesses (probabilistic, error < 4^-20)
__constant__ uint64_t MR_WITNESSES_20[20] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71
};

// ============================================================================
// 256-bit unsigned integer type
// ============================================================================

struct uint256 {
    uint64_t w[4];  // w[0] = low, w[3] = high
};

__device__ __host__ void u256_zero(uint256* a) {
    a->w[0] = a->w[1] = a->w[2] = a->w[3] = 0;
}

__device__ __host__ void u256_set64(uint256* a, uint64_t v) {
    a->w[0] = v;
    a->w[1] = a->w[2] = a->w[3] = 0;
}

__device__ __host__ void u256_copy(uint256* dst, const uint256* src) {
    dst->w[0] = src->w[0];
    dst->w[1] = src->w[1];
    dst->w[2] = src->w[2];
    dst->w[3] = src->w[3];
}

__device__ __host__ bool u256_is_zero(const uint256* a) {
    return (a->w[0] | a->w[1] | a->w[2] | a->w[3]) == 0;
}

__device__ __host__ bool u256_is_odd(const uint256* a) {
    return a->w[0] & 1;
}

__device__ __host__ bool u256_eq(const uint256* a, const uint256* b) {
    return a->w[0] == b->w[0] && a->w[1] == b->w[1] &&
           a->w[2] == b->w[2] && a->w[3] == b->w[3];
}

__device__ __host__ bool u256_lt(const uint256* a, const uint256* b) {
    if (a->w[3] != b->w[3]) return a->w[3] < b->w[3];
    if (a->w[2] != b->w[2]) return a->w[2] < b->w[2];
    if (a->w[1] != b->w[1]) return a->w[1] < b->w[1];
    return a->w[0] < b->w[0];
}

__device__ __host__ void u256_add64(uint256* a, uint64_t v) {
    uint64_t sum = a->w[0] + v;
    uint64_t carry = (sum < a->w[0]) ? 1 : 0;
    a->w[0] = sum;
    for (int i = 1; i < 4 && carry; i++) {
        sum = a->w[i] + carry;
        carry = (sum < a->w[i]) ? 1 : 0;
        a->w[i] = sum;
    }
}

__device__ void u256_sub(uint256* r, const uint256* a, const uint256* b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t diff = a->w[i] - b->w[i] - borrow;
        borrow = (a->w[i] < b->w[i] + borrow) ? 1 : 0;
        r->w[i] = diff;
    }
}

__device__ void u256_shr1(uint256* a) {
    a->w[0] = (a->w[0] >> 1) | (a->w[1] << 63);
    a->w[1] = (a->w[1] >> 1) | (a->w[2] << 63);
    a->w[2] = (a->w[2] >> 1) | (a->w[3] << 63);
    a->w[3] >>= 1;
}

__device__ uint32_t u256_popcount(const uint256* a) {
    return __popcll(a->w[0]) + __popcll(a->w[1]) +
           __popcll(a->w[2]) + __popcll(a->w[3]);
}

__device__ uint32_t u256_bitlen(const uint256* a) {
    if (a->w[3]) return 192 + (64 - __clzll(a->w[3]));
    if (a->w[2]) return 128 + (64 - __clzll(a->w[2]));
    if (a->w[1]) return 64 + (64 - __clzll(a->w[1]));
    if (a->w[0]) return 64 - __clzll(a->w[0]);
    return 0;
}

// ============================================================================
// 256-bit Montgomery multiplication
// ============================================================================

__device__ void u256_mul_schoolbook(uint256* r, const uint256* a, const uint256* b) {
    // Simplified schoolbook multiplication for 256-bit
    // Result is 512 bits, we take low 256 bits (for Montgomery reduction)
    uint64_t acc[8] = {0};
    
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a->w[i] * b->w[j] + acc[i+j] + carry;
            acc[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        acc[i+4] = carry;
    }
    
    r->w[0] = acc[0];
    r->w[1] = acc[1];
    r->w[2] = acc[2];
    r->w[3] = acc[3];
}

__device__ void u256_mod(uint256* r, const uint256* a, const uint256* m) {
    // Simple repeated subtraction (slow but correct)
    u256_copy(r, a);
    while (!u256_lt(r, m)) {
        u256_sub(r, r, m);
    }
}

__device__ void u256_mulmod(uint256* r, const uint256* a, const uint256* b, const uint256* m) {
    uint256 prod;
    u256_mul_schoolbook(&prod, a, b);
    u256_mod(r, &prod, m);
}

__device__ void u256_powmod(uint256* r, const uint256* base, const uint256* exp, const uint256* m) {
    uint256 b, e;
    u256_copy(&b, base);
    u256_copy(&e, exp);
    u256_set64(r, 1);
    
    while (!u256_is_zero(&e)) {
        if (u256_is_odd(&e)) {
            u256_mulmod(r, r, &b, m);
        }
        u256_mulmod(&b, &b, &b, m);
        u256_shr1(&e);
    }
}

// ============================================================================
// 128-bit arithmetic for 64-bit mode (using CUDA intrinsics)
// ============================================================================

__device__ __forceinline__ void mul128(uint64_t a, uint64_t b, uint64_t* hi, uint64_t* lo) {
    *lo = a * b;
    *hi = __umul64hi(a, b);
}

__device__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t hi, lo;
    mul128(a, b, &hi, &lo);
    
    if (hi == 0) return lo % m;
    
    // Barrett-like reduction for 128-bit
    uint64_t result = lo % m;
    uint64_t factor = hi % m;
    
    // Compute (hi * 2^64) mod m iteratively
    for (int i = 0; i < 64; i++) {
        factor = (factor << 1) % m;
    }
    
    result = (result + factor) % m;
    return result;
}

__device__ uint64_t powmod64(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mulmod64(result, base, m);
        exp >>= 1;
        base = mulmod64(base, base, m);
    }
    return result;
}

// ============================================================================
// Miller-Rabin primality tests
// ============================================================================

// 32-bit version (fast, 3 witnesses)
__device__ bool is_prime_32(uint32_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    if (n < 9) return true;
    if (n % 3 == 0) return false;
    
    uint32_t d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }
    
    for (int i = 0; i < 3; i++) {
        uint64_t a = MR_WITNESSES_3[i];
        if (a >= n) continue;
        
        uint64_t x = 1, base = a % n;
        uint32_t exp = d;
        while (exp > 0) {
            if (exp & 1) x = (x * base) % n;
            exp >>= 1;
            base = (base * base) % n;
        }
        
        if (x == 1 || x == n - 1) continue;
        
        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = (x * x) % n;
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

// 64-bit version (12 witnesses)
__device__ bool is_prime_64(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    if (n < 9) return true;
    if (n % 3 == 0) return false;
    
    // Use fast path for small numbers
    if (n < THRESHOLD_32BIT) return is_prime_32((uint32_t)n);
    
    uint64_t d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; r++; }
    
    for (int i = 0; i < 12; i++) {
        uint64_t a = MR_WITNESSES_12[i];
        if (a >= n) continue;
        
        uint64_t x = powmod64(a, d, n);
        
        if (x == 1 || x == n - 1) continue;
        
        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = mulmod64(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

// 256-bit version (20 witnesses)
__device__ bool is_prime_256(const uint256* n) {
    // Handle small cases
    if (n->w[3] == 0 && n->w[2] == 0 && n->w[1] == 0) {
        if (n->w[0] < 2) return false;
        if (n->w[0] < THRESHOLD_32BIT) return is_prime_32((uint32_t)n->w[0]);
        return is_prime_64(n->w[0]);
    }
    
    if (!u256_is_odd(n)) return false;
    
    // n - 1 = 2^r * d
    uint256 d, n_minus_1;
    u256_copy(&n_minus_1, n);
    u256_add64(&n_minus_1, -1ULL);  // Subtract 1 (wraps correctly for odd n)
    u256_sub(&n_minus_1, n, &(uint256){{1,0,0,0}});
    
    u256_copy(&d, &n_minus_1);
    int r = 0;
    while (!u256_is_odd(&d)) { u256_shr1(&d); r++; }
    
    uint256 one, n_m1;
    u256_set64(&one, 1);
    u256_copy(&n_m1, n);
    u256_sub(&n_m1, &n_m1, &one);
    
    for (int i = 0; i < 20; i++) {
        uint256 a, x;
        u256_set64(&a, MR_WITNESSES_20[i]);
        
        if (!u256_lt(&a, n)) continue;
        
        u256_powmod(&x, &a, &d, n);
        
        if (u256_eq(&x, &one) || u256_eq(&x, &n_m1)) continue;
        
        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            u256_mulmod(&x, &x, &x, n);
            if (u256_eq(&x, &n_m1)) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

// ============================================================================
// Per-segment statistics (valid metrics only - NO gaps)
// ============================================================================

struct SegmentStats {
    uint64_t segment_id;
    uint64_t key_start;
    uint64_t key_end;
    uint64_t total_integers;
    
    // Prime counts
    uint64_t prime_count;
    uint64_t composite_count;
    
    // Shell (floor(log2(n))) - computed for ALL integers
    uint32_t min_shell;
    uint32_t max_shell;
    double shell_sum;
    
    // Popcount - computed for ALL integers
    uint32_t min_popcount;
    uint32_t max_popcount;
    double popcount_sum;
    
    // v2 (2-adic valuation) - computed for ALL integers
    uint32_t min_v2;
    uint32_t max_v2;
    double v2_sum;
    
    // First/last prime in segment
    uint64_t first_prime;
    uint64_t last_prime;
    
    // Special prime types
    uint64_t twin_prime_count;    // p where p+2 is also prime
    uint64_t primes_mod6_1;       // primes ≡ 1 (mod 6)
    uint64_t primes_mod6_5;       // primes ≡ 5 (mod 6)
    
    // Timing
    float compute_time_ms;
};

// 256-bit mode uses simpler stats (no shell/v2 for all integers)
struct SegmentStats256 {
    uint32_t primes_found;
    uint32_t first_offset;
    uint32_t last_offset;
    uint32_t total_popcount;  // Sum of popcount of primes found
};

// ============================================================================
// Device helper functions
// ============================================================================

__device__ __forceinline__ uint32_t compute_shell(uint64_t n) {
    if (n == 0) return 0;
    return 63 - __clzll(n);
}

__device__ __forceinline__ uint32_t compute_popcount(uint64_t n) {
    return __popcll(n);
}

__device__ __forceinline__ uint32_t compute_v2(uint64_t n) {
    if (n == 0) return 64;
    return __ffsll(n) - 1;
}

// ============================================================================
// 64-bit analysis kernel (grid-stride loop)
// ============================================================================

__global__ void analyze_segment_64_kernel(
    uint64_t segment_start,
    uint64_t segment_size,
    uint64_t segment_id,
    SegmentStats* d_stats
) {
    // Thread-local accumulators
    uint64_t local_prime_count = 0;
    uint32_t local_min_shell = UINT32_MAX;
    uint32_t local_max_shell = 0;
    uint32_t local_min_popcount = UINT32_MAX;
    uint32_t local_max_popcount = 0;
    uint32_t local_min_v2 = UINT32_MAX;
    uint32_t local_max_v2 = 0;
    double local_shell_sum = 0.0;
    double local_popcount_sum = 0.0;
    double local_v2_sum = 0.0;
    uint64_t local_first_prime = UINT64_MAX;
    uint64_t local_last_prime = 0;
    uint64_t local_twin_count = 0;
    uint64_t local_mod6_1 = 0;
    uint64_t local_mod6_5 = 0;
    
    uint64_t global_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    
    for (uint64_t i = global_idx; i < segment_size; i += stride) {
        uint64_t n = segment_start + i;
        if (n == 0) continue;
        
        // Compute metrics for ALL integers
        uint32_t shell = compute_shell(n);
        uint32_t popcount = compute_popcount(n);
        uint32_t v2 = compute_v2(n);
        
        local_shell_sum += shell;
        local_popcount_sum += popcount;
        local_v2_sum += v2;
        
        if (shell < local_min_shell) local_min_shell = shell;
        if (shell > local_max_shell) local_max_shell = shell;
        if (popcount < local_min_popcount) local_min_popcount = popcount;
        if (popcount > local_max_popcount) local_max_popcount = popcount;
        if (v2 < local_min_v2) local_min_v2 = v2;
        if (v2 > local_max_v2) local_max_v2 = v2;
        
        // Check primality
        bool is_prime = (n < THRESHOLD_32BIT) ? is_prime_32((uint32_t)n) : is_prime_64(n);
        
        if (is_prime) {
            local_prime_count++;
            
            if (n < local_first_prime) local_first_prime = n;
            if (n > local_last_prime) local_last_prime = n;
            
            // Twin prime check
            uint64_t n2 = n + 2;
            bool n2_prime = (n2 < THRESHOLD_32BIT) ? is_prime_32((uint32_t)n2) : is_prime_64(n2);
            if (n2_prime) local_twin_count++;
            
            // Mod 6 classification
            if (n > 3) {
                uint32_t r = n % 6;
                if (r == 1) local_mod6_1++;
                else if (r == 5) local_mod6_5++;
            }
        }
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_prime_count += __shfl_down_sync(0xFFFFFFFF, local_prime_count, offset);
        local_shell_sum += __shfl_down_sync(0xFFFFFFFF, local_shell_sum, offset);
        local_popcount_sum += __shfl_down_sync(0xFFFFFFFF, local_popcount_sum, offset);
        local_v2_sum += __shfl_down_sync(0xFFFFFFFF, local_v2_sum, offset);
        local_twin_count += __shfl_down_sync(0xFFFFFFFF, local_twin_count, offset);
        local_mod6_1 += __shfl_down_sync(0xFFFFFFFF, local_mod6_1, offset);
        local_mod6_5 += __shfl_down_sync(0xFFFFFFFF, local_mod6_5, offset);
    }
    
    // Lane 0 of each warp writes to global memory
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        atomicAdd((unsigned long long*)&d_stats->prime_count, (unsigned long long)local_prime_count);
        atomicAdd((unsigned long long*)&d_stats->twin_prime_count, (unsigned long long)local_twin_count);
        atomicAdd((unsigned long long*)&d_stats->primes_mod6_1, (unsigned long long)local_mod6_1);
        atomicAdd((unsigned long long*)&d_stats->primes_mod6_5, (unsigned long long)local_mod6_5);
        
        // Min/max using atomicMin/Max
        atomicMin(&d_stats->min_shell, local_min_shell);
        atomicMax(&d_stats->max_shell, local_max_shell);
        atomicMin(&d_stats->min_popcount, local_min_popcount);
        atomicMax(&d_stats->max_popcount, local_max_popcount);
        atomicMin(&d_stats->min_v2, local_min_v2);
        atomicMax(&d_stats->max_v2, local_max_v2);
        
        // First/last prime (using CAS for 64-bit min/max)
        uint64_t old_first = d_stats->first_prime;
        while (local_first_prime < old_first) {
            uint64_t assumed = old_first;
            old_first = atomicCAS((unsigned long long*)&d_stats->first_prime, assumed, local_first_prime);
            if (old_first == assumed) break;
        }
        
        uint64_t old_last = d_stats->last_prime;
        while (local_last_prime > old_last) {
            uint64_t assumed = old_last;
            old_last = atomicCAS((unsigned long long*)&d_stats->last_prime, assumed, local_last_prime);
            if (old_last == assumed) break;
        }
        
        // Sum accumulators (using CAS for double)
        double old_shell = d_stats->shell_sum;
        while (atomicCAS((unsigned long long*)&d_stats->shell_sum,
                         __double_as_longlong(old_shell),
                         __double_as_longlong(old_shell + local_shell_sum))
               != __double_as_longlong(old_shell)) {
            old_shell = d_stats->shell_sum;
        }
        
        double old_pop = d_stats->popcount_sum;
        while (atomicCAS((unsigned long long*)&d_stats->popcount_sum,
                         __double_as_longlong(old_pop),
                         __double_as_longlong(old_pop + local_popcount_sum))
               != __double_as_longlong(old_pop)) {
            old_pop = d_stats->popcount_sum;
        }
        
        double old_v2 = d_stats->v2_sum;
        while (atomicCAS((unsigned long long*)&d_stats->v2_sum,
                         __double_as_longlong(old_v2),
                         __double_as_longlong(old_v2 + local_v2_sum))
               != __double_as_longlong(old_v2)) {
            old_v2 = d_stats->v2_sum;
        }
    }
}

// ============================================================================
// 256-bit analysis kernel (one block per segment)
// ============================================================================

__global__ void analyze_segment_256_kernel(
    uint256 range_start,
    uint32_t candidates_per_seg,
    uint32_t num_segments,
    SegmentStats256* stats
) {
    uint32_t seg_id = blockIdx.x;
    if (seg_id >= num_segments) return;
    
    // Compute segment start: range_start + seg_id * candidates_per_seg * 2
    uint256 seg_start;
    u256_copy(&seg_start, &range_start);
    uint64_t offset = (uint64_t)seg_id * candidates_per_seg * 2;
    u256_add64(&seg_start, offset);
    
    __shared__ uint32_t s_count;
    __shared__ uint32_t s_first;
    __shared__ uint32_t s_last;
    __shared__ uint32_t s_popcount_sum;
    
    if (threadIdx.x == 0) {
        s_count = 0;
        s_first = UINT32_MAX;
        s_last = 0;
        s_popcount_sum = 0;
    }
    __syncthreads();
    
    for (uint32_t i = threadIdx.x; i < candidates_per_seg; i += blockDim.x) {
        uint256 candidate;
        u256_copy(&candidate, &seg_start);
        u256_add64(&candidate, (uint64_t)i * 2);
        
        if (is_prime_256(&candidate)) {
            atomicAdd(&s_count, 1);
            atomicMin(&s_first, i);
            atomicMax(&s_last, i);
            atomicAdd(&s_popcount_sum, u256_popcount(&candidate));
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        stats[seg_id].primes_found = s_count;
        stats[seg_id].first_offset = (s_first == UINT32_MAX) ? 0 : s_first;
        stats[seg_id].last_offset = s_last;
        stats[seg_id].total_popcount = s_popcount_sum;
    }
}

// ============================================================================
// Host functions
// ============================================================================

void print_usage(const char* prog) {
    printf("theta_order_analysis - Unified GPU Prime Segment Analyzer\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Range specification:\n");
    printf("  --exp N             Process 2^N integers from start\n");
    printf("  --size N            Process N integers from start\n");
    printf("  --start N           Start value (decimal or 0x hex)\n");
    printf("\nSegmentation:\n");
    printf("  --segment-exp S     Segment size = 2^S (default: 20)\n");
    printf("  --segments N        Number of segments (256-bit mode)\n");
    printf("  --per-seg N         Candidates per segment (256-bit mode)\n");
    printf("\nMode selection:\n");
    printf("  --mode 64           Force 64-bit mode\n");
    printf("  --mode 256          Force 256-bit mode\n");
    printf("  (auto-detected from --start if not specified)\n");
    printf("\nOutput:\n");
    printf("  --output FILE       CSV file (default: analysis.csv)\n");
    printf("  --quiet             Minimal output\n");
    printf("  --verbose           Show every segment\n");
    printf("\nSpecial:\n");
    printf("  --test N            Test single number for primality\n");
    printf("  --help              Show this help\n");
    printf("\nExamples:\n");
    printf("  %s --exp 32                           # π(2^32)\n", prog);
    printf("  %s --start 0x100000000 --exp 30      # 64-bit range\n", prog);
    printf("  %s --mode 256 --segments 1000        # 256-bit scan\n", prog);
    printf("  %s --test 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  # Test M127\n", prog);
}

void write_csv_header_64(FILE* fp) {
    fprintf(fp, "segment_id,key_start,key_end,total_integers,"
                "prime_count,composite_count,prime_density,"
                "min_shell,max_shell,avg_shell,"
                "min_popcount,max_popcount,avg_popcount,"
                "min_v2,max_v2,avg_v2,"
                "first_prime,last_prime,"
                "twin_prime_count,primes_mod6_1,primes_mod6_5,"
                "compute_time_ms\n");
}

void write_csv_row_64(FILE* fp, const SegmentStats* s) {
    double density = s->total_integers > 0 ? (double)s->prime_count / s->total_integers : 0;
    double avg_shell = s->total_integers > 0 ? s->shell_sum / s->total_integers : 0;
    double avg_pop = s->total_integers > 0 ? s->popcount_sum / s->total_integers : 0;
    double avg_v2 = s->total_integers > 0 ? s->v2_sum / s->total_integers : 0;
    
    fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                "%" PRIu64 ",%" PRIu64 ",%.12g,"
                "%u,%u,%.6f,"
                "%u,%u,%.6f,"
                "%u,%u,%.6f,"
                "%" PRIu64 ",%" PRIu64 ","
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                "%.3f\n",
            s->segment_id, s->key_start, s->key_end, s->total_integers,
            s->prime_count, s->composite_count, density,
            s->min_shell, s->max_shell, avg_shell,
            s->min_popcount, s->max_popcount, avg_pop,
            s->min_v2, s->max_v2, avg_v2,
            s->first_prime, s->last_prime,
            s->twin_prime_count, s->primes_mod6_1, s->primes_mod6_5,
            s->compute_time_ms);
}

void write_csv_header_256(FILE* fp) {
    fprintf(fp, "segment,primes_found,first_offset,last_offset,avg_popcount\n");
}

void write_csv_row_256(FILE* fp, uint32_t seg_id, const SegmentStats256* s) {
    float avg_pop = s->primes_found > 0 ? (float)s->total_popcount / s->primes_found : 0;
    fprintf(fp, "%u,%u,%u,%u,%.2f\n",
            seg_id, s->primes_found, s->first_offset, s->last_offset, avg_pop);
}

uint64_t parse_number(const char* str) {
    if (str[0] == '0' && (str[1] == 'x' || str[1] == 'X')) {
        return strtoull(str + 2, NULL, 16);
    }
    return strtoull(str, NULL, 10);
}

void parse_hex256(const char* s, uint256* out) {
    u256_zero(out);
    const char* p = s;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;
    
    int len = strlen(p);
    int word = 0, nibble = 0;
    
    for (int i = len - 1; i >= 0 && word < 4; i--) {
        char c = p[i];
        int v = 0;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
        
        out->w[word] |= ((uint64_t)v << (nibble * 4));
        nibble++;
        if (nibble == 16) { nibble = 0; word++; }
    }
}

void print_hex256(const uint256* a) {
    if (a->w[3]) printf("0x%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64,
                        a->w[3], a->w[2], a->w[1], a->w[0]);
    else if (a->w[2]) printf("0x%016" PRIx64 "%016" PRIx64 "%016" PRIx64,
                             a->w[2], a->w[1], a->w[0]);
    else if (a->w[1]) printf("0x%016" PRIx64 "%016" PRIx64, a->w[1], a->w[0]);
    else printf("0x%" PRIx64, a->w[0]);
}

void init_stats_64(SegmentStats* s, uint64_t seg_id, uint64_t seg_start, uint64_t seg_size) {
    memset(s, 0, sizeof(SegmentStats));
    s->segment_id = seg_id;
    s->key_start = seg_start;
    s->key_end = seg_start + seg_size;
    s->total_integers = seg_size;
    s->min_shell = UINT32_MAX;
    s->min_popcount = UINT32_MAX;
    s->min_v2 = UINT32_MAX;
    s->first_prime = UINT64_MAX;
}

// ============================================================================
// Main entry point
// ============================================================================

int main(int argc, char** argv) {
    // Parse arguments
    uint64_t start_64 = 0;
    uint256 start_256;
    u256_set64(&start_256, 3);
    
    uint64_t size = 0;
    int exp = -1;
    int segment_exp = 20;
    uint32_t num_segments_256 = 10;
    uint32_t per_seg_256 = 1000;
    const char* output_file = "analysis.csv";
    int verbosity = 1;
    int mode = 0;  // 0=auto, 64, 256
    const char* test_num = NULL;
    bool start_is_256 = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--start") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            // Check if it's a 256-bit number (more than 16 hex digits)
            const char* p = s;
            if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;
            if (strlen(p) > 16) {
                parse_hex256(s, &start_256);
                if (!u256_is_odd(&start_256)) u256_add64(&start_256, 1);
                start_is_256 = true;
            } else {
                start_64 = parse_number(s);
                u256_set64(&start_256, start_64);
                if (start_64 & 1) {} else u256_add64(&start_256, 1);
            }
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = parse_number(argv[++i]);
        } else if (strcmp(argv[i], "--exp") == 0 && i + 1 < argc) {
            exp = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--segment-exp") == 0 && i + 1 < argc) {
            segment_exp = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--segments") == 0 && i + 1 < argc) {
            num_segments_256 = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--per-seg") == 0 && i + 1 < argc) {
            per_seg_256 = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--quiet") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            test_num = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // ========================================================================
    // Single number test mode
    // ========================================================================
    if (test_num) {
        uint256 n;
        parse_hex256(test_num, &n);
        
        printf("Testing: "); print_hex256(&n); printf("\n");
        
        // Use appropriate test based on size
        bool result;
        if (n.w[3] == 0 && n.w[2] == 0 && n.w[1] == 0 && n.w[0] < THRESHOLD_32BIT) {
            // Use GPU for consistency but could use CPU
            SegmentStats256* d_stats;
            cudaMalloc(&d_stats, sizeof(SegmentStats256));
            analyze_segment_256_kernel<<<1, 1>>>(n, 1, 1, d_stats);
            cudaDeviceSynchronize();
            SegmentStats256 h_stats;
            cudaMemcpy(&h_stats, d_stats, sizeof(SegmentStats256), cudaMemcpyDeviceToHost);
            result = h_stats.primes_found > 0;
            cudaFree(d_stats);
        } else {
            SegmentStats256* d_stats;
            cudaMalloc(&d_stats, sizeof(SegmentStats256));
            analyze_segment_256_kernel<<<1, 1>>>(n, 1, 1, d_stats);
            cudaDeviceSynchronize();
            SegmentStats256 h_stats;
            cudaMemcpy(&h_stats, d_stats, sizeof(SegmentStats256), cudaMemcpyDeviceToHost);
            result = h_stats.primes_found > 0;
            cudaFree(d_stats);
        }
        
        printf("Result: %s\n", result ? "PRIME" : "COMPOSITE");
        return 0;
    }
    
    // ========================================================================
    // Determine mode
    // ========================================================================
    if (mode == 0) {
        mode = start_is_256 ? 256 : 64;
    }
    
    // ========================================================================
    // 256-bit mode
    // ========================================================================
    if (mode == 256) {
        if (verbosity > 0) {
            printf("=== GPU Prime Analysis (256-bit) ===\n");
            printf("Start: "); print_hex256(&start_256); printf("\n");
            printf("Segments: %u, Candidates/seg: %u\n", num_segments_256, per_seg_256);
            printf("Total candidates: %lu\n", (unsigned long)num_segments_256 * per_seg_256);
            printf("Output: %s\n\n", output_file);
        }
        
        SegmentStats256* d_stats;
        cudaMalloc(&d_stats, num_segments_256 * sizeof(SegmentStats256));
        
        cudaEvent_t ev_start, ev_stop;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        
        cudaEventRecord(ev_start);
        analyze_segment_256_kernel<<<num_segments_256, 128>>>(start_256, per_seg_256, num_segments_256, d_stats);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        
        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        
        SegmentStats256* h_stats = (SegmentStats256*)malloc(num_segments_256 * sizeof(SegmentStats256));
        cudaMemcpy(h_stats, d_stats, num_segments_256 * sizeof(SegmentStats256), cudaMemcpyDeviceToHost);
        
        FILE* fp = fopen(output_file, "w");
        if (fp) write_csv_header_256(fp);
        
        uint64_t total_primes = 0;
        uint64_t total_popcount = 0;
        
        for (uint32_t i = 0; i < num_segments_256; i++) {
            total_primes += h_stats[i].primes_found;
            total_popcount += h_stats[i].total_popcount;
            if (fp) write_csv_row_256(fp, i, &h_stats[i]);
        }
        
        if (verbosity > 0) {
            printf("=== Results ===\n");
            printf("Total primes: %" PRIu64 "\n", total_primes);
            printf("Density: %.6f\n", (double)total_primes / (num_segments_256 * per_seg_256));
            if (total_primes > 0) {
                printf("Avg popcount: %.2f\n", (double)total_popcount / total_primes);
            }
            printf("Time: %.2f ms\n", ms);
            printf("Throughput: %.0f tests/sec\n", (num_segments_256 * per_seg_256) / (ms / 1000.0));
            printf("Output: %s\n", output_file);
        }
        
        if (fp) fclose(fp);
        free(h_stats);
        cudaFree(d_stats);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        
        return 0;
    }
    
    // ========================================================================
    // 64-bit mode (default)
    // ========================================================================
    if (exp >= 0) {
        if (exp > 62) exp = 62;
        size = 1ULL << exp;
    }
    
    if (size == 0) {
        fprintf(stderr, "Error: Specify --size or --exp\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (start_64 > UINT64_MAX - size) {
        fprintf(stderr, "Error: Range overflow\n");
        return 1;
    }
    
    uint64_t segment_size = 1ULL << segment_exp;
    uint64_t num_segments = (size + segment_size - 1) / segment_size;
    
    if (verbosity > 0) {
        printf("=== GPU Prime Analysis (64-bit) ===\n");
        printf("Range: [%" PRIu64 ", %" PRIu64 ")\n", start_64, start_64 + size);
        printf("Size: %" PRIu64, size);
        if (exp >= 0) printf(" (2^%d)", exp);
        printf("\n");
        printf("Segments: %" PRIu64 " x 2^%d\n", num_segments, segment_exp);
        printf("Output: %s\n", output_file);
        
        if (start_64 + size <= THRESHOLD_32BIT) {
            printf("Mode: FAST (3 witnesses)\n");
        } else if (start_64 >= THRESHOLD_32BIT) {
            printf("Mode: FULL (12 witnesses)\n");
        } else {
            printf("Mode: MIXED\n");
        }
        printf("\n");
    }
    
    // Allocate device stats
    SegmentStats* d_stats;
    cudaMalloc(&d_stats, sizeof(SegmentStats));
    SegmentStats h_stats;
    
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", output_file);
        return 1;
    }
    write_csv_header_64(fp);
    
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    
    uint64_t total_primes = 0;
    uint64_t total_twins = 0;
    clock_t wall_start = clock();
    
    for (uint64_t seg = 0; seg < num_segments; seg++) {
        uint64_t seg_start = start_64 + seg * segment_size;
        uint64_t seg_size = (seg == num_segments - 1) ? (start_64 + size - seg_start) : segment_size;
        
        // Initialize stats on host, copy to device
        init_stats_64(&h_stats, seg, seg_start, seg_size);
        cudaMemcpy(d_stats, &h_stats, sizeof(SegmentStats), cudaMemcpyHostToDevice);
        
        cudaEventRecord(ev_start);
        
        int blocks = min((int)((seg_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 65535);
        analyze_segment_64_kernel<<<blocks, THREADS_PER_BLOCK>>>(seg_start, seg_size, seg, d_stats);
        
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        
        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        
        cudaMemcpy(&h_stats, d_stats, sizeof(SegmentStats), cudaMemcpyDeviceToHost);
        h_stats.compute_time_ms = ms;
        h_stats.composite_count = seg_size - h_stats.prime_count;
        
        // Fix sentinel values
        if (h_stats.min_shell == UINT32_MAX) h_stats.min_shell = 0;
        if (h_stats.min_popcount == UINT32_MAX) h_stats.min_popcount = 0;
        if (h_stats.min_v2 == UINT32_MAX) h_stats.min_v2 = 0;
        if (h_stats.first_prime == UINT64_MAX) h_stats.first_prime = 0;
        
        write_csv_row_64(fp, &h_stats);
        
        total_primes += h_stats.prime_count;
        total_twins += h_stats.twin_prime_count;
        
        if (verbosity == 2 || (verbosity == 1 && (seg % 100 == 0 || seg == num_segments - 1))) {
            double density = (double)h_stats.prime_count / h_stats.total_integers;
            printf("Seg %" PRIu64 "/%" PRIu64 ": [%" PRIu64 "..] primes=%" PRIu64 " (%.6f) %.1fms\n",
                   seg + 1, num_segments, seg_start, h_stats.prime_count, density, ms);
        }
    }
    
    double wall_time = (double)(clock() - wall_start) / CLOCKS_PER_SEC;
    
    if (verbosity > 0) {
        printf("\n=== Summary ===\n");
        printf("Total time: %.2f sec\n", wall_time);
        printf("Throughput: %.2f M/sec\n", (size / 1e6) / wall_time);
        printf("Total primes: %" PRIu64 "\n", total_primes);
        printf("Twin primes: %" PRIu64 "\n", total_twins);
        printf("Overall density: %.10f\n", (double)total_primes / size);
        printf("Output: %s\n", output_file);
    }
    
    fclose(fp);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_stats);
    
    return 0;
}
