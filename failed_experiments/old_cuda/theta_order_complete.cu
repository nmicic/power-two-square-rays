// ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
// This file remains for reference only.
// Do NOT use in production. No support, no guarantees.
//
// For the current CUDA implementation, see:
//   - cuda/theta_cuda_v1.2.cuh
//   - cuda/theta_cuda_benchmark_v1.2.cu

/**
 * theta_order_complete.cu - Complete Theta Order Prime Analysis Tool
 *
 * ============================================================================
 * OVERVIEW
 * ============================================================================
 *
 * This tool performs TRUE theta order prime analysis at multiple precisions:
 *   - 32-bit mode: shells 2-31, theta positions up to 2^30
 *   - 64-bit mode: shells 2-62, theta positions up to 2^61  
 *   - 256-bit mode: shells 2-254, arbitrary precision (slower)
 *
 * Unlike natural order iteration (consecutive integers), theta order iterates
 * by angular position. Each segment contains integers from MULTIPLE shells.
 *
 * ============================================================================
 * COMPILATION
 * ============================================================================
 *
 *   # For modern GPUs (RTX 30xx, 40xx)
 *   nvcc -O3 -arch=sm_86 theta_order_complete.cu -o theta_order_complete
 *
 *   # For older GPUs (GTX 10xx, 20xx)  
 *   nvcc -O3 -arch=sm_61 theta_order_complete.cu -o theta_order_complete
 *
 *   # Generic (slower but compatible)
 *   nvcc -O3 theta_order_complete.cu -o theta_order_complete
 *
 * ============================================================================
 * USAGE MODES
 * ============================================================================
 *
 * MODE 1: scan32 - Full 32-bit theta scan
 *   Scans theta positions for shells up to 31.
 *   Use this as Phase 1 to find extreme regions.
 *
 * MODE 2: scan64 - Full 64-bit theta scan  
 *   Scans theta positions for shells up to 62.
 *   Can start from arbitrary theta position.
 *
 * MODE 3: zoom64 - Zoom into 32-bit theta region with 64-bit precision
 *   Takes a theta position from scan32 and subdivides it.
 *   Analyzes shells 32-62 within that angular region.
 *
 * MODE 4: batch - Batch mode for long-running analysis
 *   Automatically segments the full range and runs continuously.
 *   Designed for week-long runs with checkpointing.
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
#include <inttypes.h>
#include <float.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define THRESHOLD_32BIT 0x100000000ULL

// ============================================================================
// Miller-Rabin witnesses
// ============================================================================

__constant__ uint64_t MR_WITNESSES_3[3] = {2, 7, 61};
__constant__ uint64_t MR_WITNESSES_12[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
__constant__ uint64_t MR_WITNESSES_20[20] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71
};

// ============================================================================
// 256-bit type for extended precision
// ============================================================================

struct uint256 {
    uint64_t w[4];  // w[0] = low, w[3] = high
};

__device__ __host__ void u256_zero(uint256* a) {
    a->w[0] = a->w[1] = a->w[2] = a->w[3] = 0;
}

__device__ __host__ void u256_set64(uint256* a, uint64_t v) {
    a->w[0] = v; a->w[1] = a->w[2] = a->w[3] = 0;
}

__device__ __host__ void u256_copy(uint256* dst, const uint256* src) {
    dst->w[0] = src->w[0]; dst->w[1] = src->w[1];
    dst->w[2] = src->w[2]; dst->w[3] = src->w[3];
}

__device__ __host__ bool u256_is_zero(const uint256* a) {
    return (a->w[0] | a->w[1] | a->w[2] | a->w[3]) == 0;
}

__device__ __host__ bool u256_is_odd(const uint256* a) {
    return a->w[0] & 1;
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

__device__ __host__ void u256_shl(uint256* a, int bits) {
    if (bits >= 256) { u256_zero(a); return; }
    if (bits >= 192) {
        a->w[3] = a->w[0] << (bits - 192);
        a->w[2] = a->w[1] = a->w[0] = 0;
    } else if (bits >= 128) {
        int s = bits - 128;
        a->w[3] = (a->w[1] << s) | (s ? (a->w[0] >> (64 - s)) : 0);
        a->w[2] = a->w[0] << s;
        a->w[1] = a->w[0] = 0;
    } else if (bits >= 64) {
        int s = bits - 64;
        a->w[3] = (a->w[2] << s) | (s ? (a->w[1] >> (64 - s)) : 0);
        a->w[2] = (a->w[1] << s) | (s ? (a->w[0] >> (64 - s)) : 0);
        a->w[1] = a->w[0] << s;
        a->w[0] = 0;
    } else if (bits > 0) {
        a->w[3] = (a->w[3] << bits) | (a->w[2] >> (64 - bits));
        a->w[2] = (a->w[2] << bits) | (a->w[1] >> (64 - bits));
        a->w[1] = (a->w[1] << bits) | (a->w[0] >> (64 - bits));
        a->w[0] <<= bits;
    }
}

__device__ uint32_t u256_bitlen(const uint256* a) {
    if (a->w[3]) return 192 + (64 - __clzll(a->w[3]));
    if (a->w[2]) return 128 + (64 - __clzll(a->w[2]));
    if (a->w[1]) return 64 + (64 - __clzll(a->w[1]));
    if (a->w[0]) return 64 - __clzll(a->w[0]);
    return 0;
}

__device__ uint32_t u256_popcount(const uint256* a) {
    return __popcll(a->w[0]) + __popcll(a->w[1]) +
           __popcll(a->w[2]) + __popcll(a->w[3]);
}

// ============================================================================
// Bit operations
// ============================================================================

__device__ __host__ __forceinline__ uint64_t bit_reverse_64(uint64_t x, int k) {
    uint64_t result = 0;
    for (int i = 0; i < k; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

__device__ __host__ __forceinline__ uint32_t bit_reverse_32(uint32_t x, int k) {
    uint32_t result = 0;
    for (int i = 0; i < k; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// 256-bit bit reversal for extended precision theta
__device__ __host__ void u256_bit_reverse(uint256* result, const uint256* x, int k) {
    u256_zero(result);
    uint256 temp;
    u256_copy(&temp, x);
    
    for (int i = 0; i < k && i < 256; i++) {
        if (temp.w[0] & 1) {
            // Set bit (k-1-i) in result
            int bit_pos = k - 1 - i;
            int word = bit_pos / 64;
            int bit = bit_pos % 64;
            if (word < 4) result->w[word] |= (1ULL << bit);
        }
        // Shift temp right by 1
        temp.w[0] = (temp.w[0] >> 1) | (temp.w[1] << 63);
        temp.w[1] = (temp.w[1] >> 1) | (temp.w[2] << 63);
        temp.w[2] = (temp.w[2] >> 1) | (temp.w[3] << 63);
        temp.w[3] >>= 1;
    }
}

__device__ __forceinline__ uint32_t compute_shell_64(uint64_t n) {
    if (n == 0) return 0;
    return 63 - __clzll(n);
}

__device__ __forceinline__ uint32_t compute_shell_32(uint32_t n) {
    if (n == 0) return 0;
    return 31 - __clz(n);
}

__device__ __forceinline__ uint32_t compute_popcount_64(uint64_t n) {
    return __popcll(n);
}

// ============================================================================
// Theta to integer conversion
// ============================================================================

__device__ __host__ uint32_t theta_to_odd_32(uint32_t theta_pos, uint32_t shell) {
    if (shell == 0) return (theta_pos == 0) ? 1 : 0;
    if (shell == 1) return (theta_pos == 0) ? 3 : 0;
    
    uint32_t num_odds = 1U << (shell - 1);
    if (theta_pos >= num_odds) return 0;
    
    uint32_t j = bit_reverse_32(theta_pos, shell - 1);
    return (1U << shell) + 2 * j + 1;
}

__device__ __host__ uint64_t theta_to_odd_64(uint64_t theta_pos, uint32_t shell) {
    if (shell == 0) return (theta_pos == 0) ? 1 : 0;
    if (shell == 1) return (theta_pos == 0) ? 3 : 0;
    if (shell > 62) return 0;  // Can't fit in 64 bits
    
    uint64_t num_odds = 1ULL << (shell - 1);
    if (theta_pos >= num_odds) return 0;
    
    uint64_t j = bit_reverse_64(theta_pos, shell - 1);
    return (1ULL << shell) + 2 * j + 1;
}

// 256-bit version for extended precision
__device__ __host__ void theta_to_odd_256(uint256* result, const uint256* theta_pos, uint32_t shell) {
    u256_zero(result);
    
    if (shell == 0) {
        if (u256_is_zero(theta_pos)) result->w[0] = 1;
        return;
    }
    if (shell == 1) {
        if (u256_is_zero(theta_pos)) result->w[0] = 3;
        return;
    }
    
    // j = bit_reverse(theta_pos, shell - 1)
    uint256 j;
    u256_bit_reverse(&j, theta_pos, shell - 1);
    
    // result = (1 << shell) + 2*j + 1
    // = 2^shell + 2*j + 1
    u256_set64(result, 1);
    u256_shl(result, shell);  // 2^shell
    
    uint256 two_j;
    u256_copy(&two_j, &j);
    u256_shl(&two_j, 1);  // 2*j
    
    // Add 2*j
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = result->w[i] + two_j.w[i] + carry;
        carry = (sum < result->w[i]) ? 1 : 0;
        result->w[i] = sum;
    }
    
    // Add 1
    u256_add64(result, 1);
}

// ============================================================================
// Miller-Rabin primality tests
// ============================================================================

__device__ __forceinline__ void mul128(uint64_t a, uint64_t b, uint64_t* hi, uint64_t* lo) {
    *lo = a * b;
    *hi = __umul64hi(a, b);
}

__device__ uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t hi, lo;
    mul128(a, b, &hi, &lo);
    if (hi == 0) return lo % m;
    uint64_t result = lo % m;
    uint64_t factor = hi % m;
    for (int i = 0; i < 64; i++) factor = (factor << 1) % m;
    return (result + factor) % m;
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

__device__ bool is_prime_64(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    if (n < 9) return true;
    if (n % 3 == 0) return false;
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

// ============================================================================
// Statistics structure
// ============================================================================

struct ThetaStats {
    uint64_t segment_id;
    uint64_t theta_start_lo;
    uint64_t theta_start_hi;  // For 256-bit theta positions
    uint64_t theta_count;
    uint32_t min_shell;
    uint32_t max_shell;
    
    uint32_t seen_min_shell;
    uint32_t seen_max_shell;
    
    uint64_t total_tested;
    uint64_t prime_count;
    
    double shell_sum;
    uint32_t min_popcount;
    uint32_t max_popcount;
    double popcount_sum;
    
    uint64_t first_prime_lo;
    uint64_t first_prime_hi;
    uint64_t last_prime_lo;
    uint64_t last_prime_hi;
    uint64_t twin_count;
    uint64_t mod6_1;
    uint64_t mod6_5;
    
    float time_ms;
};

// ============================================================================
// KERNELS
// ============================================================================

// 32-bit theta scan kernel
__global__ void theta_kernel_32(
    uint32_t theta_start,
    uint32_t theta_count,
    uint32_t min_shell,
    uint32_t max_shell,
    ThetaStats* stats
) {
    uint64_t local_primes = 0;
    uint64_t local_total = 0;
    uint32_t local_shell_min = UINT32_MAX;
    uint32_t local_shell_max = 0;
    double local_shell_sum = 0.0;
    uint32_t local_pop_min = UINT32_MAX;
    uint32_t local_pop_max = 0;
    double local_pop_sum = 0.0;
    uint64_t local_first = UINT64_MAX;
    uint64_t local_last = 0;
    uint64_t local_twins = 0;
    uint64_t local_m1 = 0, local_m5 = 0;
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t t = idx; t < theta_count; t += stride) {
        uint32_t theta_pos = theta_start + t;
        
        for (uint32_t shell = min_shell; shell <= max_shell; shell++) {
            uint32_t odd = theta_to_odd_32(theta_pos, shell);
            if (odd == 0) continue;
            
            uint32_t s = compute_shell_32(odd);
            uint32_t p = __popc(odd);
            
            local_total++;
            local_shell_sum += s;
            local_pop_sum += p;
            
            if (s < local_shell_min) local_shell_min = s;
            if (s > local_shell_max) local_shell_max = s;
            if (p < local_pop_min) local_pop_min = p;
            if (p > local_pop_max) local_pop_max = p;
            
            bool prime = is_prime_32(odd);
            
            if (prime) {
                local_primes++;
                if (odd < local_first) local_first = odd;
                if (odd > local_last) local_last = odd;
                
                uint32_t odd2 = odd + 2;
                if (odd2 > odd && is_prime_32(odd2)) local_twins++;
                
                if (odd > 3) {
                    uint32_t r = odd % 6;
                    if (r == 1) local_m1++;
                    else if (r == 5) local_m5++;
                }
            }
        }
    }
    
    // Warp reduction
    for (int off = WARP_SIZE/2; off > 0; off /= 2) {
        local_primes += __shfl_down_sync(0xFFFFFFFF, local_primes, off);
        local_total += __shfl_down_sync(0xFFFFFFFF, local_total, off);
        local_shell_sum += __shfl_down_sync(0xFFFFFFFF, local_shell_sum, off);
        local_pop_sum += __shfl_down_sync(0xFFFFFFFF, local_pop_sum, off);
        local_twins += __shfl_down_sync(0xFFFFFFFF, local_twins, off);
        local_m1 += __shfl_down_sync(0xFFFFFFFF, local_m1, off);
        local_m5 += __shfl_down_sync(0xFFFFFFFF, local_m5, off);
    }
    
    if ((threadIdx.x % WARP_SIZE) == 0) {
        atomicAdd((unsigned long long*)&stats->prime_count, local_primes);
        atomicAdd((unsigned long long*)&stats->total_tested, local_total);
        atomicAdd((unsigned long long*)&stats->twin_count, local_twins);
        atomicAdd((unsigned long long*)&stats->mod6_1, local_m1);
        atomicAdd((unsigned long long*)&stats->mod6_5, local_m5);
        
        atomicMin(&stats->seen_min_shell, local_shell_min);
        atomicMax(&stats->seen_max_shell, local_shell_max);
        atomicMin(&stats->min_popcount, local_pop_min);
        atomicMax(&stats->max_popcount, local_pop_max);
        
        uint64_t old = stats->first_prime_lo;
        while (local_first < old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->first_prime_lo, old, local_first);
            if (prev == old) break;
            old = prev;
        }
        old = stats->last_prime_lo;
        while (local_last > old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->last_prime_lo, old, local_last);
            if (prev == old) break;
            old = prev;
        }
        
        double old_s = stats->shell_sum;
        while (atomicCAS((unsigned long long*)&stats->shell_sum,
                         __double_as_longlong(old_s),
                         __double_as_longlong(old_s + local_shell_sum))
               != __double_as_longlong(old_s)) {
            old_s = stats->shell_sum;
        }
        double old_p = stats->popcount_sum;
        while (atomicCAS((unsigned long long*)&stats->popcount_sum,
                         __double_as_longlong(old_p),
                         __double_as_longlong(old_p + local_pop_sum))
               != __double_as_longlong(old_p)) {
            old_p = stats->popcount_sum;
        }
    }
}

// 64-bit theta scan kernel
__global__ void theta_kernel_64(
    uint64_t theta_start,
    uint64_t theta_count,
    uint32_t min_shell,
    uint32_t max_shell,
    ThetaStats* stats
) {
    uint64_t local_primes = 0;
    uint64_t local_total = 0;
    uint32_t local_shell_min = UINT32_MAX;
    uint32_t local_shell_max = 0;
    double local_shell_sum = 0.0;
    uint32_t local_pop_min = UINT32_MAX;
    uint32_t local_pop_max = 0;
    double local_pop_sum = 0.0;
    uint64_t local_first = UINT64_MAX;
    uint64_t local_last = 0;
    uint64_t local_twins = 0;
    uint64_t local_m1 = 0, local_m5 = 0;
    
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    
    for (uint64_t t = idx; t < theta_count; t += stride) {
        uint64_t theta_pos = theta_start + t;
        
        for (uint32_t shell = min_shell; shell <= max_shell; shell++) {
            uint64_t odd = theta_to_odd_64(theta_pos, shell);
            if (odd == 0) continue;
            
            uint32_t s = compute_shell_64(odd);
            uint32_t p = compute_popcount_64(odd);
            
            local_total++;
            local_shell_sum += s;
            local_pop_sum += p;
            
            if (s < local_shell_min) local_shell_min = s;
            if (s > local_shell_max) local_shell_max = s;
            if (p < local_pop_min) local_pop_min = p;
            if (p > local_pop_max) local_pop_max = p;
            
            bool prime = is_prime_64(odd);
            
            if (prime) {
                local_primes++;
                if (odd < local_first) local_first = odd;
                if (odd > local_last) local_last = odd;
                
                uint64_t odd2 = odd + 2;
                if (odd2 > odd) {
                    bool p2 = is_prime_64(odd2);
                    if (p2) local_twins++;
                }
                
                if (odd > 3) {
                    uint32_t r = odd % 6;
                    if (r == 1) local_m1++;
                    else if (r == 5) local_m5++;
                }
            }
        }
    }
    
    // Warp reduction
    for (int off = WARP_SIZE/2; off > 0; off /= 2) {
        local_primes += __shfl_down_sync(0xFFFFFFFF, local_primes, off);
        local_total += __shfl_down_sync(0xFFFFFFFF, local_total, off);
        local_shell_sum += __shfl_down_sync(0xFFFFFFFF, local_shell_sum, off);
        local_pop_sum += __shfl_down_sync(0xFFFFFFFF, local_pop_sum, off);
        local_twins += __shfl_down_sync(0xFFFFFFFF, local_twins, off);
        local_m1 += __shfl_down_sync(0xFFFFFFFF, local_m1, off);
        local_m5 += __shfl_down_sync(0xFFFFFFFF, local_m5, off);
    }
    
    if ((threadIdx.x % WARP_SIZE) == 0) {
        atomicAdd((unsigned long long*)&stats->prime_count, local_primes);
        atomicAdd((unsigned long long*)&stats->total_tested, local_total);
        atomicAdd((unsigned long long*)&stats->twin_count, local_twins);
        atomicAdd((unsigned long long*)&stats->mod6_1, local_m1);
        atomicAdd((unsigned long long*)&stats->mod6_5, local_m5);
        
        atomicMin(&stats->seen_min_shell, local_shell_min);
        atomicMax(&stats->seen_max_shell, local_shell_max);
        atomicMin(&stats->min_popcount, local_pop_min);
        atomicMax(&stats->max_popcount, local_pop_max);
        
        uint64_t old = stats->first_prime_lo;
        while (local_first < old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->first_prime_lo, old, local_first);
            if (prev == old) break;
            old = prev;
        }
        old = stats->last_prime_lo;
        while (local_last > old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->last_prime_lo, old, local_last);
            if (prev == old) break;
            old = prev;
        }
        
        double old_s = stats->shell_sum;
        while (atomicCAS((unsigned long long*)&stats->shell_sum,
                         __double_as_longlong(old_s),
                         __double_as_longlong(old_s + local_shell_sum))
               != __double_as_longlong(old_s)) {
            old_s = stats->shell_sum;
        }
        double old_p = stats->popcount_sum;
        while (atomicCAS((unsigned long long*)&stats->popcount_sum,
                         __double_as_longlong(old_p),
                         __double_as_longlong(old_p + local_pop_sum))
               != __double_as_longlong(old_p)) {
            old_p = stats->popcount_sum;
        }
    }
}

// ============================================================================
// Host functions
// ============================================================================

void init_stats(ThetaStats* s, uint64_t seg, uint64_t start_lo, uint64_t start_hi,
                uint64_t count, uint32_t minS, uint32_t maxS) {
    memset(s, 0, sizeof(ThetaStats));
    s->segment_id = seg;
    s->theta_start_lo = start_lo;
    s->theta_start_hi = start_hi;
    s->theta_count = count;
    s->min_shell = minS;
    s->max_shell = maxS;
    s->seen_min_shell = UINT32_MAX;
    s->seen_max_shell = 0;
    s->min_popcount = UINT32_MAX;
    s->max_popcount = 0;
    s->first_prime_lo = UINT64_MAX;
    s->first_prime_hi = UINT64_MAX;
    s->last_prime_lo = 0;
    s->last_prime_hi = 0;
}

void write_csv_header(FILE* fp, int mode) {
    fprintf(fp, "segment,theta_start");
    if (mode == 256) fprintf(fp, "_hi,theta_start_lo");
    fprintf(fp, ",theta_end,shell_config,shell_seen,"
                "total,primes,density,avg_shell,pop_range,avg_pop,"
                "first_prime,last_prime,twins,mod6_1,mod6_5,time_ms\n");
}

void write_csv_row(FILE* fp, const ThetaStats* s, int mode) {
    double density = s->total_tested ? (double)s->prime_count / s->total_tested : 0;
    double avg_shell = s->total_tested ? s->shell_sum / s->total_tested : 0;
    double avg_pop = s->total_tested ? s->popcount_sum / s->total_tested : 0;
    
    fprintf(fp, "%" PRIu64 ",", s->segment_id);
    
    if (mode == 256) {
        fprintf(fp, "%" PRIu64 ",%" PRIu64 ",", s->theta_start_hi, s->theta_start_lo);
    } else {
        fprintf(fp, "%" PRIu64 ",", s->theta_start_lo);
    }
    
    fprintf(fp, "%" PRIu64 ",%u-%u,%u-%u,"
                "%" PRIu64 ",%" PRIu64 ",%.12f,%.2f,%u-%u,%.2f,",
            s->theta_start_lo + s->theta_count,
            s->min_shell, s->max_shell,
            s->seen_min_shell, s->seen_max_shell,
            s->total_tested, s->prime_count, density, avg_shell,
            s->min_popcount, s->max_popcount, avg_pop);
    
    if (mode == 256) {
        fprintf(fp, "%" PRIu64 ":%" PRIu64 ",%" PRIu64 ":%" PRIu64 ",",
                s->first_prime_hi, s->first_prime_lo,
                s->last_prime_hi, s->last_prime_lo);
    } else {
        fprintf(fp, "%" PRIu64 ",%" PRIu64 ",",
                s->first_prime_lo, s->last_prime_lo);
    }
    
    fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.2f\n",
            s->twin_count, s->mod6_1, s->mod6_5, s->time_ms);
}

void write_checkpoint(const char* filename, uint64_t segment, uint64_t theta_pos) {
    FILE* fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "%" PRIu64 "\n%" PRIu64 "\n", segment, theta_pos);
        fclose(fp);
    }
}

int read_checkpoint(const char* filename, uint64_t* segment, uint64_t* theta_pos) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return 0;
    int ret = fscanf(fp, "%" PRIu64 "\n%" PRIu64, segment, theta_pos);
    fclose(fp);
    return (ret == 2);
}

uint64_t parse_number(const char* str) {
    if (str[0] == '0' && (str[1] == 'x' || str[1] == 'X')) {
        return strtoull(str + 2, NULL, 16);
    }
    return strtoull(str, NULL, 10);
}

void print_usage(const char* prog) {
    printf("theta_order_complete - Complete Theta Order Prime Analysis\n");
    printf("============================================================\n\n");
    printf("Usage: %s <mode> [options]\n\n", prog);
    
    printf("MODES:\n");
    printf("  scan32    Full 32-bit theta scan (shells 2-31)\n");
    printf("  scan64    Full 64-bit theta scan (shells 2-62)\n");
    printf("  zoom64    Zoom into 32-bit region with 64-bit precision\n");
    printf("  batch     Batch mode for long-running analysis\n\n");
    
    printf("COMMON OPTIONS:\n");
    printf("  --min-shell S       Minimum shell to analyze (default: 2)\n");
    printf("  --max-shell S       Maximum shell to analyze (default: mode-dependent)\n");
    printf("  --theta-exp E       Segment size = 2^E theta positions (default: 20)\n");
    printf("  --segments N        Number of segments (0 = auto for full range)\n");
    printf("  --start-seg N       Starting segment number (for resuming)\n");
    printf("  --output FILE       Output CSV file (default: theta_output.csv)\n");
    printf("  --checkpoint FILE   Checkpoint file for resuming (default: checkpoint.txt)\n");
    printf("  -q                  Quiet mode\n");
    printf("  -v                  Verbose mode\n\n");
    
    printf("SCAN32 MODE (Full 32-bit range):\n");
    printf("  Covers theta positions 0 to 2^30 for shells 2-31.\n");
    printf("  Example: %s scan32 --theta-exp 20 --output scan32.csv\n\n", prog);
    
    printf("SCAN64 MODE (Full 64-bit range):\n");
    printf("  Covers theta positions for shells up to 62.\n");
    printf("  --theta-start T     Starting theta position (decimal or 0x hex)\n");
    printf("  Example: %s scan64 --theta-start 0x100000000 --theta-exp 24\n\n", prog);
    
    printf("ZOOM64 MODE (Zoom into specific region):\n");
    printf("  --base-theta T      Base theta from scan32 to zoom into\n");
    printf("  --sub-exp E         Sub-precision exponent (default: 20)\n");
    printf("  --sub-segments N    Number of sub-segments (default: 16)\n");
    printf("  Example: %s zoom64 --base-theta 12345678 --sub-exp 20\n\n", prog);
    
    printf("BATCH MODE (Long-running analysis):\n");
    printf("  Designed for week-long runs with automatic checkpointing.\n");
    printf("  --precision P       32 or 64 (default: 32)\n");
    printf("  --total-segments N  Total segments to process\n");
    printf("  --output-dir DIR    Output directory (default: ./theta_output)\n");
    printf("  --sleep-ms MS       Sleep between segments (default: 100)\n");
    printf("  Example: %s batch --precision 32 --total-segments 1024\n\n", prog);
    
    printf("EXAMPLES FOR WEEK-LONG RUNS:\n");
    printf("  # Full 32-bit scan with checkpointing:\n");
    printf("  %s batch --precision 32 --total-segments 1024 --output-dir ./results\n\n", prog);
    printf("  # Resume interrupted run:\n");
    printf("  %s batch --precision 32 --checkpoint ./results/checkpoint.txt\n\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* mode = argv[1];
    
    if (!strcmp(mode, "-h") || !strcmp(mode, "--help")) {
        print_usage(argv[0]);
        return 0;
    }
    
    // Default parameters
    uint32_t min_shell = 2;
    uint32_t max_shell = 0;  // 0 = auto based on mode
    int theta_exp = 20;
    int num_segments = 0;
    uint64_t start_segment = 0;
    uint64_t theta_start = 0;
    uint64_t base_theta = 0;
    int sub_exp = 20;
    int sub_segments = 16;
    int precision = 32;
    int sleep_ms = 100;
    const char* outfile = "theta_output.csv";
    const char* outdir = "./theta_output";
    const char* checkpoint_file = "checkpoint.txt";
    int verbose = 1;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--min-shell") && i+1 < argc) min_shell = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-shell") && i+1 < argc) max_shell = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--theta-exp") && i+1 < argc) theta_exp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--segments") && i+1 < argc) num_segments = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--total-segments") && i+1 < argc) num_segments = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--start-seg") && i+1 < argc) start_segment = parse_number(argv[++i]);
        else if (!strcmp(argv[i], "--theta-start") && i+1 < argc) theta_start = parse_number(argv[++i]);
        else if (!strcmp(argv[i], "--base-theta") && i+1 < argc) base_theta = parse_number(argv[++i]);
        else if (!strcmp(argv[i], "--sub-exp") && i+1 < argc) sub_exp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sub-segments") && i+1 < argc) sub_segments = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc) precision = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sleep-ms") && i+1 < argc) sleep_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1 < argc) outfile = argv[++i];
        else if (!strcmp(argv[i], "--output-dir") && i+1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--checkpoint") && i+1 < argc) checkpoint_file = argv[++i];
        else if (!strcmp(argv[i], "-q")) verbose = 0;
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Set mode-dependent defaults
    if (!strcmp(mode, "scan32")) {
        if (max_shell == 0) max_shell = 31;
        if (max_shell > 31) max_shell = 31;
    } else if (!strcmp(mode, "scan64") || !strcmp(mode, "zoom64") || !strcmp(mode, "batch")) {
        if (max_shell == 0) max_shell = (precision == 32) ? 31 : 62;
        if (precision == 32 && max_shell > 31) max_shell = 31;
        if (precision == 64 && max_shell > 62) max_shell = 62;
    }
    
    // Initialize CUDA
    cudaSetDevice(0);
    
    ThetaStats* d_stats;
    cudaMalloc(&d_stats, sizeof(ThetaStats));
    ThetaStats h_stats;
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    
    // ========================================================================
    // MODE: scan32
    // ========================================================================
    if (!strcmp(mode, "scan32")) {
        uint64_t seg_size = 1ULL << theta_exp;
        uint64_t max_theta = 1ULL << (max_shell - 1);
        
        if (num_segments == 0) {
            num_segments = (max_theta + seg_size - 1) / seg_size;
        }
        
        if (verbose) {
            printf("=== SCAN32: Full 32-bit Theta Scan ===\n");
            printf("Shells: %u to %u\n", min_shell, max_shell);
            printf("Theta range: 0 to %" PRIu64 "\n", max_theta);
            printf("Segment size: 2^%d = %" PRIu64 "\n", theta_exp, seg_size);
            printf("Total segments: %d\n", num_segments);
            printf("Starting from segment: %" PRIu64 "\n", start_segment);
            printf("Output: %s\n\n", outfile);
        }
        
        FILE* fp = fopen(outfile, start_segment > 0 ? "a" : "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", outfile); return 1; }
        if (start_segment == 0) write_csv_header(fp, 32);
        
        uint64_t total_primes = 0;
        
        for (uint64_t seg = start_segment; seg < (uint64_t)num_segments; seg++) {
            uint64_t theta_pos = seg * seg_size;
            uint64_t count = seg_size;
            if (theta_pos + count > max_theta) count = max_theta - theta_pos;
            if (count == 0) break;
            
            init_stats(&h_stats, seg, theta_pos, 0, count, min_shell, max_shell);
            cudaMemcpy(d_stats, &h_stats, sizeof(ThetaStats), cudaMemcpyHostToDevice);
            
            cudaEventRecord(t0);
            int blocks = min((int)((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 65535);
            theta_kernel_32<<<blocks, THREADS_PER_BLOCK>>>((uint32_t)theta_pos, (uint32_t)count, min_shell, max_shell, d_stats);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            
            cudaMemcpy(&h_stats, d_stats, sizeof(ThetaStats), cudaMemcpyDeviceToHost);
            h_stats.time_ms = ms;
            
            if (h_stats.seen_min_shell == UINT32_MAX) h_stats.seen_min_shell = 0;
            if (h_stats.min_popcount == UINT32_MAX) h_stats.min_popcount = 0;
            if (h_stats.first_prime_lo == UINT64_MAX) h_stats.first_prime_lo = 0;
            
            write_csv_row(fp, &h_stats, 32);
            fflush(fp);
            total_primes += h_stats.prime_count;
            
            if (verbose >= 2 || (verbose == 1 && seg % 64 == 0)) {
                double density = h_stats.total_tested ? (double)h_stats.prime_count / h_stats.total_tested : 0;
                printf("Seg %6" PRIu64 "/%d: theta=[%10" PRIu64 ",%10" PRIu64 ") primes=%8" PRIu64 " d=%.8f %.0fms\n",
                       seg, num_segments, theta_pos, theta_pos + count, h_stats.prime_count, density, ms);
            }
        }
        
        if (verbose) printf("\nTotal primes: %" PRIu64 "\n", total_primes);
        fclose(fp);
    }
    
    // ========================================================================
    // MODE: scan64
    // ========================================================================
    else if (!strcmp(mode, "scan64")) {
        uint64_t seg_size = 1ULL << theta_exp;
        
        if (verbose) {
            printf("=== SCAN64: 64-bit Theta Scan ===\n");
            printf("Shells: %u to %u\n", min_shell, max_shell);
            printf("Starting theta: %" PRIu64 " (0x%" PRIx64 ")\n", theta_start, theta_start);
            printf("Segment size: 2^%d = %" PRIu64 "\n", theta_exp, seg_size);
            printf("Segments: %d\n", num_segments);
            printf("Output: %s\n\n", outfile);
        }
        
        FILE* fp = fopen(outfile, start_segment > 0 ? "a" : "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", outfile); return 1; }
        if (start_segment == 0) write_csv_header(fp, 64);
        
        uint64_t total_primes = 0;
        
        for (uint64_t seg = start_segment; seg < (uint64_t)num_segments; seg++) {
            uint64_t theta_pos = theta_start + seg * seg_size;
            
            init_stats(&h_stats, seg, theta_pos, 0, seg_size, min_shell, max_shell);
            cudaMemcpy(d_stats, &h_stats, sizeof(ThetaStats), cudaMemcpyHostToDevice);
            
            cudaEventRecord(t0);
            int blocks = min((int)((seg_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 65535);
            theta_kernel_64<<<blocks, THREADS_PER_BLOCK>>>(theta_pos, seg_size, min_shell, max_shell, d_stats);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            
            cudaMemcpy(&h_stats, d_stats, sizeof(ThetaStats), cudaMemcpyDeviceToHost);
            h_stats.time_ms = ms;
            
            if (h_stats.seen_min_shell == UINT32_MAX) h_stats.seen_min_shell = 0;
            if (h_stats.min_popcount == UINT32_MAX) h_stats.min_popcount = 0;
            if (h_stats.first_prime_lo == UINT64_MAX) h_stats.first_prime_lo = 0;
            
            write_csv_row(fp, &h_stats, 64);
            fflush(fp);
            total_primes += h_stats.prime_count;
            
            if (verbose >= 2 || (verbose == 1 && seg % 16 == 0)) {
                double density = h_stats.total_tested ? (double)h_stats.prime_count / h_stats.total_tested : 0;
                printf("Seg %6" PRIu64 ": theta=0x%016" PRIx64 " primes=%8" PRIu64 " d=%.8f %.0fms\n",
                       seg, theta_pos, h_stats.prime_count, density, ms);
            }
        }
        
        if (verbose) printf("\nTotal primes: %" PRIu64 "\n", total_primes);
        fclose(fp);
    }
    
    // ========================================================================
    // MODE: zoom64
    // ========================================================================
    else if (!strcmp(mode, "zoom64")) {
        if (min_shell < 32) min_shell = 32;
        uint64_t sub_size = 1ULL << sub_exp;
        
        if (verbose) {
            printf("=== ZOOM64: Zoom into 32-bit Region ===\n");
            printf("Base theta (from scan32): %" PRIu64 "\n", base_theta);
            printf("Shells: %u to %u\n", min_shell, max_shell);
            printf("Sub-segment size: 2^%d = %" PRIu64 "\n", sub_exp, sub_size);
            printf("Sub-segments: %d\n", sub_segments);
            printf("Output: %s\n\n", outfile);
        }
        
        FILE* fp = fopen(outfile, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", outfile); return 1; }
        write_csv_header(fp, 64);
        
        uint64_t total_primes = 0;
        
        for (int seg = 0; seg < sub_segments; seg++) {
            uint64_t theta_pos = (base_theta << 32) + (uint64_t)seg * sub_size;
            
            init_stats(&h_stats, seg, theta_pos, 0, sub_size, min_shell, max_shell);
            cudaMemcpy(d_stats, &h_stats, sizeof(ThetaStats), cudaMemcpyHostToDevice);
            
            cudaEventRecord(t0);
            int blocks = min((int)((sub_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 65535);
            theta_kernel_64<<<blocks, THREADS_PER_BLOCK>>>(theta_pos, sub_size, min_shell, max_shell, d_stats);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            
            cudaMemcpy(&h_stats, d_stats, sizeof(ThetaStats), cudaMemcpyDeviceToHost);
            h_stats.time_ms = ms;
            
            if (h_stats.seen_min_shell == UINT32_MAX) h_stats.seen_min_shell = 0;
            if (h_stats.min_popcount == UINT32_MAX) h_stats.min_popcount = 0;
            if (h_stats.first_prime_lo == UINT64_MAX) h_stats.first_prime_lo = 0;
            
            write_csv_row(fp, &h_stats, 64);
            fflush(fp);
            total_primes += h_stats.prime_count;
            
            if (verbose) {
                double density = h_stats.total_tested ? (double)h_stats.prime_count / h_stats.total_tested : 0;
                printf("Sub %3d: theta=0x%016" PRIx64 " primes=%8" PRIu64 " d=%.8f %.0fms\n",
                       seg, theta_pos, h_stats.prime_count, density, ms);
            }
        }
        
        if (verbose) printf("\nTotal primes in zoom: %" PRIu64 "\n", total_primes);
        fclose(fp);
    }
    
    // ========================================================================
    // MODE: batch - Long-running with checkpointing
    // ========================================================================
    else if (!strcmp(mode, "batch")) {
        // Create output directory
        mkdir(outdir, 0755);
        
        char csv_path[512], ckpt_path[512];
        snprintf(csv_path, sizeof(csv_path), "%s/theta_batch_%dbit.csv", outdir, precision);
        snprintf(ckpt_path, sizeof(ckpt_path), "%s/%s", outdir, checkpoint_file);
        
        // Try to read checkpoint
        uint64_t resume_seg = 0, resume_theta = 0;
        if (read_checkpoint(ckpt_path, &resume_seg, &resume_theta)) {
            start_segment = resume_seg;
            if (verbose) printf("Resuming from checkpoint: segment %" PRIu64 "\n", start_segment);
        }
        
        uint64_t seg_size = 1ULL << theta_exp;
        uint64_t max_theta = 1ULL << (max_shell - 1);
        
        if (num_segments == 0) {
            num_segments = (max_theta + seg_size - 1) / seg_size;
        }
        
        if (verbose) {
            printf("=== BATCH MODE: Long-Running Analysis ===\n");
            printf("Precision: %d-bit\n", precision);
            printf("Shells: %u to %u\n", min_shell, max_shell);
            printf("Theta range: 0 to %" PRIu64 "\n", max_theta);
            printf("Segment size: 2^%d = %" PRIu64 "\n", theta_exp, seg_size);
            printf("Total segments: %d\n", num_segments);
            printf("Starting from: %" PRIu64 "\n", start_segment);
            printf("Output: %s\n", csv_path);
            printf("Checkpoint: %s\n", ckpt_path);
            printf("Sleep between segments: %d ms\n\n", sleep_ms);
        }
        
        FILE* fp = fopen(csv_path, start_segment > 0 ? "a" : "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", csv_path); return 1; }
        if (start_segment == 0) write_csv_header(fp, precision);
        
        uint64_t total_primes = 0;
        time_t start_time = time(NULL);
        
        for (uint64_t seg = start_segment; seg < (uint64_t)num_segments; seg++) {
            uint64_t theta_pos = seg * seg_size;
            uint64_t count = seg_size;
            if (theta_pos + count > max_theta) count = max_theta - theta_pos;
            if (count == 0) break;
            
            init_stats(&h_stats, seg, theta_pos, 0, count, min_shell, max_shell);
            cudaMemcpy(d_stats, &h_stats, sizeof(ThetaStats), cudaMemcpyHostToDevice);
            
            cudaEventRecord(t0);
            int blocks = min((int)((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 65535);
            
            if (precision == 32) {
                theta_kernel_32<<<blocks, THREADS_PER_BLOCK>>>((uint32_t)theta_pos, (uint32_t)count, min_shell, max_shell, d_stats);
            } else {
                theta_kernel_64<<<blocks, THREADS_PER_BLOCK>>>(theta_pos, count, min_shell, max_shell, d_stats);
            }
            
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            
            cudaMemcpy(&h_stats, d_stats, sizeof(ThetaStats), cudaMemcpyDeviceToHost);
            h_stats.time_ms = ms;
            
            if (h_stats.seen_min_shell == UINT32_MAX) h_stats.seen_min_shell = 0;
            if (h_stats.min_popcount == UINT32_MAX) h_stats.min_popcount = 0;
            if (h_stats.first_prime_lo == UINT64_MAX) h_stats.first_prime_lo = 0;
            
            write_csv_row(fp, &h_stats, precision);
            fflush(fp);
            total_primes += h_stats.prime_count;
            
            // Update checkpoint
            write_checkpoint(ckpt_path, seg + 1, theta_pos + count);
            
            // Progress output
            if (verbose && (seg % 64 == 0 || seg == (uint64_t)num_segments - 1)) {
                time_t now = time(NULL);
                double elapsed = difftime(now, start_time);
                double segs_done = seg - start_segment + 1;
                double eta_secs = (num_segments - seg - 1) * (elapsed / segs_done);
                
                double density = h_stats.total_tested ? (double)h_stats.prime_count / h_stats.total_tested : 0;
                printf("[%6" PRIu64 "/%d] theta=%10" PRIu64 " primes=%8" PRIu64 " d=%.8f %.0fms | ETA: %.1fh\n",
                       seg, num_segments, theta_pos, h_stats.prime_count, density, ms, eta_secs / 3600.0);
            }
            
            // Sleep between segments
            if (sleep_ms > 0) {
                struct timespec ts = {sleep_ms / 1000, (sleep_ms % 1000) * 1000000};
                nanosleep(&ts, NULL);
            }
        }
        
        if (verbose) {
            time_t end_time = time(NULL);
            double total_time = difftime(end_time, start_time);
            printf("\n=== BATCH COMPLETE ===\n");
            printf("Total primes: %" PRIu64 "\n", total_primes);
            printf("Total time: %.2f hours\n", total_time / 3600.0);
            printf("Output: %s\n", csv_path);
        }
        
        fclose(fp);
    }
    
    else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        print_usage(argv[0]);
        return 1;
    }
    
    // Cleanup
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_stats);
    
    return 0;
}
