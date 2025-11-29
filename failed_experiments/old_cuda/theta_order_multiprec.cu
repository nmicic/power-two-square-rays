// ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
// This file remains for reference only.
// Do NOT use in production. No support, no guarantees.
//
// For the current CUDA implementation, see:
//   - cuda/theta_cuda_v1.2.cuh
//   - cuda/theta_cuda_benchmark_v1.2.cu

/**
 * theta_order_multiprec.cu - Multi-Precision Theta Order Analysis
 *
 * ============================================================================
 * ZOOM WORKFLOW
 * ============================================================================
 *
 * PHASE 1: 32-bit full scan
 *   - Scan all theta positions for shells 2-31
 *   - Find segments with extreme prime densities/gaps
 *   - Output: theta regions of interest
 *
 * PHASE 2: 64-bit zoom
 *   - For each extreme theta region from phase 1
 *   - Subdivide into 2^32 finer positions (32-bit precision between neighbors)
 *   - Analyze shells 32-63
 *   - Now at 64-bit effective precision (32 + 32)
 *
 * PHASE 3: 256-bit zoom (future)
 *   - Subdivide again for shells 64-255
 *   - Another 32-bit precision layer → 96-bit total
 *
 * This is "shooting in the dark" to see if extremes cluster or distribute
 * uniformly across theta-space.
 *
 * ============================================================================
 * THETA SUBDIVISION CONCEPT
 * ============================================================================
 *
 * At shell k, there are 2^(k-1) theta positions.
 * At shell k+32, there are 2^(k+31) theta positions = 2^32 times more.
 *
 * So theta position T at shell k "expands" to positions [T*2^32, (T+1)*2^32)
 * at shell k+32.
 *
 * When we zoom, we're looking at what happens in this expanded region.
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

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define THRESHOLD_32BIT 0x100000000ULL

__constant__ uint64_t MR_WITNESSES_3[3] = {2, 7, 61};
__constant__ uint64_t MR_WITNESSES_12[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

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

__device__ __forceinline__ uint32_t compute_shell(uint64_t n) {
    if (n == 0) return 0;
    return 63 - __clzll(n);
}

__device__ __forceinline__ uint32_t compute_shell_32(uint32_t n) {
    if (n == 0) return 0;
    return 31 - __clz(n);
}

__device__ __forceinline__ uint32_t compute_popcount(uint64_t n) {
    return __popcll(n);
}

// ============================================================================
// Theta to integer conversion
// ============================================================================

/**
 * 32-bit version: Get odd integer at (theta_pos, shell) for shell <= 31
 */
__device__ __host__ uint32_t theta_to_odd_32(uint32_t theta_pos, uint32_t shell) {
    if (shell == 0) return (theta_pos == 0) ? 1 : 0;
    if (shell == 1) return (theta_pos == 0) ? 3 : 0;
    
    uint32_t num_odds = 1U << (shell - 1);
    if (theta_pos >= num_odds) return 0;
    
    uint32_t j = bit_reverse_32(theta_pos, shell - 1);
    return (1U << shell) + 2 * j + 1;
}

/**
 * 64-bit version: Get odd integer at (theta_pos, shell) for shell <= 63
 */
__device__ __host__ uint64_t theta_to_odd_64(uint64_t theta_pos, uint32_t shell) {
    if (shell == 0) return (theta_pos == 0) ? 1 : 0;
    if (shell == 1) return (theta_pos == 0) ? 3 : 0;
    
    uint64_t num_odds = 1ULL << (shell - 1);
    if (theta_pos >= num_odds) return 0;
    
    uint64_t j = bit_reverse_64(theta_pos, shell - 1);
    return (1ULL << shell) + 2 * j + 1;
}

// ============================================================================
// Miller-Rabin
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
// Statistics
// ============================================================================

struct ThetaStats {
    uint64_t segment_id;
    uint64_t theta_start;
    uint64_t theta_count;
    uint32_t min_shell;
    uint32_t max_shell;
    
    uint32_t seen_min_shell;
    uint32_t seen_max_shell;
    
    uint64_t total_tested;
    uint64_t prime_count;
    
    // For finding extremes
    double density;           // prime_count / total_tested
    uint64_t max_theta_gap;   // largest gap between consecutive primes in theta space
    
    double shell_sum;
    uint32_t min_popcount;
    uint32_t max_popcount;
    double popcount_sum;
    
    uint64_t first_prime;
    uint64_t last_prime;
    uint64_t twin_count;
    
    float time_ms;
};

// ============================================================================
// 32-BIT FULL SCAN KERNEL (Phase 1)
// ============================================================================

__global__ void theta_scan_32bit_kernel(
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
    }
    
    if ((threadIdx.x % WARP_SIZE) == 0) {
        atomicAdd((unsigned long long*)&stats->prime_count, local_primes);
        atomicAdd((unsigned long long*)&stats->total_tested, local_total);
        atomicAdd((unsigned long long*)&stats->twin_count, local_twins);
        
        atomicMin(&stats->seen_min_shell, local_shell_min);
        atomicMax(&stats->seen_max_shell, local_shell_max);
        atomicMin(&stats->min_popcount, local_pop_min);
        atomicMax(&stats->max_popcount, local_pop_max);
        
        uint64_t old = stats->first_prime;
        while (local_first < old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->first_prime, old, local_first);
            if (prev == old) break;
            old = prev;
        }
        old = stats->last_prime;
        while (local_last > old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->last_prime, old, local_last);
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
// 64-BIT ZOOM KERNEL (Phase 2)
// For zooming into a 32-bit theta region with 32-bit sub-precision
// ============================================================================

__global__ void theta_zoom_64bit_kernel(
    uint64_t theta_start,      // Base theta * 2^32 + sub_start
    uint64_t theta_count,      // Number of sub-positions to scan
    uint32_t min_shell,        // 32-63 typically
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
    
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    
    for (uint64_t t = idx; t < theta_count; t += stride) {
        uint64_t theta_pos = theta_start + t;
        
        for (uint32_t shell = min_shell; shell <= max_shell; shell++) {
            uint64_t odd = theta_to_odd_64(theta_pos, shell);
            if (odd == 0) continue;
            
            uint32_t s = compute_shell(odd);
            uint32_t p = compute_popcount(odd);
            
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
    }
    
    if ((threadIdx.x % WARP_SIZE) == 0) {
        atomicAdd((unsigned long long*)&stats->prime_count, local_primes);
        atomicAdd((unsigned long long*)&stats->total_tested, local_total);
        atomicAdd((unsigned long long*)&stats->twin_count, local_twins);
        
        atomicMin(&stats->seen_min_shell, local_shell_min);
        atomicMax(&stats->seen_max_shell, local_shell_max);
        atomicMin(&stats->min_popcount, local_pop_min);
        atomicMax(&stats->max_popcount, local_pop_max);
        
        uint64_t old = stats->first_prime;
        while (local_first < old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->first_prime, old, local_first);
            if (prev == old) break;
            old = prev;
        }
        old = stats->last_prime;
        while (local_last > old) {
            uint64_t prev = atomicCAS((unsigned long long*)&stats->last_prime, old, local_last);
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
// Host code
// ============================================================================

void init_stats(ThetaStats* s, uint64_t seg, uint64_t start, uint64_t count,
                uint32_t minS, uint32_t maxS) {
    memset(s, 0, sizeof(ThetaStats));
    s->segment_id = seg;
    s->theta_start = start;
    s->theta_count = count;
    s->min_shell = minS;
    s->max_shell = maxS;
    s->seen_min_shell = UINT32_MAX;
    s->seen_max_shell = 0;
    s->min_popcount = UINT32_MAX;
    s->max_popcount = 0;
    s->first_prime = UINT64_MAX;
    s->last_prime = 0;
}

void print_csv_header(FILE* fp) {
    fprintf(fp, "segment,theta_start,theta_end,shell_config,shell_seen,"
                "total,primes,density,avg_shell,pop_range,avg_pop,"
                "first_prime,last_prime,twins,time_ms\n");
}

void print_csv_row(FILE* fp, const ThetaStats* s) {
    double density = s->total_tested ? (double)s->prime_count / s->total_tested : 0;
    double avg_shell = s->total_tested ? s->shell_sum / s->total_tested : 0;
    double avg_pop = s->total_tested ? s->popcount_sum / s->total_tested : 0;
    
    fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u-%u,%u-%u,"
                "%" PRIu64 ",%" PRIu64 ",%.12f,%.2f,%u-%u,%.2f,"
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.2f\n",
            s->segment_id, s->theta_start, s->theta_start + s->theta_count,
            s->min_shell, s->max_shell,
            s->seen_min_shell, s->seen_max_shell,
            s->total_tested, s->prime_count, density, avg_shell,
            s->min_popcount, s->max_popcount, avg_pop,
            s->first_prime, s->last_prime, s->twin_count, s->time_ms);
}

void print_usage(const char* prog) {
    printf("theta_order_multiprec - Multi-Precision Theta Order Analysis\n\n");
    printf("Usage: %s <mode> [options]\n\n", prog);
    printf("Modes:\n");
    printf("  scan32     Full 32-bit theta scan (Phase 1)\n");
    printf("  zoom64     64-bit zoom into specific theta region (Phase 2)\n");
    printf("\nPhase 1 (scan32) options:\n");
    printf("  --min-shell S     Min shell (default: 2)\n");
    printf("  --max-shell S     Max shell (default: 31)\n");
    printf("  --theta-exp E     Segment size = 2^E (default: 20)\n");
    printf("  --segments N      Number of segments (default: covers full 2^30 range)\n");
    printf("  --output FILE     Output CSV\n");
    printf("\nPhase 2 (zoom64) options:\n");
    printf("  --base-theta T    Base theta position from phase 1\n");
    printf("  --sub-exp E       Sub-precision exponent (default: 20)\n");
    printf("  --sub-segments N  Number of sub-segments (default: 16)\n");
    printf("  --min-shell S     Min shell (default: 32)\n");
    printf("  --max-shell S     Max shell (default: 48)\n");
    printf("\nExamples:\n");
    printf("  # Phase 1: Full 32-bit scan\n");
    printf("  %s scan32 --theta-exp 20 --output phase1.csv\n\n", prog);
    printf("  # Phase 2: Zoom into theta=12345678 from phase 1\n");
    printf("  %s zoom64 --base-theta 12345678 --sub-exp 16 --output phase2.csv\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* mode = argv[1];
    
    // Default parameters
    uint32_t min_shell = 2;
    uint32_t max_shell = 31;
    int theta_exp = 20;
    int num_segments = 0;  // 0 = auto-compute for full range
    uint64_t base_theta = 0;
    int sub_exp = 20;
    int sub_segments = 16;
    const char* outfile = "theta_output.csv";
    int verbose = 1;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--min-shell") && i+1 < argc) min_shell = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-shell") && i+1 < argc) max_shell = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--theta-exp") && i+1 < argc) theta_exp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--segments") && i+1 < argc) num_segments = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--base-theta") && i+1 < argc) base_theta = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--sub-exp") && i+1 < argc) sub_exp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sub-segments") && i+1 < argc) sub_segments = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1 < argc) outfile = argv[++i];
        else if (!strcmp(argv[i], "-q")) verbose = 0;
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // ========================================================================
    // MODE: scan32 - Full 32-bit theta scan
    // ========================================================================
    if (!strcmp(mode, "scan32")) {
        if (max_shell > 31) max_shell = 31;  // 32-bit limit
        
        uint64_t seg_size = 1ULL << theta_exp;
        
        // For shell k, there are 2^(k-1) theta positions
        // For max_shell=31, that's 2^30 positions
        // We want to cover them all
        uint64_t max_theta_positions = 1ULL << (max_shell - 1);
        
        if (num_segments == 0) {
            num_segments = (max_theta_positions + seg_size - 1) / seg_size;
        }
        
        if (verbose) {
            printf("=== PHASE 1: 32-bit Full Theta Scan ===\n");
            printf("Shells: %u to %u\n", min_shell, max_shell);
            printf("Max theta positions: %" PRIu64 " (2^%u)\n", max_theta_positions, max_shell - 1);
            printf("Segment size: 2^%d = %" PRIu64 "\n", theta_exp, seg_size);
            printf("Segments: %d\n", num_segments);
            printf("Output: %s\n\n", outfile);
        }
        
        ThetaStats* d_stats;
        cudaMalloc(&d_stats, sizeof(ThetaStats));
        ThetaStats h_stats;
        
        FILE* fp = fopen(outfile, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", outfile); return 1; }
        print_csv_header(fp);
        
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        
        uint64_t total_primes = 0;
        double min_density = DBL_MAX, max_density = 0;
        uint64_t min_density_seg = 0, max_density_seg = 0;
        
        for (int seg = 0; seg < num_segments; seg++) {
            uint64_t theta_start = (uint64_t)seg * seg_size;
            uint64_t this_count = seg_size;
            if (theta_start + this_count > max_theta_positions) {
                this_count = max_theta_positions - theta_start;
            }
            if (this_count == 0) break;
            
            init_stats(&h_stats, seg, theta_start, this_count, min_shell, max_shell);
            cudaMemcpy(d_stats, &h_stats, sizeof(ThetaStats), cudaMemcpyHostToDevice);
            
            cudaEventRecord(t0);
            
            int blocks = min((int)((this_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 65535);
            theta_scan_32bit_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                (uint32_t)theta_start, (uint32_t)this_count, min_shell, max_shell, d_stats);
            
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            
            cudaMemcpy(&h_stats, d_stats, sizeof(ThetaStats), cudaMemcpyDeviceToHost);
            h_stats.time_ms = ms;
            h_stats.density = h_stats.total_tested ? (double)h_stats.prime_count / h_stats.total_tested : 0;
            
            if (h_stats.seen_min_shell == UINT32_MAX) h_stats.seen_min_shell = 0;
            if (h_stats.min_popcount == UINT32_MAX) h_stats.min_popcount = 0;
            if (h_stats.first_prime == UINT64_MAX) h_stats.first_prime = 0;
            
            print_csv_row(fp, &h_stats);
            total_primes += h_stats.prime_count;
            
            // Track extremes
            if (h_stats.density < min_density && h_stats.density > 0) {
                min_density = h_stats.density;
                min_density_seg = seg;
            }
            if (h_stats.density > max_density) {
                max_density = h_stats.density;
                max_density_seg = seg;
            }
            
            if (verbose >= 2 || (verbose == 1 && seg % 64 == 0)) {
                printf("Seg %5d: theta=[%10" PRIu64 ",%10" PRIu64 ") primes=%8" PRIu64 " density=%.10f %.0fms\n",
                       seg, theta_start, theta_start + this_count, h_stats.prime_count, h_stats.density, ms);
            }
        }
        
        if (verbose) {
            printf("\n=== PHASE 1 RESULTS ===\n");
            printf("Total primes: %" PRIu64 "\n", total_primes);
            printf("Min density segment: %5" PRIu64 " (density=%.10f, theta_start=%" PRIu64 ")\n",
                   min_density_seg, min_density, min_density_seg * seg_size);
            printf("Max density segment: %5" PRIu64 " (density=%.10f, theta_start=%" PRIu64 ")\n",
                   max_density_seg, max_density, max_density_seg * seg_size);
            printf("\nTo zoom into min density region:\n");
            printf("  %s zoom64 --base-theta %" PRIu64 " --output zoom_min.csv\n",
                   argv[0], min_density_seg * seg_size);
            printf("\nTo zoom into max density region:\n");
            printf("  %s zoom64 --base-theta %" PRIu64 " --output zoom_max.csv\n",
                   argv[0], max_density_seg * seg_size);
        }
        
        fclose(fp);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(d_stats);
        return 0;
    }
    
    // ========================================================================
    // MODE: zoom64 - 64-bit zoom into specific theta region
    // ========================================================================
    if (!strcmp(mode, "zoom64")) {
        if (min_shell < 32) min_shell = 32;  // Start at 64-bit shells
        if (max_shell > 62) max_shell = 62;  // 64-bit limit
        
        // base_theta is a 32-bit theta position from phase 1
        // We expand it to 64-bit precision by multiplying by 2^32 and adding sub-positions
        uint64_t sub_size = 1ULL << sub_exp;
        
        if (verbose) {
            printf("=== PHASE 2: 64-bit Zoom ===\n");
            printf("Base theta (from phase 1): %" PRIu64 "\n", base_theta);
            printf("Shells: %u to %u\n", min_shell, max_shell);
            printf("Sub-segment size: 2^%d = %" PRIu64 "\n", sub_exp, sub_size);
            printf("Sub-segments: %d\n", sub_segments);
            printf("Effective theta range: [%" PRIu64 "*2^32, (%" PRIu64 "+1)*2^32)\n", 
                   base_theta, base_theta);
            printf("Output: %s\n\n", outfile);
        }
        
        ThetaStats* d_stats;
        cudaMalloc(&d_stats, sizeof(ThetaStats));
        ThetaStats h_stats;
        
        FILE* fp = fopen(outfile, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", outfile); return 1; }
        print_csv_header(fp);
        
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        
        uint64_t total_primes = 0;
        double min_density = DBL_MAX, max_density = 0;
        
        for (int seg = 0; seg < sub_segments; seg++) {
            // Theta position = base_theta * 2^32 + seg * sub_size
            uint64_t theta_start = (base_theta << 32) + (uint64_t)seg * sub_size;
            
            init_stats(&h_stats, seg, theta_start, sub_size, min_shell, max_shell);
            cudaMemcpy(d_stats, &h_stats, sizeof(ThetaStats), cudaMemcpyHostToDevice);
            
            cudaEventRecord(t0);
            
            int blocks = min((int)((sub_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 65535);
            theta_zoom_64bit_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                theta_start, sub_size, min_shell, max_shell, d_stats);
            
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            
            float ms;
            cudaEventElapsedTime(&ms, t0, t1);
            
            cudaMemcpy(&h_stats, d_stats, sizeof(ThetaStats), cudaMemcpyDeviceToHost);
            h_stats.time_ms = ms;
            h_stats.density = h_stats.total_tested ? (double)h_stats.prime_count / h_stats.total_tested : 0;
            
            if (h_stats.seen_min_shell == UINT32_MAX) h_stats.seen_min_shell = 0;
            if (h_stats.min_popcount == UINT32_MAX) h_stats.min_popcount = 0;
            if (h_stats.first_prime == UINT64_MAX) h_stats.first_prime = 0;
            
            print_csv_row(fp, &h_stats);
            total_primes += h_stats.prime_count;
            
            if (h_stats.density < min_density && h_stats.density > 0) min_density = h_stats.density;
            if (h_stats.density > max_density) max_density = h_stats.density;
            
            if (verbose >= 2 || (verbose == 1 && seg % 4 == 0)) {
                printf("Sub-seg %3d: theta=0x%016" PRIx64 " primes=%8" PRIu64 " density=%.10f %.0fms\n",
                       seg, theta_start, h_stats.prime_count, h_stats.density, ms);
            }
        }
        
        if (verbose) {
            printf("\n=== PHASE 2 RESULTS ===\n");
            printf("Total primes in zoom region: %" PRIu64 "\n", total_primes);
            printf("Density range: [%.10f, %.10f]\n", min_density, max_density);
            printf("Density spread: %.2f%%\n", (max_density - min_density) / min_density * 100);
        }
        
        fclose(fp);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(d_stats);
        return 0;
    }
    
    fprintf(stderr, "Unknown mode: %s\n", mode);
    print_usage(argv[0]);
    return 1;
}
