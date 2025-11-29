// ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
// This file remains for reference only.
// Do NOT use in production. No support, no guarantees.
//
// For the current CUDA implementation, see:
//   - cuda/theta_cuda_v1.2.cuh
//   - cuda/theta_cuda_benchmark_v1.2.cu

/**
 * theta_order_prime_analysis.cu - TRUE THETA ORDER Prime Analysis
 *
 * ============================================================================
 * THE PROBLEM WITH THE ORIGINAL CODE
 * ============================================================================
 *
 * The original theta_order_analysis.cu CLAIMED theta order but actually did:
 *
 *     for (i = 0; i < segment_size; i++)
 *         n = segment_start + i;    // <- NATURAL ORDER!
 *
 * This processes consecutive integers, so each segment has:
 *   - All integers in same shell range (min_shell ≈ max_shell)
 *   - No mixing of shells within a segment
 *
 * The CSV proved this: min_shell == max_shell in every segment.
 *
 * ============================================================================
 * TRUE THETA ORDER
 * ============================================================================
 *
 * Theta order iterates by ANGULAR POSITION, not by integer value.
 *
 * Key concepts:
 *   1. theta_key(odd) = bit_reverse(odd) - determines angular position
 *   2. Same theta_key exists in EVERY shell (at different radii)
 *   3. Integers along a "ray" share the same theta_key
 *
 * TRUE theta order iteration:
 *
 *     for (theta_pos = 0; theta_pos < max_theta; theta_pos++)
 *         for (shell = min_shell; shell <= max_shell; shell++)
 *             n = theta_to_integer(theta_pos, shell);
 *             // This n comes from DIFFERENT shells as we vary shell!
 *
 * Each theta position segment now contains integers from MULTIPLE shells.
 * This is the whole point: we can analyze angular sectors across all shells.
 *
 * ============================================================================
 * WHY THIS MATTERS
 * ============================================================================
 *
 * From IntegerNative_prompt.txt:
 *
 *   FACT 15 (Lower Shell Bounds Upper):
 *   Theta-key density characteristics of shell k provide UPPER BOUND on gap
 *   sizes for all higher shells.
 *
 * To test this, we need segments that contain integers from multiple shells
 * at the same angular positions. Natural order can't do this.
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

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define THRESHOLD_32BIT 0x100000000ULL

__constant__ uint64_t MR_WITNESSES_3[3] = {2, 7, 61};
__constant__ uint64_t MR_WITNESSES_12[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

// ============================================================================
// Bit operations for theta ordering
// ============================================================================

__device__ __host__ __forceinline__ uint64_t bit_reverse(uint64_t x, int k) {
    uint64_t result = 0;
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

__device__ __forceinline__ uint32_t compute_popcount(uint64_t n) {
    return __popcll(n);
}

/**
 * Get the odd integer at angular position theta_pos within shell k.
 *
 * Shell k has 2^(k-1) odd integers (for k >= 2).
 * theta_pos indexes them in angular (theta_key) order.
 *
 * The mapping: theta_pos -> odd integer is via bit reversal.
 */
__device__ __host__ uint64_t theta_to_odd(uint64_t theta_pos, uint32_t shell) {
    if (shell == 0) return (theta_pos == 0) ? 1 : 0;
    if (shell == 1) return (theta_pos == 0) ? 3 : 0;  // Shell 1 has only odd=3
    
    // Shell k >= 2: 2^(k-1) odd integers
    uint64_t num_odds = 1ULL << (shell - 1);
    if (theta_pos >= num_odds) return 0;  // Out of range
    
    // position j in [0, 2^(k-1)) that has theta_pos as its angular index
    uint64_t j = bit_reverse(theta_pos, shell - 1);
    
    // odd integer at position j in shell k
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
// Segment statistics
// ============================================================================

struct ThetaStats {
    uint64_t segment_id;
    uint64_t theta_start;
    uint64_t theta_count;
    uint32_t config_min_shell;
    uint32_t config_max_shell;
    
    // These should show MIXED shells if iteration is correct!
    uint32_t seen_min_shell;
    uint32_t seen_max_shell;
    
    uint64_t total_tested;
    uint64_t prime_count;
    
    double shell_sum;
    uint32_t min_popcount;
    uint32_t max_popcount;
    double popcount_sum;
    
    uint64_t first_prime;
    uint64_t last_prime;
    uint64_t twin_count;
    uint64_t mod6_1;
    uint64_t mod6_5;
    
    float time_ms;
};

// ============================================================================
// TRUE THETA ORDER KERNEL
// ============================================================================

__global__ void theta_order_kernel(
    uint64_t theta_start,
    uint64_t theta_count,
    uint32_t min_shell,
    uint32_t max_shell,
    ThetaStats* stats
) {
    // Thread-local accumulators
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
    uint64_t local_m1 = 0;
    uint64_t local_m5 = 0;
    
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    
    // Iterate theta positions (angular index)
    for (uint64_t t = idx; t < theta_count; t += stride) {
        uint64_t theta_pos = theta_start + t;
        
        // For EACH shell, get the odd at this theta position
        // THIS IS THE KEY: same theta_pos maps to different integers across shells
        for (uint32_t shell = min_shell; shell <= max_shell; shell++) {
            uint64_t odd = theta_to_odd(theta_pos, shell);
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
            
            bool prime = (odd < THRESHOLD_32BIT) ? is_prime_32((uint32_t)odd) : is_prime_64(odd);
            
            if (prime) {
                local_primes++;
                if (odd < local_first) local_first = odd;
                if (odd > local_last) local_last = odd;
                
                // Twin check
                uint64_t odd2 = odd + 2;
                bool p2 = (odd2 < THRESHOLD_32BIT) ? is_prime_32((uint32_t)odd2) : is_prime_64(odd2);
                if (p2) local_twins++;
                
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
    
    // Atomic updates from lane 0
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
        
        // First/last primes
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
        
        // Sums (CAS for double)
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
    s->config_min_shell = minS;
    s->config_max_shell = maxS;
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
                "first_prime,last_prime,twins,mod6_1,mod6_5,time_ms\n");
}

void print_csv_row(FILE* fp, const ThetaStats* s) {
    double density = s->total_tested ? (double)s->prime_count / s->total_tested : 0;
    double avg_shell = s->total_tested ? s->shell_sum / s->total_tested : 0;
    double avg_pop = s->total_tested ? s->popcount_sum / s->total_tested : 0;
    
    fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u-%u,%u-%u,"
                "%" PRIu64 ",%" PRIu64 ",%.10f,%.2f,%u-%u,%.2f,"
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.2f\n",
            s->segment_id, s->theta_start, s->theta_start + s->theta_count,
            s->config_min_shell, s->config_max_shell,
            s->seen_min_shell, s->seen_max_shell,
            s->total_tested, s->prime_count, density, avg_shell,
            s->min_popcount, s->max_popcount, avg_pop,
            s->first_prime, s->last_prime,
            s->twin_count, s->mod6_1, s->mod6_5, s->time_ms);
}

int main(int argc, char** argv) {
    uint32_t min_shell = 2;
    uint32_t max_shell = 24;   // Up to ~16M per shell
    int theta_exp = 14;        // 2^14 = 16384 theta positions per segment
    int num_segments = 32;
    const char* outfile = "theta_order_output.csv";
    int verbose = 1;
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--min-shell") && i+1 < argc) min_shell = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-shell") && i+1 < argc) max_shell = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--theta-exp") && i+1 < argc) theta_exp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--segments") && i+1 < argc) num_segments = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1 < argc) outfile = argv[++i];
        else if (!strcmp(argv[i], "-q")) verbose = 0;
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("theta_order_prime_analysis - TRUE THETA ORDER iteration\n\n");
            printf("Options:\n");
            printf("  --min-shell S    Min shell (default: 2)\n");
            printf("  --max-shell S    Max shell (default: 24)\n");
            printf("  --theta-exp E    Segment size = 2^E (default: 14)\n");
            printf("  --segments N     Number of segments (default: 32)\n");
            printf("  --output FILE    Output CSV (default: theta_order_output.csv)\n");
            printf("  -q               Quiet\n");
            printf("  -v               Verbose\n\n");
            printf("VERIFICATION: If output shows shell_seen with MIXED values\n");
            printf("(min != max across most segments), theta order is working.\n");
            return 0;
        }
    }
    
    uint64_t seg_size = 1ULL << theta_exp;
    
    if (verbose) {
        printf("=== TRUE THETA ORDER Prime Analysis ===\n");
        printf("Shells: %u to %u\n", min_shell, max_shell);
        printf("Theta segment: 2^%d = %" PRIu64 " positions\n", theta_exp, seg_size);
        printf("Segments: %d\n", num_segments);
        printf("Output: %s\n\n", outfile);
        printf("VERIFICATION CHECK:\n");
        printf("  shell_seen column should show RANGE (not single value)\n");
        printf("  e.g., '2-24' not '20-20'\n\n");
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
    
    for (int seg = 0; seg < num_segments; seg++) {
        uint64_t theta_start = (uint64_t)seg * seg_size;
        
        init_stats(&h_stats, seg, theta_start, seg_size, min_shell, max_shell);
        cudaMemcpy(d_stats, &h_stats, sizeof(ThetaStats), cudaMemcpyHostToDevice);
        
        cudaEventRecord(t0);
        
        int blocks = min((int)((seg_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), 65535);
        theta_order_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            theta_start, seg_size, min_shell, max_shell, d_stats);
        
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        
        cudaMemcpy(&h_stats, d_stats, sizeof(ThetaStats), cudaMemcpyDeviceToHost);
        h_stats.time_ms = ms;
        
        if (h_stats.seen_min_shell == UINT32_MAX) h_stats.seen_min_shell = 0;
        if (h_stats.min_popcount == UINT32_MAX) h_stats.min_popcount = 0;
        if (h_stats.first_prime == UINT64_MAX) h_stats.first_prime = 0;
        
        print_csv_row(fp, &h_stats);
        total_primes += h_stats.prime_count;
        
        if (verbose >= 2 || (verbose == 1 && seg % 8 == 0)) {
            printf("Seg %2d: theta=[%6" PRIu64 ",%6" PRIu64 ") shells=%u-%u primes=%8" PRIu64 " %.0fms\n",
                   seg, theta_start, theta_start + seg_size,
                   h_stats.seen_min_shell, h_stats.seen_max_shell,
                   h_stats.prime_count, ms);
        }
    }
    
    if (verbose) {
        printf("\nTotal primes: %" PRIu64 "\n", total_primes);
        printf("Output: %s\n", outfile);
    }
    
    fclose(fp);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_stats);
    
    return 0;
}
