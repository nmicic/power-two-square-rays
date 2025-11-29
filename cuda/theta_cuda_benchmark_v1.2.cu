// Copyright (c) 2025
// Author: Nenad Micic <nenad@micic.be>
// License: MIT License
//
// This file contains the supported implementation of the Theta Toolkit
// (CUDA version). See repository root README for details.
//
// Portions of this file include AI-assisted code generation
// (ChatGPT, Claude). All work reviewed and validated by the author.

/**
 * theta_cuda_benchmark_v1.2.cu - GPU Hash/Bucket Performance Benchmark
 *
 * NOTE: Results are observations from author's test environment (RTX 4000 Ada).
 *       Community validation on different hardware is welcome.
 *
 * Features:
 *   - Correctness verification: 1000 random samples + edge cases
 *   - Pattern modes: random, powers, shifted, sequential
 *   - CLI: ./theta_bench [count_M] [buckets] [rng_mode] [pattern]
 *
 * COMPILE:
 *   nvcc -O3 theta_cuda_benchmark_v1.2.cu -o theta_bench
 *
 *   # Specific architectures:
 *   nvcc -O3 -arch=sm_86 ...   # A4000, A6000, RTX 3090
 *   nvcc -O3 -arch=sm_89 ...   # RTX 4000/4090 Ada
 *   nvcc -O3 -arch=sm_90a ...  # H100
 *
 *   # Suppress harmless "unused variable" warning:
 *   nvcc -O3 -arch=sm_89 -Xcudafe "--diag_suppress=177" ...
 *
 * RUN:
 *   ./theta_bench                  # 10M, 64 buckets, full-range, random
 *   ./theta_bench 10 64 1 0        # Same (explicit)
 *   ./theta_bench 10 64 1 1        # Powers of two pattern
 *   ./theta_bench 10 64 1 2        # Shifted base pattern
 *   ./theta_bench 10 64 1 3        # Sequential pattern
 *   ./theta_bench --help
 *
 * PATTERNS:
 *   0 = random      : Uniform random values
 *   1 = powers      : n = 1 << k (cycling k=0..31/63) - KEY 2-ADIC TEST
 *   2 = shifted     : n = base << k (random base, random k)
 *   3 = sequential  : n = 1, 2, 3, ...
 *
 * Repository: https://github.com/nmicic/power-two-square-rays/
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

//=============================================================================
// CONFIGURATION
//=============================================================================

#define RNG_ODD_ONLY   0
#define RNG_FULL_RANGE 1

#define PATTERN_RANDOM     0
#define PATTERN_POWERS     1
#define PATTERN_SHIFTED    2
#define PATTERN_SEQUENTIAL 3

#define DEFAULT_COUNT_MILLIONS 10
#define DEFAULT_NUM_BUCKETS    64
#define DEFAULT_RNG_MODE       RNG_FULL_RANGE
#define DEFAULT_PATTERN        PATTERN_RANDOM
#define DEFAULT_ITERATIONS     10
#define BLOCK_SIZE             256
#define CORRECTNESS_RANDOM_N   1000

//=============================================================================
// CUDA ERROR HANDLING
//=============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

//=============================================================================
// THETA FUNCTIONS - GPU
//=============================================================================

__device__ __forceinline__
uint32_t theta_v2_32(uint32_t n) {
    if (n == 0) return 32;
    return __ffs(n) - 1;
}

__device__ __forceinline__
uint32_t theta_v2_64(uint64_t n) {
    if (n == 0) return 64;
    return __ffsll(n) - 1;
}

__device__ __forceinline__
uint32_t theta_odd_core_32(uint32_t n) {
    if (n == 0) return 0;
    return n >> theta_v2_32(n);
}

__device__ __forceinline__
uint64_t theta_odd_core_64(uint64_t n) {
    if (n == 0) return 0;
    return n >> theta_v2_64(n);
}

__device__ __forceinline__
uint32_t theta_bit_length_32(uint32_t n) {
    if (n == 0) return 0;
    return 32 - __clz(n);
}

__device__ __forceinline__
uint32_t theta_bit_length_64(uint64_t n) {
    if (n == 0) return 0;
    return 64 - __clzll(n);
}

__device__ __forceinline__
uint32_t theta_shell_32(uint32_t n) {
    uint32_t bl = theta_bit_length_32(n);
    return bl > 0 ? bl - 1 : 0;
}

__device__ __forceinline__
uint32_t theta_shell_64(uint64_t n) {
    uint32_t bl = theta_bit_length_64(n);
    return bl > 0 ? bl - 1 : 0;
}

__device__ __forceinline__
uint32_t theta_key_32(uint32_t n) {
    if (n == 0) return 0;
    uint32_t core = theta_odd_core_32(n);
    uint32_t bits = theta_bit_length_32(core);
    uint32_t reversed = __brev(core);
    return reversed >> (32 - bits);
}

__device__ __forceinline__
uint64_t theta_key_64(uint64_t n) {
    if (n == 0) return 0;
    uint64_t core = theta_odd_core_64(n);
    uint32_t bits = theta_bit_length_64(core);
    uint32_t lo = (uint32_t)core;
    uint32_t hi = (uint32_t)(core >> 32);
    uint64_t reversed = ((uint64_t)__brev(lo) << 32) | __brev(hi);
    return reversed >> (64 - bits);
}

__device__ __forceinline__
uint32_t theta_bucket_32(uint32_t n, uint32_t num_buckets) {
    uint32_t key = theta_key_32(n);
    uint32_t v = theta_v2_32(n);
    uint32_t sh = theta_shell_32(n);
    
    uint64_t h = key;
    h = h * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)v * 0x517CC1B727220A95ULL;
    h ^= (uint64_t)sh * 0x2545F4914F6CDD1DULL;
    h = (h * 0x9E3779B97F4A7C15ULL) >> 32;
    
    return (uint32_t)(h % num_buckets);
}

__device__ __forceinline__
uint32_t theta_bucket_64(uint64_t n, uint32_t num_buckets) {
    uint64_t key = theta_key_64(n);
    uint32_t v = theta_v2_64(n);
    uint32_t sh = theta_shell_64(n);
    
    uint64_t h = key;
    h = h * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)v * 0x517CC1B727220A95ULL;
    h ^= (uint64_t)sh * 0x2545F4914F6CDD1DULL;
    h = (h * 0x9E3779B97F4A7C15ULL) >> 32;
    
    return (uint32_t)(h % num_buckets);
}

//=============================================================================
// THETA FUNCTIONS - CPU REFERENCE
//=============================================================================

inline uint32_t cpu_v2_32(uint32_t n) {
    if (n == 0) return 32;
    uint32_t count = 0;
    while ((n & 1) == 0) { count++; n >>= 1; }
    return count;
}

inline uint32_t cpu_v2_64(uint64_t n) {
    if (n == 0) return 64;
    uint32_t count = 0;
    while ((n & 1) == 0) { count++; n >>= 1; }
    return count;
}

inline uint32_t cpu_bit_length_32(uint32_t n) {
    if (n == 0) return 0;
    uint32_t bits = 0;
    while (n) { bits++; n >>= 1; }
    return bits;
}

inline uint32_t cpu_bit_length_64(uint64_t n) {
    if (n == 0) return 0;
    uint32_t bits = 0;
    while (n) { bits++; n >>= 1; }
    return bits;
}

inline uint32_t cpu_bit_reverse_32(uint32_t val, uint32_t bits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < bits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

inline uint64_t cpu_bit_reverse_64(uint64_t val, uint32_t bits) {
    uint64_t result = 0;
    for (uint32_t i = 0; i < bits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

inline uint32_t cpu_theta_key_32(uint32_t n) {
    if (n == 0) return 0;
    uint32_t v = cpu_v2_32(n);
    uint32_t core = n >> v;
    uint32_t bits = cpu_bit_length_32(core);
    return cpu_bit_reverse_32(core, bits);
}

inline uint64_t cpu_theta_key_64(uint64_t n) {
    if (n == 0) return 0;
    uint32_t v = cpu_v2_64(n);
    uint64_t core = n >> v;
    uint32_t bits = cpu_bit_length_64(core);
    return cpu_bit_reverse_64(core, bits);
}

inline uint32_t cpu_theta_bucket_32(uint32_t n, uint32_t num_buckets) {
    uint32_t key = cpu_theta_key_32(n);
    uint32_t v = cpu_v2_32(n);
    uint32_t sh = n == 0 ? 0 : cpu_bit_length_32(n) - 1;
    
    uint64_t h = key;
    h = h * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)v * 0x517CC1B727220A95ULL;
    h ^= (uint64_t)sh * 0x2545F4914F6CDD1DULL;
    h = (h * 0x9E3779B97F4A7C15ULL) >> 32;
    
    return (uint32_t)(h % num_buckets);
}

inline uint32_t cpu_theta_bucket_64(uint64_t n, uint32_t num_buckets) {
    uint64_t key = cpu_theta_key_64(n);
    uint32_t v = cpu_v2_64(n);
    uint32_t sh = n == 0 ? 0 : cpu_bit_length_64(n) - 1;
    
    uint64_t h = key;
    h = h * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)v * 0x517CC1B727220A95ULL;
    h ^= (uint64_t)sh * 0x2545F4914F6CDD1DULL;
    h = (h * 0x9E3779B97F4A7C15ULL) >> 32;
    
    return (uint32_t)(h % num_buckets);
}

//=============================================================================
// COMPARISON HASH FUNCTIONS - GPU
//=============================================================================

__device__ __forceinline__
uint32_t xxhash32(uint32_t input) {
    const uint32_t PRIME1 = 0x9E3779B1U;
    const uint32_t PRIME2 = 0x85EBCA77U;
    const uint32_t PRIME3 = 0xC2B2AE3DU;
    const uint32_t PRIME5 = 0x165667B1U;
    
    uint32_t h = PRIME5 + 4;
    h += input * PRIME3;
    h = ((h << 17) | (h >> 15)) * PRIME2;
    h ^= h >> 15;
    h *= PRIME2;
    h ^= h >> 13;
    h *= PRIME3;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__
uint64_t xxhash64(uint64_t input) {
    const uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
    const uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
    const uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;
    
    uint64_t h = PRIME5 + 8;
    uint64_t k = input * PRIME2;
    k = ((k << 31) | (k >> 33)) * PRIME1;
    h ^= k;
    h = ((h << 27) | (h >> 37)) * PRIME1 + PRIME2;
    h ^= h >> 33;
    h *= PRIME2;
    h ^= h >> 29;
    h *= PRIME1;
    h ^= h >> 32;
    return h;
}

__device__ __forceinline__
uint32_t crc32_hash(uint32_t input) {
    uint32_t crc = 0xFFFFFFFF;
    uint8_t* bytes = (uint8_t*)&input;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        crc ^= bytes[i];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return crc ^ 0xFFFFFFFF;
}

__device__ __forceinline__
uint32_t murmur3_32(uint32_t input) {
    uint32_t h = 0x971e137b;
    uint32_t k = input;
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
    h ^= 4;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__
uint64_t murmur3_64(uint64_t input) {
    uint64_t h = 0x971e137b971e137bULL;
    uint64_t k = input;
    k *= 0x87c37b91114253d5ULL;
    k = (k << 31) | (k >> 33);
    k *= 0x4cf5ad432745937fULL;
    h ^= k;
    h = (h << 27) | (h >> 37);
    h = h * 5 + 0x52dce729;
    h ^= 8;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

//=============================================================================
// BENCHMARK KERNELS
//=============================================================================

__global__ void bench_theta_key_32(uint32_t* input, uint32_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = theta_key_32(input[idx]);
}

__global__ void bench_theta_bucket_32(uint32_t* input, uint32_t* output, int n, uint32_t num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = theta_bucket_32(input[idx], num_buckets);
}

__global__ void bench_xxhash_32(uint32_t* input, uint32_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = xxhash32(input[idx]);
}

__global__ void bench_crc32(uint32_t* input, uint32_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = crc32_hash(input[idx]);
}

__global__ void bench_murmur3_32(uint32_t* input, uint32_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = murmur3_32(input[idx]);
}

__global__ void bench_theta_key_64(uint64_t* input, uint64_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = theta_key_64(input[idx]);
}

__global__ void bench_theta_bucket_64(uint64_t* input, uint32_t* output, int n, uint32_t num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = theta_bucket_64(input[idx], num_buckets);
}

__global__ void bench_xxhash_64(uint64_t* input, uint64_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = xxhash64(input[idx]);
}

__global__ void bench_murmur3_64(uint64_t* input, uint64_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = murmur3_64(input[idx]);
}

__global__ void count_buckets(uint32_t* buckets, uint32_t* counts, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(&counts[buckets[idx]], 1);
}

//=============================================================================
// DATA GENERATION KERNELS
//=============================================================================

__global__ void generate_random_32(uint32_t* data, int n, unsigned long long seed, int rng_mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        uint32_t val = curand(&state);
        if (rng_mode == RNG_ODD_ONLY) {
            val |= 1;
        } else {
            if (val == 0) val = 1;
        }
        data[idx] = val;
    }
}

__global__ void generate_random_64(uint64_t* data, int n, unsigned long long seed, int rng_mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        uint64_t lo = curand(&state);
        uint64_t hi = curand(&state);
        uint64_t val = (hi << 32) | lo;
        if (rng_mode == RNG_ODD_ONLY) {
            val |= 1;
        } else {
            if (val == 0) val = 1;
        }
        data[idx] = val;
    }
}

// Pattern: Powers of two - n = 1 << k
__global__ void generate_powers_32(uint32_t* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int k = idx % 31;
        data[idx] = 1u << k;
    }
}

__global__ void generate_powers_64(uint64_t* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int k = idx % 63;
        data[idx] = 1ULL << k;
    }
}

// Pattern: Shifted base - n = base << k
__global__ void generate_shifted_32(uint32_t* data, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        uint32_t base = (curand(&state) % 1000) + 1;
        int k = curand(&state) % 22;
        uint32_t val = base << k;
        if (val == 0) val = 1;
        data[idx] = val;
    }
}

__global__ void generate_shifted_64(uint64_t* data, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        uint64_t base = (curand(&state) % 1000) + 1;
        int k = curand(&state) % 54;
        uint64_t val = base << k;
        if (val == 0) val = 1;
        data[idx] = val;
    }
}

// Pattern: Sequential - n = 1, 2, 3, ...
__global__ void generate_sequential_32(uint32_t* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx + 1;
    }
}

__global__ void generate_sequential_64(uint64_t* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = (uint64_t)idx + 1;
    }
}

//=============================================================================
// CORRECTNESS CHECKS
//=============================================================================

bool run_correctness_checks() {
    printf("\n=== CORRECTNESS CHECKS ===\n\n");
    
    // Edge cases
    uint32_t edge_32[] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        15, 16, 17, 31, 32, 33,
        255, 256, 257,
        0xFFFFFFFF, 0x80000000, 0x55555555, 0xAAAAAAAA,
        12345678, 87654321
    };
    int n_edge_32 = sizeof(edge_32) / sizeof(edge_32[0]);
    
    uint64_t edge_64[] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        0xFFFFFFFFULL, 0x100000000ULL, 0x100000001ULL,
        0xFFFFFFFFFFFFFFFFULL, 0x8000000000000000ULL,
        0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL,
        123456789012345ULL
    };
    int n_edge_64 = sizeof(edge_64) / sizeof(edge_64[0]);
    
    // Generate random test values
    srand(42);
    uint32_t rand_32[CORRECTNESS_RANDOM_N];
    uint64_t rand_64[CORRECTNESS_RANDOM_N];
    for (int i = 0; i < CORRECTNESS_RANDOM_N; i++) {
        rand_32[i] = ((uint32_t)rand() << 16) ^ rand();
        if (rand_32[i] == 0) rand_32[i] = 1;
        rand_64[i] = ((uint64_t)rand() << 48) ^ ((uint64_t)rand() << 32) ^ ((uint64_t)rand() << 16) ^ rand();
        if (rand_64[i] == 0) rand_64[i] = 1;
    }
    
    int n_test_32 = n_edge_32 + CORRECTNESS_RANDOM_N;
    int n_test_64 = n_edge_64 + CORRECTNESS_RANDOM_N;
    
    uint32_t* test_32 = (uint32_t*)malloc(n_test_32 * sizeof(uint32_t));
    uint64_t* test_64 = (uint64_t*)malloc(n_test_64 * sizeof(uint64_t));
    
    memcpy(test_32, edge_32, n_edge_32 * sizeof(uint32_t));
    memcpy(test_32 + n_edge_32, rand_32, CORRECTNESS_RANDOM_N * sizeof(uint32_t));
    memcpy(test_64, edge_64, n_edge_64 * sizeof(uint64_t));
    memcpy(test_64 + n_edge_64, rand_64, CORRECTNESS_RANDOM_N * sizeof(uint64_t));
    
    // GPU memory
    uint32_t *d_in_32, *d_out_key_32, *d_out_bucket_32;
    uint64_t *d_in_64, *d_out_key_64;
    uint32_t *d_out_bucket_64;
    
    CUDA_CHECK(cudaMalloc(&d_in_32, n_test_32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out_key_32, n_test_32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out_bucket_32, n_test_32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_in_64, n_test_64 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_key_64, n_test_64 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_bucket_64, n_test_64 * sizeof(uint32_t)));
    
    CUDA_CHECK(cudaMemcpy(d_in_32, test_32, n_test_32 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_64, test_64, n_test_64 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    int grid = (n_test_32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bench_theta_key_32<<<grid, BLOCK_SIZE>>>(d_in_32, d_out_key_32, n_test_32);
    bench_theta_bucket_32<<<grid, BLOCK_SIZE>>>(d_in_32, d_out_bucket_32, n_test_32, 64);
    
    grid = (n_test_64 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bench_theta_key_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_out_key_64, n_test_64);
    bench_theta_bucket_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_out_bucket_64, n_test_64, 64);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint32_t* gpu_key_32 = (uint32_t*)malloc(n_test_32 * sizeof(uint32_t));
    uint32_t* gpu_bucket_32 = (uint32_t*)malloc(n_test_32 * sizeof(uint32_t));
    uint64_t* gpu_key_64 = (uint64_t*)malloc(n_test_64 * sizeof(uint64_t));
    uint32_t* gpu_bucket_64 = (uint32_t*)malloc(n_test_64 * sizeof(uint32_t));
    
    CUDA_CHECK(cudaMemcpy(gpu_key_32, d_out_key_32, n_test_32 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_bucket_32, d_out_bucket_32, n_test_32 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_key_64, d_out_key_64, n_test_64 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_bucket_64, d_out_bucket_64, n_test_64 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    int errors_32 = 0, errors_64 = 0;
    
    printf("32-bit theta_key (%d edge + %d random = %d total):\n", n_edge_32, CORRECTNESS_RANDOM_N, n_test_32);
    for (int i = 0; i < n_test_32; i++) {
        uint32_t cpu_key = cpu_theta_key_32(test_32[i]);
        uint32_t cpu_bucket = cpu_theta_bucket_32(test_32[i], 64);
        if (gpu_key_32[i] != cpu_key || gpu_bucket_32[i] != cpu_bucket) {
            if (errors_32 < 5) {
                printf("  FAIL: n=%u, GPU key=%u vs CPU key=%u\n", test_32[i], gpu_key_32[i], cpu_key);
            }
            errors_32++;
        }
    }
    printf("  CPU<->GPU mismatches: %d / %d %s\n", errors_32, n_test_32, errors_32 == 0 ? "PASSED" : "FAILED");
    
    printf("\n64-bit theta_key (%d edge + %d random = %d total):\n", n_edge_64, CORRECTNESS_RANDOM_N, n_test_64);
    for (int i = 0; i < n_test_64; i++) {
        uint64_t cpu_key = cpu_theta_key_64(test_64[i]);
        uint32_t cpu_bucket = cpu_theta_bucket_64(test_64[i], 64);
        if (gpu_key_64[i] != cpu_key || gpu_bucket_64[i] != cpu_bucket) {
            if (errors_64 < 5) {
                printf("  FAIL: n=%llu, GPU key=%llu vs CPU key=%llu\n",
                       (unsigned long long)test_64[i], (unsigned long long)gpu_key_64[i], (unsigned long long)cpu_key);
            }
            errors_64++;
        }
    }
    printf("  CPU<->GPU mismatches: %d / %d %s\n", errors_64, n_test_64, errors_64 == 0 ? "PASSED" : "FAILED");
    
    free(test_32); free(test_64);
    free(gpu_key_32); free(gpu_bucket_32);
    free(gpu_key_64); free(gpu_bucket_64);
    CUDA_CHECK(cudaFree(d_in_32));
    CUDA_CHECK(cudaFree(d_out_key_32));
    CUDA_CHECK(cudaFree(d_out_bucket_32));
    CUDA_CHECK(cudaFree(d_in_64));
    CUDA_CHECK(cudaFree(d_out_key_64));
    CUDA_CHECK(cudaFree(d_out_bucket_64));
    
    int total_errors = errors_32 + errors_64;
    printf("\nCORRECTNESS: %s\n", total_errors == 0 ? "PASS" : "FAIL");
    return total_errors == 0;
}

//=============================================================================
// BENCHMARK HELPERS
//=============================================================================

double compute_chi_square(uint32_t* counts, int num_buckets, int total) {
    double expected = (double)total / num_buckets;
    double chi_sq = 0;
    for (int i = 0; i < num_buckets; i++) {
        double diff = counts[i] - expected;
        chi_sq += (diff * diff) / expected;
    }
    return chi_sq;
}

const char* uniformity_verdict(double chi_sq, int df) {
    double std_dev = sqrt(2.0 * df);
    double deviation = fabs(chi_sq - df) / std_dev;
    if (deviation < 2) return "GOOD";
    if (deviation < 3) return "MARGINAL";
    return "POOR";
}

struct BenchResult {
    const char* name;
    double time_ms;
    double mops;
    double chi_sq;
    const char* verdict;
};

void run_hash_benchmark_32(const char* name,
                           void (*kernel)(uint32_t*, uint32_t*, int),
                           uint32_t* d_in, uint32_t* d_out, int n, int iters,
                           BenchResult* result) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    kernel<<<grid, BLOCK_SIZE>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        kernel<<<grid, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    result->name = name;
    result->time_ms = ms / iters;
    result->mops = (double)n / (result->time_ms * 1000.0);
    result->chi_sq = 0;
    result->verdict = "N/A";
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

//=============================================================================
// MAIN
//=============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [count_M] [buckets] [rng_mode] [pattern]\n", prog);
    printf("\n");
    printf("  count_M  : Millions of flow IDs (default: %d)\n", DEFAULT_COUNT_MILLIONS);
    printf("  buckets  : Number of buckets (default: %d)\n", DEFAULT_NUM_BUCKETS);
    printf("  rng_mode : 0=odd-only, 1=full-range (default: %d)\n", DEFAULT_RNG_MODE);
    printf("  pattern  : 0=random, 1=powers, 2=shifted, 3=sequential (default: 0)\n");
    printf("\n");
    printf("Patterns:\n");
    printf("  0 (random)     : Uniform random values\n");
    printf("  1 (powers)     : n = 1 << k, cycling k=0..31/63  [KEY 2-ADIC TEST]\n");
    printf("  2 (shifted)    : n = base << k, random base/k\n");
    printf("  3 (sequential) : n = 1, 2, 3, ...\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                  # 10M, 64 buckets, full-range, random\n", prog);
    printf("  %s 10 64 1 1        # Powers of two pattern\n", prog);
    printf("  %s 100 256 1 0      # 100M, 256 buckets, random\n", prog);
}

int main(int argc, char** argv) {
    if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    
    int count_millions = argc > 1 ? atoi(argv[1]) : DEFAULT_COUNT_MILLIONS;
    int num_buckets = argc > 2 ? atoi(argv[2]) : DEFAULT_NUM_BUCKETS;
    int rng_mode = argc > 3 ? atoi(argv[3]) : DEFAULT_RNG_MODE;
    int pattern = argc > 4 ? atoi(argv[4]) : DEFAULT_PATTERN;
    
    if (count_millions <= 0) count_millions = DEFAULT_COUNT_MILLIONS;
    if (num_buckets <= 0) num_buckets = DEFAULT_NUM_BUCKETS;
    if (pattern < 0 || pattern > 3) pattern = PATTERN_RANDOM;
    
    const char* pattern_names[] = {"random", "powers", "shifted", "sequential"};
    
    int n = count_millions * 1000000;
    int iters = DEFAULT_ITERATIONS;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("======================================================================\n");
    printf("  THETA CUDA BENCHMARK v1.2\n");
    printf("======================================================================\n");
    printf("  GPU: %s\n", prop.name);
    printf("  Compute: %d.%d, SMs: %d, Memory: %.1f GB\n",
           prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1e9);
    printf("  Flow IDs: %d M\n", count_millions);
    printf("  Buckets: %d\n", num_buckets);
    printf("  RNG mode: %s\n", rng_mode == RNG_ODD_ONLY ? "odd-only" : "full-range");
    printf("  Pattern: %s\n", pattern_names[pattern]);
    printf("  Iterations: %d\n", iters);
    printf("======================================================================\n");
    
    // Correctness
    bool correct = run_correctness_checks();
    if (!correct) {
        printf("\nAborting due to correctness failures.\n");
        return 1;
    }
    
    //=========================================================================
    // 32-BIT BENCHMARKS
    //=========================================================================
    printf("\n=== 32-BIT BENCHMARKS ===\n");
    
    uint32_t *d_in_32, *d_out_32;
    CUDA_CHECK(cudaMalloc(&d_in_32, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out_32, n * sizeof(uint32_t)));
    
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("\nGenerating %dM %s 32-bit values...\n", count_millions, pattern_names[pattern]);
    switch (pattern) {
        case PATTERN_POWERS:     generate_powers_32<<<grid, BLOCK_SIZE>>>(d_in_32, n); break;
        case PATTERN_SHIFTED:    generate_shifted_32<<<grid, BLOCK_SIZE>>>(d_in_32, n, 42); break;
        case PATTERN_SEQUENTIAL: generate_sequential_32<<<grid, BLOCK_SIZE>>>(d_in_32, n); break;
        default:                 generate_random_32<<<grid, BLOCK_SIZE>>>(d_in_32, n, 42, rng_mode); break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Hash speed
    printf("\n1. HASH SPEED (32-bit)\n");
    printf("   %-15s %10s %12s\n", "Hash", "Time(ms)", "M ops/sec");
    printf("   %-15s %10s %12s\n", "---------------", "----------", "------------");
    
    BenchResult res[4];
    run_hash_benchmark_32("theta_key", bench_theta_key_32, d_in_32, d_out_32, n, iters, &res[0]);
    run_hash_benchmark_32("xxhash32", bench_xxhash_32, d_in_32, d_out_32, n, iters, &res[1]);
    run_hash_benchmark_32("crc32", bench_crc32, d_in_32, d_out_32, n, iters, &res[2]);
    run_hash_benchmark_32("murmur3", bench_murmur3_32, d_in_32, d_out_32, n, iters, &res[3]);
    
    for (int i = 0; i < 4; i++) {
        printf("   %-15s %10.2f %12.1f\n", res[i].name, res[i].time_ms, res[i].mops);
    }
    
    // Bucket uniformity
    int bucket_counts[] = {64, 128, 256};
    printf("\n2. BUCKET UNIFORMITY (32-bit)\n");
    
    for (int b = 0; b < 3; b++) {
        int nb = bucket_counts[b];
        printf("\n   %d buckets:\n", nb);
        printf("   %-15s %10s %12s %10s %10s\n", "Hash", "Time(ms)", "M ops/sec", "Chi-Sq", "Verdict");
        printf("   %-15s %10s %12s %10s %10s\n", "---------------", "----------", "------------", "----------", "----------");
        
        // theta_bucket
        {
            bench_theta_bucket_32<<<grid, BLOCK_SIZE>>>(d_in_32, d_out_32, n, nb);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++) {
                bench_theta_bucket_32<<<grid, BLOCK_SIZE>>>(d_in_32, d_out_32, n, nb);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            
            uint32_t* d_counts;
            CUDA_CHECK(cudaMalloc(&d_counts, nb * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(d_counts, 0, nb * sizeof(uint32_t)));
            count_buckets<<<grid, BLOCK_SIZE>>>(d_out_32, d_counts, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            uint32_t* h_counts = (uint32_t*)malloc(nb * sizeof(uint32_t));
            CUDA_CHECK(cudaMemcpy(h_counts, d_counts, nb * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            
            double chi = compute_chi_square(h_counts, nb, n);
            double time_ms = ms / iters;
            double mops = (double)n / (time_ms * 1000.0);
            
            printf("   %-15s %10.2f %12.1f %10.1f %10s\n",
                   "theta_bucket", time_ms, mops, chi, uniformity_verdict(chi, nb-1));
            
            free(h_counts);
            CUDA_CHECK(cudaFree(d_counts));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
        
        // Other hashes
        const char* names[] = {"xxhash32", "crc32", "murmur3"};
        void (*kernels[])(uint32_t*, uint32_t*, int) = {bench_xxhash_32, bench_crc32, bench_murmur3_32};
        
        for (int h = 0; h < 3; h++) {
            kernels[h]<<<grid, BLOCK_SIZE>>>(d_in_32, d_out_32, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            uint32_t* h_out = (uint32_t*)malloc(n * sizeof(uint32_t));
            CUDA_CHECK(cudaMemcpy(h_out, d_out_32, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            
            uint32_t* counts = (uint32_t*)calloc(nb, sizeof(uint32_t));
            for (int i = 0; i < n; i++) counts[h_out[i] % nb]++;
            
            double chi = compute_chi_square(counts, nb, n);
            printf("   %-15s %10s %12s %10.1f %10s\n",
                   names[h], "-", "-", chi, uniformity_verdict(chi, nb-1));
            
            free(h_out);
            free(counts);
        }
    }
    
    CUDA_CHECK(cudaFree(d_in_32));
    CUDA_CHECK(cudaFree(d_out_32));
    
    //=========================================================================
    // 64-BIT BENCHMARKS
    //=========================================================================
    printf("\n=== 64-BIT BENCHMARKS ===\n");
    
    uint64_t *d_in_64, *d_out_64;
    uint32_t *d_bucket_out;
    CUDA_CHECK(cudaMalloc(&d_in_64, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_64, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_bucket_out, n * sizeof(uint32_t)));
    
    printf("\nGenerating %dM %s 64-bit values...\n", count_millions, pattern_names[pattern]);
    switch (pattern) {
        case PATTERN_POWERS:     generate_powers_64<<<grid, BLOCK_SIZE>>>(d_in_64, n); break;
        case PATTERN_SHIFTED:    generate_shifted_64<<<grid, BLOCK_SIZE>>>(d_in_64, n, 42); break;
        case PATTERN_SEQUENTIAL: generate_sequential_64<<<grid, BLOCK_SIZE>>>(d_in_64, n); break;
        default:                 generate_random_64<<<grid, BLOCK_SIZE>>>(d_in_64, n, 42, rng_mode); break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Hash speed
    printf("\n3. HASH SPEED (64-bit)\n");
    printf("   %-15s %10s %12s\n", "Hash", "Time(ms)", "M ops/sec");
    printf("   %-15s %10s %12s\n", "---------------", "----------", "------------");
    
    BenchResult res64[3];
    
    // theta_key_64
    {
        bench_theta_key_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_out_64, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) bench_theta_key_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_out_64, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        res64[0].name = "theta_key"; res64[0].time_ms = ms/iters; res64[0].mops = (double)n/(res64[0].time_ms*1000.0);
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // xxhash64
    {
        bench_xxhash_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_out_64, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) bench_xxhash_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_out_64, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        res64[1].name = "xxhash64"; res64[1].time_ms = ms/iters; res64[1].mops = (double)n/(res64[1].time_ms*1000.0);
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // murmur3_64
    {
        bench_murmur3_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_out_64, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) bench_murmur3_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_out_64, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        res64[2].name = "murmur3_64"; res64[2].time_ms = ms/iters; res64[2].mops = (double)n/(res64[2].time_ms*1000.0);
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    for (int i = 0; i < 3; i++) {
        printf("   %-15s %10.2f %12.1f\n", res64[i].name, res64[i].time_ms, res64[i].mops);
    }
    
    // 64-bit bucket uniformity
    printf("\n4. BUCKET UNIFORMITY (64-bit, %d buckets)\n", num_buckets);
    printf("   %-15s %10s %12s %10s %10s\n", "Hash", "Time(ms)", "M ops/sec", "Chi-Sq", "Verdict");
    printf("   %-15s %10s %12s %10s %10s\n", "---------------", "----------", "------------", "----------", "----------");
    
    {
        bench_theta_bucket_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_bucket_out, n, num_buckets);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) bench_theta_bucket_64<<<grid, BLOCK_SIZE>>>(d_in_64, d_bucket_out, n, num_buckets);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        
        uint32_t* d_counts;
        CUDA_CHECK(cudaMalloc(&d_counts, num_buckets * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_counts, 0, num_buckets * sizeof(uint32_t)));
        count_buckets<<<grid, BLOCK_SIZE>>>(d_bucket_out, d_counts, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint32_t* h_counts = (uint32_t*)malloc(num_buckets * sizeof(uint32_t));
        CUDA_CHECK(cudaMemcpy(h_counts, d_counts, num_buckets * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        
        double chi = compute_chi_square(h_counts, num_buckets, n);
        double time_ms = ms / iters;
        double mops = (double)n / (time_ms * 1000.0);
        
        printf("   %-15s %10.2f %12.1f %10.1f %10s\n",
               "theta_bucket", time_ms, mops, chi, uniformity_verdict(chi, num_buckets-1));
        
        free(h_counts);
        CUDA_CHECK(cudaFree(d_counts));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    CUDA_CHECK(cudaFree(d_in_64));
    CUDA_CHECK(cudaFree(d_out_64));
    CUDA_CHECK(cudaFree(d_bucket_out));
    
    //=========================================================================
    // SUMMARY
    //=========================================================================
    printf("\n======================================================================\n");
    printf("  SUMMARY\n");
    printf("======================================================================\n");
    printf("\n  32-bit: theta_key %.1f M/s vs xxhash %.1f M/s (%.2fx)\n",
           res[0].mops, res[1].mops, res[0].mops / res[1].mops);
    printf("  64-bit: theta_key %.1f M/s vs xxhash %.1f M/s (%.2fx)\n",
           res64[0].mops, res64[1].mops, res64[0].mops / res64[1].mops);
    printf("\n  RNG mode: %s\n", rng_mode == RNG_ODD_ONLY ? "odd-only" : "full-range");
    printf("  Pattern: %s\n", pattern_names[pattern]);
    printf("\n======================================================================\n");
    printf("  VERDICT: PASS\n");
    printf("======================================================================\n");
    
    return 0;
}
