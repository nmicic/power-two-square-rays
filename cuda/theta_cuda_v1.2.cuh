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
 * theta_cuda_v1.2.cuh - CUDA Header-Only Library for Theta Operations
 *
 * Integer-native angular encoding via 2-adic decomposition.
 * All operations use only bitwise/integer ops - no floating point in core logic.
 *
 * CORE PRINCIPLE:
 *     Every positive integer n has unique decomposition: n = 2^v2(n) × core(n)
 *     theta_key(n) = bit_reverse(odd_core(n))
 *
 * USE CASES:
 *     - Angle calculation without trigonometry
 *     - Feature hashing for ML (deterministic, GPU-friendly)
 *     - Parallel checksums
 *     - Spatial hashing / locality-sensitive hashing
 *     - 2D/3D rotation helpers
 *
 * USAGE:
 *     #include "theta_cuda_v1.2.cuh"
 *
 *     __global__ void my_kernel(uint32_t* data, uint32_t* keys, int n) {
 *         int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *         if (idx < n) {
 *             keys[idx] = theta::theta_key(data[idx]);
 *         }
 *     }
 *
 * Repository: https://github.com/nmicic/power-two-square-rays/
 *
 * DISCLAIMER:
 *     This is an AI-assisted exploratory project.
 *     Educational and experimental only - not peer-reviewed.
 */

#ifndef THETA_CUDA_CUH
#define THETA_CUDA_CUH

#include <cstdint>

namespace theta {

//=============================================================================
// DEVICE/HOST MACROS
//=============================================================================

#ifdef __CUDACC__
    #define THETA_DEVICE __device__
    #define THETA_HOST __host__
    #define THETA_INLINE __forceinline__
#else
    #define THETA_DEVICE
    #define THETA_HOST
    #define THETA_INLINE inline
#endif

#define THETA_HD THETA_HOST THETA_DEVICE


//=============================================================================
// EDGE/QUADRANT CONSTANTS
//=============================================================================

enum Edge : uint8_t {
    EDGE_TOP    = 0,  // 00 binary
    EDGE_RIGHT  = 1,  // 01 binary
    EDGE_BOTTOM = 2,  // 10 binary
    EDGE_LEFT   = 3   // 11 binary
};


//=============================================================================
// CORE PRIMITIVES - 32-bit
//=============================================================================

/**
 * Count trailing zeros (2-adic valuation)
 * v2(12) = 2 because 12 = 0b1100
 */
THETA_HD THETA_INLINE
uint32_t v2_32(uint32_t n) {
    if (n == 0) return 32;
    #ifdef __CUDA_ARCH__
        return __ffs(n) - 1;  // CUDA intrinsic: find first set bit
    #else
        return __builtin_ctz(n);  // GCC/Clang intrinsic
    #endif
}

/**
 * Extract odd core: n >> v2(n)
 * odd_core(12) = 3 because 12 = 4 × 3
 */
THETA_HD THETA_INLINE
uint32_t odd_core_32(uint32_t n) {
    if (n == 0) return 0;
    return n >> v2_32(n);
}

/**
 * Bit length = floor(log2(n)) + 1
 * bit_length(12) = 4
 */
THETA_HD THETA_INLINE
uint32_t bit_length_32(uint32_t n) {
    if (n == 0) return 0;
    #ifdef __CUDA_ARCH__
        return 32 - __clz(n);  // CUDA intrinsic: count leading zeros
    #else
        return 32 - __builtin_clz(n);
    #endif
}

/**
 * Shell index = floor(log2(n)) = bit_length - 1
 * shell(12) = 3 because 8 <= 12 < 16
 */
THETA_HD THETA_INLINE
uint32_t shell_32(uint32_t n) {
    uint32_t bl = bit_length_32(n);
    return bl > 0 ? bl - 1 : 0;
}

/**
 * Reverse bits within specified width
 * bit_reverse(0b1100, 4) = 0b0011
 */
THETA_HD THETA_INLINE
uint32_t bit_reverse_32(uint32_t val, uint32_t bits) {
    if (bits == 0) return 0;
    
    #ifdef __CUDA_ARCH__
        // Use CUDA's __brev and shift
        return __brev(val) >> (32 - bits);
    #else
        // Portable implementation
        uint32_t result = 0;
        for (uint32_t i = 0; i < bits; i++) {
            result = (result << 1) | (val & 1);
            val >>= 1;
        }
        return result;
    #endif
}

/**
 * Population count (Hamming weight)
 */
THETA_HD THETA_INLINE
uint32_t popcount_32(uint32_t n) {
    #ifdef __CUDA_ARCH__
        return __popc(n);
    #else
        return __builtin_popcount(n);
    #endif
}


//=============================================================================
// CORE PRIMITIVES - 64-bit
//=============================================================================

THETA_HD THETA_INLINE
uint32_t v2_64(uint64_t n) {
    if (n == 0) return 64;
    #ifdef __CUDA_ARCH__
        return __ffsll(n) - 1;
    #else
        return __builtin_ctzll(n);
    #endif
}

THETA_HD THETA_INLINE
uint64_t odd_core_64(uint64_t n) {
    if (n == 0) return 0;
    return n >> v2_64(n);
}

THETA_HD THETA_INLINE
uint32_t bit_length_64(uint64_t n) {
    if (n == 0) return 0;
    #ifdef __CUDA_ARCH__
        return 64 - __clzll(n);
    #else
        return 64 - __builtin_clzll(n);
    #endif
}

THETA_HD THETA_INLINE
uint32_t shell_64(uint64_t n) {
    uint32_t bl = bit_length_64(n);
    return bl > 0 ? bl - 1 : 0;
}

THETA_HD THETA_INLINE
uint64_t bit_reverse_64(uint64_t val, uint32_t bits) {
    if (bits == 0) return 0;
    
    #ifdef __CUDA_ARCH__
        return __brevll(val) >> (64 - bits);
    #else
        uint64_t result = 0;
        for (uint32_t i = 0; i < bits; i++) {
            result = (result << 1) | (val & 1);
            val >>= 1;
        }
        return result;
    #endif
}

THETA_HD THETA_INLINE
uint32_t popcount_64(uint64_t n) {
    #ifdef __CUDA_ARCH__
        return __popcll(n);
    #else
        return __builtin_popcountll(n);
    #endif
}


//=============================================================================
// THETA KEY - The Core Angular Encoding
//=============================================================================

/**
 * Compute theta key: angular position as integer
 * theta_key(n) = bit_reverse(odd_core(n), bit_length(odd_core(n)))
 * 
 * Top 2 bits encode edge: 00=TOP, 01=RIGHT, 10=BOTTOM, 11=LEFT
 */
THETA_HD THETA_INLINE
uint32_t theta_key_32(uint32_t n) {
    if (n == 0) return 0;
    uint32_t core = odd_core_32(n);
    uint32_t bits = bit_length_32(core);
    return bit_reverse_32(core, bits);
}

THETA_HD THETA_INLINE
uint64_t theta_key_64(uint64_t n) {
    if (n == 0) return 0;
    uint64_t core = odd_core_64(n);
    uint32_t bits = bit_length_64(core);
    return bit_reverse_64(core, bits);
}

// Convenience aliases
THETA_HD THETA_INLINE uint32_t theta_key(uint32_t n) { return theta_key_32(n); }
THETA_HD THETA_INLINE uint64_t theta_key(uint64_t n) { return theta_key_64(n); }

/**
 * Extract edge/quadrant from theta key (top 2 bits)
 */
THETA_HD THETA_INLINE
Edge theta_edge_32(uint32_t n) {
    uint32_t key = theta_key_32(n);
    if (key == 0) return EDGE_TOP;
    uint32_t bits = bit_length_32(key);
    return static_cast<Edge>((key >> (bits - 2)) & 0x3);
}

THETA_HD THETA_INLINE
Edge theta_edge_64(uint64_t n) {
    uint64_t key = theta_key_64(n);
    if (key == 0) return EDGE_TOP;
    uint32_t bits = bit_length_64(key);
    return static_cast<Edge>((key >> (bits - 2)) & 0x3);
}


//=============================================================================
// 2-ADIC DECOMPOSITION STRUCTURES (v1.1 - includes core_bits, uses core_shell)
//=============================================================================

/**
 * CRITICAL: core_bits is REQUIRED for reconstruction.
 * Without it, multiple integers can map to the same theta_key.
 * 
 * Naming (v1.1 spec):
 * - shell: floor(log2(n)) - shell of original integer
 * - core_bits: bit_length(core) - needed for reconstruction  
 * - core_shell: core_bits - 1 - used in codec for validation
 */
struct Decomposition32 {
    uint32_t n;
    uint32_t v2;
    uint32_t core;
    uint32_t core_bits;    // REQUIRED for reconstruction
    uint32_t shell;        // shell(n) = floor(log2(n))
    uint32_t core_shell;   // core_bits - 1, for codec validation
    uint32_t theta_key;
    Edge edge;
};

struct Decomposition64 {
    uint64_t n;
    uint32_t v2;
    uint64_t core;
    uint32_t core_bits;    // REQUIRED for reconstruction
    uint32_t shell;        // shell(n) = floor(log2(n))
    uint32_t core_shell;   // core_bits - 1, for codec validation
    uint64_t theta_key;
    Edge edge;
};

THETA_HD THETA_INLINE
Decomposition32 decompose_32(uint32_t n) {
    Decomposition32 d;
    d.n = n;
    d.v2 = v2_32(n);
    d.core = odd_core_32(n);
    d.core_bits = bit_length_32(d.core);  // CRITICAL
    d.shell = shell_32(n);
    d.core_shell = d.core_bits > 0 ? d.core_bits - 1 : 0;
    d.theta_key = theta_key_32(n);
    d.edge = theta_edge_32(n);
    return d;
}

THETA_HD THETA_INLINE
Decomposition64 decompose_64(uint64_t n) {
    Decomposition64 d;
    d.n = n;
    d.v2 = v2_64(n);
    d.core = odd_core_64(n);
    d.core_bits = bit_length_64(d.core);  // CRITICAL
    d.shell = shell_64(n);
    d.core_shell = d.core_bits > 0 ? d.core_bits - 1 : 0;
    d.theta_key = theta_key_64(n);
    d.edge = theta_edge_64(n);
    return d;
}

/**
 * Reconstruct integer from decomposition.
 * REQUIRES core_bits for correct reconstruction.
 */
THETA_HD THETA_INLINE
uint32_t recompose_32(uint32_t v2, uint32_t theta_key, uint32_t core_bits) {
    if (theta_key == 0) return 0;
    uint32_t core = bit_reverse_32(theta_key, core_bits);
    return core << v2;
}

THETA_HD THETA_INLINE
uint64_t recompose_64(uint32_t v2, uint64_t theta_key, uint32_t core_bits) {
    if (theta_key == 0) return 0;
    uint64_t core = bit_reverse_64(theta_key, core_bits);
    return core << v2;
}

/**
 * Decode with validation (v1.1)
 * Returns 0 and sets error flag if core_shell doesn't match theta_key.
 */
THETA_HD THETA_INLINE
uint32_t decode_chunk_validated_32(
    uint32_t theta_key, 
    uint32_t core_shell, 
    uint32_t v2,
    bool* valid
) {
    if (theta_key == 0) {
        *valid = true;
        return 0;
    }
    
    uint32_t computed_core_bits = bit_length_32(theta_key);
    uint32_t expected_core_shell = computed_core_bits - 1;
    
    if (core_shell != expected_core_shell) {
        *valid = false;
        return 0;
    }
    
    *valid = true;
    uint32_t core = bit_reverse_32(theta_key, computed_core_bits);
    return core << v2;
}


//=============================================================================
// ANGLE OPERATIONS (Integer approximations)
//
// WARNING: Shell boundaries introduce angular discontinuities.
// theta_key behaves like an angle, but:
//   - At shell transitions, bit-length changes
//   - Angular resolution doubles
//   - Normalized angle can jump discontinuously
//
// For ML: use (theta_key, shell) pairs, not theta_key alone.
//=============================================================================

/**
 * Normalized angle in range [0, 2^precision)
 * This gives integer angle without any floating point.
 * 
 * WARNING: NOT continuous across shell boundaries.
 * 
 * precision=8:  returns angle in [0, 256) - like a byte
 * precision=16: returns angle in [0, 65536) - fine resolution
 */
THETA_HD THETA_INLINE
uint32_t theta_angle_scaled_32(uint32_t n, uint32_t precision = 16) {
    uint32_t key = theta_key_32(n);
    if (key == 0) return 0;
    
    uint32_t bits = bit_length_32(key);
    
    // Scale key to desired precision
    if (bits >= precision) {
        return key >> (bits - precision);
    } else {
        return key << (precision - bits);
    }
}

THETA_HD THETA_INLINE
uint32_t theta_angle_scaled_64(uint64_t n, uint32_t precision = 16) {
    uint64_t key = theta_key_64(n);
    if (key == 0) return 0;
    
    uint32_t bits = bit_length_64(key);
    
    if (bits >= precision) {
        return static_cast<uint32_t>(key >> (bits - precision));
    } else {
        return static_cast<uint32_t>(key << (precision - bits));
    }
}

/**
 * Angular distance between two values (integer).
 * Returns value in [0, 2^(precision-1)].
 * 
 * WARNING: Meaningful only for values in the SAME shell.
 * For cross-shell comparison, use theta_distance_safe().
 */
THETA_HD THETA_INLINE
uint32_t theta_angular_distance_32(uint32_t a, uint32_t b, uint32_t precision = 16) {
    uint32_t angle_a = theta_angle_scaled_32(a, precision);
    uint32_t angle_b = theta_angle_scaled_32(b, precision);
    
    uint32_t diff = (angle_a > angle_b) ? (angle_a - angle_b) : (angle_b - angle_a);
    uint32_t max_angle = 1u << precision;
    
    // Return shorter arc
    if (diff > max_angle / 2) {
        diff = max_angle - diff;
    }
    return diff;
}


//=============================================================================
// FEATURE EXTRACTION (for ML)
//=============================================================================

struct Features32 {
    uint32_t theta_key;
    uint32_t shell;
    uint32_t v2;
    uint32_t edge;       // 0-3
    uint32_t popcount;
    uint32_t lo_byte;    // theta_key & 0xFF
    uint32_t hi_byte;    // (theta_key >> 8) & 0xFF
};

THETA_HD THETA_INLINE
Features32 extract_features_32(uint32_t n) {
    Features32 f;
    f.theta_key = theta_key_32(n);
    f.shell = shell_32(n);
    f.v2 = v2_32(n);
    f.edge = static_cast<uint32_t>(theta_edge_32(n));
    f.popcount = popcount_32(f.theta_key);
    f.lo_byte = f.theta_key & 0xFF;
    f.hi_byte = (f.theta_key >> 8) & 0xFF;
    return f;
}


//=============================================================================
// CHECKSUM OPERATIONS
//=============================================================================

/**
 * Parallel-friendly checksum accumulator
 * Use with atomic operations for global reduction
 */
struct ThetaChecksum {
    uint64_t xor_acc;
    uint64_t add_acc;
    uint64_t mix_acc;
    uint32_t count;
    uint32_t min_shell;
    uint32_t max_shell;
};

THETA_HD THETA_INLINE
void checksum_init(ThetaChecksum& cs) {
    cs.xor_acc = 0;
    cs.add_acc = 0;
    cs.mix_acc = 0x5A5A5A5A5A5A5A5AULL;
    cs.count = 0;
    cs.min_shell = 0xFFFFFFFF;
    cs.max_shell = 0;
}

THETA_HD THETA_INLINE
void checksum_update_32(ThetaChecksum& cs, uint32_t value) {
    uint32_t key = theta_key_32(value);
    uint32_t sh = shell_32(value);
    
    cs.xor_acc ^= key;
    cs.add_acc += key;
    
    // Mix: rotate left 5, XOR key, add shell
    cs.mix_acc = ((cs.mix_acc << 5) | (cs.mix_acc >> 59)) ^ key;
    cs.mix_acc += sh;
    
    cs.count++;
    if (sh < cs.min_shell) cs.min_shell = sh;
    if (sh > cs.max_shell) cs.max_shell = sh;
}

/**
 * Merge two checksums (for parallel reduction)
 */
THETA_HD THETA_INLINE
ThetaChecksum checksum_merge(const ThetaChecksum& a, const ThetaChecksum& b) {
    ThetaChecksum result;
    result.xor_acc = a.xor_acc ^ b.xor_acc;
    result.add_acc = a.add_acc + b.add_acc;
    result.mix_acc = a.mix_acc ^ b.mix_acc;  // Simplified merge
    result.count = a.count + b.count;
    result.min_shell = (a.min_shell < b.min_shell) ? a.min_shell : b.min_shell;
    result.max_shell = (a.max_shell > b.max_shell) ? a.max_shell : b.max_shell;
    return result;
}


//=============================================================================
// OBFUSCATION (NOT ENCRYPTION - See disclaimers)
//=============================================================================

/**
 * XOR mask the theta key.
 * 
 * WARNING: This is NOT encryption. Trivially reversible.
 * Do NOT use for security-critical applications.
 */
THETA_HD THETA_INLINE
uint32_t theta_masked_32(uint32_t n, uint32_t mask) {
    return theta_key_32(n) ^ mask;
}

/**
 * Single Feistel round.
 * 
 * WARNING: Deterministic obfuscation only. NOT cryptographically secure.
 * Vulnerable to chosen-plaintext attacks.
 */
THETA_HD THETA_INLINE
uint32_t theta_feistel_32(uint32_t n, uint32_t secret, uint32_t bits = 32) {
    uint32_t k = theta_key_32(n);
    uint32_t half = bits / 2;
    uint32_t mask = (1u << half) - 1;
    
    uint32_t l = (k >> half) & mask;
    uint32_t r = k & mask;
    
    uint32_t f = (l + secret) & mask;
    uint32_t new_r = r ^ f;
    
    return ((new_r & mask) << half) | l;
}

/**
 * Multi-round Feistel for stronger scrambling.
 * 
 * WARNING: Still deterministic. NOT cryptographically secure.
 * Use only for reversible ID scrambling, not security.
 */
THETA_HD THETA_INLINE
uint32_t theta_feistel_multi_32(uint32_t n, uint32_t secret, uint32_t rounds = 4) {
    uint32_t result = n;
    for (uint32_t r = 0; r < rounds; r++) {
        uint32_t round_secret = secret ^ (r * 0x9E3779B9u);
        result = theta_feistel_32(result, round_secret, 32);
    }
    return result;
}


//=============================================================================
// BUCKETING / SPATIAL HASHING (Fixed v2 - proper uniformity)
//=============================================================================

/**
 * Hash value to bucket with proper uniform distribution.
 * 
 * IMPORTANT: Simple theta_key % buckets has POOR uniformity because
 * theta_key is always ODD (LSB=1). This function uses proper mixing.
 * 
 * Benchmark results (10M flow IDs, 64 buckets):
 *   theta_bucket: chi-square 59.9, max deviation 0.68% - GOOD
 *   simple mod:   chi-square 10M+, max deviation 100%+ - POOR
 */
THETA_HD THETA_INLINE
uint32_t theta_bucket_32(uint32_t n, uint32_t num_buckets, uint32_t secret = 0) {
    uint32_t key = theta_key_32(n);
    uint32_t v = v2_32(n);
    uint32_t sh = shell_32(n);
    
    // Combine with 64-bit arithmetic for better mixing
    uint64_t h = key;
    h = h * 0x9E3779B97F4A7C15ULL;
    h = h ^ (uint64_t(v) * 0x517CC1B727220A95ULL);
    h = h ^ (uint64_t(sh) * 0x2545F4914F6CDD1DULL);
    h = h ^ secret;
    
    // Final mix: upper bits are better distributed
    h = (h * 0x9E3779B97F4A7C15ULL) >> 32;
    
    return uint32_t(h % num_buckets);
}

/**
 * Simple theta_key mod buckets (for comparison).
 * WARNING: POOR uniformity - theta_key is always odd!
 */
THETA_HD THETA_INLINE
uint32_t theta_bucket_simple_32(uint32_t n, uint32_t num_buckets) {
    return theta_key_32(n) % num_buckets;
}

/**
 * 2D spatial hash from two coordinates
 * Interleaves theta keys for locality preservation
 */
THETA_HD THETA_INLINE
uint64_t theta_spatial_hash_2d(uint32_t x, uint32_t y) {
    uint32_t kx = theta_key_32(x);
    uint32_t ky = theta_key_32(y);
    
    // Interleave bits (Morton code style)
    uint64_t result = 0;
    for (int i = 0; i < 32; i++) {
        result |= ((uint64_t)((kx >> i) & 1) << (2*i));
        result |= ((uint64_t)((ky >> i) & 1) << (2*i + 1));
    }
    return result;
}

/**
 * 3D spatial hash from three coordinates
 */
THETA_HD THETA_INLINE
uint64_t theta_spatial_hash_3d(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t kx = theta_key_32(x);
    uint32_t ky = theta_key_32(y);
    uint32_t kz = theta_key_32(z);
    
    // Interleave 21 bits from each (63 bits total)
    uint64_t result = 0;
    for (int i = 0; i < 21; i++) {
        result |= ((uint64_t)((kx >> i) & 1) << (3*i));
        result |= ((uint64_t)((ky >> i) & 1) << (3*i + 1));
        result |= ((uint64_t)((kz >> i) & 1) << (3*i + 2));
    }
    return result;
}


//=============================================================================
// COMPARISON / SORTING HELPERS
//=============================================================================

/**
 * Compare by theta order (for sorting)
 */
THETA_HD THETA_INLINE
int theta_compare_32(uint32_t a, uint32_t b) {
    uint32_t ka = theta_key_32(a);
    uint32_t kb = theta_key_32(b);
    return (ka > kb) - (ka < kb);
}


//=============================================================================
// KERNEL HELPERS
//=============================================================================

#ifdef __CUDACC__

/**
 * Block-level checksum reduction using shared memory
 */
__device__ inline
ThetaChecksum block_reduce_checksum(ThetaChecksum local_cs) {
    __shared__ ThetaChecksum shared_cs[256];  // Assumes max 256 threads/block
    
    int tid = threadIdx.x;
    shared_cs[tid] = local_cs;
    __syncthreads();
    
    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_cs[tid] = checksum_merge(shared_cs[tid], shared_cs[tid + stride]);
        }
        __syncthreads();
    }
    
    return shared_cs[0];
}

/**
 * Warp-level theta key reduction using shuffle
 */
__device__ inline
uint32_t warp_reduce_xor(uint32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val ^= __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

#endif  // __CUDACC__


}  // namespace theta

#endif  // THETA_CUDA_CUH
