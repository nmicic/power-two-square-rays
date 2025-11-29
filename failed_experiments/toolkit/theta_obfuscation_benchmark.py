#!/usr/bin/env python3
# ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
# This file remains for reference only.
# Do NOT use in production. No support, no guarantees.
#
# WARNING: This is a toy crypto / obfuscation benchmark.
# It is NOT cryptographically secure, MUST NOT be used to protect secrets,
# and is kept only as an experiment. Use real, vetted crypto libraries instead.
#
# For the canonical Theta Toolkit implementation and spec, see:
#   - cuda/theta_cuda_v1.2.cuh
#   - cuda/THETA_SPEC_v1.2.md
"""
theta_crypto_benchmark.py - Benchmark Theta Symmetric Encryption

Compares theta-based encryption against standard methods:
- Theta XOR (from theta_play.py)
- Fernet (cryptography library - AES-128-CBC)
- AES-GCM (if available)
- Simple XOR baseline

NOTE: Theta encryption is XOR-based OBFUSCATION, not real crypto!
This benchmark shows the performance difference, not security equivalence.

Usage:
    python3 theta_crypto_benchmark.py
"""

import time
import os
import sys
import hashlib
import struct
from typing import Callable, Tuple

#==============================================================================
# THETA ENCRYPTION (from theta_play.py)
#==============================================================================

def v2(n: int) -> int:
    if n == 0: return 0
    n = abs(int(n))
    count = 0
    while (n & 1) == 0:
        count += 1
        n >>= 1
    return count

def odd_core(n: int) -> int:
    if n == 0: return 0
    n = abs(int(n))
    return n >> v2(n)

def bit_length(n: int) -> int:
    if n == 0: return 0
    return abs(int(n)).bit_length()

def bit_reverse(val: int, bits: int) -> int:
    if bits <= 0 or val == 0: return 0
    result = 0
    for _ in range(bits):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result

def theta_key(n: int) -> int:
    if n == 0: return 0
    core = odd_core(n)
    bits = bit_length(core)
    return bit_reverse(core, bits)

def shell(n: int) -> int:
    bl = bit_length(n)
    return bl - 1 if bl > 0 else 0

def derive_key_stream(password: str, size: int) -> bytes:
    """Derive key stream from password using SHA256"""
    result = bytearray()
    counter = 0
    while len(result) < size:
        h = hashlib.sha256(f"{password}:{counter}".encode()).digest()
        result.extend(h)
        counter += 1
    return bytes(result[:size])

def xor_bytes(data: bytes, key_stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, key_stream))

def theta_encrypt(data: bytes, password: str) -> bytes:
    """Theta-based encryption: XOR + theta encoding"""
    key_stream = derive_key_stream(password, len(data))
    encrypted = xor_bytes(data, key_stream)
    
    # Theta encode (adds structure but also overhead)
    padding = (4 - len(encrypted) % 4) % 4
    padded = encrypted + bytes(padding)
    
    chunks = []
    for i in range(0, len(padded), 4):
        value = struct.unpack('<I', padded[i:i+4])[0]
        if value == 0:
            chunks.append((0, 0, 0))
        else:
            core = odd_core(value)
            core_bits = bit_length(core)
            chunks.append((theta_key(value), core_bits - 1, v2(value)))
    
    # Pack chunks
    result = struct.pack('<I', len(data))  # Original size
    for theta, sh, v in chunks:
        result += struct.pack('<IBB', theta, sh, v)
    
    return result

def theta_decrypt(data: bytes, password: str) -> bytes:
    """Theta-based decryption"""
    original_size = struct.unpack('<I', data[:4])[0]
    offset = 4
    
    chunks = []
    while offset < len(data):
        theta, sh, v = struct.unpack('<IBB', data[offset:offset+6])
        chunks.append((theta, sh, v))
        offset += 6
    
    # Decode chunks
    result = bytearray()
    for theta, sh, v in chunks:
        if theta == 0:
            value = 0
        else:
            core_bits = sh + 1
            core = bit_reverse(theta, core_bits)
            value = core << v
        result.extend(struct.pack('<I', value))
    
    encrypted = bytes(result[:original_size])
    key_stream = derive_key_stream(password, len(encrypted))
    return xor_bytes(encrypted, key_stream)

#==============================================================================
# SIMPLE XOR BASELINE
#==============================================================================

def simple_xor_encrypt(data: bytes, password: str) -> bytes:
    """Simple XOR with SHA256-derived key stream"""
    key_stream = derive_key_stream(password, len(data))
    return xor_bytes(data, key_stream)

def simple_xor_decrypt(data: bytes, password: str) -> bytes:
    return simple_xor_encrypt(data, password)  # XOR is symmetric

#==============================================================================
# THETA HASH
#==============================================================================

def theta_hash(data: bytes) -> bytes:
    """Theta-based hash (32 bytes output)"""
    padding = (4 - len(data) % 4) % 4
    padded = data + bytes(padding)
    
    xor_acc = 0
    mix_acc = 0x5A5A5A5A
    add_acc = 0
    rot_acc = 0xDEADBEEF
    
    for i in range(0, len(padded), 4):
        value = struct.unpack('<I', padded[i:i+4])[0]
        key = theta_key(value)
        sh = shell(value)
        
        xor_acc ^= key
        add_acc = (add_acc + key) & 0xFFFFFFFF
        mix_acc = ((mix_acc << 5) | (mix_acc >> 27)) & 0xFFFFFFFF
        mix_acc ^= key
        mix_acc = (mix_acc + sh) & 0xFFFFFFFF
        rot_acc = ((rot_acc << 7) | (rot_acc >> 25)) & 0xFFFFFFFF
        rot_acc ^= (key * 0x9E3779B9) & 0xFFFFFFFF
    
    # Create 32-byte output
    result = struct.pack('<IIIIIIII',
        xor_acc, add_acc, mix_acc, rot_acc,
        xor_acc ^ add_acc, mix_acc ^ rot_acc,
        (xor_acc + mix_acc) & 0xFFFFFFFF,
        (add_acc + rot_acc) & 0xFFFFFFFF
    )
    return result

#==============================================================================
# STANDARD CRYPTO (optional)
#==============================================================================

HAS_CRYPTOGRAPHY = False
HAS_HASHLIB_SHA3 = hasattr(hashlib, 'sha3_256')

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import base64
    HAS_CRYPTOGRAPHY = True
except ImportError:
    pass

def fernet_encrypt(data: bytes, password: str) -> bytes:
    """Fernet encryption (AES-128-CBC with HMAC)"""
    # Derive key from password
    key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
    f = Fernet(key)
    return f.encrypt(data)

def fernet_decrypt(data: bytes, password: str) -> bytes:
    key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
    f = Fernet(key)
    return f.decrypt(data)

def aesgcm_encrypt(data: bytes, password: str) -> bytes:
    """AES-GCM encryption"""
    key = hashlib.sha256(password.encode()).digest()
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return nonce + ciphertext

def aesgcm_decrypt(data: bytes, password: str) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    nonce = data[:12]
    ciphertext = data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)

#==============================================================================
# BENCHMARK HELPERS
#==============================================================================

def benchmark_encrypt_decrypt(name: str, encrypt_fn: Callable, decrypt_fn: Callable,
                              data: bytes, password: str, iterations: int) -> dict:
    """Benchmark encrypt/decrypt cycle"""
    
    # Warmup
    encrypted = encrypt_fn(data, password)
    decrypted = decrypt_fn(encrypted, password)
    
    # Verify correctness
    if decrypted != data:
        return {'name': name, 'error': 'Round-trip failed!'}
    
    # Benchmark encrypt
    start = time.perf_counter()
    for _ in range(iterations):
        encrypted = encrypt_fn(data, password)
    encrypt_time = time.perf_counter() - start
    
    # Benchmark decrypt
    start = time.perf_counter()
    for _ in range(iterations):
        decrypted = decrypt_fn(encrypted, password)
    decrypt_time = time.perf_counter() - start
    
    data_mb = len(data) * iterations / (1024 * 1024)
    
    return {
        'name': name,
        'encrypt_time': encrypt_time,
        'decrypt_time': decrypt_time,
        'encrypt_speed': data_mb / encrypt_time,
        'decrypt_speed': data_mb / decrypt_time,
        'expansion': len(encrypted) / len(data),
        'iterations': iterations,
        'data_size': len(data)
    }

def benchmark_hash(name: str, hash_fn: Callable, data: bytes, iterations: int) -> dict:
    """Benchmark hash function"""
    
    # Warmup
    h = hash_fn(data)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        h = hash_fn(data)
    elapsed = time.perf_counter() - start
    
    data_mb = len(data) * iterations / (1024 * 1024)
    
    return {
        'name': name,
        'time': elapsed,
        'speed': data_mb / elapsed,
        'output_size': len(h),
        'iterations': iterations
    }

#==============================================================================
# MAIN BENCHMARK
#==============================================================================

def main():
    print("=" * 70)
    print("  THETA CRYPTO BENCHMARK")
    print("=" * 70)
    print()
    print("  NOTE: Theta encryption is XOR-based OBFUSCATION, not real crypto!")
    print("        This compares PERFORMANCE, not SECURITY.")
    print()
    
    # Test parameters
    sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
    password = "benchmark_password_123"
    
    # Determine iterations based on size
    def get_iterations(size):
        if size <= 1024: return 1000
        if size <= 10240: return 100
        return 10
    
    print("-" * 70)
    print("  ENCRYPTION BENCHMARK")
    print("-" * 70)
    
    for size in sizes:
        data = os.urandom(size)
        iterations = get_iterations(size)
        
        print(f"\n  Data size: {size:,} bytes ({size/1024:.1f} KB), {iterations} iterations")
        print()
        print(f"  {'Method':<20} {'Encrypt':<12} {'Decrypt':<12} {'Expansion':<10} {'Status'}")
        print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        
        methods = [
            ('Simple XOR', simple_xor_encrypt, simple_xor_decrypt),
            ('Theta Encrypt', theta_encrypt, theta_decrypt),
        ]
        
        if HAS_CRYPTOGRAPHY:
            methods.extend([
                ('Fernet (AES-CBC)', fernet_encrypt, fernet_decrypt),
                ('AES-GCM', aesgcm_encrypt, aesgcm_decrypt),
            ])
        
        for name, enc_fn, dec_fn in methods:
            try:
                result = benchmark_encrypt_decrypt(name, enc_fn, dec_fn, data, password, iterations)
                if 'error' in result:
                    print(f"  {name:<20} {'ERROR':<12} {result['error']}")
                else:
                    print(f"  {name:<20} {result['encrypt_speed']:>8.2f} MB/s "
                          f"{result['decrypt_speed']:>8.2f} MB/s "
                          f"{result['expansion']:>6.2f}x    ✓")
            except Exception as e:
                print(f"  {name:<20} {'ERROR':<12} {str(e)[:30]}")
    
    print()
    print("-" * 70)
    print("  HASH BENCHMARK")
    print("-" * 70)
    
    for size in sizes:
        data = os.urandom(size)
        iterations = get_iterations(size) * 10  # More iterations for hash
        
        print(f"\n  Data size: {size:,} bytes, {iterations} iterations")
        print()
        print(f"  {'Method':<20} {'Speed':<15} {'Output':<10}")
        print(f"  {'-'*18} {'-'*13} {'-'*8}")
        
        hash_methods = [
            ('Theta Hash', theta_hash),
            ('SHA256', lambda d: hashlib.sha256(d).digest()),
            ('SHA512', lambda d: hashlib.sha512(d).digest()),
            ('MD5', lambda d: hashlib.md5(d).digest()),
            ('BLAKE2b', lambda d: hashlib.blake2b(d).digest()),
        ]
        
        if HAS_HASHLIB_SHA3:
            hash_methods.append(('SHA3-256', lambda d: hashlib.sha3_256(d).digest()))
        
        for name, hash_fn in hash_methods:
            try:
                result = benchmark_hash(name, hash_fn, data, iterations)
                print(f"  {name:<20} {result['speed']:>10.2f} MB/s  {result['output_size']:>3} bytes")
            except Exception as e:
                print(f"  {name:<20} ERROR: {e}")
    
    print()
    print("-" * 70)
    print("  ANALYSIS")
    print("-" * 70)
    print("""
  ENCRYPTION:
  - Simple XOR:      Fastest, but just key stream generation + XOR
  - Theta Encrypt:   Adds theta encoding overhead (1.5x expansion)
  - Fernet/AES-GCM:  Real crypto, hardware accelerated (AES-NI)
  
  HASH:
  - Theta Hash:      Integer-native, educational
  - SHA256/SHA3:     Cryptographic, battle-tested
  - BLAKE2b:         Modern, very fast
  
  VERDICT:
  - Theta is ~10-100x slower than hardware-accelerated crypto
  - Theta encryption expansion is 1.5x (vs ~1.0x for AES-GCM)
  - For LEARNING: Theta is great - simple, visual, reversible
  - For SECURITY: Use Fernet/AES-GCM (real crypto!)
  
  RECOMMENDATION:
  - Kids/Education: theta_play.py is perfect
  - Real secrets: pip install cryptography
""")
    
    if not HAS_CRYPTOGRAPHY:
        print("  TIP: pip install cryptography  (to compare with real crypto)")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
