#!/usr/bin/env python3
# ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
# This file remains for reference only.
# Do NOT use in production. No support, no guarantees.
#
# For the canonical Theta Toolkit implementation and spec, see:
#   - cuda/theta_cuda_v1.2.cuh
#   - cuda/THETA_SPEC_v1.2.md
"""
theta_play.py - Fun Theta Toolkit for Kids!

A simple tool to play with theta encoding, encryption, and hashing.

WHAT CAN YOU DO?
================

1. ENCODE - Turn any message into theta numbers
   echo "Hello" | python3 theta_play.py encode

2. DECODE - Turn theta numbers back into message  
   echo "Hello" | python3 theta_play.py encode | python3 theta_play.py decode

3. ENCRYPT - Secret messages only you and your friend can read!
   echo "Secret" | python3 theta_play.py encrypt --key "mypassword"
   
4. DECRYPT - Read secret messages
   cat secret.txt | python3 theta_play.py decrypt --key "mypassword"

5. HASH - Make a unique fingerprint of any message
   echo "Hello" | python3 theta_play.py hash

6. VERIFY - Check if message matches a hash
   python3 theta_play.py verify "Hello" abc123def

7. EXPLORE - See how theta works on numbers
   python3 theta_play.py explore 12
   python3 theta_play.py explore 1 2 3 4 5 6 7 8

HOW IT WORKS?
=============

Every number can be split into two parts:
  12 = 4 × 3 = 2² × 3

The "theta key" is the odd part (3) with its bits reversed!
This creates a unique pattern for every number.

Repository: https://github.com/nmicic/power-two-square-rays/
"""

import sys
import hashlib
import struct
from typing import Optional, List

#==============================================================================
# CORE THETA FUNCTIONS (minimal, self-contained)
#==============================================================================

def v2(n: int) -> int:
    """Count trailing zeros (how many times divisible by 2)"""
    if n == 0:
        return 0
    n = abs(int(n))
    count = 0
    while (n & 1) == 0:
        count += 1
        n >>= 1
    return count

def odd_core(n: int) -> int:
    """Get the odd part of n (remove all factors of 2)"""
    if n == 0:
        return 0
    n = abs(int(n))
    return n >> v2(n)

def bit_length(n: int) -> int:
    """How many bits to represent n"""
    if n == 0:
        return 0
    return abs(int(n)).bit_length()

def bit_reverse(val: int, bits: int) -> int:
    """Reverse the bits: 1100 -> 0011"""
    if bits <= 0 or val == 0:
        return 0
    result = 0
    for _ in range(bits):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result

def theta_key(n: int) -> int:
    """The magic theta key: reverse the bits of the odd core"""
    if n == 0:
        return 0
    core = odd_core(n)
    bits = bit_length(core)
    return bit_reverse(core, bits)

def shell(n: int) -> int:
    """Shell = floor(log2(n)) - numbers in same shell are similar size"""
    bl = bit_length(n)
    return bl - 1 if bl > 0 else 0

#==============================================================================
# ENCODING/DECODING
#==============================================================================

def encode_bytes(data: bytes) -> List[dict]:
    """Encode bytes to theta representation"""
    # Pad to 4-byte boundary
    padding = (4 - len(data) % 4) % 4
    padded = data + bytes(padding)
    
    chunks = []
    for i in range(0, len(padded), 4):
        value = struct.unpack('<I', padded[i:i+4])[0]
        if value == 0:
            chunks.append({'theta': 0, 'shell': 0, 'v2': 0})
        else:
            core = odd_core(value)
            core_bits = bit_length(core)
            chunks.append({
                'theta': theta_key(value),
                'shell': core_bits - 1,  # core_shell for reconstruction
                'v2': v2(value)
            })
    
    return chunks, len(data)

def decode_chunks(chunks: List[dict]) -> bytes:
    """Decode theta chunks back to bytes"""
    result = bytearray()
    for chunk in chunks:
        if chunk['theta'] == 0:
            value = 0
        else:
            core_bits = chunk['shell'] + 1
            core = bit_reverse(chunk['theta'], core_bits)
            value = core << chunk['v2']
        result.extend(struct.pack('<I', value))
    return bytes(result)

def format_encoded(chunks: List[dict], original_size: int) -> str:
    """Format encoded chunks as text"""
    lines = [f"THETA:{original_size}"]
    for c in chunks:
        lines.append(f"{c['theta']:08x}:{c['shell']:02x}:{c['v2']:02x}")
    return '\n'.join(lines)

def parse_encoded(text: str) -> tuple:
    """Parse text back to chunks"""
    lines = text.strip().split('\n')
    
    # Parse header
    if not lines[0].startswith('THETA:'):
        raise ValueError("Not a theta encoded message!")
    original_size = int(lines[0].split(':')[1])
    
    # Parse chunks
    chunks = []
    for line in lines[1:]:
        if ':' in line:
            parts = line.split(':')
            chunks.append({
                'theta': int(parts[0], 16),
                'shell': int(parts[1], 16),
                'v2': int(parts[2], 16)
            })
    
    return chunks, original_size

#==============================================================================
# ENCRYPTION (XOR with key-derived stream)
#==============================================================================

def derive_key_stream(password: str, size: int) -> bytes:
    """Create encryption bytes from password"""
    result = bytearray()
    counter = 0
    while len(result) < size:
        h = hashlib.sha256(f"{password}:{counter}".encode()).digest()
        result.extend(h)
        counter += 1
    return bytes(result[:size])

def xor_bytes(data: bytes, key_stream: bytes) -> bytes:
    """XOR data with key stream"""
    return bytes(a ^ b for a, b in zip(data, key_stream))

#==============================================================================
# THETA HASH
#==============================================================================

def theta_hash(data: bytes) -> str:
    """Create theta-based hash fingerprint"""
    # Process as 4-byte chunks
    padding = (4 - len(data) % 4) % 4
    padded = data + bytes(padding)
    
    xor_acc = 0
    mix_acc = 0x5A5A5A5A
    add_acc = 0
    
    for i in range(0, len(padded), 4):
        value = struct.unpack('<I', padded[i:i+4])[0]
        key = theta_key(value)
        sh = shell(value)
        
        xor_acc ^= key
        add_acc = (add_acc + key) & 0xFFFFFFFF
        mix_acc = ((mix_acc << 5) | (mix_acc >> 27)) & 0xFFFFFFFF
        mix_acc ^= key
        mix_acc = (mix_acc + sh) & 0xFFFFFFFF
    
    # Combine into final hash
    final = (xor_acc ^ add_acc ^ mix_acc) & 0xFFFFFFFF
    return f"{final:08x}-{xor_acc:08x}-{mix_acc:08x}"

#==============================================================================
# EXPLORE FUNCTION (educational)
#==============================================================================

def explore_number(n: int) -> dict:
    """Show all theta properties of a number"""
    if n <= 0:
        return {'n': n, 'error': 'Must be positive!'}
    
    core = odd_core(n)
    v2_val = v2(n)
    key = theta_key(n)
    sh = shell(n)
    
    # Binary representations
    n_bin = bin(n)[2:]
    core_bin = bin(core)[2:]
    key_bin = bin(key)[2:].zfill(len(core_bin))
    
    return {
        'n': n,
        'binary': n_bin,
        'v2': v2_val,
        'core': core,
        'core_binary': core_bin,
        'theta_key': key,
        'theta_binary': key_bin,
        'shell': sh,
        'decomposition': f"{n} = 2^{v2_val} × {core}",
        'check': (1 << v2_val) * core  # Should equal n
    }

#==============================================================================
# CLI INTERFACE
#==============================================================================

def print_help():
    """Print help message"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    🌟 THETA PLAY 🌟                                ║
║              Fun with Numbers and Secrets!                        ║
╚═══════════════════════════════════════════════════════════════════╝

COMMANDS:
---------

  encode              Turn message into theta numbers
                      echo "Hello" | python3 theta_play.py encode

  decode              Turn theta numbers back to message
                      cat encoded.txt | python3 theta_play.py decode

  encrypt --key PWD   Make a secret message
                      echo "Secret" | python3 theta_play.py encrypt --key "password"

  decrypt --key PWD   Read a secret message
                      cat secret.txt | python3 theta_play.py decrypt --key "password"

  hash                Make a fingerprint of message
                      echo "Hello" | python3 theta_play.py hash

  verify MSG HASH     Check if message matches hash
                      python3 theta_play.py verify "Hello" "abc123..."

  explore N [N2...]   See how theta works on numbers
                      python3 theta_play.py explore 12
                      python3 theta_play.py explore 1 2 3 4 5 6 7 8

  demo                Run a fun demonstration

EXAMPLES:
---------

  # Send a secret message to your friend:
  echo "Meet at the treehouse!" | python3 theta_play.py encrypt --key "clubsecret" > secret.txt
  
  # Your friend reads it:
  cat secret.txt | python3 theta_play.py decrypt --key "clubsecret"
  
  # Make sure nobody changed your message:
  echo "Important message" | python3 theta_play.py hash
  # Save the hash, later verify:
  python3 theta_play.py verify "Important message" "the-hash-you-saved"

HOW IT WORKS:
-------------

  Every number has a secret pattern!
  
  12 = 4 × 3 = 2² × 3
      ↓
  theta_key(12) = reverse_bits(3) = 3
  
  Try: python3 theta_play.py explore 12

Repository: https://github.com/nmicic/power-two-square-rays/
""")

def cmd_encode():
    """Encode stdin to theta format"""
    data = sys.stdin.buffer.read()
    chunks, size = encode_bytes(data)
    print(format_encoded(chunks, size))

def cmd_decode():
    """Decode theta format from stdin"""
    text = sys.stdin.read()
    chunks, original_size = parse_encoded(text)
    data = decode_chunks(chunks)
    sys.stdout.buffer.write(data[:original_size])

def cmd_encrypt(key: str):
    """Encrypt stdin with key"""
    data = sys.stdin.buffer.read()
    key_stream = derive_key_stream(key, len(data))
    encrypted = xor_bytes(data, key_stream)
    chunks, size = encode_bytes(encrypted)
    
    # Add marker and key hash for verification
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:8]
    print(f"ENCRYPTED:{key_hash}")
    print(format_encoded(chunks, size))

def cmd_decrypt(key: str):
    """Decrypt stdin with key"""
    text = sys.stdin.read()
    lines = text.strip().split('\n')
    
    # Check encryption marker
    if not lines[0].startswith('ENCRYPTED:'):
        print("Error: This doesn't look like an encrypted message!", file=sys.stderr)
        sys.exit(1)
    
    expected_hash = lines[0].split(':')[1]
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:8]
    
    if key_hash != expected_hash:
        print("⚠️  Warning: Key might be wrong!", file=sys.stderr)
    
    # Parse and decrypt
    theta_text = '\n'.join(lines[1:])
    chunks, original_size = parse_encoded(theta_text)
    encrypted = decode_chunks(chunks)[:original_size]
    
    key_stream = derive_key_stream(key, len(encrypted))
    decrypted = xor_bytes(encrypted, key_stream)
    sys.stdout.buffer.write(decrypted)

def cmd_hash():
    """Hash stdin"""
    data = sys.stdin.buffer.read()
    h = theta_hash(data)
    print(h)

def cmd_verify(message: str, expected_hash: str):
    """Verify message against hash"""
    h = theta_hash(message.encode())
    if h == expected_hash:
        print("✅ MATCH! Message is authentic.")
        return True
    else:
        print("❌ NO MATCH! Message may have been changed.")
        print(f"   Expected: {expected_hash}")
        print(f"   Got:      {h}")
        return False

def cmd_explore(numbers: List[int]):
    """Explore theta properties of numbers"""
    print("\n" + "="*60)
    print("  THETA EXPLORER - See the patterns!")
    print("="*60)
    
    for n in numbers:
        info = explore_number(n)
        if 'error' in info:
            print(f"\n  {n}: {info['error']}")
            continue
        
        print(f"""
  ┌─ Number: {info['n']} ─────────────────────────────
  │
  │  Binary:        {info['binary']}
  │  Decomposition: {info['decomposition']}
  │
  │  v2 (trailing zeros): {info['v2']}
  │  Core (odd part):     {info['core']} = {info['core_binary']}
  │  
  │  Theta Key:           {info['theta_key']} = {info['theta_binary']}
  │                       (core with bits reversed!)
  │  
  │  Shell:               {info['shell']} (size category)
  │
  │  Check: 2^{info['v2']} × {info['core']} = {info['check']} ✓
  └──────────────────────────────────────────────────""")
    
    print()

def cmd_demo():
    """Run demonstration"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    🎮 THETA DEMO 🎮                               ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    # 1. Explore numbers
    print("1️⃣  EXPLORING NUMBERS")
    print("   Let's see how 12 breaks down:")
    cmd_explore([12])
    
    # 2. Encoding
    print("\n2️⃣  ENCODING A MESSAGE")
    message = b"Hi!"
    chunks, size = encode_bytes(message)
    encoded = format_encoded(chunks, size)
    print(f"   Original: {message.decode()}")
    print(f"   Encoded:")
    for line in encoded.split('\n'):
        print(f"      {line}")
    
    # Decode it back
    chunks2, size2 = parse_encoded(encoded)
    decoded = decode_chunks(chunks2)[:size2]
    print(f"   Decoded:  {decoded.decode()}")
    print(f"   Match: {'✅' if decoded == message else '❌'}")
    
    # 3. Hashing
    print("\n3️⃣  CREATING A HASH (fingerprint)")
    h1 = theta_hash(b"Hello")
    h2 = theta_hash(b"Hello")
    h3 = theta_hash(b"Hello!")  # Different!
    print(f"   'Hello'  -> {h1}")
    print(f"   'Hello'  -> {h2} (same!)")
    print(f"   'Hello!' -> {h3} (different!)")
    
    # 4. Encryption
    print("\n4️⃣  SECRET MESSAGES")
    secret = b"The treasure is under the old oak tree!"
    key = "pirates123"
    
    key_stream = derive_key_stream(key, len(secret))
    encrypted = xor_bytes(secret, key_stream)
    
    print(f"   Original: {secret.decode()}")
    print(f"   Key:      {key}")
    print(f"   Encrypted bytes: {encrypted[:20].hex()}...")
    
    # Decrypt
    decrypted = xor_bytes(encrypted, key_stream)
    print(f"   Decrypted: {decrypted.decode()}")
    print(f"   Match: {'✅' if decrypted == secret else '❌'}")
    
    # 5. Pattern exploration
    print("\n5️⃣  NUMBER PATTERNS")
    print("   See how powers of 2 work:")
    for i in range(1, 9):
        n = 2 ** i
        print(f"   {n:4} = 2^{i:2}  →  theta_key = {theta_key(n):4}  shell = {shell(n)}")
    
    print("""
═══════════════════════════════════════════════════════════════════

🎉 Now try it yourself!

   python3 theta_play.py explore 100
   echo "Your message" | python3 theta_play.py encrypt --key "secret"
   python3 theta_play.py --help

═══════════════════════════════════════════════════════════════════
""")

def main():
    args = sys.argv[1:]
    
    if not args or args[0] in ('-h', '--help', 'help'):
        print_help()
        return
    
    cmd = args[0]
    
    if cmd == 'encode':
        cmd_encode()
    
    elif cmd == 'decode':
        cmd_decode()
    
    elif cmd == 'encrypt':
        if '--key' not in args:
            print("Error: Need --key for encryption!", file=sys.stderr)
            print("Example: echo 'msg' | python3 theta_play.py encrypt --key 'password'", file=sys.stderr)
            sys.exit(1)
        key_idx = args.index('--key')
        key = args[key_idx + 1]
        cmd_encrypt(key)
    
    elif cmd == 'decrypt':
        if '--key' not in args:
            print("Error: Need --key for decryption!", file=sys.stderr)
            sys.exit(1)
        key_idx = args.index('--key')
        key = args[key_idx + 1]
        cmd_decrypt(key)
    
    elif cmd == 'hash':
        cmd_hash()
    
    elif cmd == 'verify':
        if len(args) < 3:
            print("Usage: python3 theta_play.py verify 'message' 'hash'", file=sys.stderr)
            sys.exit(1)
        success = cmd_verify(args[1], args[2])
        sys.exit(0 if success else 1)
    
    elif cmd == 'explore':
        if len(args) < 2:
            print("Usage: python3 theta_play.py explore 12 [more numbers...]", file=sys.stderr)
            sys.exit(1)
        numbers = [int(x) for x in args[1:]]
        cmd_explore(numbers)
    
    elif cmd == 'demo':
        cmd_demo()
    
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print("Run: python3 theta_play.py --help", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
