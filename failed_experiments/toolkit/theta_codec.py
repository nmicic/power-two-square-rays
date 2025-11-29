#!/usr/bin/env python3
# ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
# This file remains for reference only.
# Do NOT use in production. No support, no guarantees.
#
# For the canonical Theta Toolkit implementation and spec, see:
#   - cuda/theta_cuda_v1.2.cuh
#   - cuda/THETA_SPEC_v1.2.md
"""
theta_codec.py - Reversible Binary Encoding via Theta Transform

A complete encode/decode system built on the theta-key concept:
- Bijective: every byte sequence maps to unique theta encoding and back
- Integrity: built-in checksum verifies data hasn't been corrupted
- Encryption: XOR with secret key provides symmetric encryption
- Streaming: works with stdin/stdout or files

USAGE:
    # File to file
    python3 theta_codec.py encode input.bin output.theta
    python3 theta_codec.py decode output.theta recovered.bin
    
    # Stdin/stdout (use - for stdin/stdout)
    cat input.bin | python3 theta_codec.py encode - - > output.theta
    cat output.theta | python3 theta_codec.py decode - - > recovered.bin
    
    # With encryption
    python3 theta_codec.py encode input.bin output.theta --key "secret"
    
    # Verify only
    python3 theta_codec.py verify output.theta
    
    # Text format (like uuencode)
    python3 theta_codec.py encode input.bin output.txt --text

Repository: https://github.com/nmicic/power-two-square-rays/
"""

from __future__ import annotations
import struct
import hashlib
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterator, Optional, BinaryIO, Tuple

# Import from local theta_toolkit
try:
    from theta_toolkit import (
        v2, odd_core, bit_length, shell, bit_reverse, theta_key,
        recompose_from_core
    )
except ImportError:
    # Inline minimal implementation if toolkit not available
    def v2(n):
        if n == 0: return 0
        return (abs(n) & -abs(n)).bit_length() - 1
    
    def odd_core(n):
        if n == 0: return 0
        n = abs(int(n))
        return n >> v2(n)
    
    def bit_length(n):
        if n == 0: return 0
        return abs(int(n)).bit_length()
    
    def shell(n):
        bl = bit_length(n)
        return bl - 1 if bl > 0 else 0
    
    def bit_reverse(val, bits):
        if bits <= 0 or val == 0: return 0
        result = 0
        for _ in range(bits):
            result = (result << 1) | (val & 1)
            val >>= 1
        return result
    
    def theta_key(n):
        if n == 0: return 0
        core = odd_core(n)
        bits = bit_length(core)
        return bit_reverse(core, bits)
    
    def recompose_from_core(v2_val, core):
        return core << v2_val


#==============================================================================
# CONSTANTS
#==============================================================================

MAGIC = b'THETA1'
VERSION = 1
DEFAULT_CHUNK_BITS = 32


#==============================================================================
# DATA STRUCTURES
#==============================================================================

@dataclass
class ThetaHeader:
    """Header for theta-encoded files"""
    magic: bytes
    version: int
    chunk_bits: int
    original_size: int
    encrypted: bool
    key_hash: str
    xor_checksum: int
    mix_checksum: int
    num_chunks: int


@dataclass 
class ThetaChunk:
    """Single encoded chunk"""
    theta_key: int
    core_shell: int  # bit_length(core) - 1
    v2: int


#==============================================================================
# CORE CODEC
#==============================================================================

class ThetaCodec:
    """Reversible binary encoding using theta transform."""
    
    def __init__(self, chunk_bits: int = DEFAULT_CHUNK_BITS):
        if chunk_bits not in (8, 16, 32, 64):
            raise ValueError("chunk_bits must be 8, 16, 32, or 64")
        self.chunk_bits = chunk_bits
        self.chunk_bytes = chunk_bits // 8
        self.struct_format = {8: 'B', 16: 'H', 32: 'I', 64: 'Q'}[chunk_bits]
    
    def encode_chunk(self, value: int) -> ThetaChunk:
        """Encode single integer to theta representation."""
        if value == 0:
            return ThetaChunk(theta_key=0, core_shell=0, v2=0)
        
        v2_val = v2(value)
        core = odd_core(value)
        core_bits = bit_length(core)
        key = bit_reverse(core, core_bits)
        
        return ThetaChunk(
            theta_key=key,
            core_shell=core_bits - 1,
            v2=v2_val
        )
    
    def decode_chunk(self, chunk: ThetaChunk) -> int:
        """Decode theta chunk back to integer."""
        if chunk.theta_key == 0:
            if chunk.core_shell != 0 or chunk.v2 != 0:
                import warnings
                warnings.warn(f"Suspicious zero encoding: core_shell={chunk.core_shell}, v2={chunk.v2}")
            return 0
        
        core_bits = chunk.core_shell + 1
        core = bit_reverse(chunk.theta_key, core_bits)
        return recompose_from_core(chunk.v2, core)
    
    def _derive_key_stream(self, key: str, size: int) -> bytes:
        """Derive encryption key stream from password."""
        result = bytearray()
        counter = 0
        while len(result) < size:
            h = hashlib.sha256(f"{key}:{counter}".encode()).digest()
            result.extend(h)
            counter += 1
        return bytes(result[:size])
    
    def encode_stream(self, data: bytes, key: Optional[str] = None) -> Tuple[ThetaHeader, list]:
        """Encode bytes to theta chunks."""
        # Pad to chunk boundary
        padding = (self.chunk_bytes - len(data) % self.chunk_bytes) % self.chunk_bytes
        padded = data + bytes(padding)
        
        # Encrypt if key provided
        encrypted = False
        key_hash = ""
        if key:
            encrypted = True
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            key_stream = self._derive_key_stream(key, len(padded))
            padded = bytes(a ^ b for a, b in zip(padded, key_stream))
        
        # Encode chunks
        chunks = []
        xor_checksum = 0
        mix_checksum = 0x5A5A5A5A
        
        for i in range(0, len(padded), self.chunk_bytes):
            value = struct.unpack(self.struct_format, padded[i:i+self.chunk_bytes])[0]
            chunk = self.encode_chunk(value)
            chunks.append(chunk)
            
            xor_checksum ^= chunk.theta_key
            mix_checksum = ((mix_checksum << 5) | (mix_checksum >> 27)) & 0xFFFFFFFF
            mix_checksum ^= chunk.theta_key
        
        header = ThetaHeader(
            magic=MAGIC,
            version=VERSION,
            chunk_bits=self.chunk_bits,
            original_size=len(data),
            encrypted=encrypted,
            key_hash=key_hash,
            xor_checksum=xor_checksum,
            mix_checksum=mix_checksum,
            num_chunks=len(chunks)
        )
        
        return header, chunks
    
    def decode_stream(self, header: ThetaHeader, chunks: list, key: Optional[str] = None) -> bytes:
        """Decode theta chunks back to bytes."""
        # Verify key if encrypted
        if header.encrypted:
            if not key:
                raise ValueError("File is encrypted but no key provided")
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            if key_hash != header.key_hash:
                raise ValueError("Incorrect encryption key")
        
        # Verify checksums
        xor_checksum = 0
        mix_checksum = 0x5A5A5A5A
        for chunk in chunks:
            xor_checksum ^= chunk.theta_key
            mix_checksum = ((mix_checksum << 5) | (mix_checksum >> 27)) & 0xFFFFFFFF
            mix_checksum ^= chunk.theta_key
        
        if xor_checksum != header.xor_checksum or mix_checksum != header.mix_checksum:
            raise ValueError("Checksum mismatch - data corrupted")
        
        # Decode chunks
        result = bytearray()
        for chunk in chunks:
            value = self.decode_chunk(chunk)
            result.extend(struct.pack(self.struct_format, value))
        
        # Decrypt if needed
        if header.encrypted and key:
            key_stream = self._derive_key_stream(key, len(result))
            result = bytearray(a ^ b for a, b in zip(result, key_stream))
        
        # Remove padding
        return bytes(result[:header.original_size])
    
    def encode_to_binary(self, header: ThetaHeader, chunks: list) -> bytes:
        """Serialize header and chunks to binary format."""
        out = bytearray()
        out.extend(header.magic)
        out.extend(struct.pack('<BBQI', header.version, header.chunk_bits, 
                               header.original_size, header.num_chunks))
        out.extend(struct.pack('<B', 1 if header.encrypted else 0))
        out.extend(header.key_hash.encode().ljust(16, b'\x00')[:16])
        out.extend(struct.pack('<II', header.xor_checksum, header.mix_checksum))
        
        for chunk in chunks:
            if self.chunk_bits == 64:
                out.extend(struct.pack('<QBB', chunk.theta_key, chunk.core_shell, chunk.v2))
            else:
                out.extend(struct.pack('<IBB', chunk.theta_key, chunk.core_shell, chunk.v2))
        
        return bytes(out)
    
    def decode_from_binary(self, data: bytes) -> Tuple[ThetaHeader, list]:
        """Deserialize binary format to header and chunks."""
        if not data.startswith(MAGIC):
            raise ValueError("Invalid theta file format")
        
        offset = len(MAGIC)
        version, chunk_bits, original_size, num_chunks = struct.unpack('<BBQI', data[offset:offset+14])
        offset += 14
        
        encrypted = struct.unpack('<B', data[offset:offset+1])[0] == 1
        offset += 1
        
        key_hash = data[offset:offset+16].rstrip(b'\x00').decode()
        offset += 16
        
        xor_checksum, mix_checksum = struct.unpack('<II', data[offset:offset+8])
        offset += 8
        
        header = ThetaHeader(
            magic=MAGIC, version=version, chunk_bits=chunk_bits,
            original_size=original_size, encrypted=encrypted, key_hash=key_hash,
            xor_checksum=xor_checksum, mix_checksum=mix_checksum, num_chunks=num_chunks
        )
        
        self.chunk_bits = chunk_bits
        self.chunk_bytes = chunk_bits // 8
        self.struct_format = {8: 'B', 16: 'H', 32: 'I', 64: 'Q'}[chunk_bits]
        
        chunks = []
        chunk_size = 10 if chunk_bits == 64 else 6
        for _ in range(num_chunks):
            if chunk_bits == 64:
                key, cs, v = struct.unpack('<QBB', data[offset:offset+chunk_size])
            else:
                key, cs, v = struct.unpack('<IBB', data[offset:offset+chunk_size])
            chunks.append(ThetaChunk(theta_key=key, core_shell=cs, v2=v))
            offset += chunk_size
        
        return header, chunks
    
    def encode_file(self, input_path: str, output_path: str, key: Optional[str] = None, verbose: bool = False):
        """Encode file to theta format."""
        data = Path(input_path).read_bytes()
        header, chunks = self.encode_stream(data, key)
        binary = self.encode_to_binary(header, chunks)
        Path(output_path).write_bytes(binary)
        
        if verbose:
            print(f"Encoded: {len(data)} bytes → {len(binary)} bytes ({len(chunks)} chunks)")
        return header
    
    def decode_file(self, input_path: str, output_path: str, key: Optional[str] = None, verbose: bool = False):
        """Decode theta file back to original."""
        binary = Path(input_path).read_bytes()
        header, chunks = self.decode_from_binary(binary)
        data = self.decode_stream(header, chunks, key)
        Path(output_path).write_bytes(data)
        
        if verbose:
            print(f"Decoded: {len(binary)} bytes → {len(data)} bytes")
        return True, header
    
    def verify_file(self, path: str, verbose: bool = False) -> Tuple[bool, Optional[ThetaHeader]]:
        """Verify theta file integrity."""
        try:
            binary = Path(path).read_bytes()
            header, chunks = self.decode_from_binary(binary)
            
            xor_checksum = 0
            mix_checksum = 0x5A5A5A5A
            for chunk in chunks:
                xor_checksum ^= chunk.theta_key
                mix_checksum = ((mix_checksum << 5) | (mix_checksum >> 27)) & 0xFFFFFFFF
                mix_checksum ^= chunk.theta_key
            
            valid = (xor_checksum == header.xor_checksum and 
                     mix_checksum == header.mix_checksum)
            
            if verbose:
                print(f"File: {path}")
                print(f"  Original size: {header.original_size}")
                print(f"  Chunks: {header.num_chunks}")
                print(f"  Encrypted: {header.encrypted}")
                print(f"  Checksum: {'✓ VALID' if valid else '✗ INVALID'}")
            
            return valid, header
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            return False, None


#==============================================================================
# TEXT FORMAT (like uuencode)
#==============================================================================

class ThetaTextCodec:
    """Text-based encoding for email/terminal safe transmission."""
    
    def __init__(self):
        self.codec = ThetaCodec(chunk_bits=32)
    
    def encode_to_text(self, input_path: str, key: Optional[str] = None) -> str:
        """Encode file to text format."""
        data = Path(input_path).read_bytes()
        header, chunks = self.codec.encode_stream(data, key)
        
        lines = [
            f"begin-theta {header.original_size} {Path(input_path).name}",
            f"version {header.version}",
            f"chunks {header.num_chunks}",
            f"encrypted {'yes' if header.encrypted else 'no'}",
            f"checksum {header.xor_checksum:08x} {header.mix_checksum:08x}",
            ""
        ]
        
        for i, chunk in enumerate(chunks):
            lines.append(f"{chunk.theta_key:08x} {chunk.core_shell:02x} {chunk.v2:02x}")
            if (i + 1) % 8 == 0:
                lines.append("")
        
        lines.append("")
        lines.append("end-theta")
        
        return "\n".join(lines)
    
    def decode_from_text(self, text: str, output_path: str, key: Optional[str] = None):
        """Decode text format back to file."""
        lines = text.strip().split('\n')
        
        # Parse header
        first = lines[0].split()
        original_size = int(first[1])
        
        chunks = []
        encrypted = False
        xor_cs = mix_cs = 0
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            if line.startswith('version'):
                continue
            if line.startswith('chunks'):
                continue
            if line.startswith('encrypted'):
                encrypted = 'yes' in line
            if line.startswith('checksum'):
                parts = line.split()
                xor_cs = int(parts[1], 16)
                mix_cs = int(parts[2], 16)
            if line.startswith('end-theta'):
                break
            
            # Chunk line
            parts = line.split()
            if len(parts) == 3:
                try:
                    key_val = int(parts[0], 16)
                    cs = int(parts[1], 16)
                    v = int(parts[2], 16)
                    chunks.append(ThetaChunk(theta_key=key_val, core_shell=cs, v2=v))
                except ValueError:
                    continue
        
        header = ThetaHeader(
            magic=MAGIC, version=1, chunk_bits=32,
            original_size=original_size, encrypted=encrypted, key_hash="",
            xor_checksum=xor_cs, mix_checksum=mix_cs, num_chunks=len(chunks)
        )
        
        data = self.codec.decode_stream(header, chunks, key)
        Path(output_path).write_bytes(data)


#==============================================================================
# STDIN/STDOUT SUPPORT
#==============================================================================

def encode_stdin_stdout(key: Optional[str] = None, text_format: bool = False):
    """Encode from stdin to stdout."""
    if sys.stdin.isatty():
        print("Error: No input on stdin", file=sys.stderr)
        sys.exit(1)
    
    data = sys.stdin.buffer.read()
    codec = ThetaCodec()
    header, chunks = codec.encode_stream(data, key)
    
    if text_format:
        # Text format
        lines = [
            f"begin-theta {header.original_size} stdin",
            f"version {header.version}",
            f"chunks {header.num_chunks}",
            f"encrypted {'yes' if header.encrypted else 'no'}",
            f"checksum {header.xor_checksum:08x} {header.mix_checksum:08x}",
            ""
        ]
        for chunk in chunks:
            lines.append(f"{chunk.theta_key:08x} {chunk.core_shell:02x} {chunk.v2:02x}")
        lines.append("")
        lines.append("end-theta")
        sys.stdout.write("\n".join(lines))
    else:
        # Binary format
        binary = codec.encode_to_binary(header, chunks)
        sys.stdout.buffer.write(binary)


def decode_stdin_stdout(key: Optional[str] = None):
    """Decode from stdin to stdout."""
    if sys.stdin.isatty():
        print("Error: No input on stdin", file=sys.stderr)
        sys.exit(1)
    
    data = sys.stdin.buffer.read()
    codec = ThetaCodec()
    
    # Check if text or binary
    if data.startswith(b'begin-theta'):
        text_codec = ThetaTextCodec()
        # Write to temp, decode, read back
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        text_codec.decode_from_text(data.decode('utf-8'), tmp_path, key)
        result = Path(tmp_path).read_bytes()
        os.unlink(tmp_path)
    else:
        header, chunks = codec.decode_from_binary(data)
        result = codec.decode_stream(header, chunks, key)
    
    sys.stdout.buffer.write(result)


#==============================================================================
# CLI
#==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Theta Codec - Reversible binary encoding via theta transform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # File to file
  %(prog)s encode input.bin output.theta
  %(prog)s decode output.theta recovered.bin
  
  # Stdin/stdout (use - for stdin/stdout)
  cat input.bin | %(prog)s encode - - > output.theta
  cat output.theta | %(prog)s decode - - > recovered.bin
  
  # With encryption
  %(prog)s encode input.bin output.theta --key "secret"
  
  # Text format (like uuencode)
  %(prog)s encode input.bin output.txt --text
"""
    )
    
    parser.add_argument('command', choices=['encode', 'decode', 'verify', 'demo'],
                       help='Command to execute')
    parser.add_argument('input', nargs='?', default='-',
                       help='Input file (- for stdin)')
    parser.add_argument('output', nargs='?', default='-',
                       help='Output file (- for stdout)')
    parser.add_argument('--key', '-k', help='Encryption key')
    parser.add_argument('--text', '-t', action='store_true',
                       help='Use text format instead of binary')
    
    args = parser.parse_args()
    
    if args.command == 'encode':
        if args.input == '-' or args.output == '-':
            encode_stdin_stdout(args.key, args.text)
        else:
            if args.text:
                codec = ThetaTextCodec()
                text = codec.encode_to_text(args.input, args.key)
                Path(args.output).write_text(text)
                print(f"Encoded to text: {args.output}", file=sys.stderr)
            else:
                codec = ThetaCodec()
                codec.encode_file(args.input, args.output, args.key, verbose=True)
    
    elif args.command == 'decode':
        if args.input == '-' or args.output == '-':
            decode_stdin_stdout(args.key)
        else:
            content = Path(args.input).read_bytes()
            if content.startswith(b'begin-theta'):
                codec = ThetaTextCodec()
                codec.decode_from_text(content.decode('utf-8'), args.output, args.key)
                print(f"Decoded from text: {args.output}", file=sys.stderr)
            else:
                codec = ThetaCodec()
                codec.decode_file(args.input, args.output, args.key, verbose=True)
    
    elif args.command == 'verify':
        if args.input == '-':
            print("Error: Cannot verify stdin", file=sys.stderr)
            sys.exit(1)
        codec = ThetaCodec()
        valid, _ = codec.verify_file(args.input, verbose=True)
        sys.exit(0 if valid else 1)
    
    elif args.command == 'demo':
        print("=" * 60)
        print("  THETA CODEC DEMONSTRATION")
        print("=" * 60)
        
        test_data = b"Hello, Theta World! This is a test."
        
        codec = ThetaCodec()
        header, chunks = codec.encode_stream(test_data)
        
        print(f"\nOriginal: {len(test_data)} bytes")
        print(f"Encoded: {len(chunks)} chunks")
        print(f"Checksum: {header.xor_checksum:08x}")
        
        recovered = codec.decode_stream(header, chunks)
        print(f"Round-trip: {'✓ PASS' if recovered == test_data else '✗ FAIL'}")
        
        # Show a few chunks
        print(f"\nFirst 5 chunks:")
        for i, chunk in enumerate(chunks[:5]):
            print(f"  {i}: theta_key={chunk.theta_key:08x} shell={chunk.core_shell} v2={chunk.v2}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
