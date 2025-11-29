# ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
# This file remains for reference only.
# Do NOT use in production. No support, no guarantees.

# Theta Order Prime Analysis - Complete HOWTO

## Overview

This tool performs **TRUE theta order** prime analysis, iterating by angular position rather than integer value. Each segment contains integers from **multiple shells**, enabling analysis of how prime properties at one shell relate to higher shells.

---

## Quick Start

```bash
# 1. Compile
nvcc -O3 -arch=sm_86 theta_order_complete.cu -o theta_order_complete

# 2. Full 32-bit scan (~days on modern GPU)
./theta_order_batch.sh scan32

# 3. Check progress anytime
./theta_order_batch.sh status

# 4. Resume after interruption
./theta_order_batch.sh resume

# 5. Zoom into extreme regions
./theta_order_batch.sh analyze_extremes
```

---

## Compilation

```bash
# For RTX 30xx/40xx series
nvcc -O3 -arch=sm_86 theta_order_complete.cu -o theta_order_complete

# For RTX 20xx series
nvcc -O3 -arch=sm_75 theta_order_complete.cu -o theta_order_complete

# For GTX 10xx series
nvcc -O3 -arch=sm_61 theta_order_complete.cu -o theta_order_complete

# Generic (slower but compatible)
nvcc -O3 theta_order_complete.cu -o theta_order_complete
```

---

## Command Reference

### Mode: scan32

Full 32-bit theta scan covering shells 2-31.

```bash
./theta_order_complete scan32 [options]

Options:
  --min-shell S       Minimum shell (default: 2)
  --max-shell S       Maximum shell (default: 31, max: 31)
  --theta-exp E       Segment size = 2^E theta positions (default: 20)
  --segments N        Number of segments (0 = auto for full range)
  --start-seg N       Starting segment (for resuming)
  --output FILE       Output CSV file
  -q                  Quiet mode
  -v                  Verbose mode
```

**Examples:**

```bash
# Full scan with default settings (covers 2^30 theta positions)
./theta_order_complete scan32 --output full_32bit.csv

# Custom segment size (2^16 = 65536 positions per segment)
./theta_order_complete scan32 --theta-exp 16 --output fine_32bit.csv

# Limited shell range
./theta_order_complete scan32 --min-shell 10 --max-shell 25 --output mid_shells.csv

# Resume from segment 500
./theta_order_complete scan32 --start-seg 500 --output resumed.csv
```

---

### Mode: scan64

64-bit theta scan for higher shells.

```bash
./theta_order_complete scan64 [options]

Options:
  --min-shell S       Minimum shell (default: 2)
  --max-shell S       Maximum shell (default: 62, max: 62)
  --theta-start T     Starting theta position (decimal or 0x hex)
  --theta-exp E       Segment size = 2^E (default: 20)
  --segments N        Number of segments
  --output FILE       Output CSV file
```

**Examples:**

```bash
# Start from theta = 2^32
./theta_order_complete scan64 --theta-start 0x100000000 --segments 1024

# High shells only
./theta_order_complete scan64 --min-shell 40 --max-shell 55 --segments 256

# Large segments for faster coverage
./theta_order_complete scan64 --theta-exp 24 --segments 1024
```

---

### Mode: zoom64

Zoom into a 32-bit theta region with 64-bit precision. This "expands" a single theta position from scan32 into 2^32 sub-positions.

```bash
./theta_order_complete zoom64 [options]

Options:
  --base-theta T      Base theta position from scan32 to zoom into
  --sub-exp E         Sub-segment size = 2^E (default: 20)
  --sub-segments N    Number of sub-segments (default: 16)
  --min-shell S       Minimum shell (default: 32)
  --max-shell S       Maximum shell (default: 62)
  --output FILE       Output CSV file
```

**Examples:**

```bash
# Zoom into theta position 12345678 from scan32
./theta_order_complete zoom64 --base-theta 12345678 --sub-segments 64

# Fine zoom with small sub-segments
./theta_order_complete zoom64 --base-theta 12345678 --sub-exp 16 --sub-segments 256

# Focus on specific shell range
./theta_order_complete zoom64 --base-theta 12345678 --min-shell 32 --max-shell 40
```

---

### Mode: batch

Long-running batch mode with automatic checkpointing. Designed for week-long runs.

```bash
./theta_order_complete batch [options]

Options:
  --precision P       32 or 64 (default: 32)
  --min-shell S       Minimum shell
  --max-shell S       Maximum shell
  --theta-exp E       Segment size = 2^E
  --total-segments N  Total segments to process
  --start-seg N       Starting segment (for resuming)
  --output-dir DIR    Output directory (default: ./theta_output)
  --checkpoint FILE   Checkpoint filename (default: checkpoint.txt)
  --sleep-ms MS       Sleep between segments (default: 100)
```

**Examples:**

```bash
# Full 32-bit batch run
./theta_order_complete batch --precision 32 --total-segments 1024 --output-dir ./results

# Resume interrupted run
./theta_order_complete batch --precision 32 --checkpoint ./results/checkpoint.txt

# 64-bit batch with longer sleep for thermal management
./theta_order_complete batch --precision 64 --sleep-ms 500 --total-segments 2048
```

---

## Bash Script Usage

The `theta_order_batch.sh` script wraps the binary for common workflows.

```bash
# Full 32-bit scan
./theta_order_batch.sh scan32

# Full 64-bit scan  
./theta_order_batch.sh scan64

# Resume from checkpoint
./theta_order_batch.sh resume

# Find and zoom into extreme regions
./theta_order_batch.sh analyze_extremes

# Full workflow: scan32 → find extremes → zoom
./theta_order_batch.sh multi_precision

# Check current status
./theta_order_batch.sh status

# Show help
./theta_order_batch.sh help
```

### Environment Variables

```bash
OUTPUT_DIR=./my_results      # Output directory
BINARY=./theta_order_complete # Binary path
PRECISION=32                  # 32 or 64
MIN_SHELL=2                   # Minimum shell
MAX_SHELL_32=31              # Max shell for 32-bit
MAX_SHELL_64=48              # Max shell for 64-bit
THETA_EXP=20                 # Segment size exponent
TOTAL_SEGMENTS=1024          # Total segments (0=auto)
SLEEP_MS=100                 # Sleep between segments

# Example with custom settings
OUTPUT_DIR=./long_run TOTAL_SEGMENTS=4096 ./theta_order_batch.sh scan32
```

---

## Your Original Workflow (Adapted)

Your original script did segments of 2^32 integers in 64-bit mode. Here's how to do that with theta order:

```bash
#!/bin/bash

# Equivalent to your original script but in THETA ORDER

Seg=1024
OUTPUT_DIR="./theta_output"
mkdir -p "$OUTPUT_DIR"

for ((i=0; i<Seg; i++)); do
    # Calculate theta start for this segment
    # Each segment covers 2^32 worth of theta positions
    ThetaStart=$(python3 -c "print($i * (2**32))")
    ThetaHex=$(printf "0x%016x" $ThetaStart)
    
    echo "Segment $i/$Seg: theta_start=$ThetaHex"
    
    ./theta_order_complete scan64 \
        --theta-start "$ThetaStart" \
        --theta-exp 20 \
        --segments 1 \
        --min-shell 32 \
        --max-shell 48 \
        --output "$OUTPUT_DIR/analysis_64bit_${i}_${Seg}.csv" \
        -q
    
    sleep 5
    sync
done
```

Or use the built-in batch mode (recommended):

```bash
./theta_order_complete batch \
    --precision 64 \
    --total-segments 1024 \
    --theta-exp 32 \
    --min-shell 32 \
    --max-shell 48 \
    --output-dir ./theta_output \
    --sleep-ms 5000
```

---

## Multi-Resolution Workflow

Your idea of zooming into extreme regions:

### Step 1: Full 32-bit Scan

```bash
./theta_order_complete batch \
    --precision 32 \
    --total-segments 1024 \
    --output-dir ./phase1
```

### Step 2: Find Extremes

```bash
# Find min/max density segments
cd ./phase1

# Min density segment
tail -n +2 theta_batch_32bit.csv | sort -t',' -k8 -n | head -5

# Max density segment  
tail -n +2 theta_batch_32bit.csv | sort -t',' -k8 -rn | head -5
```

### Step 3: Zoom into Extremes

```bash
# Say segment 789 had min density, segment 234 had max density
# Segment 789 corresponds to theta_start = 789 * 2^20

./theta_order_complete zoom64 \
    --base-theta $((789 * 1048576)) \
    --sub-exp 20 \
    --sub-segments 64 \
    --min-shell 32 \
    --max-shell 48 \
    --output ./phase2_min_density.csv

./theta_order_complete zoom64 \
    --base-theta $((234 * 1048576)) \
    --sub-exp 20 \
    --sub-segments 64 \
    --min-shell 32 \
    --max-shell 48 \
    --output ./phase2_max_density.csv
```

### Step 4: Compare

```bash
# Compare density spreads
echo "Min density region spread:"
tail -n +2 phase2_min_density.csv | awk -F',' '{print $8}' | sort -n | head -1
tail -n +2 phase2_min_density.csv | awk -F',' '{print $8}' | sort -n | tail -1

echo "Max density region spread:"
tail -n +2 phase2_max_density.csv | awk -F',' '{print $8}' | sort -n | head -1
tail -n +2 phase2_max_density.csv | awk -F',' '{print $8}' | sort -n | tail -1
```

---

## Output CSV Columns

| Column | Description |
|--------|-------------|
| segment | Segment number |
| theta_start | Starting theta position |
| theta_end | Ending theta position |
| shell_config | Configured shell range (e.g., "2-31") |
| shell_seen | **Actually observed shells** (should span range if theta order works!) |
| total | Total integers tested |
| primes | Prime count |
| density | Prime density (primes/total) |
| avg_shell | Average shell of tested integers |
| pop_range | Popcount min-max |
| avg_pop | Average popcount |
| first_prime | First prime found |
| last_prime | Last prime found |
| twins | Twin prime count |
| mod6_1 | Primes ≡ 1 (mod 6) |
| mod6_5 | Primes ≡ 5 (mod 6) |
| time_ms | Computation time |

---

## Verification: Is Theta Order Working?

**The key check:** Look at `shell_seen` vs `shell_config`.

```bash
# If theta order is WRONG (natural order):
shell_config,shell_seen
2-31,20-20        # ← Same shell! BAD
2-31,21-21        # ← Same shell! BAD

# If theta order is CORRECT:
shell_config,shell_seen
2-31,2-31         # ← Full range! GOOD
2-31,2-31         # ← Full range! GOOD
```

In theta order, each segment processes integers from ALL configured shells because the same angular position exists at every radius.

---

## Performance Tuning

### Segment Size (--theta-exp)

| Value | Segment Size | Notes |
|-------|--------------|-------|
| 16 | 65,536 | Fine granularity, more CSV rows |
| 20 | 1,048,576 | Default, good balance |
| 24 | 16,777,216 | Fewer segments, longer per segment |
| 28 | 268,435,456 | Coarse, fastest coverage |

### Sleep Time (--sleep-ms)

- **0-50ms**: Maximum speed, GPU may throttle
- **100ms** (default): Good balance
- **500-1000ms**: Thermal management for long runs
- **5000ms**: Conservative, matches your original script

### Shell Range

More shells = more work per theta position:

| Shells | Integers/Theta | Time Factor |
|--------|----------------|-------------|
| 2-20 | 19 | 0.6x |
| 2-31 | 30 | 1x (baseline) |
| 2-48 | 47 | 1.6x |
| 32-62 | 31 | 1x (but 64-bit ops are slower) |

---

## Estimated Run Times

On RTX 3080 (rough estimates):

| Mode | Range | Time |
|------|-------|------|
| scan32 full | 2^30 theta, shells 2-31 | ~24-48 hours |
| scan64 | 2^32 theta, shells 32-48 | ~3-7 days |
| zoom64 | 2^32 sub-positions | ~30-60 min |

---

## Tips for Week-Long Runs

1. **Use batch mode** - automatic checkpointing
2. **Set sleep-ms=500** - prevents thermal throttling
3. **Monitor with status** - `./theta_order_batch.sh status`
4. **Run in screen/tmux** - survives SSH disconnection
5. **Check disk space** - CSV files grow over time

```bash
# Run in screen
screen -S theta_run
./theta_order_batch.sh scan32
# Ctrl+A, D to detach

# Reconnect later
screen -r theta_run
```

---

## Troubleshooting

### "shell_seen shows same value for all segments"
→ This means natural order, not theta order. Use the corrected code.

### "CUDA out of memory"
→ Reduce theta-exp to use smaller segments.

### "Very slow performance"
→ Check nvidia-smi for thermal throttling. Increase sleep-ms.

### "Resume doesn't work"
→ Ensure checkpoint.txt exists and has correct format.

---

## Files

- `theta_order_complete.cu` - Main CUDA source
- `theta_order_batch.sh` - Wrapper script
- `theta_order_prime_analysis.cu` - Simpler version (32-bit only)
- `THETA_ORDER_CORRECTION_README.md` - Explanation of the fix
