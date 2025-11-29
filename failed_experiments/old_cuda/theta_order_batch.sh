#!/bin/bash
# ARCHIVED / EXPERIMENTAL CODE — NOT MAINTAINED
# This file remains for reference only.
# Do NOT use in production. No support, no guarantees.
#
# theta_order_batch.sh - Production script for week-long theta order analysis
#
# ============================================================================
# QUICK START
# ============================================================================
#
#   # Compile
#   nvcc -O3 -arch=sm_86 theta_order_complete.cu -o theta_order_complete
#
#   # Run full 32-bit scan (takes ~days depending on GPU)
#   ./theta_order_batch.sh scan32
#
#   # Resume interrupted run
#   ./theta_order_batch.sh resume
#
#   # Zoom into extreme regions (after scan32 completes)
#   ./theta_order_batch.sh analyze_extremes
#
# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./theta_results}"

# Binary location
BINARY="${BINARY:-./theta_order_complete}"

# Precision: 32 or 64
PRECISION="${PRECISION:-32}"

# Shell range
MIN_SHELL="${MIN_SHELL:-2}"
MAX_SHELL_32="${MAX_SHELL_32:-31}"
MAX_SHELL_64="${MAX_SHELL_64:-48}"

# Segment configuration
THETA_EXP="${THETA_EXP:-20}"          # 2^20 = 1M theta positions per segment
TOTAL_SEGMENTS="${TOTAL_SEGMENTS:-1024}"  # Total segments (0 = auto for full range)

# Sleep between segments (ms) - helps with thermal management
SLEEP_MS="${SLEEP_MS:-100}"

# Checkpoint interval (segments)
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"

# Log file
LOG_FILE="${OUTPUT_DIR}/theta_batch.log"

# ============================================================================
# FUNCTIONS
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $*"
    exit 1
}

check_binary() {
    if [ ! -x "$BINARY" ]; then
        error "Binary not found: $BINARY. Please compile first."
    fi
}

setup_output_dir() {
    mkdir -p "$OUTPUT_DIR" || error "Cannot create output directory"
    log "Output directory: $OUTPUT_DIR"
}

# ============================================================================
# SCAN32: Full 32-bit theta scan
# ============================================================================

scan32() {
    log "=== Starting SCAN32: Full 32-bit Theta Scan ==="
    
    check_binary
    setup_output_dir
    
    local max_shell="$MAX_SHELL_32"
    local segments="${TOTAL_SEGMENTS}"
    
    # Calculate total theta positions: 2^(max_shell - 1)
    local max_theta=$((1 << (max_shell - 1)))
    local seg_size=$((1 << THETA_EXP))
    
    if [ "$segments" -eq 0 ]; then
        segments=$(( (max_theta + seg_size - 1) / seg_size ))
    fi
    
    log "Configuration:"
    log "  Shells: $MIN_SHELL to $max_shell"
    log "  Theta range: 0 to $max_theta"
    log "  Segment size: 2^$THETA_EXP = $seg_size"
    log "  Total segments: $segments"
    log "  Sleep between segments: ${SLEEP_MS}ms"
    
    "$BINARY" batch \
        --precision 32 \
        --min-shell "$MIN_SHELL" \
        --max-shell "$max_shell" \
        --theta-exp "$THETA_EXP" \
        --total-segments "$segments" \
        --sleep-ms "$SLEEP_MS" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "=== SCAN32 Complete ==="
}

# ============================================================================
# SCAN64: Full 64-bit theta scan
# ============================================================================

scan64() {
    log "=== Starting SCAN64: Full 64-bit Theta Scan ==="
    
    check_binary
    setup_output_dir
    
    local max_shell="$MAX_SHELL_64"
    local segments="${TOTAL_SEGMENTS}"
    local theta_start="${THETA_START:-0}"
    
    log "Configuration:"
    log "  Shells: $MIN_SHELL to $max_shell"
    log "  Starting theta: $theta_start"
    log "  Segment size: 2^$THETA_EXP"
    log "  Segments: $segments"
    
    "$BINARY" batch \
        --precision 64 \
        --min-shell "$MIN_SHELL" \
        --max-shell "$max_shell" \
        --theta-exp "$THETA_EXP" \
        --total-segments "$segments" \
        --sleep-ms "$SLEEP_MS" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "=== SCAN64 Complete ==="
}

# ============================================================================
# RESUME: Resume interrupted run
# ============================================================================

resume() {
    log "=== Resuming from checkpoint ==="
    
    check_binary
    
    local ckpt_file="$OUTPUT_DIR/checkpoint.txt"
    
    if [ ! -f "$ckpt_file" ]; then
        error "No checkpoint file found: $ckpt_file"
    fi
    
    local last_seg=$(head -1 "$ckpt_file")
    log "Resuming from segment $last_seg"
    
    local max_shell="$MAX_SHELL_32"
    if [ "$PRECISION" -eq 64 ]; then
        max_shell="$MAX_SHELL_64"
    fi
    
    "$BINARY" batch \
        --precision "$PRECISION" \
        --min-shell "$MIN_SHELL" \
        --max-shell "$max_shell" \
        --theta-exp "$THETA_EXP" \
        --total-segments "$TOTAL_SEGMENTS" \
        --start-seg "$last_seg" \
        --sleep-ms "$SLEEP_MS" \
        --output-dir "$OUTPUT_DIR" \
        --checkpoint checkpoint.txt \
        2>&1 | tee -a "$LOG_FILE"
    
    log "=== Resume Complete ==="
}

# ============================================================================
# ANALYZE_EXTREMES: Find and zoom into extreme regions
# ============================================================================

analyze_extremes() {
    log "=== Analyzing extremes from scan32 results ==="
    
    local csv_file="$OUTPUT_DIR/theta_batch_32bit.csv"
    
    if [ ! -f "$csv_file" ]; then
        error "Scan32 results not found: $csv_file"
    fi
    
    # Find segments with min/max density
    log "Finding extreme density segments..."
    
    # Skip header, sort by density (column 8)
    local min_seg=$(tail -n +2 "$csv_file" | sort -t',' -k8 -n | head -1 | cut -d',' -f1)
    local max_seg=$(tail -n +2 "$csv_file" | sort -t',' -k8 -rn | head -1 | cut -d',' -f1)
    
    local seg_size=$((1 << THETA_EXP))
    local min_theta=$((min_seg * seg_size))
    local max_theta=$((max_seg * seg_size))
    
    log "Min density segment: $min_seg (theta_start=$min_theta)"
    log "Max density segment: $max_seg (theta_start=$max_theta)"
    
    # Zoom into min density region
    log "Zooming into min density region..."
    "$BINARY" zoom64 \
        --base-theta "$min_theta" \
        --sub-exp 20 \
        --sub-segments 64 \
        --min-shell 32 \
        --max-shell 48 \
        --output "$OUTPUT_DIR/zoom_min_density.csv" \
        2>&1 | tee -a "$LOG_FILE"
    
    # Zoom into max density region
    log "Zooming into max density region..."
    "$BINARY" zoom64 \
        --base-theta "$max_theta" \
        --sub-exp 20 \
        --sub-segments 64 \
        --min-shell 32 \
        --max-shell 48 \
        --output "$OUTPUT_DIR/zoom_max_density.csv" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "=== Extreme Analysis Complete ==="
    log "Results:"
    log "  Min density zoom: $OUTPUT_DIR/zoom_min_density.csv"
    log "  Max density zoom: $OUTPUT_DIR/zoom_max_density.csv"
}

# ============================================================================
# CUSTOM: Custom segment range (like your original script)
# ============================================================================

custom_range() {
    local total_seg="${1:-1024}"
    local seg_exp="${2:-32}"        # Each segment covers 2^32 integers worth of theta
    local precision="${3:-64}"
    
    log "=== Custom Range Scan ==="
    log "Total segments: $total_seg"
    log "Segment exponent: $seg_exp"
    log "Precision: $precision"
    
    check_binary
    setup_output_dir
    
    local i=0
    while [ $i -lt $total_seg ]; do
        local theta_start=$(python3 -c "print($i * (2**$seg_exp))")
        local theta_hex=$(printf "0x%016x" "$theta_start")
        
        log "Segment $i/$total_seg: theta_start=$theta_hex"
        
        local outfile="$OUTPUT_DIR/analysis_${precision}bit_seg${i}_of${total_seg}.csv"
        
        if [ "$precision" -eq 32 ]; then
            "$BINARY" scan32 \
                --theta-start "$theta_start" \
                --theta-exp 20 \
                --segments 1 \
                --output "$outfile" \
                -q
        else
            "$BINARY" scan64 \
                --theta-start "$theta_start" \
                --theta-exp 20 \
                --segments 1 \
                --min-shell 32 \
                --max-shell 48 \
                --output "$outfile" \
                -q
        fi
        
        log "  Saved: $outfile"
        
        i=$((i + 1))
        sleep $((SLEEP_MS / 1000))
        sync
    done
    
    log "=== Custom Range Complete ==="
}

# ============================================================================
# MULTI_PRECISION: Run full workflow (32-bit → extremes → 64-bit zoom)
# ============================================================================

multi_precision() {
    log "=== Starting Multi-Precision Workflow ==="
    
    # Phase 1: Full 32-bit scan
    log "Phase 1: Full 32-bit scan"
    scan32
    
    # Phase 2: Analyze and zoom into extremes
    log "Phase 2: Zoom into extremes"
    analyze_extremes
    
    log "=== Multi-Precision Workflow Complete ==="
    log "Results in: $OUTPUT_DIR"
}

# ============================================================================
# STATUS: Show current progress
# ============================================================================

status() {
    echo "=== Theta Order Analysis Status ==="
    echo
    
    if [ -f "$OUTPUT_DIR/checkpoint.txt" ]; then
        local seg=$(head -1 "$OUTPUT_DIR/checkpoint.txt")
        local theta=$(tail -1 "$OUTPUT_DIR/checkpoint.txt")
        echo "Checkpoint: segment $seg, theta $theta"
    else
        echo "No checkpoint found"
    fi
    echo
    
    if [ -f "$OUTPUT_DIR/theta_batch_32bit.csv" ]; then
        local lines=$(wc -l < "$OUTPUT_DIR/theta_batch_32bit.csv")
        echo "32-bit results: $((lines - 1)) segments completed"
        
        # Show density stats
        if [ $lines -gt 1 ]; then
            echo "Density range:"
            tail -n +2 "$OUTPUT_DIR/theta_batch_32bit.csv" | \
                awk -F',' '{sum+=$8; if(min=="" || $8<min) min=$8; if($8>max) max=$8} END {print "  Min: " min "\n  Max: " max "\n  Avg: " sum/NR}'
        fi
    fi
    echo
    
    if [ -f "$LOG_FILE" ]; then
        echo "Last 5 log entries:"
        tail -5 "$LOG_FILE"
    fi
}

# ============================================================================
# HELP
# ============================================================================

help() {
    cat << 'EOF'
theta_order_batch.sh - Production script for theta order prime analysis

USAGE:
    ./theta_order_batch.sh <command> [options]

COMMANDS:
    scan32          Full 32-bit theta scan (shells 2-31)
    scan64          Full 64-bit theta scan (shells 32-48+)
    resume          Resume interrupted run from checkpoint
    analyze_extremes Find and zoom into extreme density regions
    multi_precision  Full workflow: scan32 → analyze → zoom
    custom_range    Custom segment range (like original script)
    status          Show current progress
    help            Show this help

ENVIRONMENT VARIABLES:
    OUTPUT_DIR      Output directory (default: ./theta_results)
    BINARY          Path to theta_order_complete (default: ./theta_order_complete)
    PRECISION       32 or 64 (default: 32)
    MIN_SHELL       Minimum shell (default: 2)
    MAX_SHELL_32    Maximum shell for 32-bit (default: 31)
    MAX_SHELL_64    Maximum shell for 64-bit (default: 48)
    THETA_EXP       Segment size exponent (default: 20)
    TOTAL_SEGMENTS  Total segments (default: 1024, 0=auto)
    SLEEP_MS        Sleep between segments in ms (default: 100)

EXAMPLES:
    # Compile first
    nvcc -O3 -arch=sm_86 theta_order_complete.cu -o theta_order_complete

    # Full 32-bit scan (week-long run)
    ./theta_order_batch.sh scan32

    # Check progress
    ./theta_order_batch.sh status

    # Resume after interrupt (Ctrl+C or power loss)
    ./theta_order_batch.sh resume

    # After scan32 completes, zoom into extremes
    ./theta_order_batch.sh analyze_extremes

    # Custom configuration
    OUTPUT_DIR=./my_results TOTAL_SEGMENTS=2048 ./theta_order_batch.sh scan32

    # Full multi-precision workflow
    ./theta_order_batch.sh multi_precision

OUTPUT FILES:
    theta_results/theta_batch_32bit.csv   - Main scan results
    theta_results/checkpoint.txt           - Resume checkpoint
    theta_results/theta_batch.log          - Execution log
    theta_results/zoom_min_density.csv     - Zoom into min density region
    theta_results/zoom_max_density.csv     - Zoom into max density region

CSV COLUMNS:
    segment         - Segment number
    theta_start     - Starting theta position
    theta_end       - Ending theta position
    shell_config    - Configured shell range
    shell_seen      - Actually observed shell range (should differ if theta order works!)
    total           - Total integers tested
    primes          - Prime count
    density         - Prime density (primes/total)
    avg_shell       - Average shell of tested integers
    pop_range       - Popcount min-max
    avg_pop         - Average popcount
    first_prime     - First prime found in segment
    last_prime      - Last prime found in segment
    twins           - Twin prime count
    mod6_1          - Primes ≡ 1 (mod 6)
    mod6_5          - Primes ≡ 5 (mod 6)
    time_ms         - Computation time

VERIFICATION:
    To verify theta order is working correctly, check that shell_seen differs
    from shell_config. In natural order, shell_seen would be a narrow range
    like "20-20". In theta order, it should span the full configured range
    like "2-31".

MULTI-RESOLUTION ZOOM WORKFLOW:
    1. Run scan32 to get full 32-bit theta coverage
    2. Find extreme segments (min/max density)
    3. Run zoom64 on those regions with 32-bit sub-precision
    4. Analyze if extremes cluster or distribute uniformly

    This is "shooting in the dark" to see if extreme prime densities
    occur in specific angular regions of theta-space.
EOF
}

# ============================================================================
# MAIN
# ============================================================================

case "${1:-help}" in
    scan32)
        scan32
        ;;
    scan64)
        scan64
        ;;
    resume)
        resume
        ;;
    analyze_extremes|analyze|extremes)
        analyze_extremes
        ;;
    multi_precision|multi|full)
        multi_precision
        ;;
    custom_range|custom)
        shift
        custom_range "$@"
        ;;
    status)
        status
        ;;
    help|--help|-h)
        help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
