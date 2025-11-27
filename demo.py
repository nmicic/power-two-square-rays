#!/usr/bin/env python3
disclaimer = """

> **Warning / Disclaimer**  
>This is an AI-assisted exploratory visualization project.
>It is educational and experimental only—not peer-reviewed, not mathematical research, and not a new theorem.
>It does not serve as evidence for or against conjectures.
>This repository is math-art/visualization, not a research paper.
>Its purpose is educational exploration and aesthetic interest.
>Any visual patterns are artifacts of the construction.
 
"""
print(disclaimer)

"""
demo.py

Power-of-two square rays: coordinates, visualization, and analysis.

INTEGER-NATIVE COMPUTATION: All core analysis uses INTEGER/BITWISE operations only.
                           No trigonometry in core logic. Trig only for SVG display.

Commands:

  # SVG VISUALIZATION
  # -----------------
  # Small labeled map):, N = 2^8 = 256
  python3 demo.py small-map --exp 8
  python3 demo.py small-map --exp 8 --with-rays
  
  # Small labeled map with paths (rainbow colored)
  python3 demo.py small-map --exp 8 --ants-path
  python3 demo.py small-map --exp 8 --no-rays --ants-path

  # Odd-core rays visualization, N = 2^12 = 4096
  python3 demo.py odd-core-rays --exp 12

  # Ray structure visualization (alternative view)
  python3 demo.py ray-structure --exp 9                    # default horizontal
  python3 demo.py ray-structure --exp 9 --orientation horizontal
  python3 demo.py ray-structure --exp 9 --orientation vertical

  # COORDINATE ANALYSIS
  # -------------------
  # Shows binary patterns and ray relationships (uses theta_key, not theta_deg)
  python3 demo.py coord-analysis --exp 8
  python3 demo.py coord-analysis --exp 8 --csv coords.csv --coordinate both
  python3 demo.py coord-analysis --exp 8 --csv coords_sorted.csv --sort-by-theta
  python3 demo.py coord-analysis --exp 8 --ray 3              # filter to specific ray
  python3 demo.py coord-analysis --exp 8 --primes-only        # show only primes
  python3 demo.py coord-analysis --exp 8 --coordinate edge    # edge+offset encoding

  # PRIME THETA DENSITY ANALYSIS (CORE)
  # -----------------------------------
  # Prime density in THETA ORDER (angular order, not natural order)
  # Uses only integer/bitwise metrics - CUDA compatible
  python3 demo.py prime-theta-density --exp 12
  python3 demo.py prime-theta-density --exp 16 --csv prime_theta.csv
"""

__version__ = "1.0.0"
__author__ = "Nenad"

import argparse
import csv
import math
import os
from typing import Dict, Tuple, List, Optional

from xml.sax.saxutils import escape


# ------------------ basic helpers ------------------ #

def is_prime(n: int) -> bool:
    """Simple primality test, OK for moderate N."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def odd_core_and_v2(n: int) -> Tuple[int, int]:
    """
    2-adic decomposition: return (a, k) where n = 2^k * a and a is odd.
    """
    k = 0
    while n > 0 and n % 2 == 0:
        n //= 2
        k += 1
    return n, k


def popcount(n: int) -> int:
    """Number of 1-bits in n (Python 3.9+ compatible)."""
    return bin(n).count("1")


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between equal-length bit strings."""
    if len(a) != len(b):
        raise ValueError(f"Strings must have equal length: {len(a)} vs {len(b)}")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


# ------------------ perimeter parameterization ------------------ #

def coord_int(n: int) -> Tuple[int, int]:
    """
    [CORE] Integer coordinate map.
    
    Returns exact integer grid position (x, y) for integer n.
    Uses only integer arithmetic (no trig, no floats in core logic).
    
    For n in shell k (2^k <= n < 2^(k+1)):
      - R = 2^k (shell radius)
      - offset = n - 2^k (position along perimeter, 0 to 2^k - 1)
      - Maps to perimeter of square with corners at (±R, ±R)
      
    The perimeter traversal (clockwise from top-left):
      - TOP edge:    x from -R to +R, y = +R
      - RIGHT edge:  x = +R, y from +R to -R  
      - BOTTOM edge: x from +R to -R, y = -R
      - LEFT edge:   x = -R, y from -R to +R
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n == 1:
        return (0, 0)
    
    # Shell index k and radius R = 2^k
    k = n.bit_length() - 1
    R = 1 << k
    
    # Position within shell: offset in [0, 2^k)
    offset = n - R
    
    # Each edge has 2*R positions, total 8*R for full perimeter
    # But we only have 2^k integers, each occupies "8 units" of perimeter
    # So we scale: perimeter_pos = offset * 8 (conceptually)
    # Edge length in terms of offset: R/4 per edge = 2^(k-2)
    
    # Simpler approach: use the fractional parameter t and scale
    # t = offset / R, then s = t * 8R = offset * 8
    s = offset << 3  # s = offset * 8
    
    # Now s is in [0, 8*R) and we walk the perimeter
    edge_len = R << 1  # 2R per edge
    
    if s < edge_len:
        # TOP edge: x from -R to +R, y = R
        x = -R + s
        y = R
    elif s < (edge_len << 1):
        # RIGHT edge: x = R, y from R to -R
        x = R
        y = R - (s - edge_len)
    elif s < edge_len * 3:
        # BOTTOM edge: x from R to -R, y = -R
        x = R - (s - (edge_len << 1))
        y = -R
    else:
        # LEFT edge: x = -R, y from -R to R
        x = -R
        y = -R + (s - edge_len * 3)
    
    return (x, y)


def compute_ant_path(n: int) -> List[Tuple[int, int]]:
    """
    [VISUALIZATION] Compute a Manhattan path from origin to position n.
    
    Uses only cardinal moves: UP, DOWN, LEFT, RIGHT.
    Returns list of (x, y) coordinates including start and end.
    
    Path strategy: Manhattan path - first horizontal, then vertical.
    (Used for visualization only - demonstrates integer-native navigation.)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    
    target_x, target_y = coord_int(n)
    
    path = [(0, 0)]  # Start at origin
    
    # Walk horizontally first
    x, y = 0, 0
    dx = 1 if target_x > 0 else -1 if target_x < 0 else 0
    while x != target_x:
        x += dx
        path.append((x, y))
    
    # Then walk vertically
    dy = 1 if target_y > 0 else -1 if target_y < 0 else 0
    while y != target_y:
        y += dy
        path.append((x, y))
    
    return path


def odd_core_to_key(a: int, exp: int) -> int:
    """
    Convert odd core to theta-key using pure bitwise operations.
    
    THETA_KEY EXPLAINED:
    ====================
    The theta_key is an INTEGER that encodes angular position around the
    square perimeter. It's computed using ONLY bitwise operations - no trig.
    
    For an odd number a with binary: 1.bbbbb (leading 1 + fractional bits)
    The key is the fractional bits, left-padded to (exp-1) bits.
    
    KEY STRUCTURE (exp-1 bits total):
    ┌──────────┬───────────────────────────────────────┐
    │ Top 2    │ Remaining bits                        │
    │ bits     │ (position within edge)                │
    ├──────────┼───────────────────────────────────────┤
    │ 00       │ TOP edge    (angles ~135° to ~45°)    │
    │ 01       │ RIGHT edge  (angles ~45° to ~-45°)    │
    │ 10       │ BOTTOM edge (angles ~-45° to ~-135°)  │
    │ 11       │ LEFT edge   (angles ~-135° to ~180°)  │
    └──────────┴───────────────────────────────────────┘
    
    EXAMPLE (exp=12, so key is 11 bits):
      odd core a = 21 = 0b10101
      bit_length k = 5
      fractional bits f = 0101 (remove leading 1)
      key = 0101 << (12-5) = 0101 << 7 = 0b01010000000 = 640
      
      Top 2 bits = 01 → RIGHT edge ✓
      (21 maps to coordinates (4,4) on shell 4, which is RIGHT edge)
    
    WALKING THETA ORDER:
      key=0 → key=1 → key=2 → ... → key=2^(exp-1)-1
      This walks clockwise around the square perimeter.
      
    WHY THIS WORKS:
      The binary fraction 0.bbbbb directly maps to perimeter position t.
      Padding/normalizing to fixed width allows integer comparison
      to determine angular order without any trigonometry.
    
    Args:
        a: Odd number (odd core)
        exp: Exponent defining normalization (key will be exp-1 bits)
        
    Returns:
        Theta-key for this odd number in [0, 2^(exp-1))
    """
    if a == 1:
        return 0
    
    k = a.bit_length()
    f = a ^ (1 << (k - 1))  # Remove leading 1 bit
    return f << (exp - k)   # Normalize to (exp-1) bits


def rainbow_color_from_index(idx: int, total: int, saturation: float = 75.0, lightness: float = 50.0) -> str:
    """
    Generate rainbow HSL color based on index.
    
    Distributes colors evenly across the hue spectrum.
    
    Args:
        idx: Index of item (0 to total-1)
        total: Total number of items
        saturation: HSL saturation percentage
        lightness: HSL lightness percentage
        
    Returns:
        CSS HSL color string
    """
    if total <= 0:
        total = 1
    hue = (360.0 * idx) / total
    return hsl(hue, saturation, lightness)


def perimeter_point(R: float, t: float) -> Tuple[float, float]:
    """
    Parameterize the perimeter of a square of radius R (L∞ ball) by t ∈ [0,1).

    Total perimeter length = 8R.
    We map t -> s = t * 8R, and travel:
      top    (left→right),
      right  (top→bottom),
      bottom (right→left),
      left   (bottom→top).
    """
    if R <= 0.0:
        return (0.0, 0.0)
    t = t % 1.0
    s = t * 8.0 * R

    if 0.0 <= s < 2.0 * R:
        # top edge
        x = -R + s
        y = R
    elif 2.0 * R <= s < 4.0 * R:
        # right edge
        x = R
        y = R - (s - 2.0 * R)
    elif 4.0 * R <= s < 6.0 * R:
        # bottom edge
        x = R - (s - 4.0 * R)
        y = -R
    else:
        # left edge
        x = -R
        y = -R + (s - 6.0 * R)

    return (x, y)


def coord(n: int) -> Tuple[float, float]:
    """
    Coordinate map C(n) for the power-of-two square geometry.

    - For n=1: C(1) = (0,0).
    - For n>=2:
        k = floor(log2(n))
        R = 2^k
        t = (n - 2^k)/2^k
        C(n) = perimeter_point(R, t)

    This yields:
        C(2n) = 2 * C(n)   for all n >= 2 (1 is a special base point at origin).
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n)}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if n == 1:
        return (0.0, 0.0)

    k = int(math.floor(math.log2(n)))
    R = float(2 ** k)
    base = 2 ** k
    t = (n - base) / base  # in [0,1)
    return perimeter_point(R, t)


def angle_from_coord(n: int) -> float:
    """
    [VISUALIZATION HELPER - uses trig]
    Angle θ(n) (in radians) via coord(n).
    Used by SVG functions for display only.
    """
    x, y = coord(n)
    return math.atan2(y, x)


# ------------------ SVG utilities ------------------ #

def ensure_svgs_dir() -> str:
    outdir = os.path.join(os.path.dirname(__file__), "svgs")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def hsl(h: float, s: float, l: float) -> str:
    """CSS HSL string."""
    return f"hsl({h:.1f} {s:.1f}% {l:.1f}%)"


# ------------------ Visualization: small labeled map ------------------ #

def svg_small_map(N: int, filename: str, size_px: int = 1600, 
                  with_rays: bool = False, with_ants_path: bool = False) -> None:
    """
    [VISUALIZATION] Generate an SVG with labeled bubbles for numbers 1..N.

    - Coordinates via coord(n).
    - Primes in red, composites in dark gray.
    - Light axes.
    - Optionally draw odd-core rays as colored polylines in background (--with-rays).
    - Optionally draw Manhattan paths as rainbow-colored polylines (--ants-path).
    
    Manhattan paths show the walk from origin to each number's position
    using cardinal moves only (UP/DOWN/LEFT/RIGHT).
    Rainbow colors are assigned based on theta_key (angular position).
    """
    coords: Dict[int, Tuple[float, float]] = {n: coord(n) for n in range(1, N + 1)}

    xs = [x for x, y in coords.values()]
    ys = [y for x, y in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    margin = 60
    width = height = size_px

    span_x = max_x - min_x if max_x > min_x else 1.0
    span_y = max_y - min_y if max_y > min_y else 1.0
    scale = min((width - 2 * margin) / span_x,
                (height - 2 * margin) / span_y)

    def to_svg_xy(x: float, y: float) -> Tuple[float, float]:
        X = (x - min_x) * scale + margin
        Y = (max_y - y) * scale + margin
        return (X, Y)

    font_family = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    title_extra = ""
    if with_rays:
        title_extra = " (with rays)"
    elif with_ants_path:
        title_extra = " (with paths)"
    parts.append(
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" '
        f'font-family="{escape(font_family)}" font-size="18" font-weight="700">'
        f'Power-of-Two Square Map • 1–{N}{title_extra}</text>'
    )
    
    subtitle = 'Primes in red • Composites in dark gray'
    if with_ants_path:
        subtitle = 'Manhattan paths: rainbow colors by angular position • Primes in red'
    parts.append(
        f'<text x="{width/2:.1f}" y="52" text-anchor="middle" '
        f'font-family="{escape(font_family)}" font-size="11" fill="#555">'
        f'{subtitle}</text>'
    )

    # faint axes
    X0, Y0 = to_svg_xy(0.0, 0.0)
    parts.append('<g>')
    parts.append(
        f'<line x1="{X0:.1f}" y1="{margin}" x2="{X0:.1f}" y2="{height-margin}" '
        f'stroke="#e0e0e0" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{margin}" y1="{Y0:.1f}" x2="{width-margin}" y2="{Y0:.1f}" '
        f'stroke="#e0e0e0" stroke-width="1"/>'
    )
    parts.append('</g>')

    # Optional rays in background
    if with_rays:
        # Group n by odd core
        rays: Dict[int, List[int]] = {}
        for n in range(1, N + 1):
            core, v2 = odd_core_and_v2(n)
            if core % 2 == 1 and core >= 3:
                rays.setdefault(core, []).append(n)
        prime_cores = sorted(core for core in rays.keys() if is_prime(core))

        parts.append('<g stroke-linecap="round" stroke-linejoin="round">')
        num_cores = len(prime_cores)
        for idx, core in enumerate(prime_cores):
            hue = (360.0 * idx) / max(num_cores, 1)
            color = hsl(hue, 70.0, 45.0)
            nums_on_ray = sorted(rays[core])
            pts = []
            for n in nums_on_ray:
                x, y = coords[n]
                X, Y = to_svg_xy(x, y)
                pts.append((X, Y))
            if len(pts) < 2:
                continue
            d = " ".join(f"{x:.2f},{y:.2f}" for (x, y) in pts)
            parts.append(
                f'<polyline points="{d}" fill="none" stroke="{color}" '
                f'stroke-width="1.2" stroke-opacity="0.5"/>'
            )
        parts.append('</g>')

    # Optional Manhattan paths (rainbow colored by angular position)
    if with_ants_path:
        # Compute exp from N (N = 2^exp)
        exp = N.bit_length() - 1 if N > 1 else 1
        
        # Collect all numbers except 1 (which is at origin), sorted by theta_key
        path_data: List[Tuple[int, int]] = []  # (key, n) pairs
        for n in range(2, N + 1):
            core, v2 = odd_core_and_v2(n)
            if core % 2 == 1:
                key = odd_core_to_key(core, exp)
                path_data.append((key, n))
        
        # Sort by key for consistent rainbow coloring
        path_data.sort(key=lambda x: x[0])
        total_items = len(path_data)
        
        parts.append('<g stroke-linecap="round" stroke-linejoin="round">')
        for idx, (key, n) in enumerate(path_data):
            # Rainbow color based on index in theta-sorted order
            color = rainbow_color_from_index(idx, total_items, saturation=70.0, lightness=50.0)
            
            # Compute Manhattan path from origin to n's position
            path = compute_ant_path(n)
            
            if len(path) < 2:
                continue
            
            # Convert path to SVG coordinates
            svg_pts = []
            for px, py in path:
                X, Y = to_svg_xy(float(px), float(py))
                svg_pts.append((X, Y))
            
            d = " ".join(f"{x:.2f},{y:.2f}" for (x, y) in svg_pts)
            parts.append(
                f'<polyline points="{d}" fill="none" stroke="{color}" '
                f'stroke-width="0.8" stroke-opacity="0.6"/>'
            )
        parts.append('</g>')

    # bubbles + labels
    parts.append(f'<g font-family="{escape(font_family)}" font-size="11">')
    for n, (x, y) in coords.items():
        X, Y = to_svg_xy(x, y)
        prime_flag = is_prime(n)
        fill = "#d00000" if prime_flag else "#222222"
        # bubble
        parts.append(
            f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="7.5" '
            f'fill="#ffffff" stroke="#cccccc" stroke-width="0.7"/>'
        )
        # label
        parts.append(
            f'<text x="{X:.1f}" y="{Y+3:.1f}" text-anchor="middle" '
            f'font-size="11" font-weight="600" fill="{fill}">{n}</text>'
        )
    parts.append('</g>')

    parts.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ------------------ Visualization: odd-core rays ------------------ #

def svg_prime_rays(N: int, filename: str, size_px: int = 2000) -> None:
    """
    Generate an SVG showing odd-core rays up to N:

    - Gray dots for all 1..N
    - Colored polylines for each prime odd core 'a' connecting a, 2a, 4a, ...
    """
    coords: Dict[int, Tuple[float, float]] = {n: coord(n) for n in range(1, N + 1)}
    xs = [x for x, y in coords.values()]
    ys = [y for x, y in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    margin = 40
    width = height = size_px

    span_x = max_x - min_x if max_x > min_x else 1.0
    span_y = max_y - min_y if max_y > min_y else 1.0
    scale = min((width - 2 * margin) / span_x,
                (height - 2 * margin) / span_y)

    def to_svg_xy(x: float, y: float) -> Tuple[float, float]:
        X = (x - min_x) * scale + margin
        Y = (max_y - y) * scale + margin
        return (X, Y)

    font_family = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"

    # group numbers by odd core
    rays: Dict[int, List[int]] = {}
    for n in range(1, N + 1):
        core, v2 = odd_core_and_v2(n)
        if core % 2 == 1 and core >= 3:  # odd core >= 3
            rays.setdefault(core, []).append(n)

    prime_cores = sorted(core for core in rays.keys() if is_prime(core))

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    parts.append(
        f'<text x="{width/2:.1f}" y="26" text-anchor="middle" '
        f'font-family="{escape(font_family)}" font-size="16" font-weight="700">'
        f'Prime Rays on Power-of-Two Square Grid • 1–{N}</text>'
    )
    parts.append(
        f'<text x="{width/2:.1f}" y="44" text-anchor="middle" '
        f'font-family="{escape(font_family)}" font-size="11" fill="#555">'
        f'Each colored polyline = ray of multiples of a prime odd core a (a, 2a, 4a, ...)</text>'
    )

    # background dots
    parts.append('<g>')
    for n, (x, y) in coords.items():
        X, Y = to_svg_xy(x, y)
        parts.append(
            f'<circle cx="{X:.2f}" cy="{Y:.2f}" r="1.5" '
            f'fill="#999999" fill-opacity="0.25"/>'
        )
    parts.append('</g>')

    # odd-core rays
    parts.append('<g stroke-linecap="round" stroke-linejoin="round">')
    num_cores = len(prime_cores)
    for idx, core in enumerate(prime_cores):
        hue = (360.0 * idx) / max(num_cores, 1)
        color = hsl(hue, 70.0, 45.0)

        nums_on_ray = sorted(rays[core])
        pts = []
        for n in nums_on_ray:
            x, y = coords[n]
            X, Y = to_svg_xy(x, y)
            pts.append((X, Y))
        if len(pts) < 2:
            continue

        d = " ".join(f"{x:.2f},{y:.2f}" for (x, y) in pts)
        parts.append(
            f'<polyline points="{d}" fill="none" stroke="{color}" '
            f'stroke-width="1.4" stroke-opacity="0.9"/>'
        )

        # mark the core itself
        cx, cy = coords[core]
        cX, cY = to_svg_xy(cx, cy)
        parts.append(
            f'<circle cx="{cX:.2f}" cy="{cY:.2f}" r="2.4" '
            f'fill="{color}" stroke="#222222" stroke-width="0.7"/>'
        )

    parts.append('</g>')
    parts.append('</svg>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ------------------ Ray Structure Visualization ------------------ #

def ray_coord(n: int, X: float = 100.0, Y: float = 500.0) -> Tuple[float, float]:
    """
    Ray-structure coordinate system.
    
    Construction:
    1. Place 1 at (0, Y)
    2. Shell k positions at x = X * (2 - 2^(1-k))
    3. Each odd core defines a ray with slope from (0, Y)
    4. Core 5 ray is horizontal (slope = 0)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    
    if n == 1:
        return (0.0, Y)
    
    core, v2 = odd_core_and_v2(n)
    shell = n.bit_length() - 1
    
    # X position based on shell
    if shell == 0:
        x_pos = 0.0
    else:
        x_pos = X * (2.0 - 2.0**(1-shell))
    
    # Slope based on odd core
    if core == 1:
        slope = -1.0
    elif core == 3:
        slope = +1.0
    elif core == 5:
        slope = 0.0  # HORIZONTAL
    elif core == 7:
        slope = 4.0 / 3.0
    else:
        core_shell = core.bit_length() - 1
        if core_shell >= 3:
            shell_start = 1 << core_shell
            num_new_odds = 1 << (core_shell - 1)
            odd_index = (core - shell_start - 1) // 2
            min_slope = -1.0
            max_slope = 2.0
            slope = min_slope + (max_slope - min_slope) * (odd_index + 0.5) / num_new_odds
        else:
            slope = 0.0
    
    y_pos = Y + slope * x_pos
    return (x_pos, y_pos)


def rainbow_color(index: int, total: int) -> str:
    """Generate rainbow color for index out of total items."""
    if total <= 1:
        hue = 0
    else:
        hue = (index * 360) / total
    return f"hsl({hue:.0f}, 85%, 45%)"


def svg_ray_structure(N: int, filename: str, orientation: str = "horizontal",
                      width: int = 1400, height: int = 900, max_labels: int = 31) -> None:
    """
    Generate ray-structure SVG visualization.
    
    Args:
        N: Maximum integer
        filename: Output SVG path
        orientation: "horizontal" (default) or "vertical"
        width, height: SVG dimensions
        max_labels: How many integers to label (default 31)
    """
    X, Y = 100.0, 500.0
    
    # Compute coordinates
    coords = {}
    max_shell = 0
    for n in range(1, N + 1):
        coords[n] = ray_coord(n, X, Y)
        shell = n.bit_length() - 1
        max_shell = max(max_shell, shell)
    
    # For horizontal orientation, swap x and y
    if orientation == "horizontal":
        coords = {n: (y, x) for n, (x, y) in coords.items()}
    
    # Get bounds
    xs = [x for x, y in coords.values()]
    ys = [y for x, y in coords.values()]
    min_x, max_x = min(xs) - 30, max(xs) + 30
    min_y, max_y = min(ys) - 30, max(ys) + 30
    
    # Scale
    scale_x = (width - 100) / (max_x - min_x) if max_x > min_x else 1
    scale_y = (height - 100) / (max_y - min_y) if max_y > min_y else 1
    scale = min(scale_x, scale_y)
    
    def to_svg(x: float, y: float) -> Tuple[float, float]:
        sx = 50 + (x - min_x) * scale
        sy = 50 + (max_y - y) * scale
        return (sx, sy)
    
    def dot_radius(n: int, is_prime_n: bool) -> float:
        if n == 1:
            return 5.0
        shell = n.bit_length() - 1
        if max_shell <= 1:
            base = 3.0
        else:
            t = (shell - 1) / (max_shell - 1)
            base = 3.0 * (1 - t) + 0.5 * t
        return base * 1.4 if is_prime_n else base
    
    # Get primes
    primes = sorted([n for n in range(2, N + 1) if is_prime(n)])
    num_primes = len(primes)
    prime_set = set(primes)
    
    # Group by odd core
    rays: Dict[int, List[int]] = {}
    for n in range(1, N + 1):
        core, _ = odd_core_and_v2(n)
        if core not in rays:
            rays[core] = []
        rays[core].append(n)
    
    # Prime cores for rainbow
    prime_cores = sorted([c for c in rays.keys() if is_prime(c)])
    num_prime_cores = len(prime_cores)
    prime_core_rainbow = {c: i for i, c in enumerate(prime_cores)}
    
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    
    # Title
    orient_label = "Horizontal" if orientation == "horizontal" else "Vertical"
    parts.append(f'<text x="{width//2}" y="25" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="16" font-weight="bold">'
                 f'Ray Structure 1-{N} ({orient_label})</text>')
    parts.append(f'<text x="{width//2}" y="45" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="11" fill="#666">'
                 f'{num_primes} primes • Rainbow = prime-core rays • Black = composite-core rays</text>')
    
    # Draw rays
    parts.append('<g fill="none">')
    for core in sorted(rays.keys()):
        members = sorted(rays[core])
        if len(members) < 2:
            continue
        
        if core in prime_core_rainbow:
            idx = prime_core_rainbow[core]
            color = rainbow_color(idx, num_prime_cores)
            stroke_width = 1.5
            opacity = 0.85
        elif core == 1:
            color = "#333333"
            stroke_width = 1.0
            opacity = 0.5
        else:
            color = "#000000"
            stroke_width = 0.5
            opacity = 0.2
        
        points = [to_svg(*coords[n]) for n in members]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{d}" stroke="{color}" '
                     f'stroke-width="{stroke_width}" stroke-opacity="{opacity}"/>')
    parts.append('</g>')
    
    # Draw points
    parts.append('<g>')
    for n in range(1, N + 1):
        sx, sy = to_svg(*coords[n])
        is_prime_n = n in prime_set
        r = dot_radius(n, is_prime_n)
        
        if is_prime_n:
            fill, stroke = "#dd0000", "#990000"
        elif n == 1:
            fill, stroke = "#000000", "#000000"
        else:
            fill, stroke = "#777777", "#555555"
        
        stroke_w = max(0.1, r * 0.12)
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r:.2f}" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w:.2f}"/>')
    parts.append('</g>')
    
    # Labels
    parts.append('<g font-family="sans-serif">')
    for n in range(1, min(N + 1, max_labels + 1)):
        sx, sy = to_svg(*coords[n])
        is_prime_n = n in prime_set
        r = dot_radius(n, is_prime_n)
        
        if is_prime_n:
            fill, font_size, weight = "#cc0000", 10, "bold"
        elif n == 1:
            fill, font_size, weight = "#000000", 10, "bold"
        else:
            fill, font_size, weight = "#444444", 8, "normal"
        
        offset = r + 3
        
        if orientation == "horizontal":
            # Labels below/above for horizontal layout
            parts.append(f'<text x="{sx:.1f}" y="{sy+offset+8:.1f}" text-anchor="middle" '
                         f'fill="{fill}" font-size="{font_size}" font-weight="{weight}">{n}</text>')
        else:
            # Labels to the right for vertical layout
            parts.append(f'<text x="{sx+offset:.1f}" y="{sy+3:.1f}" '
                         f'fill="{fill}" font-size="{font_size}" font-weight="{weight}">{n}</text>')
    parts.append('</g>')
    
    # Legend
    ly = height - 35
    parts.append(f'<g font-family="sans-serif" font-size="10">')
    for i in range(7):
        c = rainbow_color(i, 7)
        parts.append(f'<line x1="{50 + i*12}" y1="{ly}" x2="{58 + i*12}" y2="{ly}" '
                     f'stroke="{c}" stroke-width="2"/>')
    parts.append(f'<text x="140" y="{ly+4}">Prime-core rays</text>')
    parts.append(f'<line x1="260" y1="{ly}" x2="300" y2="{ly}" stroke="#000000" stroke-width="0.8" stroke-opacity="0.3"/>')
    parts.append(f'<text x="305" y="{ly+4}">Composite-core rays</text>')
    parts.append(f'<circle cx="440" cy="{ly}" r="3" fill="#dd0000"/>')
    parts.append(f'<text x="448" y="{ly+4}">Prime</text>')
    parts.append(f'<circle cx="500" cy="{ly}" r="1.5" fill="#777777"/>')
    parts.append(f'<text x="508" y="{ly+4}">Composite</text>')
    parts.append('</g>')
    
    parts.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(parts))




# ------------------ Coordinate Analysis ------------------ #

def encode_edge_offset(x: int, y: int, shell: int) -> Tuple[int, str, int, int]:
    """
    Encode (x, y) coordinate as edge + offset.
    
    For shell k, R = 2^k. Returns:
      (encoded_value, edge_name, edge_bits, offset)
    
    Edge encoding (2 bits): TOP=00, RIGHT=01, BOTTOM=10, LEFT=11
    Offset: position along edge (0 to 2R-1)
    Total bits: 2 + (k+1) bits
    """
    if shell == 0:
        return (0, "ORIGIN", 0, 0)
    
    R = 1 << shell
    
    if y == R:  # TOP edge
        edge_bits = 0b00
        edge_name = "TOP"
        offset = x + R
    elif x == R:  # RIGHT edge
        edge_bits = 0b01
        edge_name = "RIGHT"
        offset = R - y
    elif y == -R:  # BOTTOM edge
        edge_bits = 0b10
        edge_name = "BOTTOM"
        offset = R - x
    else:  # LEFT edge (x == -R)
        edge_bits = 0b11
        edge_name = "LEFT"
        offset = y + R
    
    # Encode: edge (2 bits) | offset (k+1 bits for range 0 to 2R-1)
    offset_bits = shell + 1
    encoded = (edge_bits << offset_bits) | offset
    
    return (encoded, edge_name, edge_bits, offset)


def signed_binary(val: int, width: int) -> str:
    """Format integer as signed binary string."""
    if val == 0:
        return "0".zfill(width)
    elif val > 0:
        return f"+{val:0{width-1}b}"
    else:
        return f"-{abs(val):0{width-1}b}"


def ants_coordinate_analysis(N: int, csv_path: Optional[str] = None, 
                              coordinate_format: str = "both",
                              ray_filter: Optional[int] = None,
                              primes_only: bool = False,
                              sort_by_theta: bool = False) -> None:
    """
    [ANALYSIS] Generate coordinate analysis showing binary patterns and ray relationships.
    
    Columns:
      n, core, v2, shell, is_prime, ray_pos,
      binary, octal, binary_rev,
      theta_key, key_bin (angular position - bitwise, no trig)
      x, y, x_bin, y_bin,
      edge, offset, edge_encoded,
      next_on_ray, delta_bits
    
    The key insight: numbers on the same odd-core ray have coordinates
    that differ by exactly one bit shift: C(2n) = 2 * C(n)
    """
    exp = N.bit_length()
    key_bits = exp - 1  # theta_key width
    
    print(f"[+] Running coordinate analysis for N={N}")
    if ray_filter:
        print(f"    Filtering to ray (odd core) = {ray_filter}")
    if primes_only:
        print(f"    Showing primes only")
    if sort_by_theta:
        print(f"    Sorting by theta_key (angular position)")
    
    max_shell = N.bit_length() - 1
    max_bits = N.bit_length()
    coord_bits = max_shell + 2  # enough bits for largest coordinate
    
    # Build rows
    rows = []
    
    # Group by rays for delta analysis
    rays: Dict[int, List[int]] = {}
    for n in range(1, N + 1):
        core, v2 = odd_core_and_v2(n)
        rays.setdefault(core, []).append(n)
    
    for n in range(1, N + 1):
        core, v2 = odd_core_and_v2(n)
        
        # Apply filters
        if ray_filter is not None and core != ray_filter:
            continue
        if primes_only and not is_prime(n):
            continue
        
        shell = 0 if n == 1 else n.bit_length() - 1
        prime_flag = is_prime(n)
        
        # Position within ray (0 = core itself, 1 = core*2, etc.)
        ray_pos = v2
        
        # Binary representations
        bin_str = format(n, f"0{max_bits}b")
        oct_str = format(n, f"o")
        bin_rev = bin_str[::-1]
        
        # Theta key (angular position) - pure bitwise, no trig!
        theta_key = odd_core_to_key(core, exp)
        key_bin = format(theta_key, f'0{key_bits}b')
        
        # Coordinates
        x, y = coord_int(n)
        x_bin = signed_binary(x, coord_bits)
        y_bin = signed_binary(y, coord_bits)
        
        # Edge encoding
        encoded, edge_name, edge_bits, offset = encode_edge_offset(x, y, shell)
        edge_enc_str = format(encoded, f"0{shell+3}b") if shell > 0 else "0"
        
        # Next on ray and delta
        next_n = n << 1  # 2n
        if next_n <= N:
            next_on_ray = next_n
            # Delta in coordinates is exactly 2x (one bit shift)
            delta_bits = "<<1"
        else:
            next_on_ray = None
            delta_bits = "-"
        
        row = {
            'n': n,
            'core': core,
            'v2': v2,
            'shell': shell,
            'is_prime': int(prime_flag),
            'ray_pos': ray_pos,
            'binary': bin_str,
            'octal': oct_str,
            'binary_rev': bin_rev,
            'theta_key': theta_key,
            'key_bin': key_bin,
            'x': x,
            'y': y,
            'x_bin': x_bin,
            'y_bin': y_bin,
            'edge': edge_name,
            'offset': offset,
            'edge_encoded': edge_enc_str,
            'next_on_ray': next_on_ray if next_on_ray else "",
            'delta_bits': delta_bits
        }
        rows.append(row)
    
    # Select columns based on coordinate format
    if coordinate_format == "xy":
        columns = ['n', 'core', 'v2', 'shell', 'is_prime', 'ray_pos',
                   'binary', 'octal', 'binary_rev',
                   'theta_key', 'key_bin',
                   'x', 'y', 'x_bin', 'y_bin',
                   'next_on_ray', 'delta_bits']
    elif coordinate_format == "edge":
        columns = ['n', 'core', 'v2', 'shell', 'is_prime', 'ray_pos',
                   'binary', 'octal', 'binary_rev',
                   'theta_key', 'key_bin',
                   'edge', 'offset', 'edge_encoded',
                   'next_on_ray', 'delta_bits']
    else:  # both
        columns = ['n', 'core', 'v2', 'shell', 'is_prime', 'ray_pos',
                   'binary', 'octal', 'binary_rev',
                   'theta_key', 'key_bin',
                   'x', 'y', 'x_bin', 'y_bin',
                   'edge', 'offset', 'edge_encoded',
                   'next_on_ray', 'delta_bits']
    
    # Sort by theta if requested
    if sort_by_theta:
        rows.sort(key=lambda r: r['theta_key'])
        print(f"[+] Sorted {len(rows)} rows by theta_key")
    
    # Output
    if csv_path:
        print(f"[+] Writing to {csv_path}")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f"[+] Wrote {len(rows)} rows")
    else:
        # Print to stdout in nice format
        print(f"\n{'='*100}")
        print(f"ANT COORDINATE ANALYSIS (N={N})")
        print(f"{'='*100}")
        
        # Header
        print(f"{'n':>5} {'core':>5} {'v2':>3} {'P':>1} {'binary':>{max_bits}} {'key_bin':>{key_bits}} ", end="")
        if coordinate_format in ("xy", "both"):
            print(f"{'(x,y)':>12} {'x_bin':>{coord_bits+1}} {'y_bin':>{coord_bits+1}} ", end="")
        if coordinate_format in ("edge", "both"):
            print(f"{'edge':>8} {'off':>4} ", end="")
        print(f"{'next':>6} {'Δ':>4}")
        print("-" * 110)
        
        for row in rows[:64]:  # Limit stdout to first 64 rows
            print(f"{row['n']:>5} {row['core']:>5} {row['v2']:>3} {row['is_prime']:>1} {row['binary']:>{max_bits}} {row['key_bin']:>{key_bits}} ", end="")
            if coordinate_format in ("xy", "both"):
                coord_str = f"({row['x']},{row['y']})"
                print(f"{coord_str:>12} {row['x_bin']:>{coord_bits+1}} {row['y_bin']:>{coord_bits+1}} ", end="")
            if coordinate_format in ("edge", "both"):
                print(f"{row['edge']:>8} {row['offset']:>4} ", end="")
            print(f"{str(row['next_on_ray']):>6} {row['delta_bits']:>4}")
        
        if len(rows) > 64:
            print(f"... ({len(rows) - 64} more rows, use --csv to export all)")
        
        # Show ray comparison summary
        print(f"\n{'='*100}")
        print("RAY COMPARISON: Prime vs Non-Prime on same ray")
        print(f"{'='*100}")
        
        # Pick a few interesting rays
        interesting_cores = [c for c in sorted(rays.keys()) if len(rays[c]) >= 3 and c <= 15][:5]
        
        for core in interesting_cores:
            ray_nums = sorted(rays[core])
            primes_on_ray = [n for n in ray_nums if is_prime(n)]
            composites = [n for n in ray_nums if not is_prime(n) and n > 1]
            
            print(f"\nRay {core}: {ray_nums}")
            print(f"  Primes: {primes_on_ray if primes_on_ray else 'none'}")
            print(f"  Composites: {composites if composites else 'none'}")
            
            # Show bit shift pattern
            if len(ray_nums) >= 2:
                print(f"  Bit pattern (C(2n) = 2*C(n)):")
                for n in ray_nums[:4]:
                    x, y = coord_int(n)
                    print(f"    n={n:>3}: ({x:>4},{y:>4}) = ({signed_binary(x, 6)}, {signed_binary(y, 6)})")


# ------------------ PRIME THETA DENSITY ANALYSIS ------------------ #
# 
# GOAL: Analyze prime distribution in THETA ORDER (angular order around
#       the square perimeter) - NOT in natural order (1,2,3,4...).
#
# WHY THIS MATTERS:
#   Standard prime gap analysis uses natural order: 2,3,5,7,11,13...
#   measuring gaps like 5-3=2, 11-7=4, etc.
#   
#   In theta order, we ask: when primes are sorted by
#   their angular position (theta_key), what do "gaps" look like?
#   This is a DIFFERENT question - changing the prism through which
#   we view primes.
#
# KEY PROPERTY (from construction):
#   Each prime defines EXACTLY ONE odd-core ray.
#   All other numbers on that ray (prime*2, prime*4, prime*8...) are composite.
#   So sorting primes by theta = sorting odd-core rays by angular position.
#
# VALID METRICS (all CUDA-compatible, bitwise/integer only):
#   ┌──────────────────┬────────────────────────────────────┬─────────────┐
#   │ Metric           │ Formula                            │ CUDA Cost   │
#   ├──────────────────┼────────────────────────────────────┼─────────────┤
#   │ theta_key        │ bitwise key from odd core          │ O(1) __clz  │
#   │ delta_theta_key  │ key2 - key1 (angular gap)          │ O(1) sub    │
#   │ hamming_primes   │ popcount(p1 XOR p2)                │ O(1) __popc │
#   │ hamming_keys     │ popcount(key1 XOR key2)            │ O(1) __popc │
#   └──────────────────┴────────────────────────────────────┴─────────────┘
#
# INVALID METRICS (do NOT use for theta-order analysis):
#   - delta_prime (p2 - p1): Natural-order thinking, meaningless in theta order
#   - theta_deg: requires atan2() - violates integer-native rule
#   - any floating point computation
#
# INTERPRETATION:
#   Results are visual/structural artifacts of this embedding.
#   No claims about number theory or prime structure.
#   We're just looking at primes through a different lens.
# ------------------ 

def prime_theta_density_analysis(N: int, csv_path: Optional[str] = None) -> None:
    """
    Analyze prime density in THETA ORDER (angular order).
    
    For consecutive primes p1, p2 in theta order (NOT natural order),
    computes bitwise/integer metrics suitable for CUDA implementation.
    
    Output columns:
      rank            - position in theta-sorted prime list (1-indexed)
      prime           - the prime number
      prime_bin       - binary representation of prime
      theta_key       - angular position as integer key (see odd_core_to_key docs)
      key_bin         - binary of theta_key (top 2 bits = edge: 00=TOP,01=RIGHT,10=BOTTOM,11=LEFT)
      delta_theta_key - gap to next prime in key units (angular gap)
      hamming_primes  - popcount(this_prime XOR next_prime)
      hamming_keys    - popcount(this_key XOR next_key)
    """
    exp = N.bit_length()
    num_keys = 1 << (exp - 1)
    key_bits = exp - 1  # theta_key is (exp-1) bits
    
    print(f"[+] Prime theta density analysis for N={N} (exp={exp})")
    print(f"    Total keys (angular positions): {num_keys:,}")
    print(f"    Key format: {key_bits} bits (top 2 bits = edge)")
    
    # Step 1: Collect all primes and their theta_keys
    # Using theta_key ordering means we get primes in angular order
    prime_data = []  # list of (theta_key, prime)
    
    print(f"    Collecting primes...", end='', flush=True)
    for n in range(2, N + 1):
        if is_prime(n):
            # Prime n has odd_core = n (since primes > 2 are odd, and 2 is special)
            # For prime p, the ray is labeled by p itself
            core, v2 = odd_core_and_v2(n)
            key = odd_core_to_key(core, exp)
            prime_data.append((key, n))
    
    # Sort by theta_key (angular order)
    prime_data.sort(key=lambda x: x[0])
    
    num_primes = len(prime_data)
    print(f" found {num_primes:,} primes")
    
    # Step 2: Compute metrics for consecutive prime pairs
    rows = []
    max_bits = N.bit_length()
    
    print(f"    Computing gaps between consecutive primes in theta order...")
    
    for i in range(num_primes):
        key_i, prime_i = prime_data[i]
        prime_bin = format(prime_i, f'0{max_bits}b')
        key_bin = format(key_i, f'0{key_bits}b')
        
        # For all but last, compute gap to next
        if i < num_primes - 1:
            key_next, prime_next = prime_data[i + 1]
            delta_theta_key = key_next - key_i
            hamming_primes = popcount(prime_i ^ prime_next)
            hamming_keys = popcount(key_i ^ key_next)
        else:
            # Last prime - wrap around to first
            key_next, prime_next = prime_data[0]
            delta_theta_key = (num_keys - key_i) + key_next  # wrap around
            hamming_primes = popcount(prime_i ^ prime_next)
            hamming_keys = popcount(key_i ^ key_next)
        
        row = {
            'rank': i + 1,
            'prime': prime_i,
            'prime_bin': prime_bin,
            'theta_key': key_i,
            'key_bin': key_bin,
            'delta_theta_key': delta_theta_key,
            'hamming_primes': hamming_primes,
            'hamming_keys': hamming_keys
        }
        rows.append(row)
    
    # Step 3: Output
    columns = ['rank', 'prime', 'prime_bin', 'theta_key', 'key_bin', 'delta_theta_key', 
               'hamming_primes', 'hamming_keys']
    
    if csv_path:
        print(f"[+] Writing to {csv_path}")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[+] Wrote {len(rows)} rows")
    
    # Step 4: Summary statistics
    delta_theta_keys = [r['delta_theta_key'] for r in rows[:-1]]  # exclude wrap-around
    hamming_p = [r['hamming_primes'] for r in rows[:-1]]
    hamming_k = [r['hamming_keys'] for r in rows[:-1]]
    
    print(f"\n{'='*70}")
    print(f"PRIME THETA DENSITY SUMMARY (N={N}, {num_primes:,} primes)")
    print(f"{'='*70}")
    
    print(f"\nAngular gap (delta_theta_key) statistics:")
    print(f"  Min:    {min(delta_theta_keys):,}")
    print(f"  Max:    {max(delta_theta_keys):,}")
    print(f"  Mean:   {sum(delta_theta_keys)/len(delta_theta_keys):.2f}")
    print(f"  Expected (uniform): {num_keys / num_primes:.2f}")
    
    print(f"\nHamming distance (primes) statistics:")
    print(f"  Min:    {min(hamming_p)}")
    print(f"  Max:    {max(hamming_p)}")
    print(f"  Mean:   {sum(hamming_p)/len(hamming_p):.2f}")
    
    print(f"\nHamming distance (keys) statistics:")
    print(f"  Min:    {min(hamming_k)}")
    print(f"  Max:    {max(hamming_k)}")
    print(f"  Mean:   {sum(hamming_k)/len(hamming_k):.2f}")
    
    # Show largest gaps (potential "theta deserts")
    print(f"\nLargest angular gaps (theta deserts):")
    sorted_by_gap = sorted(enumerate(rows[:-1]), key=lambda x: x[1]['delta_theta_key'], reverse=True)
    for idx, row in sorted_by_gap[:5]:
        next_row = rows[idx + 1]
        print(f"  gap={row['delta_theta_key']:>6}  key:{row['key_bin']} → {next_row['key_bin']}")
    
    # Show smallest gaps (clustering)
    print(f"\nSmallest angular gaps (theta clustering):")
    sorted_by_gap_asc = sorted(enumerate(rows[:-1]), key=lambda x: x[1]['delta_theta_key'])
    for idx, row in sorted_by_gap_asc[:5]:
        next_row = rows[idx + 1]
        print(f"  gap={row['delta_theta_key']:>6}  key:{row['key_bin']} → {next_row['key_bin']}")
    
    if not csv_path:
        # Print first 20 rows to stdout
        print(f"\nFirst 20 primes in theta order:")
        print(f"{'rank':>5} {'prime':>8} {'prime_bin':<{max_bits}} {'key_bin':<{key_bits}} {'Δkey':>6} {'H(p)':>5} {'H(k)':>5}")
        print("-" * (45 + max_bits + key_bits))
        for row in rows[:20]:
            print(f"{row['rank']:>5} {row['prime']:>8,} {row['prime_bin']:<{max_bits}} "
                  f"{row['key_bin']:<{key_bits}} {row['delta_theta_key']:>6} "
                  f"{row['hamming_primes']:>5} {row['hamming_keys']:>5}")


# ------------------ CLI ------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Power-of-two square rays v{__version__}: visualization and analysis",
        epilog=f"Author: {__author__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # small-map
    p_small = subparsers.add_parser("small-map", help="Generate small labeled SVG map")
    p_small.add_argument("--exp", type=int, default=8,
                         help="Use N = 2^exp (default: 8 → N=256)")
    p_small.add_argument("--with-rays", action="store_true",
                         help="Draw odd-core rays in background")
    p_small.add_argument("--no-rays", action="store_true",
                         help="Exclude rays (use with --ants-path)")
    p_small.add_argument("--ants-path", action="store_true",
                         help="Draw paths from origin to each number (rainbow colored, "
                              "cardinal moves only: UP/DOWN/LEFT/RIGHT).")

    # odd-core-rays (keep old name for compatibility)
    p_rays = subparsers.add_parser("prime-rays", help="Generate odd-core rays SVG (legacy name)")
    p_rays.add_argument("--exp", type=int, default=12,
                        help="Use N = 2^exp (default: 12 → N=4096)")

    # Also add new name as alias
    p_rays2 = subparsers.add_parser("odd-core-rays", help="Generate odd-core rays SVG")
    p_rays2.add_argument("--exp", type=int, default=12,
                        help="Use N = 2^exp (default: 12 → N=4096)")

    # ray-structure (alternative view)
    p_rs = subparsers.add_parser("ray-structure",
        help="Ray structure visualization (alternative view, horizontal or vertical)")
    p_rs.add_argument("--exp", type=int, default=9,
                      help="Use N = 2^exp (default: 9 → N=512)")
    p_rs.add_argument("--orientation", type=str, default="horizontal",
                      choices=["horizontal", "vertical"],
                      help="Layout orientation: horizontal (default) or vertical")
    p_rs.add_argument("--width", type=int, default=1400,
                      help="SVG width (default: 1400)")
    p_rs.add_argument("--height", type=int, default=900,
                      help="SVG height (default: 900)")
    p_rs.add_argument("--max-labels", type=int, default=31,
                      help="Maximum number of labels to show (default: 31)")

    # coord-analysis (keep old name for compatibility)
    p_ants = subparsers.add_parser("ants-analysis", 
        help="Coordinate analysis with binary patterns (legacy name)")
    p_ants.add_argument("--exp", type=int, default=8,
                        help="Use N = 2^exp (default: 8 → N=256)")
    p_ants.add_argument("--csv", type=str, default=None,
                        help="CSV output path (default: prints to stdout)")
    p_ants.add_argument("--coordinate", type=str, default="xy",
                        choices=["xy", "edge", "both"],
                        help="Coordinate format: xy=(x,y), edge=edge+offset encoding, both=all formats")
    p_ants.add_argument("--ray", type=int, default=None,
                        help="Filter to show only specific ray (odd core)")
    p_ants.add_argument("--primes-only", action="store_true",
                        help="Show only prime numbers")
    p_ants.add_argument("--sort-by-theta", action="store_true",
                        help="Sort rows by theta_key (angular position) instead of n")

    # Also add new name as alias
    p_coords = subparsers.add_parser("coord-analysis", 
        help="Coordinate analysis with binary patterns and ray comparisons")
    p_coords.add_argument("--exp", type=int, default=8,
                        help="Use N = 2^exp (default: 8 → N=256)")
    p_coords.add_argument("--csv", type=str, default=None,
                        help="CSV output path (default: prints to stdout)")
    p_coords.add_argument("--coordinate", type=str, default="xy",
                        choices=["xy", "edge", "both"],
                        help="Coordinate format: xy=(x,y), edge=edge+offset encoding, both=all formats")
    p_coords.add_argument("--ray", type=int, default=None,
                        help="Filter to show only specific ray (odd core)")
    p_coords.add_argument("--primes-only", action="store_true",
                        help="Show only prime numbers")
    p_coords.add_argument("--sort-by-theta", action="store_true",
                        help="Sort rows by theta_key (angular position) instead of n")

    # prime-theta-density (CORE ANALYSIS)
    p_ptd = subparsers.add_parser("prime-theta-density", 
        help="Prime density analysis in THETA ORDER (angular order, not natural order)")
    p_ptd.add_argument("--exp", type=int, default=12,
                       help="Use N = 2^exp (default: 12 → N=4096)")
    p_ptd.add_argument("--csv", type=str, default=None,
                       help="CSV output path (default: prints summary to stdout)")

    args = parser.parse_args()
    outdir = ensure_svgs_dir()

    if args.command == "small-map":
        N = 2 ** args.exp
        
        # Determine visualization mode
        with_rays = args.with_rays and not args.no_rays
        with_ants_path = args.ants_path
        
        # Generate filename based on mode
        if with_ants_path:
            suffix = "_paths"
        elif with_rays:
            suffix = "_with_rays"
        else:
            suffix = ""
        
        outfile = os.path.join(outdir, f"map_{N}_labeled{suffix}.svg")
        
        mode_desc = ""
        if with_ants_path:
            mode_desc = " with paths"
        elif with_rays:
            mode_desc = " with rays"
            
        print(f"[+] Generating small labeled map for N={N}{mode_desc} → {outfile}")
        svg_small_map(N, outfile, with_rays=with_rays, with_ants_path=with_ants_path)

    elif args.command in ("prime-rays", "odd-core-rays"):
        N = 2 ** args.exp
        outfile = os.path.join(outdir, f"odd_core_rays_{N}.svg")
        print(f"[+] Generating odd-core rays for N={N} → {outfile}")
        svg_prime_rays(N, outfile)

    elif args.command == "ray-structure":
        N = 2 ** args.exp
        orient = args.orientation
        outfile = os.path.join(outdir, f"ray_structure_{N}_{orient}.svg")
        print(f"[+] Generating ray structure ({orient}) for N={N} → {outfile}")
        svg_ray_structure(N, outfile, orientation=orient,
                          width=args.width, height=args.height,
                          max_labels=args.max_labels)

    elif args.command in ("ants-analysis", "coord-analysis"):
        N = 2 ** args.exp
        ants_coordinate_analysis(
            N, 
            csv_path=args.csv, 
            coordinate_format=args.coordinate,
            ray_filter=args.ray,
            primes_only=args.primes_only,
            sort_by_theta=args.sort_by_theta
        )

    elif args.command == "prime-theta-density":
        N = 2 ** args.exp
        prime_theta_density_analysis(N, csv_path=args.csv)


if __name__ == "__main__":
    main()
