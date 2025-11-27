#!/usr/bin/env python3
disclaimer = """

> **Warning / Disclaimer**  
> **Disclaimer**  
>This is an AI-assisted exploratory visualization project.
>It is educational and experimental only—not peer-reviewed, not mathematical research, and not a new theorem.
>It does not serve as evidence for or against conjectures.
>This repository is math-art/visualization, not a research paper.
>Its purpose is educational exploration and aesthetic interest.
>Any visual patterns are artifacts of the construction.

"""

print(disclaimer)

"""
demo3d.py

3D Extension of Power-of-Two Square Rays: Cube Shell Mapping

This module extends the 2D square-shell construction to 3D cube-shells,
mapping natural numbers to points on the surface of cubes in the L∞ norm.

INTEGER-NATIVE 3D MODEL
-----------------------
In the 3D extension, an agent can move in six cardinal directions:
    UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD

The same constraint applies: all core computations use only integers,
bitwise operations, and comparisons. No trigonometry in the coordinate
system itself.

The decomposition n = 2^v₂(n) × core(n) still applies:
    - v₂(n) determines the shell (which cube surface)
    - core(n) determines the position on that shell

Key Properties:
    - Scaling: C₃(2n) = 2·C₃(n) for all n ≥ 2
    - Shell structure: n ∈ [2^k, 2^(k+1)) maps to cube of radius 2^k
    - Parameter preservation: t(2n) = t(n), enabling exact scaling

Potential Applications (exploratory):
    - Integer-to-direction encoding for 3D space
    - Approximate spherical coordinates from integers
    - Multi-resolution spatial hashing with hierarchical structure
    - Direction quantization in machine learning

Resolution Estimates (order-of-magnitude, equal-area assumption):
    For exp=24 (N = 2²⁴ ≈ 16.7M points):
        - Linear resolution on Earth: ~5-6 km (neighbourhood-level)

    For exp=32 (N = 2³² ≈ 4.3B points):
        - Linear resolution on Earth: ~350 m (block-level)

    Note: Actual resolution varies due to cube-to-sphere non-uniformity.

Limitations and Trade-offs:
    - Non-uniform distribution on sphere (cube projection distortion, ~2× variation)
    - Corners/edges have different density than face centers
    - NOT equal-area like HEALPix or S2 geometry
    - For strict equal-area or high-precision geodesy, use HEALPix/S2 instead
    
    BUT: Maintains exact dyadic scaling property + simple integer hierarchy,
    which may be valuable for certain multi-resolution or hierarchical applications.

Usage:
    python3 demo3d.py                    # Run all demos and tests
    python3 demo3d.py --plot             # Show 3D visualization
    python3 demo3d.py --test             # Run validation tests
    python3 demo3d.py --resolution       # Analyze resolution for various exp

Author: Nenad Micic
Repository: https://github.com/nmicic/power-two-square-rays
"""

__version__ = "1.1.0"
__author__ = "Nenad Micic"

import math
import argparse
from typing import Tuple, List, Optional


# ============================================================
# CORE MAPPING FUNCTIONS
# ============================================================

def cube_surface_point(t: float, R: float = 1.0) -> Tuple[float, float, float]:
    """
    Map parameter t ∈ [0, 1) to a point on the surface of an L∞ cube.
    
    The cube has "radius" R, meaning max(|x|, |y|, |z|) = R.
    The cube surface area is 6 × (2R)² = 24R².
    
    Parameterization Strategy:
        We traverse the 6 faces of the cube in order:
        - t ∈ [0/6, 1/6): Top face    (+z = R)
        - t ∈ [1/6, 2/6): Front face  (+y = R)
        - t ∈ [2/6, 3/6): Right face  (+x = R)
        - t ∈ [3/6, 4/6): Back face   (-y = R)
        - t ∈ [4/6, 5/6): Left face   (-x = R)
        - t ∈ [5/6, 6/6): Bottom face (-z = R)
        
        Within each face, we use a simple (u, v) parameterization
        that sweeps the square from corner to corner.
    
    Args:
        t: Parameter in [0, 1) representing position on cube surface
        R: "Radius" of the cube (half edge length)
    
    Returns:
        (x, y, z) tuple on the cube surface with max(|x|, |y|, |z|) = R
    
    Example:
        >>> cube_surface_point(0.0, 1.0)   # Top face, corner
        (-1.0, -1.0, 1.0)
        >>> cube_surface_point(0.5, 1.0)   # Right face, somewhere
        (1.0, ...)
    """
    # Ensure t is in [0, 1)
    t = t % 1.0
    
    # Determine which face (0-5) and local parameter within face
    face_idx = int(t * 6)
    if face_idx >= 6:
        face_idx = 5  # Handle edge case t = 0.9999...
    
    # Local parameter within face: 0 to 1
    t_local = (t * 6) - face_idx
    
    # Convert t_local to (u, v) coordinates in [-1, 1] × [-1, 1]
    # Simple row-column scan for uniform coverage
    # We use a snake pattern for better continuity at face boundaries
    
    # Let's use a simple grid with N divisions and continuous interpolation
    N = 1024  # Virtual grid size for mapping
    idx = int(t_local * N * N)
    row = idx // N
    col = idx % N
    
    # Boustrophedon (snake) pattern for better continuity
    if row % 2 == 1:
        col = N - 1 - col
    
    # Map to [-1, 1] × [-1, 1]
    u = 2.0 * col / (N - 1) - 1.0 if N > 1 else 0.0
    v = 2.0 * row / (N - 1) - 1.0 if N > 1 else 0.0
    
    # Scale to [-R, R]
    u *= R
    v *= R
    
    # Assign coordinates based on face
    if face_idx == 0:    # Top face: z = +R
        x, y, z = u, v, R
    elif face_idx == 1:  # Front face: y = +R
        x, y, z = u, R, v
    elif face_idx == 2:  # Right face: x = +R
        x, y, z = R, u, v
    elif face_idx == 3:  # Back face: y = -R
        x, y, z = -u, -R, v
    elif face_idx == 4:  # Left face: x = -R
        x, y, z = -R, -u, v
    else:                # Bottom face: z = -R (face_idx == 5)
        x, y, z = u, -v, -R
    
    return (x, y, z)


def coord3d(n: int) -> Tuple[float, float, float]:
    """
    Map integer n to a point on the 3D cube shell structure.
    
    This is the 3D analogue of coord(n) in the 2D construction.
    
    Properties:
        - n = 1 → (0, 0, 0) (origin)
        - n ∈ [2^k, 2^(k+1)) → point on cube of radius 2^k
        - Scaling: coord3d(2n) = 2 · coord3d(n) for n ≥ 2
    
    The scaling property is preserved because:
        - Shell k contains n ∈ [2^k, 2^(k+1))
        - Parameter t = (n - 2^k) / 2^k
        - For 2n: shell is k+1, t = (2n - 2^(k+1)) / 2^(k+1) = (n - 2^k) / 2^k = same t!
        - So coord3d(2n) = P_{2R}(t) = 2 · P_R(t) = 2 · coord3d(n)
    
    Args:
        n: Positive integer ≥ 1
    
    Returns:
        (x, y, z) tuple on the cube shell
    
    Example:
        >>> coord3d(1)
        (0.0, 0.0, 0.0)
        >>> coord3d(2)  # Shell 1, first point
        (-1.0, -1.0, 1.0)
        >>> x, y, z = coord3d(100)
        >>> max(abs(x), abs(y), abs(z))  # Should be 2^6 = 64
        64.0
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    
    if n == 1:
        return (0.0, 0.0, 0.0)
    
    # Determine shell: k = floor(log2(n))
    k = n.bit_length() - 1
    R = float(2 ** k)
    
    # Compute parameter t ∈ [0, 1)
    base = 2 ** k
    t = (n - base) / base
    
    # Map to cube surface
    return cube_surface_point(t, R)


def coord3d_unit(n: int) -> Tuple[float, float, float]:
    """
    Map integer n to a unit vector (approximate direction on sphere).
    
    This normalizes coord3d(n) to unit length, projecting the cube
    surface onto the unit sphere. Useful for direction encoding.
    
    Note on Non-Uniformity:
        The projection from cube to sphere is NOT equal-area:
        - Face centers project to less area on sphere
        - Corners project to more area on sphere
        - This creates ~2x density variation
        
        For applications requiring equal-area, consider HEALPix.
        For applications requiring dyadic scaling, use this.
    
    Args:
        n: Positive integer ≥ 1
    
    Returns:
        (x, y, z) unit vector with x² + y² + z² ≈ 1
        For n=1, returns (0, 0, 0) as special case (no direction)
    
    Example:
        >>> x, y, z = coord3d_unit(12345)
        >>> abs(x*x + y*y + z*z - 1.0) < 1e-10
        True
    """
    x, y, z = coord3d(n)
    
    if n == 1:
        return (0.0, 0.0, 0.0)  # Origin has no direction
    
    # Normalize to unit length
    length = math.sqrt(x*x + y*y + z*z)
    if length < 1e-15:
        return (0.0, 0.0, 0.0)
    
    return (x / length, y / length, z / length)


def unit_to_spherical(x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Convert unit vector to spherical coordinates (azimuth, elevation).
    
    Args:
        x, y, z: Unit vector components
    
    Returns:
        (azimuth, elevation) in radians
        azimuth ∈ [-π, π]: angle from +x axis in xy-plane
        elevation ∈ [-π/2, π/2]: angle from xy-plane
    
    For astro navigation:
        azimuth → longitude or right ascension
        elevation → latitude or declination
    """
    # Elevation (latitude): angle from xy-plane
    elevation = math.asin(max(-1.0, min(1.0, z)))
    
    # Azimuth (longitude): angle in xy-plane from +x axis
    azimuth = math.atan2(y, x)
    
    return (azimuth, elevation)


def coord3d_spherical(n: int) -> Tuple[float, float]:
    """
    Map integer n directly to spherical coordinates.
    
    Args:
        n: Positive integer ≥ 1
    
    Returns:
        (azimuth, elevation) in radians
        Returns (0, 0) for n=1
    """
    if n == 1:
        return (0.0, 0.0)
    
    x, y, z = coord3d_unit(n)
    return unit_to_spherical(x, y, z)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def shell_3d(n: int) -> int:
    """Return the shell index for integer n."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return 0
    return n.bit_length() - 1


def shell_radius_3d(n: int) -> float:
    """Return the shell radius for integer n."""
    return float(2 ** shell_3d(n)) if n > 1 else 0.0


def odd_core_and_v2(n: int) -> Tuple[int, int]:
    """
    Compute the 2-adic factorization n = 2^v2 * core.
    
    Returns:
        (core, v2) where core is odd and n = 2^v2 * core
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    
    v2 = 0
    while n % 2 == 0:
        n //= 2
        v2 += 1
    
    return (n, v2)


# ============================================================
# GENERATION AND ANALYSIS FUNCTIONS
# ============================================================

def generate_points_3d(exp: int, 
                       max_points: Optional[int] = None,
                       include_spherical: bool = False) -> List[dict]:
    """
    Generate 3D points for n in [1, 2^exp].
    
    Args:
        exp: Generate points up to N = 2^exp
        max_points: Optional limit on number of points (random subsample)
        include_spherical: Also compute spherical coordinates
    
    Returns:
        List of dicts with keys: n, x, y, z, shell, [azimuth, elevation]
    """
    import random
    
    N = 2 ** exp
    
    if max_points is not None and max_points < N:
        # Random subsample
        indices = sorted(random.sample(range(1, N + 1), max_points))
    else:
        indices = range(1, N + 1)
    
    points = []
    for n in indices:
        x, y, z = coord3d(n)
        point = {
            'n': n,
            'x': x,
            'y': y,
            'z': z,
            'shell': shell_3d(n)
        }
        
        if include_spherical and n > 1:
            ux, uy, uz = coord3d_unit(n)
            az, el = unit_to_spherical(ux, uy, uz)
            point['azimuth'] = az
            point['elevation'] = el
        
        points.append(point)
    
    return points


def analyze_resolution(exp: int) -> dict:
    """
    Analyze the effective resolution for a given exponent.
    
    For direction encoding / sphere coverage, we estimate:
    - Number of distinct directions
    - Angular resolution (degrees between adjacent points)
    - Approximate linear resolution if used for Earth surface
    
    Args:
        exp: Exponent (N = 2^exp points)
    
    Returns:
        Dictionary with resolution metrics
    """
    N = 2 ** exp
    
    # Approximate angular resolution (assuming uniform distribution)
    # Total solid angle = 4π steradians
    # Solid angle per point ≈ 4π / N
    # Angular separation ≈ sqrt(4π / N) radians
    
    solid_angle_per_point = 4 * math.pi / N
    angular_resolution_rad = math.sqrt(solid_angle_per_point)
    angular_resolution_deg = math.degrees(angular_resolution_rad)
    
    # For Earth surface (radius ≈ 6371 km)
    earth_radius_km = 6371
    earth_surface_km2 = 4 * math.pi * earth_radius_km ** 2
    area_per_point_km2 = earth_surface_km2 / N
    linear_resolution_km = math.sqrt(area_per_point_km2)
    
    return {
        'exp': exp,
        'N': N,
        'solid_angle_per_point_sr': solid_angle_per_point,
        'angular_resolution_rad': angular_resolution_rad,
        'angular_resolution_deg': angular_resolution_deg,
        'area_per_point_km2': area_per_point_km2,
        'linear_resolution_km': linear_resolution_km,
        'linear_resolution_m': linear_resolution_km * 1000,
        'memory_for_lookup_MB': N * 12 / (1024 * 1024)  # 3 × 4-byte floats per point
    }


# ============================================================
# VISUALIZATION FUNCTIONS (require matplotlib)
# ============================================================

def plot_shell_3d(exp: int, sample: int = 5000, 
                  show_shells: bool = True,
                  save_path: Optional[str] = None):
    """
    Create a 3D scatter plot of the cube shell mapping.
    
    Args:
        exp: Exponent (visualize up to N = 2^exp)
        sample: Maximum number of points to plot (for performance)
        show_shells: Color-code by shell
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return
    
    print(f"Generating 3D plot for exp={exp} (up to {sample} points)...")
    
    points = generate_points_3d(exp, max_points=sample)
    
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    zs = [p['z'] for p in points]
    shells = [p['shell'] for p in points]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if show_shells:
        scatter = ax.scatter(xs, ys, zs, c=shells, cmap='viridis', 
                            s=1, alpha=0.6)
        plt.colorbar(scatter, label='Shell (log₂ scale)', shrink=0.6)
    else:
        ax.scatter(xs, ys, zs, s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Cube Shell Mapping (n = 1 to 2^{exp})\n'
                 f'Showing {len(points)} points')
    
    # Equal aspect ratio
    max_range = max(max(abs(min(xs)), abs(max(xs))),
                   max(abs(min(ys)), abs(max(ys))),
                   max(abs(min(zs)), abs(max(zs))))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_spherical_projection(exp: int, sample: int = 5000,
                              save_path: Optional[str] = None):
    """
    Create a scatter plot of the spherical projection.
    
    Args:
        exp: Exponent
        sample: Maximum points
        save_path: Optional save path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return
    
    print(f"Generating spherical projection for exp={exp}...")
    
    points = generate_points_3d(exp, max_points=sample, include_spherical=True)
    
    # Skip n=1 which has no direction
    points = [p for p in points if p['n'] > 1]
    
    azimuths = [math.degrees(p['azimuth']) for p in points]
    elevations = [math.degrees(p['elevation']) for p in points]
    shells = [p['shell'] for p in points]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    scatter = ax.scatter(azimuths, elevations, c=shells, cmap='viridis',
                        s=1, alpha=0.6)
    plt.colorbar(scatter, label='Shell (log₂ scale)')
    
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Elevation (degrees)')
    ax.set_title(f'Spherical Projection of Cube Shell Mapping\n'
                f'n = 2 to 2^{exp}, {len(points)} points')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


# ============================================================
# VALIDATION TESTS
# ============================================================

def test_scaling_property(max_n: int = 10000) -> bool:
    """
    Verify C₃(2n) = 2·C₃(n) for all n in range.
    
    This is the fundamental property that makes the construction
    compatible with the 2-adic hierarchy.
    
    Returns True if all tests pass.
    """
    print(f"Testing scaling property C₃(2n) = 2·C₃(n) for n up to {max_n}...")
    
    tolerance = 1e-10
    failures = 0
    
    for n in range(2, max_n + 1):
        x1, y1, z1 = coord3d(n)
        x2, y2, z2 = coord3d(2 * n)
        
        # Check if coord3d(2n) = 2 * coord3d(n)
        dx = abs(x2 - 2 * x1)
        dy = abs(y2 - 2 * y1)
        dz = abs(z2 - 2 * z1)
        
        if dx > tolerance or dy > tolerance or dz > tolerance:
            if failures < 5:
                print(f"  FAIL at n={n}:")
                print(f"    coord3d({2*n}) = ({x2}, {y2}, {z2})")
                print(f"    2 * coord3d({n}) = ({2*x1}, {2*y1}, {2*z1})")
            failures += 1
    
    if failures == 0:
        print(f"  ✓ All {max_n - 1} tests passed!")
        return True
    else:
        print(f"  ✗ {failures} failures out of {max_n - 1} tests")
        return False


def test_shell_radius(max_n: int = 10000) -> bool:
    """
    Verify that points lie on correct shell: max(|x|,|y|,|z|) = 2^k.
    
    Returns True if all tests pass.
    """
    print(f"Testing shell radius for n up to {max_n}...")
    
    tolerance = 1e-10
    failures = 0
    
    for n in range(2, max_n + 1):
        x, y, z = coord3d(n)
        expected_R = shell_radius_3d(n)
        actual_R = max(abs(x), abs(y), abs(z))
        
        if abs(actual_R - expected_R) > tolerance:
            if failures < 5:
                print(f"  FAIL at n={n}:")
                print(f"    Expected radius: {expected_R}")
                print(f"    Actual max|coord|: {actual_R}")
            failures += 1
    
    if failures == 0:
        print(f"  ✓ All {max_n - 1} tests passed!")
        return True
    else:
        print(f"  ✗ {failures} failures out of {max_n - 1} tests")
        return False


def test_unit_vectors(max_n: int = 1000) -> bool:
    """
    Verify that coord3d_unit returns unit vectors.
    """
    print(f"Testing unit vector normalization for n up to {max_n}...")
    
    tolerance = 1e-10
    failures = 0
    
    for n in range(2, max_n + 1):
        x, y, z = coord3d_unit(n)
        length_sq = x*x + y*y + z*z
        
        if abs(length_sq - 1.0) > tolerance:
            if failures < 5:
                print(f"  FAIL at n={n}:")
                print(f"    Length² = {length_sq} (expected 1.0)")
            failures += 1
    
    if failures == 0:
        print(f"  ✓ All {max_n - 1} tests passed!")
        return True
    else:
        print(f"  ✗ {failures} failures out of {max_n - 1} tests")
        return False


def run_all_tests(max_n: int = 10000) -> bool:
    """Run all validation tests."""
    print("=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)
    
    results = []
    results.append(("Scaling property", test_scaling_property(max_n)))
    results.append(("Shell radius", test_shell_radius(max_n)))
    results.append(("Unit vectors", test_unit_vectors(min(max_n, 1000))))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    return all_passed


# ============================================================
# DEMO FUNCTIONS
# ============================================================

def demo_basic():
    """Show basic usage examples."""
    print("=" * 60)
    print("BASIC DEMO: 3D Cube Shell Mapping")
    print("=" * 60)
    
    print("\nSample points:")
    for n in [1, 2, 3, 4, 8, 16, 100, 1000]:
        x, y, z = coord3d(n)
        shell = shell_3d(n)
        R = shell_radius_3d(n)
        print(f"  n={n:4d}: ({x:8.2f}, {y:8.2f}, {z:8.2f})  shell={shell}, R={R}")
    
    print("\nScaling property demonstration:")
    for n in [5, 17, 100]:
        x1, y1, z1 = coord3d(n)
        x2, y2, z2 = coord3d(2*n)
        print(f"  coord3d({n:3d})   = ({x1:8.3f}, {y1:8.3f}, {z1:8.3f})")
        print(f"  coord3d({2*n:3d}) = ({x2:8.3f}, {y2:8.3f}, {z2:8.3f})")
        print(f"  2 × coord3d({n:3d}) = ({2*x1:8.3f}, {2*y1:8.3f}, {2*z1:8.3f})")
        print(f"  Match: {abs(x2-2*x1) < 1e-10 and abs(y2-2*y1) < 1e-10 and abs(z2-2*z1) < 1e-10}")
        print()


def demo_spherical():
    """Demo spherical coordinate conversion."""
    print("=" * 60)
    print("SPHERICAL COORDINATES DEMO")
    print("=" * 60)
    
    print("\nInteger → Direction mapping:")
    for n in [2, 100, 1000, 10000, 100000]:
        ux, uy, uz = coord3d_unit(n)
        az, el = coord3d_spherical(n)
        print(f"  n={n:6d}: direction=({ux:+.4f}, {uy:+.4f}, {uz:+.4f})")
        print(f"           azimuth={math.degrees(az):+7.2f}°, elevation={math.degrees(el):+7.2f}°")
        print()


def demo_resolution():
    """Demo resolution analysis for different exponents."""
    print("=" * 70)
    print("RESOLUTION ANALYSIS")
    print("=" * 70)
    print("\nFor direction encoding / Earth surface approximation:")
    print("(Order-of-magnitude estimates using equal-area assumption)")
    print()
    
    print(f"{'Exp':>4} | {'N':>12} | {'Linear Scale':>14} | {'Qualitative':>18} | {'Memory':>10}")
    print(f"{'':>4} | {'':>12} | {'(approx)':>14} | {'':>18} | {'(MB)':>10}")
    print("-" * 75)
    
    qualitative = {
        12: "region-level",
        16: "large-city",
        20: "city-level",
        24: "neighbourhood",
        28: "district-level",
        32: "block-level"
    }
    
    for exp in [12, 16, 20, 24, 28, 32]:
        res = analyze_resolution(exp)
        if res['linear_resolution_km'] >= 1:
            scale_str = f"~{res['linear_resolution_km']:.0f} km"
        else:
            scale_str = f"~{res['linear_resolution_m']:.0f} m"
        qual = qualitative.get(exp, "")
        print(f"{exp:4d} | {res['N']:12,d} | {scale_str:>14} | {qual:>18} | {res['memory_for_lookup_MB']:10.1f}")
    
    print()
    print("Notes:")
    print("  • Linear scale = sqrt(Earth surface area / N)")
    print("  • Actual resolution varies due to cube-to-sphere non-uniformity (~2× variation)")
    print("  • For strict equal-area needs, consider HEALPix or S2 Geometry")
    print("  • This construction's advantage: exact dyadic scaling + integer hierarchy")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="3D Cube Shell Mapping - Extension of Power-of-Two Square Rays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 demo3d.py                    # Run all demos
  python3 demo3d.py --test             # Run validation tests
  python3 demo3d.py --plot --exp 12    # Show 3D visualization
  python3 demo3d.py --resolution       # Show resolution analysis
  python3 demo3d.py --spherical        # Demo spherical projection
        """
    )
    
    parser.add_argument('--test', action='store_true',
                        help='Run validation tests')
    parser.add_argument('--plot', action='store_true',
                        help='Show 3D visualization')
    parser.add_argument('--spherical', action='store_true',
                        help='Show spherical projection')
    parser.add_argument('--resolution', action='store_true',
                        help='Show resolution analysis')
    parser.add_argument('--exp', type=int, default=10,
                        help='Exponent for visualization (default: 10)')
    parser.add_argument('--sample', type=int, default=5000,
                        help='Number of sample points for visualization (default: 5000)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file')
    
    args = parser.parse_args()
    
    if args.test:
        run_all_tests()
    elif args.plot:
        plot_shell_3d(args.exp, sample=args.sample, save_path=args.save)
    elif args.spherical:
        plot_spherical_projection(args.exp, sample=args.sample, save_path=args.save)
    elif args.resolution:
        demo_resolution()
    else:
        # Run all demos
        demo_basic()
        print()
        demo_spherical()
        print()
        demo_resolution()
        print()
        print("Run with --test to validate, --plot to visualize")


if __name__ == "__main__":
    main()
