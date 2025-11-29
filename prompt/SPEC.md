SPEC.md

Comprehensive Technical Specification for Power-of-Two Square Rays

⸻

0. Overview

This document contains the full detail of every definition, construction step, and derived property.
SYSTEM.md is the strict contract; SPEC.md expands definitions for reference.

⸻

1. INTEGER STRUCTURE

1.1 2-Adic Decomposition

(Same as SYSTEM.md, with elaboration.)

Example:

n = 40
v2(40) = 3
core(40) = 40 >> 3 = 5
Shell = bit_length(40)-1 = 5


⸻

2. SHELLS

2.1 Definition

Shell k = all n with:

2^k ≤ n < 2^(k+1)

Contains exactly:
	•	2^k integers total
	•	2^(k-1) odd (new rays)
	•	2^(k-1) even (existing rays)

2.2 Properties
	•	Δx halves each shell
	•	coordinates rational
	•	doubling moves integer one shell outward

⸻

3. THETA SYSTEM (expanded)

3.1 Theta Position (theta_pos)

Theta positions subdivide angular directions.
At each shell:
	•	Previous directions persist
	•	New odd numbers insert new directions between these

Structure forms a complete binary tree.

3.2 Theta Key (bit reversal)

bit_reverse is standard FFT-style reversal.
The MSB/LSB pattern determines quadrant/edge.

Edge decoding (optional):
	•	top edge: leading bits “00”
	•	right edge: “01”
	•	bottom: “10”
	•	left: “11”

⸻

4. SQUARE-PERIMETER GEOMETRY

4.1 Perimeter Parameterization

Parameter t runs clockwise from north-west corner.

Quadrants:
	•	top edge: t ∈ [0, 0.25)
	•	right:    t ∈ [0.25, 0.5)
	•	bottom:   t ∈ [0.5, 0.75)
	•	left:     t ∈ [0.75, 1)

4.2 Exact coordinate formulas

Not repeated here — project code defines P_R(t).
All coordinates use integer divisions by powers of two.

4.3 Scaling

C(2n) = 2*C(n)

Direct result of shell doubling.

⸻

5. RAY GEOMETRY

5.1 Ray membership

Given odd core a, all ray members have binary form:

a, a0, a00, a000, ...

5.2 Slope Definition

The project defines slopes m_a for visualization:
	•	1 → slope -1
	•	3 → +1
	•	5 → 0
	•	7 → +4/3
Etc.

This is not derived mathematically; it is chosen.

⸻

6. ITERATION STRATEGIES

6.1 Theta Order Traversal

Two nested loops:
	•	outer: theta_pos
	•	inner: shell

This walks across directions, not magnitudes.

6.2 Verification Heuristic

To test whether theta reveals structure:
	•	compare theta-order against random permutations
	•	compare against natural order
	•	compare across shells

⸻

7. DERIVED FACTS (construction consequences)
	•	Shell k has x-position:
X(2 - 2^(1-k))
	•	Δx halves each shell
	•	Lower shells embed into higher
	•	Gaps shrink as shells refine
	•	All coordinate denominators are powers of 2

These are consequences of definitions, not discoveries.

⸻

End of SPEC.md
