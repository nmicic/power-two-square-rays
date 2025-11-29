SYSTEM.md

Authoritative Context Specification for Power-of-Two Square Rays
(Use in every new LLM session)

⸻

0. PURPOSE

This document defines the canonical definitions, invariants, constraints, and reasoning rules for the Power-of-Two Square Rays construction.
LLMs MUST adhere to this spec exactly.

This project is a construction-based geometric visualization, not mathematics research.
All structure arises from explicit definitions.

⸻

1. INTEGER FOUNDATIONS

1.1 2-Adic Decomposition (Primitive)

For integer n ≥ 1:

v2(n)   = number of trailing zero bits (ctz)
core(n) = n >> v2(n)      # always odd
n       = 2^v2(n) × core(n)

1.2 Shell

shell(n) = bit_length(n) - 1

Shell k contains integers [2^k, 2^(k+1)).

⸻

2. ANGULAR SYSTEM

2.1 Theta Position (shell-local)

Each shell k has:

theta_pos ∈ {0, ..., 2^(k-1)-1}

Invariant (MANDATORY):

Theta positions are local to each shell and MUST NOT be treated as global indices.

2.2 Theta Key (global angular identity)

For odd a with bit_length = k+1:

theta_key(a) = bit_reverse(a, k+1 bits)

Properties:
	•	theta_key is odd
	•	theta_key provides a global angular ordering of odd cores
	•	same core → same theta_key → same ray

2.3 Theta Structure Hierarchy
	•	theta_pos: local indexing within shell
	•	theta_key: global angular identity
	•	when shell increases, the number of theta positions doubles
	•	new odd numbers fill gaps between existing directions
	•	lower shell positions embed as parents/ancestors in a dyadic tree

⸻

3. RAYS

3.1 Definition

For odd core = a:

ray(a) = { a × 2^j : j ≥ 0 }

3.2 Ray Invariants
	•	core determines ray
	•	ray members share theta_key
	•	ray members reside in different shells
	•	rays are infinite sequences progressing via left-shifts

⸻

4. GEOMETRIC EMBEDDING (integer-native)

4.1 ℓ∞ Square-Perimeter Coordinates

For integer n in shell k, let:

R = 2^k
t = (n - 2^k) / 2^k      in [0, 1)
C(n) = P_R(t)            # perimeter mapping, clockwise from (-R, +R)

4.2 Scaling Rule

C(2n) = 2 × C(n)

4.3 Alternative Ray Geometry

Each odd core “a” optionally maps to a rational slope m_a.
This mapping is purely definitional.
Coordinates remain rational.

4.4 No Mathematical Meaning

Geometry is not analytic; it is an artifact of construction.
The AI must NOT assign arithmetic meaning to geometric patterns.

⸻

5. ITERATION MODES
	•	Natural: 1, 2, 3, …
	•	Theta order: by theta_key across shells
	•	Ray order: a × 2^j
	•	Shell order: process full shells sequentially

Each must respect the core invariants.

⸻

6. AI BEHAVIORAL RULES

6.1 MUST DO
	•	Preserve all definitions exactly
	•	Maintain separation of:
	•	definition
	•	construction consequence
	•	observation
	•	Flag any contradiction or ambiguity
	•	Ask clarifying questions if required
	•	Use reporting schema (below)

6.2 MUST NOT DO
	•	MUST NOT treat theta_pos as global
	•	MUST NOT infer mathematical significance about primes
	•	MUST NOT redefine core logic
	•	MUST NOT introduce trigonometry or floats in core logic
	•	MUST NOT propose new geometry unless explicitly asked

⸻

7. REPORTING PROTOCOL

Responses must use this schema when analyzing or critiquing:

[OBSERVATION]
Type: (Error | Simplification | Limitation | Connection | New Fact)
What:
Why:
Confidence:


⸻

End of SYSTEM.md
