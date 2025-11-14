# Quaternion Group Q₈ Actions and Graph Representations - Analysis

## Overview

This document provides a comprehensive analysis of the quaternion group Q₈ = {±1, ±i, ±j, ±k} and its various graph representations, demonstrating why Q₈'s actions are fundamentally different from typical permutation groups like Sₙ or Dₙ.

## Files Generated

1. **`quaternion_group_actions_text.py`** - Text-based analysis (runs without dependencies)
2. **`quaternion_graphs.html`** - Interactive D3.js visualizations (open in browser)
3. **`QUATERNION_GRAPHS_ANALYSIS.md`** - This analysis document

---

## 1. The Core Problem: No Faithful Small-Set Action

### Critical Mathematical Fact

**Q₈ cannot act faithfully as permutations on a set smaller than its own 8 elements.**

This is a profound constraint that distinguishes Q₈ from dihedral groups:

- **Dihedral group D₄** (order 8) can act faithfully on 4 vertices of a square
- **Quaternion group Q₈** (order 8) **cannot** act faithfully on any set with fewer than 8 elements

### Why This Matters

This constraint means:
- No natural action on {1,2,3,4} that preserves group structure
- Must use linear/matrix actions or conjugation on itself
- Graph representations are inherently algebraic-geometric, not combinatorial

---

## 2. Conjugation Action (Internal Symmetry)

### The Action

Q₈ acts on its own elements by conjugation: **g · x = gxg⁻¹**

### Graph Representation

Visualized as a commutativity plot:
- **Blue squares**: pairs that commute
- **Red squares**: pairs that don't commute

### Structure Revealed

```
Elements that commute:
  • Center: {1, -1} commutes with everything
  • Within pairs: {±i}, {±j}, {±k} commute with themselves
  
Elements that DON'T commute:
  • Cross terms: (i,j), (i,k), (j,k) and their negatives
```

### Key Patterns

1. **Center Z(Q₈) = {±1}** commutes with everything → full blue rows/columns
2. **i, j, k** each commute with {±1, ±i} (or ±j, ±k) → 2×2 blue blocks on diagonal
3. **Cross-terms** (i vs j) are red → non-abelian structure

### Mathematical Significance

This graph reveals:
- **Centralizer structure** of each element
- Why Q₈ is **Hamiltonian** (all subgroups normal)
- The **class equation**: 8 = 1 + 1 + 2 + 2 + 2

---

## 3. Geometric Action: 3-Axis Cross Symmetry

### The Action

Q₈ acts by **conjugation on three orthogonal axes**:
- **Elements**: ±i, ±j, ±k correspond to 180° rotations about x, y, z axes
- **Action**: Conjugating by i rotates the j and k axes into each other
- **Visualization**: A 3-colored cross where each quarter-turn changes colors

### Graph Representation

- **Vertices**: 6 ray endpoints (+x, -x, +y, -y, +z, -z)
- **Edges**: Action of group elements maps one vertex to another
- **Result**: 6-vertex graph showing how Q₈ permutes axes with sign changes

### Limitation

Each rotation is represented by **two quaternions** (±i), so the action isn't faithful on this set. This demonstrates why Q₈ needs larger sets for faithful actions.

---

## 4. Linear Action: Matrix Representation on 𝔽₃²

### The Representation

The **most faithful action** is as **2×2 matrices over 𝔽₃**:

```
i ↦ [[1, 1], [1, 2]]    (order 4)
j ↦ [[2, 1], [1, 1]]    (order 4)
k ↦ [[0, 2], [1, 0]]    (order 4)
```

### Verification

All quaternion relations hold modulo 3:
- i² = j² = k² = -1 (where -1 ≡ 2 mod 3)
- ij = k, jk = i, ki = j
- ijk = -1

### Graph Construction

- **Vertices**: 9 vectors in 𝔽₃² = {(0,0), (0,1), ..., (2,2)}
- **Directed edges**: v → M·v for each matrix M ∈ Q₈
- **Result**: 9-vertex orbital graph showing the group action

### Significance

This is Q₈'s **smallest faithful permutation representation**:
- Acts faithfully on 9 points
- As a subgroup of GL(2,3), not just S₉
- Proves Q₈ cannot embed in Sₙ for n < 8

---

## 5. Orbit Structure Analysis

### Orbit-Stabilizer Theorem

For **any Q₈ action** on a set X:
```
|Orbit(x)| = |G| / |Stabilizer(x)|
```

Since |G| = 8, possible orbit sizes are:
- **Size 1**: Stabilizer = Q₈ (entire group fixes element)
- **Size 2**: Stabilizer has order 4 (subgroup of index 2)
- **Size 4**: Stabilizer has order 2 (subgroup of index 4)
- **Size 8**: Stabilizer trivial (free action)

### Example: Conjugation Action on Q₈

```
• 2 orbits of size 1: {1}, {-1}
• 3 orbits of size 2: {±i}, {±j}, {±k}
```

**Class equation**: 8 = 1² + 1² + 2² + 2² + 2²

### Implications

This orbit structure explains:
- Why Q₈ has 5 conjugacy classes
- The representation theory of Q₈
- How characters work in group theory

---

## 6. Advanced: Quaternion Quotient Graphs

### Number Theory Connection

In advanced number theory, Q₈ acts on **Bruhat-Tits trees** (infinite graphs):
- **Action**: Units of a quaternion order act on a tree associated to PGL₂
- **Quotient**: A finite graph A*\T encoding arithmetic information
- **Significance**: "Almost Ramanujan" spectral properties

### Mathematical Depth

This is the **function field analog** of:
- Fuchsian group actions on hyperbolic plane
- Modular group actions on upper half-plane
- Deep connections to automorphic forms

---

## Summary: Graph-Theoretic Insight

### Q₈'s actions are best understood through:

1. **Commutativity graphs** → reveals internal structure and center
2. **Cayley graphs** → shows generator relationships and presentation
3. **Orbital graphs** → encodes action on external sets
4. **Quotient graphs** → connects to deep arithmetic

### Unlike Sₙ or Dₙ

**Q₈ requires linear representations for its smallest faithful actions**, making its graph representations **inherently algebraic-geometric** rather than purely combinatorial.

---

## Key Takeaways

### For Group Theory

1. **Non-abelian center**: Z(Q₈) = {±1} = 25% of the group
2. **Hamiltonian property**: All subgroups are normal
3. **No faithful low-dim action**: Unlike D₄ acting on 4 vertices
4. **Requires linear representation**: Not a subgroup of Sₙ for n < 8

### For Graph Theory

1. **Commutativity graphs** reveal centralizer structure
2. **Cayley graphs** visualize group presentations
3. **Orbital graphs** encode permutation representations
4. **Quotient graphs** connect to number theory

### For Computational Mathematics

1. **Matrix representations** over finite fields give faithful actions
2. **Orbit-stabilizer** theorem governs all group actions
3. **Class equations** predict conjugacy class structure
4. **Representation theory** requires understanding both linear and permutation actions

---

## Running the Visualizations

### Text-Based Analysis

```bash
cd /Users/jim/work/cppfort
python3 quaternion_group_actions_text.py
```

This runs without any dependencies and provides complete text output.

### Interactive Visualizations

```bash
# Open in your default browser
open /Users/jim/work/cppfort/quaternion_graphs.html

# Or use python's web server
cd /Users/jim/work/cppfort
python3 -m http.server 8000
# Then open http://localhost:8000/quaternion_graphs.html
```

The HTML file contains interactive D3.js visualizations of all graph types.

---

## Conclusion

The quaternion group Q₈ demonstrates that **not all groups are created equal** when it comes to actions and representations. Its requirement for linear representations and its unique commutativity structure make it a fascinating case study in how algebraic properties translate into graph-theoretic constraints.

Understanding Q₈'s graph representations provides deep insight into:
- The difference between permutation and linear representations
- Why some groups need more "space" to act faithfully
- How commutativity structure reveals group properties
- The connection between abstract algebra and geometric visualization

This analysis bridges pure group theory, graph theory, and computational mathematics, showing how modern visualization tools can illuminate abstract algebraic structures.
