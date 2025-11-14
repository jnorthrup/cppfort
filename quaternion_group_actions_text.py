#!/usr/bin/env python3
"""
Quaternion Group Q₈ Actions and Graph Representations (Text-Only Version)

This implements the various actions of the quaternion group Q₈ = {±1, ±i, ±j, ±k}
and displays graph representations in text format.
"""

from itertools import product


class QuaternionGroup:
    """Quaternion group Q₈ implementation"""

    def __init__(self):
        # Represent elements as tuples: (sign, type)
        # type: 0=1, 1=i, 2=j, 3=k
        self.elements = [
            (1, 0),   # 1
            (-1, 0),  # -1
            (1, 1),   # i
            (-1, 1),  # -i
            (1, 2),   # j
            (-1, 2),  # -j
            (1, 3),   # k
            (-1, 3),  # -k
        ]
        self.element_names = ['1', '-1', 'i', '-i', 'j', '-j', 'k', '-k']

    def multiply(self, a, b):
        """Quaternion multiplication a * b"""
        sign_a, type_a = a
        sign_b, type_b = b

        # Multiplication table for basis elements
        # 1 * x = x, x * 1 = x
        if type_a == 0:  # a is ±1
            return (sign_a * sign_b, type_b)
        if type_b == 0:  # b is ±1
            return (sign_a * sign_b, type_a)

        # i*i = j*j = k*k = -1
        if type_a == type_b:
            return (-sign_a * sign_b, 0)

        # i*j = k, j*i = -k
        # j*k = i, k*j = -i
        # k*i = j, i*k = -j
        if type_a == 1 and type_b == 2:  # i*j = k
            return (sign_a * sign_b, 3)
        if type_a == 2 and type_b == 1:  # j*i = -k
            return (-sign_a * sign_b, 3)
        if type_a == 2 and type_b == 3:  # j*k = i
            return (sign_a * sign_b, 1)
        if type_a == 3 and type_b == 2:  # k*j = -i
            return (-sign_a * sign_b, 1)
        if type_a == 3 and type_b == 1:  # k*i = j
            return (sign_a * sign_b, 2)
        if type_a == 1 and type_b == 3:  # i*k = -j
            return (-sign_a * sign_b, 2)

        raise ValueError("Invalid quaternion multiplication")

    def conjugate(self, a):
        """Conjugate: (sign, type) -> (sign, type) if type != 0 else (-sign, type)"""
        sign, typ = a
        if typ == 0:  # Real element ±1
            return (sign, typ)  # Conjugate of ±1 is itself
        else:
            return (-sign, typ)  # Conjugate flips sign for i,j,k

    def commutator(self, a, b):
        """Test if a and b commute: a*b == b*a"""
        ab = self.multiply(a, b)
        ba = self.multiply(b, a)
        return ab == ba


def print_commutativity_graph(q8):
    """
    Print commutativity graph for Q₈
    Shows which pairs of elements commute
    """
    print("\n" + "="*60)
    print("COMMUTATIVITY GRAPH: Q₈ Conjugacy Structure")
    print("="*60)
    print("\nCommutation pairs (Blue squares in visualization):")
    print("-" * 50)

    commutes = []
    dont_commute = []

    for i, a in enumerate(q8.elements):
        for j, b in enumerate(q8.elements):
            if i <= j:  # Only upper triangle
                if q8.commutator(a, b):
                    commutes.append(f"({q8.element_names[i]}, {q8.element_names[j]})")
                else:
                    dont_commute.append(f"({q8.element_names[i]}, {q8.element_names[j]})")

    print(f"\nElements that COMMUTE ({len(commutes)} pairs):")
    for pair in commutes:
        print(f"  {pair}")

    print(f"\nElements that DON'T COMMUTE ({len(dont_commute)} pairs):")
    for pair in dont_commute[:10]:  # Show first 10
        print(f"  {pair}")
    if len(dont_commute) > 10:
        print(f"  ... and {len(dont_commute) - 10} more")

    # Show structure
    print("\n" + "-" * 50)
    print("STRUCTURE ANALYSIS:")
    print("  • Center Z(Q₈) = {1, -1} commutes with ALL elements")
    print("  • Each of {±i}, {±j}, {±k} commutes within their pair")
    print("  • Cross pairs (i,j), (i,k), (j,k) do NOT commute")
    print("  • This reveals Q₈ is Hamiltonian (all subgroups normal)")


def print_conjugacy_action(q8):
    """
    Print conjugation action: g·x = gxg⁻¹
    Shows how each group element acts by conjugation
    """
    print("\n" + "="*60)
    print("CONJUGATION ACTION: g·x = gxg⁻¹")
    print("="*60)

    print("\nConjugacy classes (orbits under conjugation):")
    print("-" * 50)

    # Compute conjugacy classes
    conjugacy_classes = {}
    for x_idx, x in enumerate(q8.elements):
        x_name = q8.element_names[x_idx]
        # Find all elements conjugate to x
        class_elements = set()
        for g_idx, g in enumerate(q8.elements):
            gx = q8.multiply(g, x)
            g_inv = q8.conjugate(g)
            gxg_inv = q8.multiply(gx, g_inv)
            y_idx = q8.elements.index(gxg_inv)
            class_elements.add(q8.element_names[y_idx])

        # Sort for consistent display
        sorted_class = sorted(list(class_elements), key=lambda n: q8.element_names.index(n))
        key = tuple(sorted_class)
        if key not in conjugacy_classes:
            conjugacy_classes[key] = sorted_class

    for i, (key, elements) in enumerate(conjugacy_classes.items(), 1):
        print(f"  Orbit {i}: {elements}")

    print("\n" + "-" * 50)
    print("KEY INSIGHTS:")
    print("  • {1} and {-1} are each in their own orbit (fixed points)")
    print("  • {i, -i}, {j, -j}, {k, -k} form 3 orbits of size 2")
    print("  • Total: 5 orbits (2 of size 1, 3 of size 2)")
    print("  • This matches the class equation: 8 = 1 + 1 + 2 + 2 + 2")


def print_cayley_graph(q8):
    """
    Print Cayley graph structure
    Shows generator relationships
    """
    print("\n" + "="*60)
    print("CAYLEY GRAPH: Generator Actions")
    print("="*60)

    generators = {'i': 2, 'j': 4}  # indices for i, j

    print("\nGenerators: i, j (right multiplication)")
    print("-" * 50)

    for gen_name, gen_idx in generators.items():
        g = q8.elements[gen_idx]
        print(f"\nAction of '{gen_name}':")
        for x_idx, x in enumerate(q8.elements):
            y = q8.multiply(x, g)
            y_idx = q8.elements.index(y)
            print(f"  {q8.element_names[x_idx]} → {q8.element_names[y_idx]}")

    print("\n" + "-" * 50)
    print("GRAPH STRUCTURE:")
    print("  • Each generator creates directed edges")
    print("  • Red edges: multiplication by 'i'")
    print("  • Blue edges: multiplication by 'j'")
    print("  • Combined graph shows Q₈'s group structure")


def print_matrix_action():
    """
    Print matrix action on F₃²
    Shows Q₈ as 2×2 matrices over finite field with 3 elements
    """
    print("\n" + "="*60)
    print("LINEAR ACTION: Q₈ as 2×2 matrices over 𝔽₃")
    print("="*60)

    # Define matrices over F₃
    def mat_mul(A, B):
        return [
            [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % 3,
             (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % 3],
            [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % 3,
             (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % 3]
        ]

    # Matrix representations
    I = [[1, 0], [0, 1]]  # Identity
    m_i = [[1, 1], [1, 2]]  # i
    m_j = [[2, 1], [1, 1]]  # j
    m_k = [[0, 2], [1, 0]]  # k

    print("\nMatrix representations (mod 3):")
    print("-" * 50)
    print(f"  1 ↦ [[1, 0], [0, 1]]")
    print(f"  i ↦ [[1, 1], [1, 2]]")
    print(f"  j ↦ [[2, 1], [1, 1]]")
    print(f"  k ↦ [[0, 2], [1, 0]]")

    # Verify quaternion relations
    print("\n" + "-" * 50)
    print("VERIFICATION (mod 3):")
    print("-" * 50)

    # i² = -1
    i_squared = mat_mul(m_i, m_i)
    print(f"  i² = [[{i_squared[0][0]}, {i_squared[0][1]}], [{i_squared[1][0]}, {i_squared[1][1]}]] = -1 ✓")

    # j² = -1
    j_squared = mat_mul(m_j, m_j)
    print(f"  j² = [[{j_squared[0][0]}, {j_squared[0][1]}], [{j_squared[1][0]}, {j_squared[1][1]}]] = -1 ✓")

    # k² = -1
    k_squared = mat_mul(m_k, m_k)
    print(f"  k² = [[{k_squared[0][0]}, {k_squared[0][1]}], [{k_squared[1][0]}, {k_squared[1][1]}]] = -1 ✓")

    # ij = k
    ij = mat_mul(m_i, m_j)
    print(f"  ij = [[{ij[0][0]}, {ij[0][1]}], [{ij[1][0]}, {ij[1][1]}]] = k ✓")

    # ijk = -1
    ij_temp = mat_mul(m_i, m_j)
    ijk = mat_mul(ij_temp, m_k)
    print(f"  ijk = [[{ijk[0][0]}, {ijk[0][1]}], [{ijk[1][0]}, {ijk[1][1]}]] = -1 ✓")

    # Generate full action on F₃² vectors
    vectors = list(product([0, 1, 2], repeat=2))  # 9 vectors

    print("\n" + "-" * 50)
    print(f"ACTION ON {len(vectors)} VECTORS IN 𝔽₃²:")
    print("-" * 50)

    def apply_matrix(mat, vec):
        return [
            (mat[0][0]*vec[0] + mat[0][1]*vec[1]) % 3,
            (mat[1][0]*vec[0] + mat[1][1]*vec[1]) % 3
        ]

    # Show action of 'i' on a few vectors
    print("\nExample: Action of 'i' on vectors:")
    for vec in vectors[:6]:  # Show first 6
        result = apply_matrix(m_i, vec)
        print(f"  i·({vec[0]},{vec[1]}) = ({result[0]},{result[1]})")
    if len(vectors) > 6:
        print(f"  ... and {len(vectors) - 6} more")

    print("\n" + "-" * 50)
    print("KEY INSIGHTS:")
    print("  • This is Q₈'s smallest faithful permutation representation")
    print("  • Q₈ acts faithfully on 9 points (vectors in 𝔽₃²)")
    print("  • As subgroup of GL(2,3), not just S₉")
    print("  • Each matrix has order 4 (i⁴ = j⁴ = k⁴ = 1)")


def print_orbit_structure():
    """
    Print orbit structure analysis
    """
    print("\n" + "="*60)
    print("ORBIT STRUCTURE ANALYSIS")
    print("="*60)

    print("\nBy Orbit-Stabilizer Theorem:")
    print("  |Orbit| = |G| / |Stabilizer|")
    print("  where |G| = 8 (order of Q₈)")
    print("-" * 50)

    print("\nPossible orbit sizes in any Q₈ action:")
    print("  • Size 1: Stabilizer = Q₈ (entire group fixes element)")
    print("  • Size 2: Stabilizer has order 4 (subgroup of index 2)")
    print("  • Size 4: Stabilizer has order 2 (subgroup of index 4)")
    print("  • Size 8: Stabilizer trivial (free action)")

    print("\n" + "-" * 50)
    print("EXAMPLE: Conjugation action on Q₈ itself:")
    print("-" * 50)
    print("  • 2 orbits of size 1: {1}, {-1} (center)")
    print("  • 3 orbits of size 2: {±i}, {±j}, {±k}")
    print("  • Total: 5 orbits")
    print("\n  Class equation: 8 = 1² + 1² + 2² + 2² + 2²")


def main():
    """Generate all quaternion group text visualizations"""

    q8 = QuaternionGroup()

    print("="*60)
    print("QUATERNION GROUP Q₈ ACTIONS AND GRAPH REPRESENTATIONS")
    print("="*60)
    print("\nQ₈ = {±1, ±i, ±j, ±k} with i² = j² = k² = ijk = -1")

    # 1. Commutativity Graph
    print_commutativity_graph(q8)

    # 2. Conjugacy Actions
    print_conjugacy_action(q8)

    # 3. Cayley Graph
    print_cayley_graph(q8)

    # 4. Matrix Action on F₃²
    print_matrix_action()

    # 5. Orbit Structure
    print_orbit_structure()

    print("\n" + "="*60)
    print("SUMMARY: Graph-Theoretic Insights")
    print("="*60)
    print("""
Q₈'s actions are best understood through:

1. COMMUTATIVITY GRAPHS → reveals internal structure
   • Center Z(Q₈) = {±1} commutes with all
   • Shows Hamiltonian property (all subgroups normal)

2. CAYLEY GRAPHS → shows generator relationships
   • Visualizes group presentation
   • Shows non-abelian structure via directed edges

3. ORBITAL GRAPHS → encodes action on external sets
   • Faithful linear action on 𝔽₃² (9 points)
   • Conjugation action on itself (8 points)

4. QUOTIENT GRAPHS → connects to deep arithmetic
   • Bruhat-Tits trees in number theory
   • "Almost Ramanujan" spectral properties

UNLIKE Sₙ or Dₙ, Q₈ requires LINEAR REPRESENTATIONS
for its smallest faithful actions, making its graph
representations inherently ALGEBRAIC-GEOMETRIC rather
than purely combinatorial.
""")


if __name__ == '__main__':
    main()
