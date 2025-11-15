#!/usr/bin/env python3
"""
Quaternion Group Q₈ Actions and Graph Representations (No NumPy Version)

This implements the various actions of the quaternion group Q₈ = {±1, ±i, ±j, ±k}
and generates graph visualizations for each type of action.
"""

import networkx as nx
import matplotlib.pyplot as plt
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


def generate_commutativity_graph(q8):
    """
    Generate commutativity graph for Q₈
    Vertices: group elements
    Edge colors: blue if commute, red if don't commute
    """
    G = nx.Graph()

    # Add nodes
    for i, name in enumerate(q8.element_names):
        G.add_node(i, label=name)

    # Add edges with colors
    edge_colors = []
    for i, a in enumerate(q8.elements):
        for j, b in enumerate(q8.elements):
            if i <= j:  # Only upper triangle
                G.add_edge(i, j)
                if q8.commutator(a, b):
                    edge_colors.append('blue')
                else:
                    edge_colors.append('red')

    return G, edge_colors


def generate_conjugacy_action(q8):
    """
    Conjugation action: g·x = gxg⁻¹
    Returns adjacency matrices for each group element's action
    """
    actions = {}

    for g_idx, g in enumerate(q8.elements):
        action_matrix = [[0 for _ in range(8)] for _ in range(8)]
        for x_idx, x in enumerate(q8.elements):
            # Compute gxg⁻¹
            gx = q8.multiply(g, x)
            g_inv = q8.conjugate(g)  # g⁻¹ = conjugate for unit quaternions
            gxg_inv = q8.multiply(gx, g_inv)

            # Find index of result
            y_idx = q8.elements.index(gxg_inv)
            action_matrix[x_idx][y_idx] = 1

        actions[q8.element_names[g_idx]] = action_matrix

    return actions


def generate_cayley_graph(q8, generators=None):
    """
    Generate Cayley graph for Q₈
    Default generators: i, j (or their indices)
    """
    if generators is None:
        generators = [2, 4]  # indices for i, j

    G = nx.DiGraph()

    # Add nodes
    for i, name in enumerate(q8.element_names):
        G.add_node(i, label=name)

    # Add directed edges for each generator
    edge_colors = []
    for gen_idx in generators:
        g = q8.elements[gen_idx]
        color = 'red' if gen_idx == 2 else 'blue'  # i=red, j=blue

        for x_idx, x in enumerate(q8.elements):
            y = q8.multiply(x, g)
            y_idx = q8.elements.index(y)
            G.add_edge(x_idx, y_idx, color=color)
            edge_colors.append(color)

    return G, edge_colors


def matrix_action_f3_squared():
    """
    Q₈ as 2×2 matrices over F₃ (finite field with 3 elements)
    This is the smallest faithful permutation representation
    """
    # Define matrices over F₃
    # Using integers modulo 3
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

    # Generate full action on F₃² vectors
    vectors = list(product([0, 1, 2], repeat=2))  # 9 vectors

    actions = {}
    matrices = {
        '1': I,
        'i': m_i,
        'j': m_j,
        'k': m_k,
    }

    # Generate ± versions
    for name, mat in list(matrices.items()):
        if name == '1':
            matrices['-1'] = [[(-x) % 3 for x in row] for row in mat]
        else:
            matrices[f'-{name}'] = [[(-x) % 3 for x in row] for row in mat]

    def apply_matrix(mat, vec):
        return [
            (mat[0][0]*vec[0] + mat[0][1]*vec[1]) % 3,
            (mat[1][0]*vec[0] + mat[1][1]*vec[1]) % 3
        ]

    # Compute action graph
    for name, mat in matrices.items():
        action = {}
        for i, vec in enumerate(vectors):
            result = apply_matrix(mat, vec)
            j = vectors.index(result)
            action[i] = j
        actions[name] = action

    return vectors, actions


def visualize_graph(G, edge_colors=None, node_labels=None, title=""):
    """Visualize a NetworkX graph"""
    plt.figure(figsize=(10, 10))

    # Layout
    if len(G) <= 8:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, iterations=50)

    # Draw edges
    if edge_colors:
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6)
    else:
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

    # Draw labels
    if node_labels:
        labels = {i: node_labels[i] for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
    else:
        nx.draw_networkx_labels(G, pos, {i: i for i in G.nodes()}, font_size=10)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    return plt


def main():
    """Generate all quaternion group visualizations"""

    q8 = QuaternionGroup()

    print("Quaternion Group Q₈ Actions and Graph Representations")
    print("=" * 60)

    # 1. Commutativity Graph
    print("\n1. Commutativity Graph (conjugacy structure)")
    print("   Blue edges: elements commute")
    print("   Red edges: elements don't commute")

    G_comm, colors = generate_commutativity_graph(q8)
    plt1 = visualize_graph(G_comm, colors, q8.element_names,
                          "Q₈ Commutativity: Blue=Commute, Red=Don't Commute")
    plt1.savefig('/Users/jim/work/cppfort/q8_commutativity.png', dpi=150, bbox_inches='tight')
    print("   Saved to: q8_commutativity.png")

    # 2. Conjugacy Actions
    print("\n2. Conjugation Actions: g·x = gxg⁻¹")
    actions = generate_conjugacy_action(q8)

    # Show one example
    g_name = 'i'
    action_i = actions[g_name]
    print(f"   Action of '{g_name}' demonstrates non-trivial inner automorphism")
    print(f"   Matrix representation (rows: from, cols: to):")
    for row in action_i:
        print(f"     {row}")

    # 3. Cayley Graph
    print("\n3. Cayley Graph (generator actions)")
    print("   Red edges: right multiplication by 'i'")
    print("   Blue edges: right multiplication by 'j'")

    G_cayley, cayley_colors = generate_cayley_graph(q8, [2, 4])  # i, j
    plt3 = visualize_graph(G_cayley, cayley_colors, q8.element_names,
                          "Q₈ Cayley Graph: Red=i, Blue=j")
    plt3.savefig('/Users/jim/work/cppfort/q8_cayley.png', dpi=150, bbox_inches='tight')
    print("   Saved to: q8_cayley.png")

    # 4. Matrix Action on F₃²
    print("\n4. Linear Action: Q₈ as 2×2 matrices over 𝔽₃")
    print("   This is the smallest faithful permutation representation")

    vectors, mat_actions = matrix_action_f3_squared()
    print(f"   Acts on {len(vectors)} vectors in 𝔽₃²")
    print(f"   Example: action of 'i' maps vectors according to matrix [[1,1],[1,-1]]")

    # Create graph for this action
    G_lin = nx.DiGraph()
    node_labels = [f"{v[0]},{v[1]}" for v in vectors]

    for i, label in enumerate(node_labels):
        G_lin.add_node(i, label=label)

    # Add edges for 'i' action
    for src, dst in mat_actions['i'].items():
        G_lin.add_edge(src, dst)

    plt4 = visualize_graph(G_lin, ['red'] * len(G_lin.edges()), node_labels,
                          "Q₈ Action on 𝔽₃² (via matrix 'i')")
    plt4.savefig('/Users/jim/work/cppfort/q8_linear_action.png', dpi=150, bbox_inches='tight')
    print("   Saved to: q8_linear_action.png")

    # 5. Orbit Structure
    print("\n5. Orbit Structure (conjugacy classes)")
    orbits = {
        'Center': ['1', '-1'],
        'Conjugacy class i': ['i', '-i'],
        'Conjugacy class j': ['j', '-j'],
        'Conjugacy class k': ['k', '-k'],
    }
    print(f"   Q₈ has {len(orbits)} orbits under conjugation:")
    for name, elements in orbits.items():
        print(f"     {name}: {elements}")

    print("\n" + "=" * 60)
    print("Graphs saved as PNG files in: /Users/jim/work/cppfort/")
    print("\nKey Insights:")
    print("  • Q₈'s non-abelian structure visible in commutativity graph")
    print("  • Conjugation reveals inner automorphisms")
    print("  • Cayley graph shows generator relationships")
    print("  • Matrix action gives smallest faithful representation (size 9)")
    print("  • Orbit structure shows 4 conjugacy classes")


if __name__ == '__main__':
    main()
