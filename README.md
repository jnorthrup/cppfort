# cppfort - Sea-of-Nodes Cpp2 Transpiler

cppfort is a Cpp2-to-C++ transpiler built around a Sea-of-Nodes (SoN) intermediate representation powered by MLIR, implementing the TrikeShed compositional manifold strategy.

## TrikeShed Manifold Architecture

### Mathematical Foundation

**Definition 1.1: Compiler Manifold**
Let $\mathcal{P}$ denote the space of all syntactically valid programs. The compiler manifold $M$ is a smooth manifold of dimension $n$ where:

$$M = \bigsqcup_{i \in I} U_i$$

with an atlas $\mathcal{A} = \{(U_i, \phi_i)\}_{i \in I}$ where each chart $\phi_i: U_i \to \mathbb{R}^n$ maps program representations to Euclidean space.

**Definition 1.2: Semantic Coordinate Space**
Let $S \subseteq M$ be the subspace of semantic coordinates - program representations preserving semantic equivalence under transformation.

**Definition 1.3: Chart as Manifold Mapping**
A chart $(U, \phi)$ satisfies:
1. $U \subseteq M$ is an open set of program representations
2. $\phi: U \to \mathbb{R}^n$ is a homeomorphism
3. $\phi$ is smooth (infinitely differentiable)
4. $\phi(U)$ is an open subset of $\mathbb{R}^n$

**Definition 1.4: Atlas as Coordinate System Collection**
An atlas $\mathcal{A}$ is a collection of charts $\{(U_\alpha, \phi_\alpha)\}_{\alpha \in A}$ covering $M$ with smooth transition functions $\tau_{\alpha\beta} = \phi_\beta \circ \phi_\alpha^{-1}$.

**Theorem 1.1: Manifold Structure Theorem**
The compiler manifold $M$ admits a unique smooth structure compatible with its atlas $\mathcal{A}$.

**Proof:** The cocycle condition $\tau_{\alpha\beta} \circ \tau_{\beta\gamma} \circ \tau_{\gamma\alpha} = \text{id}$ on triple overlaps ensures consistency. $\square$

**Definition 1.5: Tangent Space at a Point**
For any point $p \in M$, the tangent space $T_pM$ consists of velocity vectors of smooth curves through $p$:

$$T_pM = \left\{ \left.\frac{d}{dt}\right|_{t=0} \gamma(t) : \gamma \text{ smooth curve through } p \right\}$$

**Definition 1.6: Vector Field as Rewrite Rule**
A vector field $X$ on $M$ assigns a tangent vector $X_p \in T_pM$ to each point $p \in M$. Rewrite rules are vector fields whose integral curves represent transformation sequences.

**Definition 1.7: Riemannian Metric**
A metric $g$ on $M$ is a smooth symmetric positive-definite bilinear form on each tangent space:

$$g_p: T_pM \times T_pM \to \mathbb{R}$$

**Definition 1.7.1: Optimization Landscape**
The cost function $C: M \to \mathbb{R}$ induces gradient vector field $\nabla C$ satisfying:

$$g(\nabla C, X) = dC(X)$$

**Theorem 1.2: Geodesic Principle**
Geodesics on the compiler manifold represent locally optimal sequences of program transformations minimizing the cost functional.

**Definition 1.8: Geodesic**
A geodesic $\gamma: I \to M$ satisfies:

$$\frac{D}{dt}\frac{d\gamma}{dt} = 0$$

### Core Manifold Operations

**1. Coordinates Operation** - `cpp2.coordinates`
Semantic coordinate literals: `coords[1.0, 2.0]`

**2. Chart Project Operation** - `cpp2.chart_project`
Point to local coordinates: `chart.project(point)`

**3. Chart Embed Operation** - `cpp2.chart_embed`
Local to point: `chart.embed(local)`

**4. Atlas Operation** - `cpp2.atlas`
Chart collection: `atlas[chart1, chart2]`

**5. Manifold Operation** - `cpp2.manifold`
Smooth coordinate space: `manifold name = atlas[...]`

**6. Transition Operation** - `cpp2.transition`
Chart reprojection: `manifold.transition(from, to, coords)`

**7. Lower Dense Operation** - `cpp2.lower_dense`
Explicit dense view: `coords.lowered()`

### Architecture Flow

```
Semantic Coordinates (coords[...])
    ↓ (chart.project)
Local Coordinates (chart space)
    ↓ (manifold.transition)
Reprojected Coordinates (new chart space)
    ↓ (lowered)
Dense Storage (materialized view)
```

### TrikeShed Principles

✅ **Semantic objects first**: All operations work on semantic types
✅ **Dense views second**: Lowering is explicit and separate
✅ **Early normalization**: Surface syntax → canonical operations
✅ **Zero-cost abstraction**: Compile-time optimization possible

## Project Structure

- `include/Cpp2SONDialect.td` - MLIR dialect definitions (7 operations)
- `lib/Dialect/Cpp2SONDialect.cpp` - Dialect implementation
- `lib/Passes/SoNConstantProp.cpp` - Constant propagation pass
- `tests/smoke/` - 24 dogfooding smoke tests
- `docs/manifold-architecture.md` - Detailed architecture documentation
- `expanded_cpp2_spec.md` - Kotlin-to-cpp2 mapping
- `docs/sea-of-nodes/` - SeaOfNodes chapter documentation (01-24)

## Build System

```bash
cmake -S . -B build -G Ninja
ninja -C build Cpp2SONDialect Cpp2SONPasses
```

## Status

✅ Cpp2SON MLIR dialect with 7 manifold operations
✅ Constant propagation pass implemented
✅ 24 dogfooding smoke tests created
✅ Full SeaOfNodes chapter documentation (01-24)
✅ Mathematical foundation documented

**In Progress:**
- Parser implementation (hand-written, 1650+ lines)
- End-to-end test pipeline

**Working:**
- Expression parsing: `42`, `x`, `x = 42`
- Unified declaration: `main: () -> int = { return 42; }` ✅ DOGFOODING MINIMUM
- MLIR dialect builds
- CMake configuration

**Next:**
- Colon declaration syntax: `main: () -> int = { return 42; }`
- Full TrikeShed surface syntax
... we'll se if this holds up as we implement the rest of the architecture!