# cppfort - Sea-of-Nodes Cpp2 Transpiler

cppfort is a Cpp2-to-C++ transpiler built around a Sea-of-Nodes (SoN) intermediate representation powered by MLIR.

## TrikeShed Manifold Architecture

### Core Manifold Operations (7 total)

**1. Coordinates Operation**

- Purpose: Semantic coordinate literals
- Syntax: `coords[1.0, 2.0]`
- Type: Semantic coordinate type (NOT dense storage)
- MLIR: `cpp2.coordinates`

**2. Chart Project Operation**

- Purpose: Point → Local coordinates
- Syntax: `chart.project(point)`
- Direction: Forward mapping
- MLIR: `cpp2.chart_project`

**3. Chart Embed Operation**

- Purpose: Local coordinates → Point
- Syntax: `chart.embed(local)`
- Direction: Reverse mapping
- MLIR: `cpp2.chart_embed`

**4. Atlas Operation**

- Purpose: Collection of charts
- Syntax: `atlas[chart1, chart2]`
- Type: Indexed chart collection
- MLIR: `cpp2.atlas`

**5. Manifold Operation**

- Purpose: Smooth coordinate space
- Syntax: `manifold name = atlas[...]`
- Type: Composed from atlas
- MLIR: `cpp2.manifold`

**6. Transition Operation**

- Purpose: Chart reprojection
- Syntax: `manifold.transition(from, to, coords)`
- Behavior: Reproject through semantic point
- MLIR: `cpp2.transition`

**7. Lower Dense Operation**

- Purpose: Explicit materialized view
- Syntax: `coords.lowered()`
- Property: Never aliases semantic type
- MLIR: `cpp2.lower_dense`

## Mathematical Foundation

The compiler manifold $M$ is a smooth manifold where:

- Points represent program representations
- Charts represent different IR coordinate systems
- Atlas is a collection of charts
- Manifold is the smooth coordinate space
- Transition functions are diffeomorphisms between charts

### Architecture Flow

```
Semantic Coordinates
    ↓ (chart.project)
Local Coordinates (chart space)
    ↓ (manifold.transition)
Reprojected Coordinates (new chart space)
    ↓ (lowered)
Dense Storage (materialized)
```

## Status

- ✅ Cpp2SON MLIR dialect with 7 manifold operations
- ✅ 24 dogfooding smoke tests
- ✅ Full SeaOfNodes chapter documentation (01-24)
- ✅ Build system verified

## Documentation

- `docs/manifold-architecture.md` - Complete architecture documentation
- `expanded_cpp2_spec.md` - Kotlin-to-cpp2 mapping
- `docs/sea-of-nodes/` - SeaOfNodes chapter documentation

## Build

```bash
cmake -S . -B build -G Ninja
ninja -C build Cpp2SONDialect Cpp2SONPasses
```
