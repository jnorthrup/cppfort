# TrikeShed Manifold Architecture

## Overview

The manifold system in cppfort implements smooth coordinate spaces using the TrikeShed compositional strategy. This document details how manifolds are shaping out across the codebase.

## Core Components

### 1. Semantic Coordinates (`cpp2.coordinates`)

**Purpose**: Represent semantic positions in chart domains without materializing storage.

**Kotlin syntax**: `coords[1.0, 2.0]`
**MLIR op**: `cpp2.coordinates`
**Type**: Semantic coordinate type (not dense storage)

**Key property**: Coordinates are semantic objects first, dense views second.

### 2. Charts (`cpp2.chart_project`, `cpp2.chart_embed`)

**Purpose**: Bidirectional mapping between point space and local coordinate space.

**Kotlin syntax**:
```kotlin
chart identity(point: f64) {
  contains point <= 20.0
  project -> coords[point]
  embed(local) -> local[0]
}
```

**MLIR ops**:
- `cpp2.chart_project`: point → local coordinates
- `cpp2.chart_embed`: local coordinates → point

**Key property**: Charts define smooth mappings with domain constraints (`contains`).

### 3. Atlas (`cpp2.atlas`)

**Purpose**: Collection of charts covering a manifold.

**Kotlin syntax**: `atlas[shifted, identity]`
**MLIR op**: `cpp2.atlas`

**Key property**: Atlas is an indexed collection of charts for manifold construction.

### 4. Manifold (`cpp2.manifold`)

**Purpose**: Smooth coordinate space composed from atlas of charts.

**Kotlin syntax**: `manifold line = atlas[shifted, identity]`
**MLIR op**: `cpp2.manifold`

**Key property**: Manifold is the semantic object that provides coordinate transformations.

### 5. Transition (`cpp2.transition`)

**Purpose**: Reproject coordinates between charts through semantic point.

**Kotlin syntax**: `manifold.transition("identity", "shifted", coords[17.0])`
**MLIR op**: `cpp2.transition`

**Key property**: Transition preserves semantic meaning while changing chart representation.

### 6. Dense View (`cpp2.lower_dense`)

**Purpose**: Explicit materialized view of semantic coordinates.

**Kotlin syntax**: `coords.lowered()`
**MLIR op**: `cpp2.lower_dense`

**Key property**: Dense view never aliases the semantic coordinate type.

## Architecture Flow

```
Semantic Coordinates (coords[...])
    ↓ (chart.project)
Local Coordinates (chart-local space)
    ↓ (manifold.transition)
Reprojected Coordinates (new chart space)
    ↓ (lowered)
Dense Storage (materialized view)
```

## Composition Strategy

The manifold system follows TrikeShed principles:

1. **Semantic objects first**: Coordinates, charts, atlases, manifolds are semantic
2. **Dense views second**: Lowering happens only when needed
3. **Early normalization**: Surface syntax normalizes to canonical operations
4. **Zero-cost abstraction**: Semantic objects compile away to optimal code

## MLIR Implementation

### Dialect Operations (include/Cpp2SONDialect.td)

- `cpp2.coordinates` - semantic coordinate literal
- `cpp2.chart_project` - point to local projection
- `cpp2.chart_embed` - local to point embedding
- `cpp2.atlas` - chart collection
- `cpp2.manifold` - smooth coordinate space
- `cpp2.transition` - chart reprojection
- `cpp2.lower_dense` - explicit dense view

### Dogfooding Tests (tests/smoke/)

- `chapter01_manifold.cpp2` - basic manifold operations
- `chapter02_indexed.cpp2` - indexed operations
- Chapters 03-24 - progressive manifold complexity

## Example: Line Manifold

From expanded_cpp2_spec.md:

```cpp2
// Define charts with domain constraints
let shifted = chart shifted(point: f64) {
  contains point > 5.0
  project -> coords[point - 10.0]
  embed(local) -> local[0] + 10.0
}

let identity = chart identity(point: f64) {
  contains point <= 20.0
  project -> coords[point]
  embed(local) -> local[0]
}

// Compose manifold from atlas
let line = manifold line = atlas[shifted, identity]

// Transition between charts
let local = line.transition("identity", "shifted", coords[17.0])

// Materialize dense view
let dense = local.lowered()
```

## Current Status

✅ **Complete**:
- All 7 manifold operations implemented in MLIR dialect
- 24 dogfooding smoke tests created
- Full SeaOfNodes chapter documentation (01-24)
- Build system verified

🔄 **In Progress**:
- Parser implementation for cpp2 manifold syntax
- SoN lowering from canonical AST to MLIR
- End-to-end compilation pipeline

## Next Steps

1. Implement parser for manifold syntax
2. Wire canonical types to build system
3. Create end-to-end test: cpp2 → canonical AST → SoN → MLIR → compiled output
4. Verify zero-cost abstraction property
