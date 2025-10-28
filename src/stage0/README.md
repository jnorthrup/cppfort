# Stage 0 – Orbit-Only Scanner

> **Context:** Stage 0 has been trimmed to the minimum viable scanning surface. It no longer
> builds an AST, emitter, or TableGen pipeline. Everything in this directory now revolves
> around orbit detection, lattice classification, and producing metadata that later stages
> (or external tools) can translate into language-specific projections.

## Deliverables

| Artifact | Description |
| --- | --- |
| `liborbit_scanner.a` | Static library that exposes wide scanner, Rabin–Karp hashes, confix detection, and multi‑grammar pattern catalogs. |
| `stage0_cli` | Minimal CLI that scans source text and prints anchor/boundary statistics with lattice masks and orbit confidence. |

## Key Components

- `wide_scanner.[h|cpp]`
  - Generates alternating UTF‑8 anchors (64/32 stride) and performs SIMD sweeps.
  - `scanAnchorsWithOrbits` enriches each boundary with:
    - `lattice_mask` (`cppfort::stage0::LatticeClasses`) for byte-level classification.
    - `orbit_confidence` (0.0–1.0) computed from `OrbitContext` depth balance.
- `confix_fishy_detector.cpp2`
  - cpp2 definition of confix tracking structs used when the scanner is embedded in a cpp2 environment.
- `orbit_mask.[h|cpp]`
  - Maintains `OrbitContext`, used by both the C++ scanner and the cpp2 definitions.
- `rabin_karp.[h|cpp]`
  - Tracks hierarchical hashes of orbit depth vectors for pattern lookups.
- `multi_grammar_loader`, `tblgen_patterns`
  - Lightweight pattern catalogs (C / C++ / cpp2) used to label orbit signatures.

## What’s Explicitly Gone

- Stage 0 no longer exports an AST, emitter, or Sea-of-Nodes lowering.
- The static library `libstage0.a` and the multitude of chapter/unit tests have been removed.
- Stage 1 / Stage 2 / IR directories exist for historical reference but are excluded from the default build.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The build produces:

- `build/liborbit_scanner.a`
- `build/stage0_cli`

No other stage binaries are generated in this configuration.

## CLI Usage

```bash
./build/stage0_cli scan path/to/source.cpp2

# Sample output
# Generated 108 anchor points
# Found 599 boundaries
#   pos=0 delim=  conf=1 mask=0x1202
#   pos=68 delim=, conf=1 mask=0x0004
```

The masks correspond to `cppfort::stage0::LatticeClasses` and can be decoded to infer
structural hints (delimiter, identifier, whitespace, etc.). A confidence below ~0.6 typically
indicates imbalanced confix depth and should be treated as a “needs reviewer” region.

## Integration Guidance

1. **Feed-orbit-first:** Run `WideScanner::scanAnchorsWithOrbits` and persist the boundary
   stream. Do not attempt AST reconstruction inside Stage 0.
2. **Projection happens elsewhere:** Build separate modules that consume the boundary stream to
   render cpp2 / C++ / C text. Stage 0 should remain agnostic of the target syntax.
3. **Handle uncertainty:** When `orbit_confidence` drops or conflicting lattice bits appear,
   downstream tools should flag the span for human review or fall back to a conservative copy.

## Next Steps / TODOs

- Document a stable serialization format for boundary streams (`.orbits`?).
- Provide tiny helper utilities for translating lattice masks into human-readable tags.
- Restore regression tests focused solely on scanner accuracy (golden boundary snapshots).
- Gate optional builds for Stage 1/Stage 2 behind explicit CMake flags if they need to return.

---

Maintainers: keep Stage 0 focused. Anything that smells like parsing or emission belongs in a
different stage or module.
