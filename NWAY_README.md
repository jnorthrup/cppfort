# xai_4_2 N-Way Orbit Orchestration

The original GPT-training redux has been refactored into an explainable, telemetry-first pipeline.
Stage 0 supplies orbit metadata; xai_4_2 fuses that with the optimized evidence heuristics in
`build/x.txt`, and the n-way renderers consume the fused stream to emit cpp2/cpp/C (or any future
target).

## Core Flow

1. **Scan once** – `WideScanner::scanAnchorsWithOrbits` yields `(pos, delimiter, mask, confidence)`.
2. **Fuse evidence** – accumulate the spans into confidence vectors using the xai_4_2 blueprint.
3. **Project n-way** – renderers pull spans plus evidence traits and choose an emission strategy.

```text
source → anchors → orbit stream → xai_4_2 evidence → {render_cpp2, render_cpp, render_c, …}
```

The evidence layer is explainable: every emitted token is backed by lattice bits, SIMD-derived
confidence vectors, and gap records for human review.

## Evidence Planes

| Plane | Description | Consumption |
| --- | --- | --- |
| Anchor confidence | SIMD vectors derived from NEON heuristics (`build/x.txt`) | Drives span ranking and pruning |
| Lattice mask stack | Structural bits from Stage 0 | Selects renderer macro strategies |
| Gap ledger | Spans below threshold with orbit imbalances | Propagated as `// xai-gap` markers |

Renderers must ingest all planes so decisions remain auditable.

## Renderer Contracts

All renderers operate on the unified span model:

```cpp
struct OrbitSpan {
    size_t start;
    size_t end;
    uint16_t lattice_mask;
    double confidence;
    std::string_view text;
    xai_4_2::EvidenceVector evidence; // defined by the anchor evidence blueprint
};
```

Key expectations:

1. **cpp2 renderer**
   - Promote spans with function signatures into cpp2 contracts (`name: (params) -> type = {}`).
   - Convert includes/imports when evidence ≥ 0.82; otherwise wrap with `// xai-gap`.
2. **C++ renderer**
   - Mirror original tokens when confidence ≥ 0.7, otherwise surface TODO breadcrumbs.
3. **C renderer**
   - Downgrade templates/refs to C-safe constructs; emit gap markers when lattice bits signal
     templates or generics.

Emitters that target other environments (e.g., Betanet mixin modules) should follow the same
evidence contract.

## Self-Fidelity Loop

1. Emit cpp2 via xai_4_2.
2. Re-scan with Stage 0 to rebuild the orbit/evidence stream.
3. Compare with the original stream (confidence vectors + masks).
4. Divergence beyond tolerances requires either gap escalation or manual span pinning.

This loop keeps explainability intact without reaching for an AST.

## Operating Notes

- Store serialized spans as `.orbit.jsonl` with embedded evidence for diffing and dashboards.
- Feed lattice statistics into CI: track `% of spans ≥ 0.9` and `% of gaps auto-closed`.
- Integrate with Betanet transport by piping the serialized evidence through the
  `BetanetCASResolver` when provenance is required.

The mantra remains: **scan once, explain always, project everywhere.**
