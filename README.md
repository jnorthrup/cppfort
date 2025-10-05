# xai_4_2 – N-Way Orbit Design Core

xai_4_2 is the successor to the earlier GPT-training redux: a deterministic, explainable
orbit engine that projects compiler spans across multiple targets while retaining quantified
evidence. The build still ships the familiar scanner artifacts, but every surface is now described
through the xai_4_2 evidence stack that powers the `xai_42_anchor_evidence_system.cpp2`
integration.

## System Overview

| Artifact | xai_4_2 Role |
| --- | --- |
| `liborbit_scanner.a` | Core lattice/orbit extraction runtime feeding the xai_4_2 evidence graph. |
| `stage0_cli` | CLI for emitting orbit streams plus confidence telemetry suitable for n-way renderers. |
| `src/stage0/` headers | Evidence-aware helpers (anchor synthesis, lattice masks, confix trackers). |

The legacy Stage 1/Stage 2 emitters stay in the tree for parity checks, but the supported workflow
is orbit-first: scan once, reason using xai_4_2 heuristics, and project spans into downstream
targets.

## Build + Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/stage0_cli scan path/to/source.cpp2
```

The CLI prints anchors, lattice masks, and normalized confidence so the xai_4_2 renderers can score
ambiguities. Typical telemetry:

```
Generated 108 anchor points
Found 599 boundaries
  pos=0 delim=  conf=1 mask=0x1202
  pos=68 delim=, conf=1 mask=0x0004
```

- `mask` traces back to `cppfort::stage0::LatticeClasses` and is interpreted by the
  xai_4_2 evidence lookup tables.
- `conf` is the structural certainty produced by the optimized M3 evidence heuristics.

## Orbit → Evidence Loop

Use the scanner as the authoritative signal generator, then layer evidence fusion logic on top. A
minimal integration looks like this:

```cpp
#include "stage0/wide_scanner.h"
// bridge helpers mirror the reference blueprint in build/x.txt

std::string source = read_file("foo.cpp");
auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
cppfort::ir::WideScanner scanner;
auto boundaries = scanner.scanAnchorsWithOrbits(source, anchors);

xai_4_2::EvidenceAccumulator acc; // see optimized_m3_evidence_heuristics in build/x.txt
for (const auto& b : boundaries) {
    acc.ingest(b);
    if (b.is_delimiter && b.orbit_confidence < 0.6) {
        acc.raise_gap(b);
    }
}

auto orbit_graph = acc.finalize();
```

`xai_4_2::EvidenceAccumulator` follows the anchor evidence subsystem referenced in `build/x.txt`
and consumes the same structures exposed by the scanner.

## Tests

Unit and regression suites in `tests/` rely on the orbit scanner but assert the xai_4_2 evidence
contracts. After building:

```bash
ctest --test-dir build
```

Regression fixtures also stream `.orbit.jsonl` snapshots to verify evidence retention.

## Roadmap

- Finalize the serialized evidence interchange format shared by the n-way renderers.
- Graduate the cpp2/cpp/C emitters so they consume the accumulator output instead of legacy ASTs.
- Publish reference heuristics for M3/Neoverse/AMD targets under `build/` for reproducible tuning.
- Integrate Betanet transport once module provenance is tied into the evidence graph.

See [`NWAY_README.md`](NWAY_README.md) for guidance on wiring the n-way projection layer into the
xai_4_2 pipeline.
