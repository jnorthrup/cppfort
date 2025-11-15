# CPPfort Architecture - Post-Mortem & Redesign

**Date:** 2025-11-13
**Status:** Critical issues identified, GraphNode redesign proposed

---

## What Went Wrong

### The Bypass
**Root Cause:** Production build used `main_minimal.cpp` instead of `main.cpp`

```cmake
# CMakeLists.txt (WRONG)
add_executable(stage0_cli main_minimal.cpp)  # Line-by-line text processing

# Should be:
add_executable(stage0_cli main.cpp)  # Orbit-based graph processing
```

**Impact:** Entire orbit infrastructure orphaned. 20,000+ lines of graph scanner code unused.

### The Disconnect
**Root Cause:** Pattern matching never connected to orbit system

```cpp
// orbit_pipeline.cpp line 94 (HARDCODED)
orbit->set_selected_pattern("default");  // All orbits get "default"

// Should be:
auto matches = DepthPatternMatcher::find_matches(orbit_text, patterns);
orbit->set_selected_pattern(matches[0].pattern->name);
```

**Impact:** Scanner built correct graph, but patterns never matched to nodes.

### The Abstraction Explosion
**Root Cause:** 15+ overlapping classes simulating graph operations

**Redundant abstractions:**
- `Orbit`, `ConfixOrbit`, `FunctionOrbit` - 3 classes for tree nodes
- `OrbitFragment`, `EvidenceSpan`, `SpanMemento` - 3 types for text ranges
- `OrbitPipeline`, `OrbitIterator`, `OrbitEmitter` - 3 classes for traversal
- `WideScanner`, `OrbitScanner`, `RBCursiveScanner` - 3 scanners
- `PatternData`, `GrammarTree` - 2 pattern representations

**Impact:** Complexity prevented seeing pattern matching = graph isomorphism.

---

## What Actually Works

### Orbit Scanner ✓
```
WideScanner.scanAnchorsWithOrbits(source)
→ Finds confix boundaries {}, (), [], <>
→ Builds tree of OrbitFragments with [start, end) spans
→ Correctly handles multi-line constructs
```

### Evidence Extraction ✓
```
ConfixOrbit.extract_evidence(source, start, end)
→ Extracts text spans between delimiters
→ Preserves source locations
→ Ready for pattern matching
```

### Pattern Loading ✓
```
PatternLoader.load_yaml("patterns/bnfc_cpp2_complete.yaml")
→ 16 patterns loaded
→ Anchors, evidence types, templates all present
```

### What Didn't Connect
- Orbit tree → Pattern matcher (line 94 hardcode)
- Evidence spans → Pattern anchors (never compared)
- Matched patterns → Transforms (always "default")

---

## The GraphNode Solution

### Single Unified Type

```cpp
struct GraphNode {
    // Identity
    NodeType type;  // CONFIX, PATTERN, EVIDENCE, MLIR_OP
    std::string id;

    // Span
    size_t start_pos, end_pos;
    double confidence;

    // Payload
    std::variant<
        ConfixPayload,      // {open, close} delimiter
        PatternPayload,     // Pattern match metadata
        EvidencePayload,    // Text span + constraints
        MLIRPayload        // MLIR operation/region/block
    > data;

    // Graph structure
    std::vector<GraphNode*> children;
    GraphNode* parent;
    std::unordered_map<std::string, GraphNode*> edges;  // Named edges

    // Grammar context
    GrammarType grammar;  // C, CPP, CPP2
};
```

### What Collapses

| Old Abstraction | New Representation |
|----------------|-------------------|
| `ConfixOrbit` | `GraphNode{type=CONFIX}` |
| `FunctionOrbit` | Removed - just metadata in CONFIX node |
| `OrbitFragment` | `GraphNode{type=FRAGMENT}` |
| `EvidenceSpan` | `GraphNode{type=EVIDENCE}` |
| `PatternData` | `GraphNode{type=PATTERN}` |
| `OrbitPipeline` | `GraphTraversal` (functions, not class) |
| `OrbitIterator` | `GraphIterator` |
| `WideScanner` | `GraphBuilder` |

### Operations

**Pattern Matching = Subgraph Isomorphism**
```cpp
bool SubgraphMatcher::match(GraphNode* pattern, GraphNode* source) {
    // Pattern is a tree, source is a tree
    // Match structure + constraints
    if (pattern->type != source->type) return false;
    if (!match_evidence(pattern, source)) return false;

    for (size_t i = 0; i < pattern->children.size(); ++i) {
        if (!match(pattern->children[i], source->children[i]))
            return false;
    }

    return true;
}
```

**Transform = Graph Rewrite**
```cpp
GraphNode* GraphTransformer::apply(GraphNode* matched, Transform& t) {
    // Build replacement subgraph from template
    auto* replacement = instantiate_template(t.template_str, matched);

    // Preserve source locations
    replacement->start_pos = matched->start_pos;
    replacement->end_pos = matched->end_pos;

    // Splice into parent
    replace_child(matched->parent, matched, replacement);
    return replacement;
}
```

**Emit = Graph Walk**
```cpp
void GraphEmitter::emit(GraphNode* root, std::ostream& out) {
    for (auto* node : traverse_preorder(root)) {
        if (node->type == MLIR_OP) {
            emit_mlir(node, out);
        } else {
            emit_source(node, out);
        }
    }
}
```

---

## MLIR Integration Path

GraphNode maps directly to MLIR concepts:

| GraphNode | MLIR Concept |
|-----------|-------------|
| `CONFIX{scope}` | `mlir::Region` |
| `CONFIX{function_body}` | `mlir::Block` |
| `FRAGMENT` | `mlir::Operation` |
| `EVIDENCE` | Source location metadata |

**Direct conversion:**
```cpp
mlir::ModuleOp GraphToMLIR::convert(GraphNode* root) {
    for (auto* node : traverse(root)) {
        switch (node->type) {
            case CONFIX:
                if (is_scope(node)) {
                    node->data = MLIRPayload{
                        .region = builder.createRegion()
                    };
                }
                break;
            case FRAGMENT:
                auto ops = lower_to_ops(builder, node);
                node->data = MLIRPayload{.op = ops.front()};
                break;
        }
    }
}
```

---

## Migration Plan

**Phase 1: Minimal Fix (DONE)**
- ✓ Switch CMakeLists to main.cpp
- ✓ Connect pattern matcher in orbit_pipeline.cpp line 94
- Result: Basic transpilation works, 1/20 tests pass

**Phase 2: GraphNode Foundation**
- Write `graph_node.h` with variant payload
- Implement `SubgraphMatcher` for pattern matching
- Implement `GraphBuilder` wrapping WideScanner

**Phase 3: Pattern Port**
- Convert YAML patterns to GraphNode trees
- Replace string-based matching with structure matching
- Add missing patterns (type alias, inspect, is)

**Phase 4: Transform Rewrite**
- Replace string substitution with graph rewrite
- Preserve all evidence spans
- Add proper multi-line handling

**Phase 5: MLIR Backend**
- Add MLIR payload to GraphNode
- Implement GraphToMLIR converter
- Direct code generation from graph

**Phase 6: Cleanup**
- Remove Orbit/Fragment/Pipeline classes
- Remove emit_depth_based and main_minimal.cpp
- Consolidate to graph-only operations

---

## Current Test Status

**Passing:** 1/20 (simple_mixed.cpp2)
**Failing:** 19/20

**Failure categories:**
1. Multi-line constructs (6) - Fixed by orbit scanner
2. Type aliases (4) - Need pattern
3. Missing features (4) - inspect, is, forward, templates
4. Type definitions (2) - i32, cpp2 aliases
5. Headers (2) - cpp2_inline.h dependency
6. Statement transforms (1) - Non-declaration statements

**Next actions:**
1. Add type alias pattern to YAML
2. Test on pure2-type-safety-1.cpp2 (multi-line main)
3. Verify pattern matching works on complex cases
4. Begin GraphNode implementation

---

## Key Insights

1. **Graph scanner wasn't broken** - it was bypassed by build system
2. **Pattern matching wasn't broken** - it was never connected
3. **Abstractions hide the graph** - 15 classes doing graph ops without admitting it
4. **Pattern matching = graph isomorphism** - once you see it, everything simplifies
5. **MLIR integration is natural** - GraphNode maps directly to regions/blocks/ops

The codebase had all the pieces but they were:
- Not connected (hardcoded "default")
- Not built (main_minimal.cpp instead of main.cpp)
- Not recognized as graph operations (abstraction overload)

GraphNode makes the graph explicit and eliminates impedance mismatch.
