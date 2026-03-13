# Product Guidelines

## Implementation Rules

### 1. Zero-Cost Abstractions via Front-End Sugar

**Core Principle**: Front-end sugar IS the abstraction mechanism. Surface syntax (operators, underscore patterns, manifold notation) is the PRIMARY user interface and must compile to zero-cost abstractions.

**Zero-Cost Definition**: 
- Type alias = free hoisted vtable in real-world front IR
- Surface syntax normalization produces optimal code with no abstraction overhead
- SoN optimization proves zero-cost through constant propagation and alias analysis

**Front-End Sugar as Core**:
- TrikeShed operators (`j`, `α`, `**`, `++`, `*[]`) are semantic primitives
- Manifold notation (`coords[...]`, `chart.project()`, `manifold.transition()`) is first-class
- Early normalization preserves semantic intent while enabling SoN optimization

**Canonical AST Design**:
- Small, repo-owned types: `indexed`, `series`, `tensor`, `dense_tensor`, `atlas`, `manifold`
- Templates normalize cleanly into SoN without source changes
- SoN/MLIR does the smashing at compile time

**DO:**
```cpp2
template<typename I, typename F>
class indexed { ... };
```

**DON'T:**
```cpp2
consteval auto make_indexed(...) { ... }  // No constexpr factories
```

### 2. Parser API Contract

- Legacy `cppfort_parser.h` / `src/parser.cpp` path has been removed
- Active parser bootstrap lives in `src/selfhost/` as pure cpp2
- Dogfood route is `cppfront`-boosted: transpile selfhost cpp2 into the build tree, then host-compile and run smoke coverage
- Must be 100% hand-written in parser logic (no LLM-generated parser internals)

### 3. Early Normalization

- Surface sugar (operators, underscore patterns) normalizes immediately
- Target is small canonical AST: `indexed`, `series`, `tensor`, `dense_tensor`, `atlas`, `manifold`
- Do not leak sugar into SoN pipeline

### 4. SoN/MLIR Integration

- Template instantiation becomes concrete SoN nodes
- All template parameters become constant attributes
- Constant propagation happens in MLIR passes, not source

### 5. Dense/Lowered Separation

- Semantic objects (`indexed`, `series`, `manifold`) stay high-level
- Dense views (`dense_tensor`, `memref`) are explicit lowering
- Never conflate the two in one type

### 6. Gradient Protocol

- Use C++ concepts for `grad_expr`, not inheritance hierarchies
- `grad_backend` is a protocol, not a class hierarchy
- AD lowering happens in MLIR passes

### 7. CAS Internment

- Constants deduplicated via linker sections, not runtime
- Follow Java classfile constant pool pattern
- Section name: `.cas_pool`

### 8. Manifold Guidance

- `manifold` means algebraic/process structure for compiler phases
- NOT: learned embeddings, token classification, statistical inference
- Charts, atlases, coordinates, transitions describe semantic routing

## Tooling and Standards

### Build Tools

- **Supported**: `cmake` and `ninja` as the primary build tools
- **Build target**: `selfhost_bootstrap_smoke` is the authoritative top-level build target
- **Verified build path**:
  ```bash
  cmake -S . -B build -G Ninja
  ninja -C build selfhost_bootstrap_smoke
  ```

### Contribution Standards

Based on [`CONTRIBUTING.md`](../CONTRIBUTING.md:1):

- **Contributor License Agreement**: Required for all contributions
  - Sign at [cla.developers.google.com](https://cla.developers.google.com/)
  - Retains copyright, grants permission to use and redistribute

- **Community Guidelines**: Follow [Google's Open Source Community Guidelines](https://opensource.google/conduct/)

- **Code Review Process**:
  - All submissions require review via GitHub pull requests
  - Project members also require review

### Tooling Options

- **Parser Combinators**: Complete implementation available
  - 121 grammar rules, 815 lines
  - Follows NarseseBbcursive pattern
  - Next: AST construction, error recovery, integration

- **Semantic Preservation**: Validated with 99% accuracy
  - Average semantic loss: 0.124 (target: < 0.15)
  - 189 regression tests
  - Production-grade fidelity confirmed

## File Organization

```
cppfort/
├── conductor/           # Track management
│   ├── product.md       # Product description
│   ├── product-guidelines.md  # Implementation rules
│   └── tracks/          # Track plans
├── selfhost/            # New cpp2-native bootstrap
│   ├── canonical_types.cpp2  # Template definitions
│   └── bootstrap_tags.cpp2   # Node tags
├── old/                 # Archived legacy code
├── include/             # MLIR dialect definitions
├── lib/                 # Pass implementations
└── CMakeLists.txt      # Build system
```

## Track Process

1. Read track `spec.md` and `plan.md`
2. Execute one small batch (3-5 tasks)
3. Validate with targeted tests
4. Update track `plan.md` with progress
5. Leave clear handoff notes

## Quality Gates

- Each new surface must rebuild via CMake
- Semantic identity must stay distinct from dense views
- No training or model-language concepts in accepted truth
- External spec informs design, but acceptance requires repo code/tests

## Honest Implementation Tracking

**DO NOT claim implementation exists until:**
- Source files with actual logic exist (not just headers)
- CMake target builds successfully
- Test validates the feature works end-to-end

**Current state (2026-03-12):**
- ❌ Parser: header-only API, no implementation
- ❌ MLIR SoN dialect: TableGen defined but disabled in build
- ❌ CAS internment: declared in headers, not implemented
- ✅ Bootstrap tags: integer constants that build (in `src/selfhost/bootstrap_tags.cpp2`)
- ❌ Canonical types: declared in `src/selfhost/canonical_types.cpp2`, not wired to transpilation
- ❌ Transpiler: `old/cppfort` binary missing (should exist per spec but doesn't)
