# Product Guidelines

## Implementation Rules

### 1. Template-Based Canonical Types

- Use raw C++ templates for canonical semantic types, not constexpr factories
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

- `cppfort_parser.h` is the public API contract
- Must be 100% hand-written (no LLM-generated parser internals)
- Internal parser implementation lives in `src/parser.cpp` (archived to `old/`)
- New parser work goes in `selfhost/` as cpp2-native bootstrap

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
- ✅ Bootstrap tags: integer constants that build
- ❌ Canonical types: declared in .cpp2, not wired to transpilation
