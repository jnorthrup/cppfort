# TrikeShed Signals - Clean Architectural Extraction

## Source
Extracted from `/Users/jim/work/TrikeShed/conductor/grok_share_bGVnYWN5_21edd44f-9e25-434b-9bcb-2d036feee2dc.md`
- Original: 2274-line transcript with chat noise, code dumps, and repetitive explanations
- Cleaned: Core architectural signals only

## Core Architectural Principles (Lines 7-12)

### 1. Semantic Objects First, Dense Lowered Views Second
- Canonical types represent mathematical/algorithmic intent
- Separate representation for optimized runtime data
- Never conflate semantic and dense layers in one type

### 2. TrikeShed Notation as Front-End Sugar Only
- Operators and underscore patterns are notation only
- Normalize early into a small canonical AST
- Front-end sugar IS the abstraction mechanism (zero-cost)

### 3. Sea-of-Nodes + Constant Propagation as Semantic Engine
- NOT surface-language identity
- SoN does the real type/effect recovery
- AA/Simple discipline for optimization

### 4. Temporary cppfront Ride Only
- Bootstrap compatibility only
- Removable the second native parser lands
- No long-term dependency

### 5. Templates Over Constexpr (User Preference Confirmed)
- Raw generics stay in source
- Compiler smashes to constants via SoN
- No constexpr factories or reflection gymnastics

### 6. 100% Hand-Written Parser
- No LLM-generated parser internals
- Active parser logic lives in `src/selfhost/rbcursive.cpp2`
- Current dogfood path is `cppfront`-boosted selfhost smoke

## Front-End Sugar as Zero-Cost Abstraction Core

**User Clarification**: "that front-end sugar should be the core of compositional compiler zero cost abstractions"

**Interpretation**:
1. Surface syntax (TrikeShed operators, manifold notation) IS the primary abstraction mechanism
2. Compiler's job is to normalize to canonical forms while preserving semantic intent
3. Zero-cost means optimal code with no abstraction overhead (type alias = free hoisted vtable)
4. SoN optimization proves zero-cost through constant propagation and alias analysis

## Manifold Applicability to SoN Compilation

**Unique Challenge**: Lifecycle memory management without the maths

**Manifold Guidance**:
- Charts, atlases, coordinates, transitions describe semantic routing
- Guides normalization and lowering phases
- NOT learned embeddings or statistical inference

**SoN Integration**:
- Manifold types lower to SoN operations
- Lifecycle analysis happens in SoN passes
- Unique challenge: managing memory across chart transitions

## I/O Strategy Decision

**User Question**: "does stdio or mmap between build stages with channelized reactor like io gonna help or hinder clarity?"

**Decision**: Use stdio for clarity
- Simple stdio between build stages
- Memory-mapped I/O only for large intermediate representations
- Channelized reactor adds complexity for marginal benefit

## cppfort Integration Strategy

**TrikeShed as Source of Truth**:
- Treat TrikeShed Kotlin as reference surface
- Selectively transpile to C++26 equivalents
- Minimize gaps from most-primitive C++26 (ratified or aspirational)

**cppfront Role**:
- Temporary benchmark/validator only
- No cppfront linking, calling, or inclusion in build
- Removable once native parser lands

## Code Dumps for Debate

The transcript contains large code dumps that should be debated between SOTA math LLM and final judge:

### Reference Implementations (For Debate)
1. **SoNConstantProp.cpp** - Template parameter folding, dead code elimination
2. **GradADLowering.cpp** - Protocol-to-arithmetic lowering for AD
3. **JacobianMatrixMulLowering.cpp** - Manifold chain rule to fused multiply-add
4. **Lowering hooks** - `lower_canonical_to_son` implementation

### Debate Questions
1. Should these passes be implemented as MLIR passes or custom SoN transformations?
2. Is the constant propagation strategy optimal for manifold types?
3. Can lifecycle analysis be integrated into existing SoN infrastructure?

## Normalization Flow

```
TrikeShed Surface Syntax (front-end sugar)
    ↓ (early normalization - semantic preservation)
Canonical AST (small, repo-owned, zero-cost)
    ↓ (Sea-of-Nodes + constant propagation)
Optimized IR (zero-cost abstractions proven)
    ↓ (MLIR lowering)
Target Code (optimal, no abstraction overhead)
```

## Immediate Action Items

1. **Implement parser** with TrikeShed sugar support
2. **Wire canonical types** to build system
3. **Implement one SoN pass** (SoNConstantProp or GradADLowering)
4. **Create manifold SoN integration** - wire guidance to compilation phases
5. **Verify zero-cost**: Prove surface syntax compiles to optimal code

## Verification Criteria

- Surface syntax parses to canonical AST
- Canonical AST lowers to SoN without semantic loss
- SoN optimization produces zero-cost abstractions
- Manifold guidance affects compilation phases
- Lifecycle memory management works across chart transitions
