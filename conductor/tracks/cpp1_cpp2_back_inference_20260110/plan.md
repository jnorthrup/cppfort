# Track: Back-Inferring C++1 to C++2 Semantic Equivalence

**Objective**: Leverage the existing `cppfort` codebase capabilities to back-infer C++1 to C++2 semantic equivalence, aiming for a semantic loss target of < 0.15. This involves triangulating between C++2 source, C++1 original code (via Clang), and C++front transpiled C++1 code.

## Available Infrastructure

### 1. **Clang AST Analysis Tools**

| Tool | Purpose | Key Features |
|------|---------|--------------|
| [`compare_ast_dumps.py`](../../tools/compare_ast_dumps.py) | Direct AST comparison | Parses clang AST dumps, computes structural similarity and semantic loss |
| [`extract_ast_isomorphs.py`](../../tools/extract_ast_isomorphs.py) | Pattern extraction | Extracts canonical graph isomorphs, computes signatures for isomorphism detection |
| [`refine_cpp2_ast_from_clang.py`](../../tools/refine_cpp2_ast_from_clang.py) | AST refinement | Refines C++2 AST from Clang output |

### 2. **Semantic Equivalence Metrics**

The project uses a multi-dimensional semantic loss metric (from [`SEMANTIC_PRESERVATION_REPORT.md`](../../SEMANTIC_PRESERVATION_REPORT.md)):

- **Structural Distance (50%)**: Graph edit distance of normalized AST patterns
- **Type Distance (30%)**: Type inference mismatches
- **Operation Distance (20%)**: MLIR region classification differences

### 3. **AST Structure Capabilities**

From [`slim_ast.hpp`](../../include/slim_ast.hpp), the project has:

- **Arena-based parse tree** - No heap allocation per node
- **Left-Child Right-Sibling (LCRS) topology** - Efficient tree traversal
- **NodeKind enum** - 70+ node types covering expressions, statements, types, contracts, patterns
- **Thread-local TreeBuilder** - For incremental parsing with checkpoint/restore

### 4. **MLIR/Sea of Nodes Integration**

Available tools include:
- [`tag_mlir_regions.py`](../../tools/tag_mlir_regions.py) - Tags MLIR regions in AST dumps
- [`batch_tag_mlir_regions.sh`](../../tools/batch_tag_mlir_regions.sh) - Batch processing for MLIR tagging
- [`build_isomorph_database.py`](../../tools/build_isomorph_database.py) - Builds database of AST patterns

## Implementation Workflow

### Phase 1: Reference & Candidate Generation
1.  **Generate Reference AST**: Use `cppfront` to transpile C++2 source to C++1, then dump its AST via Clang.
2.  **Generate Candidate AST**: Parse the original C++1 code (or candidate back-inference output) using Clang to generate its AST.

### Phase 2: Structural & Semantic Extraction
1.  **Extract Isomorphs**: Run [`extract_ast_isomorphs.py`](../../tools/extract_ast_isomorphs.py) on both ASTs to get canonical graph structures and SHA256 signatures.
2.  **Tag MLIR Regions**: Use [`tag_mlir_regions.py`](../../tools/tag_mlir_regions.py) to capture operation semantics and control flow patterns (if/while/for).

### Phase 3: Comparison & Scoring
1.  **Compare**: Use [`compare_ast_dumps.py`](../../tools/compare_ast_dumps.py) for structural analysis side-by-side.
2.  **Score**: Run [`score_semantic_loss.py`](../../tools/score_semantic_loss.py) to calculate the composite semantic loss.

### Phase 4: Batch Validation
-   Use [`batch_compare_ast_dumps.sh`](../../tools/batch_compare_ast_dumps.sh) and [`score_corpus_semantics.sh`](../../tools/score_corpus_semantics.sh) to validate across the entire corpus.
-   **Target**: Achieve average semantic loss < 0.15 (Current baseline: ~0.124).
-   **Goal**: Achieve nearly single semantic weights (identical isomorph signatures) for both, implying perfect isomorphism.

## Key Capabilities for Back-Inference

### 1. **Structural Isomorphism Detection**
- Canonical signature computation via SHA256 hashing of node kind sequences
- Graph edit distance approximation
- Pattern matching by node kind (FunctionDecl, IfStmt, WhileStmt, CallExpr, etc.)

### 2. **Type Equivalence Verification**
- Type attribute comparison in matched isomorphs
- Structure-level type inference validation
- Template argument matching

### 3. **Operation Semantics**
- MLIR region classification comparison
- Control flow pattern extraction (if, while, for, do)
- Expression pattern matching (calls, operators)
