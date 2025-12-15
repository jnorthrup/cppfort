# Clang AST Graph Isomorph → MLIR Region Mapping Specification

**Objective**: Capture Clang AST graph isomorphisms that correspond to normalized MLIR base-dialect region expressions, using hsutter/cppfront as the reference transpiler for semantic loss scoring.

## Overview

This specification defines a methodology for:

1. **Extracting AST graph isomorphs** from Clang's AST representation of C++ code
2. **Mapping isomorphs to MLIR regions** using normalized base-dialect expressions
3. **Scoring semantic loss** by comparing cppfort output against cppfront reference

## Definitions

### Clang AST Graph Isomorph

A **graph isomorph** is a structural pattern in the Clang AST that has a canonical mapping to an MLIR region. Key properties:

- **Structural equivalence**: Two AST subgraphs are isomorphic if they have the same node types and edge relationships
- **Semantic preservation**: Isomorphic graphs represent the same computation
- **Normalization**: Multiple C++ syntactic forms may map to the same isomorph

**Examples**:

```
AST Isomorph 1: Function with single return
  FunctionDecl
  └─CompoundStmt
    ├─VarDecl (initialization)
    └─ReturnStmt
      └─DeclRefExpr (variable reference)

Normalized MLIR Region:
  cpp2.func {
    %var = cpp2.var ...
    cpp2.return %var
  }
```

```
AST Isomorph 2: If-else branch
  IfStmt
  ├─BinaryOperator (condition)
  ├─CompoundStmt (then branch)
  │ └─ReturnStmt
  └─CompoundStmt (else branch)
    └─ReturnStmt

Normalized MLIR Region:
  cpp2.if %cond {
    cpp2.return %then_val
  } else {
    cpp2.return %else_val
  }
```

### MLIR Base-Dialect Region Expression

A **region expression** in MLIR is a sequence of operations bounded by a scope. For Cpp2Dialect:

- **Region delimiters**: Entry block, terminator operation
- **SSA form**: Single static assignment for all values
- **Normalized representation**: Canonical form regardless of source syntax

**Base dialect operations used**:
- Control flow: `cpp2.start`, `cpp2.if`, `cpp2.region`, `cpp2.loop`, `cpp2.return`
- Data flow: `cpp2.constant`, `cpp2.phi`, `cpp2.add`, `cpp2.sub`, `cpp2.mul`, `cpp2.div`
- Memory: `cpp2.new`, `cpp2.load`, `cpp2.store`

### Semantic Loss Metric

**Semantic loss** measures divergence between cppfort and cppfront transpiler outputs:

$$
\text{Loss} = \frac{\sum_{i=1}^{n} \text{distance}(\text{AST}_{\text{cppfort}}^i, \text{AST}_{\text{cppfront}}^i)}{n}
$$

Where `distance` is defined by:
- **Structural distance**: Edit distance between AST graphs
- **Type distance**: Mismatches in inferred types
- **Operation distance**: Differences in generated operations

**Loss categories**:
- **Zero loss (0.0)**: Identical AST structure and semantics
- **Low loss (<0.1)**: Minor syntactic differences (whitespace, const placement)
- **Medium loss (0.1-0.5)**: Different code generation strategies (same semantics)
- **High loss (>0.5)**: Semantic differences (incorrect translation)

## Reference Transpiler: hsutter/cppfront

**cppfront** serves as the ground truth for Cpp2 → C++1 transpilation.

### Role

1. **Golden outputs**: cppfront's C++1 output is the reference implementation
2. **AST baseline**: Clang AST from cppfront output defines canonical isomorphs
3. **Validation source**: Our transpiler (cppfort) is scored against cppfront

### Workflow

```
Cpp2 source file (e.g., mixed-hello.cpp2)
    │
    ├─→ cppfront transpiler
    │   └─→ reference.cpp (C++1 output)
    │       └─→ clang++ -ast-dump
    │           └─→ reference.ast.txt (GOLDEN AST)
    │               └─→ Extract graph isomorphs
    │                   └─→ Tag with MLIR region patterns
    │
    └─→ cppfort transpiler
        └─→ candidate.cpp (C++1 output)
            └─→ clang++ -ast-dump
                └─→ candidate.ast.txt
                    └─→ Extract graph isomorphs
                        └─→ Compare against reference
                            └─→ Calculate semantic loss
```

## Corpus Processing Pipeline

### Phase 1: Reference Generation

For each `.cpp2` file in `corpus/inputs/`:

1. **Transpile with cppfront**:
   ```bash
   cppfront input.cpp2 -o corpus/reference/input.cpp
   ```

2. **Generate Clang AST**:
   ```bash
   clang++ -std=c++20 -Xclang -ast-dump -fsyntax-only \
     corpus/reference/input.cpp > corpus/reference/input.ast.txt
   ```

3. **Extract isomorphs**:
   ```bash
   ./tools/extract_ast_isomorphs.py \
     --ast corpus/reference/input.ast.txt \
     --output corpus/isomorphs/input.isomorph.json
   ```

4. **Tag with MLIR patterns**:
   ```bash
   ./tools/tag_mlir_regions.py \
     --isomorphs corpus/isomorphs/input.isomorph.json \
     --dialect include/Cpp2Dialect.td \
     --output corpus/tagged/input.tagged.json
   ```

### Phase 2: Candidate Evaluation

For each `.cpp2` file:

1. **Transpile with cppfort**:
   ```bash
   ./build/src/cppfort input.cpp2 output.cpp
   ```

2. **Generate Clang AST**:
   ```bash
   clang++ -std=c++20 -Xclang -ast-dump -fsyntax-only \
     output.cpp > corpus/candidate/input.ast.txt
   ```

3. **Extract isomorphs**:
   ```bash
   ./tools/extract_ast_isomorphs.py \
     --ast corpus/candidate/input.ast.txt \
     --output corpus/isomorphs/input.candidate.json
   ```

4. **Score semantic loss**:
   ```bash
   ./tools/score_semantic_loss.py \
     --reference corpus/tagged/input.tagged.json \
     --candidate corpus/isomorphs/input.candidate.json \
     --output corpus/scores/input.loss.json
   ```

### Phase 3: Aggregation

```bash
./tools/aggregate_corpus_scores.py \
  --scores-dir corpus/scores/ \
  --output corpus/CORPUS_LOSS_REPORT.md
```

## Isomorph Extraction Algorithm

### Step 1: Parse Clang AST

```python
def parse_ast(ast_text: str) -> ASTNode:
    """Parse Clang AST text dump into tree structure."""
    # Parse line-by-line, build tree from indentation
    # Each line: "NodeType <location> attributes"
    pass
```

### Step 2: Identify Subgraph Patterns

```python
def extract_subgraphs(ast: ASTNode) -> List[Subgraph]:
    """Extract all maximal connected subgraphs."""
    subgraphs = []

    # Pattern 1: Function definitions
    for func in ast.find_all("FunctionDecl"):
        subgraphs.append(extract_function_subgraph(func))

    # Pattern 2: Control flow structures
    for stmt in ast.find_all(["IfStmt", "WhileStmt", "ForStmt"]):
        subgraphs.append(extract_control_flow_subgraph(stmt))

    # Pattern 3: Expression trees
    for expr in ast.find_all(["BinaryOperator", "CallExpr"]):
        subgraphs.append(extract_expression_subgraph(expr))

    return subgraphs
```

### Step 3: Compute Graph Signature

```python
def compute_signature(subgraph: Subgraph) -> str:
    """Compute canonical signature for isomorphism checking."""
    # Signature = hash of (node_types, edge_structure, semantic_attributes)

    nodes = sorted([n.kind for n in subgraph.nodes])
    edges = sorted([(e.src.kind, e.dst.kind) for e in subgraph.edges])

    signature = f"{':'.join(nodes)}|{':'.join(map(str, edges))}"
    return hashlib.sha256(signature.encode()).hexdigest()
```

### Step 4: Normalize Isomorphs

```python
def normalize_isomorph(subgraph: Subgraph) -> NormalizedIsomorph:
    """Convert subgraph to normalized form."""

    # Normalize variable names (α-renaming)
    rename_map = create_alpha_renaming(subgraph)

    # Normalize type annotations
    types = normalize_types(subgraph)

    # Normalize control flow structure
    cfg = build_control_flow_graph(subgraph)
    normalized_cfg = normalize_cfg(cfg)

    return NormalizedIsomorph(
        signature=compute_signature(subgraph),
        nodes=normalize_nodes(subgraph.nodes, rename_map),
        edges=normalize_edges(subgraph.edges, rename_map),
        types=types,
        cfg=normalized_cfg
    )
```

## MLIR Region Tagging

### Tagging Schema

```json
{
  "isomorph_id": "sha256:abc123...",
  "ast_pattern": {
    "root_kind": "FunctionDecl",
    "structure": ["CompoundStmt", "VarDecl", "ReturnStmt"],
    "signature": "func:1ret"
  },
  "mlir_region": {
    "dialect": "cpp2",
    "operations": [
      {"op": "cpp2.func", "role": "entry"},
      {"op": "cpp2.var", "role": "data"},
      {"op": "cpp2.return", "role": "terminator"}
    ],
    "region_type": "function_body",
    "ssa_form": true
  },
  "confidence": 0.95,
  "examples": [
    {
      "cpp2_snippet": "name: () -> int = { x := 42; return x; }",
      "ast_location": "file.cpp:5:1",
      "mlir_template": "cpp2.func @name() -> i32 { %x = cpp2.constant 42; cpp2.return %x }"
    }
  ]
}
```

### Tagging Rules

| AST Pattern | MLIR Region Template | Confidence |
|-------------|---------------------|------------|
| `FunctionDecl` → `CompoundStmt` | `cpp2.func { ... }` | 1.0 |
| `IfStmt` with both branches | `cpp2.if { ... } else { ... }` | 1.0 |
| `WhileStmt` | `cpp2.loop { cpp2.if %cond { ... } else { cpp2.break } }` | 0.9 |
| `ForStmt` (range-based) | `cpp2.loop { ... }` | 0.85 |
| `ReturnStmt` | `cpp2.return` | 1.0 |
| `VarDecl` with init | `cpp2.var` | 1.0 |
| `CallExpr` (free function) | `cpp2.call` or `cpp2.ufcs_call` | 0.9 |
| `BinaryOperator` (+/-/*//) | `cpp2.add` / `cpp2.sub` / `cpp2.mul` / `cpp2.div` | 1.0 |

## Semantic Loss Scoring

### Structural Distance

**Edit distance** between AST graphs:

```python
def structural_distance(ref_ast: ASTGraph, cand_ast: ASTGraph) -> float:
    """Compute normalized graph edit distance."""

    # Operations: insert node, delete node, relabel node
    operations = compute_edit_script(ref_ast, cand_ast)

    max_nodes = max(len(ref_ast.nodes), len(cand_ast.nodes))
    distance = len(operations) / max_nodes

    return distance
```

### Type Distance

**Type mismatches** between reference and candidate:

```python
def type_distance(ref_types: Dict[str, Type], cand_types: Dict[str, Type]) -> float:
    """Measure type inference divergence."""

    mismatches = 0
    for var, ref_type in ref_types.items():
        cand_type = cand_types.get(var)
        if not cand_type or not types_equivalent(ref_type, cand_type):
            mismatches += 1

    return mismatches / len(ref_types)
```

### Operation Distance

**Differences in generated operations**:

```python
def operation_distance(ref_ops: List[Op], cand_ops: List[Op]) -> float:
    """Measure operation sequence divergence."""

    # Align operation sequences
    alignment = align_sequences(ref_ops, cand_ops)

    differences = 0
    for ref_op, cand_op in alignment:
        if ref_op != cand_op:
            differences += 1

    return differences / len(alignment)
```

### Combined Loss

```python
def semantic_loss(ref: Reference, cand: Candidate) -> float:
    """Compute overall semantic loss."""

    structural = structural_distance(ref.ast, cand.ast)
    type_diff = type_distance(ref.types, cand.types)
    op_diff = operation_distance(ref.ops, cand.ops)

    # Weighted combination
    loss = 0.5 * structural + 0.3 * type_diff + 0.2 * op_diff

    return loss
```

## Output Formats

### Isomorph Database

`corpus/isomorphs/database.json`:

```json
{
  "version": "1.0",
  "corpus_size": 195,
  "total_isomorphs": 1247,
  "isomorphs": [
    {
      "id": "iso_001",
      "signature": "sha256:...",
      "occurrence_count": 42,
      "files": ["mixed-hello.cpp2", "..."],
      "ast_pattern": { ... },
      "mlir_region": { ... }
    }
  ]
}
```

### Loss Report

`corpus/scores/CORPUS_LOSS_REPORT.md`:

```markdown
# Semantic Loss Report: cppfort vs cppfront

**Total files**: 195
**Average loss**: 0.12
**Median loss**: 0.08
**Zero-loss files**: 87 (44.6%)

## Loss Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 0.0 (perfect) | 87 | 44.6% |
| 0.0-0.1 (low) | 73 | 37.4% |
| 0.1-0.5 (medium) | 28 | 14.4% |
| >0.5 (high) | 7 | 3.6% |

## High-Loss Files (>0.5)

| File | Loss | Primary Issue |
|------|------|---------------|
| pure2-inspect-generic.cpp2 | 0.72 | Pattern matching not implemented |
| pure2-metafunction.cpp2 | 0.68 | Template expansion differs |
...
```

## Tools to Build

1. **`tools/extract_ast_isomorphs.py`**: Extract subgraphs from Clang AST
2. **`tools/tag_mlir_regions.py`**: Tag isomorphs with MLIR patterns
3. **`tools/score_semantic_loss.py`**: Compare candidate against reference
4. **`tools/aggregate_corpus_scores.py`**: Generate corpus-wide report
5. **`tools/build_cppfront.sh`**: Build hsutter/cppfront from source

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Average corpus loss | <0.15 | TBD |
| Zero-loss files | >40% | TBD |
| High-loss files | <5% | TBD |
| Isomorphs extracted | >1000 | TBD |
| MLIR region coverage | 100% of dialect ops | TBD |

## References

- **Graph Isomorphism**: VF2 algorithm (Cordella et al., 2004)
- **Edit Distance**: Zhang-Shasha tree edit distance
- **MLIR Regions**: https://mlir.llvm.org/docs/LangRef/#regions
- **Cpp2 Spec**: https://github.com/hsutter/cppfront

---

**Status**: Specification complete, tooling in development
