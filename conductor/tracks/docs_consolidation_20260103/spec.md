# Specification: EBNF-to-Combinator Documentation Consolidation

## Overview

Consolidate scattered documentation into `conductor/` to establish a single backbone: **EBNF grammar → combinator parser/lexer mappings → cppfront AST isomorph normalization → semantic loss scoring**.

## Problem Statement

Documentation drift has created inconsistent sources of truth across:
- Root-level status files
- `docs/` with overlapping specifications
- `grammar/` with combinator definitions
- Scattered AST mapping and corpus infrastructure docs

This makes it difficult to understand how EBNF grammar drives parser combinator orchestration.

## Functional Requirements

### 1. Create `conductor/PARSER_ORCHESTRATION.md` (Single Backbone)

The unified document must contain five sections:

| Section | Content | Source |
|---------|---------|--------|
| **EBNF Design** | Formal EBNF grammar specification for Cpp2 | `grammar/cpp2.combinators.md`, `docs/CPP2_GRAMMAR.md` |
| **Combinator Mappings** | How each EBNF rule maps to parser combinators (lexers + parsers) | `docs/COMBINATORS.md`, `docs/COMBINATOR_VERIFICATION.md` |
| **EBNF Standard Tracking** | Versioned EBNF standard as cppfort source of truth | `grammar/cpp2.combinators.md` |
| **AST Isomorph Normalization** | How cppfront AST dumps normalize to isomorphic patterns | `docs/AST_ISOMORPH_MAPPING_SPEC.md`, `docs/AST_MAPPING_STATUS.md` |
| **Semantic Loss Scoring** | Idempotent scoring against isomorphs | `docs/CORPUS_INFRASTRUCTURE_STATUS.md`, `docs/REGRESSION_TESTING.md` |

### 2. Source Files to Consolidate and Delete

| File | Action |
|------|--------|
| `grammar/cpp2.combinators.md` | Merge → delete |
| `docs/CPP2_GRAMMAR.md` | Merge → delete |
| `docs/COMBINATORS.md` | Merge → delete |
| `docs/COMBINATOR_VERIFICATION.md` | Merge → delete |
| `docs/AST_ISOMORPH_MAPPING_SPEC.md` | Merge → delete |
| `docs/AST_MAPPING_STATUS.md` | Merge → delete |
| `docs/CORPUS_INFRASTRUCTURE_STATUS.md` | Merge → delete |
| `docs/REGRESSION_TESTING.md` | Merge → delete |
| `docs/ARCHITECTURE.md` | Delete (outdated) |
| `docs/MAPPING_SPEC.md` | Delete (superseded) |
| `docs/MAPPING_TASK.md` | Delete (superseded) |
| `docs/CRDT_SEMANTIC_MAPPING.md` | Delete (superseded) |
| `CPP2_FEATURES_STATUS.md` | Consolidate info → delete |
| `IMPLEMENTATION_STATUS.md` | Consolidate info → delete |
| `TEST_SUMMARY.md` | Consolidate info → delete |
| `regression_analysis.md` | Consolidate info → delete |
| `regression_summary.md` | Consolidate info → delete |
| `corpus_scan_results.txt` | Preserve (data file) |

### 3. Preserve Unchanged

| Directory | Reason |
|-----------|--------|
| `docs/cpp2/` | Cpp2 language reference (external) |
| `docs/cppfront/` | Cppfront reference (external) |
| `docs/sea-of-nodes/` | Sea-of-Nodes reference (external) |
| `docs/Simple/` | Simple IR reference (external) |
| `conductor/tracks/` | Active track structure |
| **MLIR/SON docs** | Protected for `cpp2_mlir_son_20251222` track |
| **AST dump files** | Idempotent dumps excluded due to repository bloat |

### 4. Root Directory Final State

After consolidation, root contains only:
- `README.md`

## Non-Functional Requirements

- **DRY**: Zero duplication between PARSER_ORCHESTRATION.md and any other doc
- **Idempotent**: AST dumps produce same normalized output for same source
- **Deterministic**: Loss scoring is repeatable

## Acceptance Criteria

1. `conductor/PARSER_ORCHESTRATION.md` exists as single source of truth
2. All files in section (2) are deleted
3. `docs/` contains only: `cpp2/`, `cppfront/`, `sea-of-nodes/`, `Simple/`
4. Root contains only: `README.md`, plus code directories
5. Zero content duplication
6. All internal links resolve
7. `conductor/product.md` references updated to new location
8. **MLIR/SON documentation untouched**

## Out of Scope

- Modifying `docs/cpp2/`, `docs/cppfront/`, `docs/sea-of-nodes/`, `docs/Simple/`
- Altering track structure (`conductor/tracks/*/spec.md`, `plan.md`)
- Changing third-party documentation
- Archiving (no archive directory - consolidate or delete only)
