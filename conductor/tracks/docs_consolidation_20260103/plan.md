# Plan: EBNF-to-Combinator Documentation Consolidation

## Phase 1: Inventory and Analysis

- [x] Task: Catalog all documentation files to consolidate
  - [x] List all `*.md` files in root, `docs/`, `grammar/`
  - [x] Identify content overlap and duplications
  - [x] Map content sections to destination in PARSER_ORCHESTRATION.md
  - [x] **PROTECT**: All MLIR dialect docs (preserve for cpp2_mlir_son track)
  - [x] **PROTECT**: All Sea-of-Nodes docs (preserve for future SON work)
  - [x] **IGNORE**: AST dump files (idempotent dumps cause repository bloat)

## Phase 2: Create PARSER_ORCHESTRATION.md

- [x] Task: Create unified document structure
  - [x] Section 1: EBNF Design (formal grammar)
  - [x] Section 2: Combinator Mappings (EBNF → parser combinators)
  - [x] Section 3: EBNF Standard Tracking (versioning)
  - [x] Section 4: AST Isomorph Normalization
  - [x] Section 5: Semantic Loss Scoring
- [x] Task: Migrate content from source files
  - [x] Merge `grammar/cpp2.combinators.md` content
  - [x] Merge `docs/CPP2_GRAMMAR.md` content
  - [x] Merge `docs/COMBINATORS.md` content
  - [x] Merge `docs/COMBINATOR_VERIFICATION.md` content
  - [x] Merge `docs/AST_ISOMORPH_MAPPING_SPEC.md` content
  - [x] Merge `docs/AST_MAPPING_STATUS.md` content
  - [x] Merge `docs/CORPUS_INFRASTRUCTURE_STATUS.md` content
  - [x] Merge `docs/REGRESSION_TESTING.md` content
  - [x] Merge root-level status files content
- [x] Task: Verify internal links and structure
  - [x] Add table of contents
  - [x] Add internal section links
  - [x] Verify all links resolve

## Phase 3: Cleanup

- [x] Task: Delete source files after consolidation
  - [x] Delete `grammar/cpp2.combinators.md`
  - [x] Delete `docs/CPP2_GRAMMAR.md`
  - [x] Delete `docs/COMBINATORS.md`
  - [x] Delete `docs/COMBINATOR_VERIFICATION.md`
  - [x] Delete `docs/AST_ISOMORPH_MAPPING_SPEC.md`
  - [x] Delete `docs/AST_MAPPING_STATUS.md`
  - [x] Delete `docs/CORPUS_INFRASTRUCTURE_STATUS.md`
  - [x] Delete `docs/REGRESSION_TESTING.md`
  - [x] Delete `docs/ARCHITECTURE.md` (outdated)
  - [x] Delete `docs/MAPPING_SPEC.md` (superseded)
  - [x] Delete `docs/MAPPING_TASK.md` (superseded)
  - [x] Delete `docs/CRDT_SEMANTIC_MAPPING.md` (superseded)
  - [x] Delete `CPP2_FEATURES_STATUS.md` (info consolidated)
  - [x] Delete `IMPLEMENTATION_STATUS.md` (info consolidated)
  - [x] Delete `TEST_SUMMARY.md` (info consolidated)
  - [x] Delete `regression_analysis.md` (info consolidated)
  - [x] Delete `regression_summary.md` (info consolidated)
  - [x] **PRESERVE**: `grammar/cpp2.ebnf` (canonical EBNF)
  - [x] **PRESERVE**: All MLIR and SON related documentation in place

## Phase 4: Verification

- [x] Task: Verify consolidation
  - [x] Confirm `docs/` contains only: `cpp2/`, `cppfront/`, `sea-of-nodes/`, `Simple/`
  - [x] Confirm root contains only: `README.md` (plus code directories)
  - [x] Verify all links in PARSER_ORCHESTRATION.md resolve
  - [x] Update `conductor/product.md` documentation references
  - [x] **VERIFY**: No MLIR or SON documentation was disturbed
  - [x] **VERIFY**: Zero content duplication remains
  - [x] Deleted `docs/CORPUS_BUILDER.md` (content merged into PARSER_ORCHESTRATION.md)
