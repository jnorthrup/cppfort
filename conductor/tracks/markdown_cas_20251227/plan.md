# Plan: Markdown Comments with CAS-Linked Module Stubs

## Phase 1: Lexer Support for Markdown Blocks

- [x] **Task:** Add MARKDOWN_BLOCK token type to lexer.
    - [x] **Sub-task:** Write tests for recognizing markdown block delimiters.
    - [x] **Sub-task:** Implement markdown block tokenization in lexer.
- [x] **Task:** Implement markdown block content extraction.
    - [x] **Sub-task:** Write tests for content extraction with various delimiters.
    - [x] **Sub-task:** Implement content capture between triple backticks.
- [ ] **Task:** Conductor - User Manual Verification 'Lexer Support for Markdown Blocks' (Protocol in workflow.md)

## Phase 2: SHA256 Computation

- [ ] **Task:** Implement trim-and-concatenate algorithm for markdown content.
    - [ ] **Sub-task:** Write tests for line trimming and concatenation.
    - [ ] **Sub-task:** Implement string processing algorithm.
- [ ] **Task:** Integrate SHA256 hashing for markdown blocks.
    - [ ] **Sub-task:** Write tests with known SHA256 test vectors.
    - [ ] **Sub-task:** Reuse existing SHA256 implementation from semantic_hash.cpp.
- [ ] **Task:** Conductor - User Manual Verification 'SHA256 Computation' (Protocol in workflow.md)

## Phase 3: AST Metadata Support

- [ ] **Task:** Define MarkdownBlockAttr structure in AST.
    - [ ] **Sub-task:** Write tests for metadata structure creation.
    - [ ] **Sub-task:** Add MarkdownBlockAttr to include/ast.hpp.
- [ ] **Task:** Attach markdown metadata to AST nodes during parsing.
    - [ ] **Sub-task:** Write tests for metadata attachment to declarations.
    - [ ] **Sub-task:** Implement parser integration with markdown blocks.
- [ ] **Task:** Conductor - User Manual Verification 'AST Metadata Support' (Protocol in workflow.md)

## Phase 4: Code Generation - Module Stubs

- [ ] **Task:** Implement empty C++20 module stub generation.
    - [ ] **Sub-task:** Write tests for module stub output format.
    - [ ] **Sub-task:** Implement module emission with export module directive.
- [ ] **Task:** Add SHA256 constant to generated modules.
    - [ ] **Sub-task:** Write tests for SHA256 constant emission.
    - [ ] **Sub-task:** Implement inline constexpr char array generation.
- [ ] **Task:** Integrate module stub emission into code generator.
    - [ ] **Sub-task:** Write integration tests with full Cpp2 files.
    - [ ] **Sub-task:** Connect AST metadata to code generation pipeline.
- [ ] **Task:** Conductor - User Manual Verification 'Code Generation - Module Stubs' (Protocol in workflow.md)

## Phase 5: Testing and Validation

- [ ] **Task:** Create comprehensive test suite for markdown blocks.
    - [ ] **Sub-task:** Write unit tests for lexer edge cases (empty blocks, Unicode).
    - [ ] **Sub-task:** Write unit tests for SHA256 computation edge cases.
    - [ ] **Sub-task:** Write integration tests with complete Cpp2 programs.
- [ ] **Task:** Verify generated modules compile with C++20 compiler.
    - [ ] **Sub-task:** Create build tests with g++/clang++.
    - [ ] **Sub-task:** Verify module imports work correctly.
- [ ] **Task:** Conductor - User Manual Verification 'Testing and Validation' (Protocol in workflow.md)
