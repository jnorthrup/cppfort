# Spec: Markdown Comments with CAS-Linked Module Stubs

## 1. Overview

Implement support for markdown comments in the Cpp2 dialect that generate empty C++20 module stubs with embedded SHA256 hashes. The markdown blocks are extracted during lexing, and their SHA256 hashes are used for Content-Addressable Storage (CAS) linking and module identification.

This feature enables the cppfront dialect to produce CAS-aware module stubs that survive compilation while maintaining reproducibility through content-based hashing.

## 2. Key Features & Requirements

### 2.1 Markdown Block Syntax

Markdown comment blocks use triple-backtick syntax:

```cpp
/*
``` <optional-name>
<markdown content here>
```
*/
```

- The block starts with `/*` followed immediately by `` ``` `` and optional identifier
- The block ends with `` ``` `` followed immediately by `*/`
- Content between backticks is the markdown text
- Blocks can appear anywhere a comment can appear in Cpp2 source

### 2.2 Lex-Time Extraction

- **Token Recognition:** Lexer recognizes `` ```<optional-name>`` `...` `` ``` `` as a distinct token type (`MARKDOWN_BLOCK`)
- **Content Capture:** Extract the full markdown content between backticks
- **Location Tracking:** Track source location for error reporting
- **Nesting:** Markdown blocks do NOT nest (syntax error if `` ``` `` appears inside content)

### 2.3 SHA256 Computation

Algorithm for computing SHA256:
1. Split markdown content into lines
2. Trim whitespace from each line (both leading and trailing)
3. Concatenate all lines with single `\n` (LF, 0x0A) between each
4. Compute SHA256 hash of the resulting byte sequence
5. Encode as hexadecimal string (64 lowercase hex characters)

**Example:**
```
Input:
"""
  Hello world
  Foo bar
"""
Trimmed lines: ["Hello world", "Foo bar"]
Concatenated: "Hello world\nFoo bar\n"
SHA256: <hash of bytes>
```

### 2.4 AST Metadata Storage

- **Token to AST:** During parsing, attach `MarkdownBlock` metadata to AST nodes
- **Metadata Fields:**
  - `sha256`: Hex-encoded SHA256 string
  - `content`: Original markdown content
  - `name`: Optional identifier from opening `` ```<name>``
  - `location`: Source location
- **Node Association:** Metadata attached to nearest enclosing declaration or statement

### 2.5 Code Generation

**Generated Module Stub Format:**

```cpp
export module <name>;

inline constexpr char cas_sha256[] = "<64-char-hex-hash>";
```

**Where:**
- `<name>` is derived from the optional identifier or generated as `__cas_<hash>`
- Module is otherwise empty (no other declarations)
- SHA256 constant is named `cas_sha256` for CAS linking

**Module Export:**
- Generated in same translation unit as the source file
- Emitted after all other code generation
- Module name added to global module import map

### 2.6 CAS Linking

- **Hash as Key:** SHA256 serves as content-based key for CAS lookups
- **Module Import:** Other modules can import via SHA256 for reproducible builds
- **Deduplication:** Identical markdown blocks across files generate same module reference
- **Incremental Builds:** Unchanged content means unchanged SHA256, enables cache hits

## 3. Technical Implementation

### 3.1 Lexer Changes

```cpp
// Token type enum
enum TokenType {
  // ... existing tokens
  MARKDOWN_BLOCK,
};

// Token value
struct Token {
  TokenType type;
  std::string value;  // For MARKDOWN_BLOCK: full markdown content
  SourceLocation location;
};
```

**Lexer State Machine:**
1. On `/*`, check if next characters are `` ``` ``
2. If yes, enter MARKDOWN_BLOCK state
3. Read until `` ``` `` followed by `*/`
4. Emit `MARKDOWN_BLOCK` token with content

### 3.2 Parser Changes

```cpp
// AST node attribute
struct MarkdownBlockAttr {
  std::string sha256;
  std::string content;
  std::string name;  // optional
  SourceLocation location;
};

// Attached to declarations
class Declaration {
  // ... existing fields
  std::vector<MarkdownBlockAttr> markdown_blocks;
};
```

### 3.3 SHA256 Implementation

Use existing SHA256 from `include/semantic_hash.hpp`:
- `SHA256::hash(const std::string& input)` returns hex-encoded hash
- Reuse pure C++ SHA256 implementation (no OpenSSL dependency)

### 3.4 Code Generator Changes

```cpp
// For each markdown block in AST:
void emitMarkdownModule(const MarkdownBlockAttr& block) {
  std::string module_name = block.name.empty()
    ? "__cas_" + block.sha256.substr(0, 16)
    : block.name;

  out << "export module " << module_name << ";\n\n";
  out << "inline constexpr char cas_sha256[] = \"" << block.sha256 << "\";\n";
}
```

## 4. Acceptance Criteria

### 4.1 Functional Requirements
- [ ] Lexer recognizes `` ```...``` `` blocks as `MARKDOWN_BLOCK` tokens
- [ ] SHA256 computed correctly using trim-and-concatenate algorithm
- [ ] Parser attaches metadata to AST nodes
- [ ] Code generator emits empty module stubs with SHA256 constant
- [ ] Generated modules compile with C++20 compiler

### 4.2 Testing Requirements
- [ ] Unit tests for lexer markdown block recognition
- [ ] Unit tests for SHA256 computation with known test vectors
- [ ] Unit tests for parser metadata attachment
- [ ] Integration tests for code generation output
- [ ] Tests for edge cases (empty blocks, special characters, Unicode)

### 4.3 Code Quality
- [ ] Code coverage >20% for markdown block handling
- [ ] Follows existing C++ coding style
- [ ] Documentation for public APIs

## 5. Out of Scope

- Markdown content processing/rendering (content is opaque string)
- Markdown validation (any content is valid)
- Nested markdown blocks (syntax error)
- CAS resolution/linking at runtime (compile-time only)

## 6. Deliverables

1. **Lexer Support:** `src/lexer.cpp` with `MARKDOWN_BLOCK` token handling
2. **Parser Support:** `include/ast.hpp` with `MarkdownBlockAttr` metadata
3. **Code Generator:** `src/code_generator.cpp` with module stub emission
4. **Test Suite:** `tests/test_markdown_blocks.cpp` with comprehensive tests
5. **SHA256 Reuse:** Integration with existing `src/semantic_hash.cpp`
