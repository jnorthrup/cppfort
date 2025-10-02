# Orbit Scanner Architecture - Jumpstart Reference

**Source:** Extracted from `ororoboros-couchduck` project (Sept 2024)
**Purpose:** Reference implementation patterns for Story 1.1 (Rabin-Karp Orbit Scanner)

---

## 1. Core Data Structures

### OrbitResult
```cpp
struct OrbitResult {
    u32 orbit_id;                    // Protocol/language identifier
    u64 pattern_mask;                // Bitmask of detected patterns (64-bit)
    double confidence;               // Detection confidence (0.0-1.0)
    std::string protocol_version;    // Version detected (e.g., "C++20", "QUIC_v1")
    std::map<std::string, std::any> orbit_metadata;  // Extended metadata
    u64 timestamp;                   // Detection timestamp
    u32 pattern_strength;            // Pattern strength score (0-100)
};
```

### OrbitPattern
```cpp
struct OrbitPattern {
    std::string name;                           // Pattern name ("cpp20_concepts")
    u32 orbit_id;                               // Orbit identifier
    std::vector<std::string> signature_patterns; // Keywords/markers
    std::vector<std::string> protocol_indicators; // Protocol hints
    std::vector<std::string> version_patterns;   // Version strings
    double weight;                              // Pattern importance (0.0-1.0)
};
```

### OrbitMatch (Simplified)
```cpp
struct OrbitMatch {
    size_t position;      // Byte offset in input
    OrbitType type;       // OpenBrace, CloseBrace, OpenBracket, etc.
    double confidence;    // Balance-based confidence score
    std::string snippet;  // Matched text fragment
};
```

### OrbitType Enum
```cpp
enum class OrbitType {
    None = 0,
    OpenBrace, CloseBrace,      // { }
    OpenBracket, CloseBracket,  // [ ]
    OpenAngle, CloseAngle,      // < > (inferred from Story 1.1)
    OpenParen, CloseParen,      // ( ) (inferred from Story 1.1)
    Quote,                      // "
    NumberStart, NumberEnd,     // Numeric literals
    Unknown
};
```

---

## 2. Key Architecture Patterns

### 2.1 Hierarchical Cascading Typevidence

**From `professional_orbit_scanner.cpp2` design notes:**

```
// ADDITIONAL NOTE: Hierarchical cascading "typevidence" and confix scopes
// Scanning wide is punctuated by hierarchical, cascading "typevidence":
// - Each detector maintains an open "orbit" for a scope while typevidence
//   accumulates. The orbit remains open until the scope is explicitly
//   closed or the evidence fails (timeout/contradiction).
// - Confix regions (byte-span markers that delimit sub-contexts) should
//   roll forward, capturing typevidence for the sub-context between the
//   confix bytes.
```

**Key Concepts:**
- **Confix regions**: Delimiter-bounded contexts (e.g., `{ }`, `[ ]`)
- **Typevidence**: Accumulated evidence for language/protocol detection
- **Scope lifecycle**: `open → accumulate → close/fail`
- **Subscopes**: Nested contexts with independent evidence

### 2.2 Context Subscope Pattern

```cpp
// CONTEXT SUBSCOPE: orbit scoping, evidence elimination, and ranking
// When a confix or other delimiter opens a sub-context, create a new
// "context subscope" which owns a fresh set of candidate orbits. The
// subscope must:
//  - collect incremental evidence (anchors, prefix-hashes, nibble projections)
//  - expose operations to strengthen or eliminate candidates
//  - when closed, merge survivors into parent orbit according to policy
//
// Ranking policy (suggested): sort surviving candidates by matched
// token chain length first, then by confidence, then by pattern weight.
// This ensures the longest-legal-chain wins when grammars are ambiguous.
```

**Disambiguation Priority Order:**
1. **Longest match** (token chain length)
2. **Confidence score**
3. **Pattern weight**

### 2.3 Lazy AST / Token-Combinator System

```cpp
// The scanner is organized as a hierarchical, lazy AST/token-combinator
// system. Detectors should treat the input as a stream of tokens and
// build combinators (OR/AND/sequence/WAM-unification style) lazily:
// - Anchors: detectors continuously collect "anchors" (keywords, markers)
// - Lazy AST: compose combinators into a DAG of possibilities
// - Token prefix hashing: maintain fixed-size prefix hashes (32/64-bit)
// - Minefield projections: maintain small nibble projections (1/2/4-bit)
```

**Components:**
- **Anchors**: Keyword/marker collection during scan
- **Prefix hashes**: 32/64-bit hashes for token equivalence
- **Nibble projections**: 1/2/4-bit projections for quick filtering
- **Combinator DAG**: OR/AND lanes with WAM-style unification

---

## 3. Hierarchical Hash Tree (from `stage2_hierarchical_hash.cpp`)

### Hash Tree Structure
```cpp
class hierarchical_hash_tree {
private:
    struct hash_node {
        std::string hash_value;
        std::vector<std::shared_ptr<hash_node>> children;
        std::unordered_map<std::string, std::string> metadata;
        size_t depth;
        std::chrono::steady_clock::time_point timestamp;
    };

    std::unordered_map<std::string, std::shared_ptr<hash_node>> trees;
    std::mutex tree_mutex;  // Thread-safe

public:
    void create_tree_with_reactor(const std::string& tree_name,
                                   size_t max_depth,
                                   const std::string& strategy);

    void add_evidence_reactive(const std::string& tree_name,
                               const std::string& data,
                               const std::unordered_map<std::string, std::string>& metadata);

    std::string get_root_hash_consistent(const std::string& tree_name);
};
```

**Key Features:**
- Dynamic reactor integration
- Metadata per node
- Timestamp tracking
- Thread-safe operations
- Reactive notifications on evidence updates

---

## 4. Orbit Chunk Processing (from `orbit_scanner_test.cpp`)

### 64-Char Chunk Strategy
```cpp
std::vector<OrbitMatch> scan_orbits(const std::string& input) {
    std::vector<OrbitMatch> matches;
    size_t chunk_size = 64;
    int depth = 0;  // Balance tracking

    for (size_t i = 0; i < input.size(); i += chunk_size) {
        size_t end = std::min(i + chunk_size, input.size());
        std::string_view chunk = input.substr(i, end - i);

        // Bitmask: Set bits for delimiters
        OrbitMask mask = 0;  // uint64_t

        for (size_t j = 0; j < chunk.size(); ++j) {
            char ch = chunk[j];
            size_t pos = i + j;

            if (ch == '{') {
                mask |= (1ULL << 0);
                matches.push_back(OrbitMatch{pos, OrbitType::OpenBrace, 1.0, "{"});
                ++depth;
            }
            // ... other delimiters ...
        }

        // Confidence: balance check
        bool balanced = (depth >= 0);
        double conf = balanced ? 1.0 : (bitcount(mask) / 64.0);
    }

    return matches;
}
```

**Design Rationale:**
- **64-char chunks** align with CPU cache lines
- **Bitmask** (uint64_t) tracks delimiter presence efficiently
- **Depth tracking** for balance validation
- **Confidence scoring** based on structural balance

---

## 5. Multi-Protocol Detection Pattern

### Protocol Detector Interface
```cpp
class QuicOrbitDetector {
    std::vector<OrbitPattern> quic_patterns;

    std::optional<OrbitResult> detect_quic_orbit(std::span<u8> data) {
        if (data.size() < 4) return std::nullopt;

        // Binary pattern check (first byte & 0x80)
        if ((data[0] & 0x80) != 0) {
            u32 version = extract_version(data[1..4]);

            return OrbitResult{
                .orbit_id = 3,  // QUIC long header
                .pattern_mask = 1ULL << 3,
                .confidence = 0.9,
                .protocol_version = format("QUIC_{:08X}", version),
                .timestamp = dense_timestamp(),
                .pattern_strength = 90
            };
        }

        return std::nullopt;
    }
};
```

**Pattern:**
- Each detector owns pattern definitions
- Binary + text-based detection
- Returns `std::optional<OrbitResult>`
- Early-exit on insufficient data
- Confidence scoring per detection

---

## 6. Orbit Language Versions (from orbit files)

### C++17 Orbit
```cpp
// 17_orbit.cpp2
core_features = {
    "structured_bindings",
    "constexpr_if",
    "fold_expressions",
    "inline_variables"
};

library_features = {
    "std_filesystem",
    "std_variant",
    "std_optional",
    "std_any"
};
```

### C++20 Orbit
```cpp
// 20_orbit.cpp2
major_features = {
    "concepts",
    "modules",
    "coroutines",
    "ranges"
};

library_features = {
    "std_concepts",
    "std_ranges",
    "std_coroutine",
    "std_format"
};
```

### C++23 Orbit
```cpp
// 23_orbit.cpp2
core_features = {
    "modules",
    "executors",
    "networking",
    "multidim_subscript"
};

library_features = {
    "std_flat_map",
    "std_flat_set",
    "std_mdspan",
    "std_expected"
};
```

### C++26 Orbit
```cpp
// cpp26_orbit.cpp2
advanced_features = {
    "hazard_pointers",
    "flat_map",
    "flat_set",
    "function_ref"
};

delta_from_cpp23 = {
    "reflection_support",
    "pattern_matching",
    "contracts"
};
```

### CPP2 Orbit (Parser)
```cpp
// 30_orbit.cpp2 - Cpp2 Parser Orbit (Tblgen Tokenization)
enum TokenType {
    ID, KEYWORD, COLON, EQUALS, LBRACE, RBRACE, AT,
    FLOAT_LIT, INT_LIT, UNKNOWN
};

struct NoopAtom {
    string content;
    int depth = 0;
    double energy = 0.0;  // Speculative energy
};
```

---

## 7. TableGen Integration Pattern

### Pattern to TableGen String
```cpp
std::string orbit_patterns_to_tablegen(
    std::vector<OrbitPattern> patterns,
    std::string dialect
) {
    std::string out = format("// Auto-generated TableGen for dialect: {}\n\n", dialect);

    for (auto& p : patterns) {
        std::string id = sanitize_ident(p.name);

        out += format("def {} : OrbitPattern {{\n", id);
        out += format("  let name = \"{}\";\n", escape(p.name));
        out += format("  let orbit_id = {}u;\n", p.orbit_id);
        out += "  let signature_patterns = [";

        for (auto& sig : p.signature_patterns) {
            out += format("\"{}\"", escape(sig));
        }
        out += "];\n";
        out += format("  let weight = {:.3f};\n", p.weight);
        out += "}\n\n";
    }

    return out;
}
```

---

## 8. Implementation Priorities for Story 1.1

### Phase 1 Priorities (from ororoboros-couchduck learnings)

1. **Rabin-Karp Core** (Layer 1)
   - Implement power-of-2 hierarchical windows (1,2,4,8,16,32,64)
   - Rolling hash with O(1) updates per level
   - Use **64-bit hashes** (uint64_t) not 32-bit
   - Prime = 31 (from reference implementations)

2. **Orbit Context System** (Layer 3) - Can start in parallel
   - OrbitType enum: 8 types (open/close × 4 brackets)
   - OrbitMatch structure with position/type/confidence/snippet
   - **Depth tracking** for balance validation
   - Confidence scoring based on structural balance

3. **Pattern Database** (Layer 4) - Can start in parallel
   - OrbitPattern structure with weights
   - Signature patterns (vector<string>)
   - Protocol indicators
   - Version patterns
   - **YAML storage format** (reference uses this)

### Key Design Constraints

✅ **From ororoboros-couchduck evidence:**
- Chunk size: **64 bytes** (aligns with cache lines)
- Hash type: **uint64_t** (64-bit masks)
- Confidence range: **0.0-1.0** (not percentage)
- Pattern strength: **0-100** (integer score)
- Timestamp: **nanosecond precision** (chrono::high_resolution_clock)

✅ **Disambiguation strategy:**
1. Longest match wins (token chain length)
2. Confidence score
3. Pattern weight
4. Grammar priority (CPP2 > C++ > C)

✅ **Context tracking:**
- Depth counter for balance checking
- Bitmask per 64-char chunk
- Metadata map per node
- Scope lifecycle hooks (open/close/fail)

---

## 9. Performance Targets

**From `stage2_hierarchical_hash.cpp` benchmarks:**
- 1000 nodes added in ~100-200ms
- Throughput: ~5000-10000 nodes/second
- Memory: ~64-128 bytes per node (with metadata)
- Threading: Mutex-protected for concurrent access

**Story 1.1 targets:**
- <100ms for 10KB files with 3 grammars active
- O(n) scaling confirmed via benchmarking
- 7 hash levels maintained simultaneously

---

## 10. Quick Reference: Key Files to Study

### From ororoboros-couchduck:
1. **`src/endgame/professional_orbit_scanner.cpp2`** (1527 lines)
   - Complete multi-protocol detection
   - OrbitResult/OrbitPattern structures
   - Detector pattern (QUIC, CouchDB, OpenAPI, Git, SQL, IPFS)
   - TableGen conversion utilities

2. **`orbit_scanner_test.cpp`** (140 lines)
   - Simple 64-char chunk processing
   - OrbitMatch/OrbitType definitions
   - Balance-based confidence scoring
   - Minimal working implementation

3. **`stage2_hierarchical_hash.cpp`** (272 lines)
   - Hash tree with metadata
   - Dynamic reactor integration
   - Thread-safe operations
   - Performance benchmarks

4. **`hierarchical_growing_hash_simple.cpp`** (164 lines)
   - Simplified hash tree (no reactors)
   - Evidence integrity tracking
   - Performance metrics

5. **`cppfort/src/orbits/*.cpp2`** (7 files)
   - Language version orbit definitions
   - Feature delta tracking
   - Energy level encoding

---

## 11. Critical Insights

### Don't Reinvent - Adapt

**ororoboros-couchduck already solved:**
✅ Multi-protocol simultaneous detection
✅ Confidence-based ranking
✅ Orbit bitmask encoding (64-bit)
✅ Hierarchical evidence accumulation
✅ TableGen conversion utilities
✅ 64-char chunk processing
✅ Balance-based confidence scoring

**Story 1.1 must add:**
🔨 Rabin-Karp hierarchical hashing (not in ororoboros)
🔨 Power-of-2 window sizes (1,2,4,8,16,32,64)
🔨 O(1) rolling hash updates
🔨 C/C++/CPP2 grammar disambiguation
🔨 Integration with cppfort PatternMatcher/Machine

### Architecture Alignment

**ororoboros-couchduck uses:**
- Confidence scoring (0.0-1.0)
- Pattern weights (doubles)
- 64-bit orbit masks
- Metadata maps (string → any)
- Subscope lifecycle (open/close/fail)

**Story 1.1 design uses:** ✅ **Same patterns!**
- Confidence (0.0-1.0) ✓
- Pattern weights ✓
- Orbit masks ✓
- TableGen middle tuple ✓
- Context validation ✓

---

## 12. Implementation Checklist (Phase 1)

### Layer 1: Rabin-Karp Core
- [ ] `RabinKarp` class with 7-level hash array (std::array<uint64_t, 7>)
- [ ] `initialize()` method for all 7 windows
- [ ] `update(char remove, char add)` with O(1) complexity
- [ ] `hashAt(size_t level)` accessor
- [ ] Prime = 31 (confirmed from references)
- [ ] Unit tests: hash correctness, rolling updates, collision handling

### Layer 3: Orbit Context System
- [ ] `OrbitType` enum (8 types: open/close × 4)
- [ ] `OrbitMatch` struct with position/type/confidence/snippet
- [ ] `OrbitContext` class with depth tracking
- [ ] `validates(Match)` method for context checking
- [ ] Balance-based confidence scoring
- [ ] Unit tests: context validation for `[` in expr/lambda/attribute

### Layer 4: TableGen Storage
- [ ] `OrbitPattern` struct (name, orbit_id, signatures, weight)
- [ ] `PatternDatabase` class for storage/queries
- [ ] YAML parser (use existing C++ YAML library)
- [ ] Create `patterns/c_patterns.yaml`
- [ ] Create `patterns/cpp_patterns.yaml`
- [ ] Create `patterns/cpp2_patterns.yaml`
- [ ] Populate with 10-20 patterns per grammar (use orbit files as reference)
- [ ] Unit tests: pattern load, query, serialization

---

## 13. Next Steps

1. **Copy reference implementations** (with attribution):
   - `OrbitResult` → `src/stage0/orbit_result.h`
   - `OrbitPattern` → `src/stage0/orbit_pattern.h`
   - `OrbitMatch` → `src/stage0/orbit_mask.h`

2. **Implement Rabin-Karp** (net new):
   - `src/stage0/rabin_karp.{h,cpp}`
   - 7-level hierarchical hash array
   - O(1) rolling updates

3. **Start YAML pattern files**:
   - `patterns/c_patterns.yaml` - Use 17_orbit as guide
   - `patterns/cpp_patterns.yaml` - Combine 17/20/23 orbits
   - `patterns/cpp2_patterns.yaml` - Use 30_orbit + cpp26_orbit

4. **Write unit tests first** (TDD):
   - `src/stage0/test_rabin_karp.cpp`
   - `src/stage0/test_orbit_mask.cpp`
   - `src/stage0/test_pattern_database.cpp`

---

**Status:** ✅ **Orbit languages acquired. Ready for Phase 1 implementation.**

**Evidence base:** 7 orbit files + 4 scanner implementations + 2 hash tree implementations analyzed.

**Next action:** Implement Layer 1 (Rabin-Karp Core) or Layer 3/4 in parallel.
