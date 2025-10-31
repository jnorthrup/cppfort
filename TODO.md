# TODO: Self-Hosting Path

## TEST STATUS (2025-10-28)

### Stage0 Unit Tests: 6/6 passing (100%)

- test_reality_check: PASS
- test_confix_depth: PASS
- test_correlation: PASS
- test_pattern_match: PASS
- test_tblgen_integration: PASS
- test_depth_matcher: PASS

### Reality Check (Feature Tests): 8/8 passing (100%)

PASSING:

- simple_main: CPP2 function with variable declaration → C++
- parameter_inout: inout parameter lowered to `std::string&` with include emission
- template_alias: template type alias lowers to correct `using` declaration
- include_generation: std::vector usage emits required #include
- walrus_operator: := auto variable declaration
- forward_declaration: extern declaration + definition pairing
- nested_patterns: lambda generation inside function bodies
- contracts: pre-condition attributes emitted with function signature

Run `cd src/stage0/build && ./test_reality_check` for current feature status.

## CURRENT STATE (2025-10-16)

### Existing Infrastructure

- YAML pattern loading (pattern_loader.cpp)
- Anchor-based segment extraction
- Pattern-driven substitution
- OrbitIterator + ConfixOrbit pipeline
- PackratCache infrastructure
- Tblgen pattern matcher (tblgen_pattern_matcher.cpp)
- Depth-based pattern matcher (depth_pattern_matcher.cpp)
- Grammar classification (correlator.cpp)
- CPP2 emitter (cpp2_emitter.cpp)

### Missing Functionality

- Parameter transformation (inout/in/out/move/forward)
- Include generation
- Forward declarations
- Bidirectional patterns (C++ → CPP2)
- Full recursive pattern application
- Template specialization handling

### Architecture (Actual Implementation)

**Pattern Matching System:**

- **TblgenPatternMatcher**: Anchor-based segment extraction
  - Finds literal anchors in pattern (non-$N text)
  - Extracts segments between anchors from input
  - No regex - direct string search and substring

- **DepthPatternMatcher**: Depth-aware pattern validation
  - Builds confix depth map (tracks {}, (), [], <> nesting)
  - Validates evidence doesn't cross confix boundaries
  - Filters overlapping matches

- **ConfixTracker**: Nesting depth tracking
  - Processes (), {}, [], <> characters
  - Handles template angle brackets vs comparison operators
  - Returns depth map for validation

**Pattern Data Structure:**

- **PatternData**: Alternating anchor/evidence patterns
  - alternating_anchors: fixed literal strings
  - evidence_types: segment types between anchors
  - substitution_templates: $0, $1, $2 placeholders per grammar
  - AnchorSegment: ordinal positions with delimiters

- **TblgenSemanticUnit**: N-way pattern storage
  - c_pattern, cpp_pattern, cpp2_pattern
  - segments: shared structure across grammars

**Transformation System:**

- **CPP2Emitter**: Pattern-driven substitution
  - emit_depth_based(): deterministic depth matching
  - extract_alternating_segments(): anchor-based extraction
  - Applies substitution templates with segment placeholders

**Orbit Infrastructure (structural):**

- ConfixOrbit, FunctionOrbit: AST-like structure representation
- OrbitIterator, OrbitPipeline: iteration framework
- Used for structure tracking, not primary pattern matching

### Code Standards

- CMake build system
- One ninja per directory
- Direct anchor-based pattern matching (no libraries)
- Data-driven pattern configuration

### Constraints

- Clean room implementation
- N-way graph mapping transpiler
- No external dependencies beyond build tools

### Architectural Principles (For AI Agents)

**NO REGEX. NO STD::REGEX. NO BOOST.REGEX:**

- RBCursiveScanner is the combinator-based pattern matcher
- Direct string search (find, substr) only
- Character-by-character processing via ConfixTracker
- All regex violations converted to RBCursive (2025-10-18)

**SEMICOLON HANDLING:**

- Semicolons are whitespace-like: can appear in any order or congregation size
- CPP2 uses 0-length semicolons (automatically added)
- Parser should treat multiple semicolons as single separator
- No semantic difference between `;` `;;` `;;;`

**DO NOT add Boost.Spirit, Karma, or similar parser combinator libraries:**

- Violates clean room constraint
- Massive template metaprogramming overhead
- 30+ minute compile times
- Megabytes of headers for what 75 lines does

**Current approach is correct:**

- tblgen_pattern_matcher.cpp: Direct string search for anchors
- depth_pattern_matcher.cpp: Confix depth validation
- confix_tracker.h: Nesting level tracking
- Total: ~200 lines that compile instantly

**N-way conversion without libraries:**

- TblgenSemanticUnit already has c_pattern, cpp_pattern, cpp2_pattern
- Anchor-based extraction: find literal strings, extract segments between
- Template substitution: replace $0, $1, $2 with extracted segments
- Works for CPP2 → C++, C → CPP2 requires reverse anchors (future)

**Why this is better:**

- Zero dependencies = clean room compliance
- Instant compile vs 30 min Spirit builds
- Debuggable: trace exact string positions
- Extensible: add pattern = add tblgen entry + anchors
- Known good: 5/6 tests passing on working code

**The metal matters:**

```cpp
// This is enough:
size_t pos = input.find(anchor, start);
std::string segment = input.substr(pos, next_pos - pos);

// Don't add this:
qi::rule<Iterator, std::string(), qi::locals<char>> identifier
    = qi::lexeme[qi::alpha >> *qi::alnum];
```

## Phase 1: Core Orbit Infrastructure

### 1.1 Create Orbit base class with grammar children

- [x] Add class Orbit to orbit_ring.h
- [x] Add OrbitType enum (Confix, Keyword, Operator, Identifier, Literal)
- [x] Add `std::vector<EvidenceSpan>` evidence member
- [x] Add std::map<GrammarType, Orbit*> grammar_children member
- [x] Add void assign_child(GrammarType g, Orbit* child) method
- [x] Add Orbit* get_child(GrammarType g) method
- [x] Add void parameterize_children(const PatternData& pattern) method
- [x] Add virtual bool matches(const EvidenceSpan& e) method
- [x] Add size_t start_pos member
- [x] Add size_t end_pos member
- [x] Add double confidence member

### 1.2 Create OrbitIterator in orbit_iterator.cpp

- [x] Create orbit_iterator.h header file
- [x] Create orbit_iterator.cpp implementation file
- [x] Add class OrbitIterator declaration
- [x] Add Orbit* next() method
- [x] Add Orbit* current() const method
- [x] Add void reset() method
- [x] Add bool has_next() const method
- [x] Add std::vector<Orbit*> orbits member
- [x] Add size_t current_index member
- [x] Add constructor OrbitIterator()
- [x] Add destructor ~OrbitIterator()

### 1.3 Create OrbitFragment struct

- [x] Add struct OrbitFragment to orbit_ring.h
- [x] Add std::string c_text member
- [x] Add std::string cpp_text member
- [x] Add std::string cpp2_text member
- [x] Add size_t start_pos member
- [x] Add size_t end_pos member
- [x] Add uint16_t lattice_mask member
- [x] Add double confidence member

### 1.4 Create ConfixOrbit as derived class

- [x] Create confix_orbit.h header file
- [x] Create confix_orbit.cpp implementation file
- [x] Add class ConfixOrbit : public Orbit
- [x] Add char open_char member
- [x] Add char close_char member
- [x] Add int depth_counter member
- [x] Override bool matches(const EvidenceSpan& e) method
- [x] Add bool validate_pair(char open, char close) method

## Phase 2: Wide Scanner Packrat Architecture

### 2.1 Create PackratCache for memoization

- [x] Add struct PackratEntry with position, orbit_id, result
- [x] Add std::unordered_map<size_t, PackratEntry> cache
- [x] Add bool has_cached(size_t pos, OrbitType type)
- [x] Add PackratEntry* get_cached(size_t pos, OrbitType type)
- [x] Add void store_cache(size_t pos, OrbitType type, result)

### 2.2 Build n-way tree from grammar data

- [x] Load patterns/bnfc_cpp2_complete.yaml into tree root
- [x] Create branch node for each UnifiedOrbitCategory
- [x] Add C-specific leaves under C branch
- [x] Add CPP-specific leaves under CPP branch
- [x] Add CPP2-specific leaves under CPP2 branch
- [x] Share common trunk nodes between all three

### 2.3 Pattern matching (actual implementation)

- [x] Depth-aware pattern matching via DepthPatternMatcher
- [x] Anchor-based extraction via TblgenPatternMatcher
- [x] Confix boundary validation via ConfixTracker
- [x] Longest deterministic match wins (no speculation)

## Phase 3: Combinator Integration (Infrastructure exists, not primary mechanism)

**Note:** RBCursiveScanner and combinator infrastructure exists but pattern matching actually uses TblgenPatternMatcher and DepthPatternMatcher.

### 3.1 Add combinator to ConfixOrbit

- [x] Add forward declaration class RBCursiveScanner
- [x] Add RBCursiveScanner* combinator member
- [x] Add void set_combinator(RBCursiveScanner* c) method
- [x] Add RBCursiveScanner* get_combinator() const method
- [x] Initialize combinator to nullptr in constructor

### 3.2 Create CombinatorPool class

- [x] Add class CombinatorPool to rbcursive.h
- [x] Add `std::vector<RBCursiveScanner>` pool member
- [x] Add RBCursiveScanner* allocate() method
- [x] Add void release(RBCursiveScanner* c) method
- [x] Add size_t available() const method
- [x] Add constructor with initial size parameter

### 3.3 Connect combinators to orbits

- [x] Add CombinatorPool pool to OrbitIterator
- [x] In OrbitIterator constructor, allocate combinators
- [x] In OrbitIterator::next(), assign combinator to orbit
- [x] In orbit destructor, release combinator back to pool
- [ ] **Actually use combinators for pattern matching** (currently uses TblgenPatternMatcher instead)

## Phase 4: Pattern Loading

### 4.1 Create PatternLoader class

- [x] Add class PatternLoader to pattern_loader.h
- [x] Add bool load_yaml(const std::string& path) method
- [x] Add `std::vector<PatternData>` patterns member
- [x] Add struct PatternData with name, anchors, segments fields
- [x] Add size_t pattern_count() const method

### 4.2 Parse YAML patterns

- [x] Open patterns/bnfc_cpp2_complete.yaml
- [x] Parse each pattern node
- [x] Extract name field
- [x] Extract unified_signatures array
- [x] Extract grammar_variants map
- [x] Store in PatternData struct

### 4.3 Convert patterns to orbits

- [x] For each PatternData create ConfixOrbit
- [x] Set orbit name from pattern.name
- [x] Create RBCursiveScanner combinator for pattern
- [x] Assign combinator to orbit
- [x] Add orbit to OrbitIterator

## Phase 5: Enable Orbit Recursion Across Fragments ✅ COMPLETED

### 5.1 Wire RBCursiveScanner to continue across fragment boundaries

- [x] Add class FragmentCorrelator to correlator.h
- [x] Add void correlate(OrbitFragment& f) method
- [x] Add bool is_cpp2_syntax(const std::string& text) const
- [x] Add bool is_cpp_syntax(const std::string& text) const
- [x] Add bool is_c_syntax(const std::string& text) const

### 5.2 Implement cross-fragment recursion in RBCursiveScanner ✅ COMPLETED

- [x] Modify RBCursiveScanner::speculate() to accept fragment list - implemented speculate_across_fragments()
- [x] Add logic to continue pattern matching across fragment boundaries - scope expansion captures full constructs
- [x] Use PackratCache to memoize results across fragments - integrated PackratCache in speculate_across_fragments
- [x] Return SpeculativeMatch with confidence when pattern spans multiple fragments - confidence based on match length

### 5.3 Update OrbitIterator to handle cross-fragment patterns ✅ COMPLETED

- [x] Modify populate_iterator() to pass fragment list to combinators - evaluate_fragment calls speculate_across_fragments
- [x] Allow orbits to consume multiple fragments during recursion - single fragment for now, framework ready for expansion
- [x] Track fragment consumption during orbit traversal - fragment positions adjusted in speculate_across_fragments
- [x] Update confidence calculation for multi-fragment matches - confidence penalized for cross-fragment matches

### 5.4 Populate OrbitFragment from orbit traversal results ✅ COMPLETED

- [x] OrbitFragment.{c_text, cpp_text, cpp2_text} set by successful orbit recursion - grammar set based on pattern name
- [x] Remove manual correlation functions - use orbit output directly - orbit pipeline sets grammar from patterns
- [x] Each grammar variant (C/CPP/CPP2) gets its text from corresponding orbit path - grammar classification working
- [x] Confidence scores propagate from orbit matching to fragment selection - confidence propagated through orbit system

## Phase 6: Evidence Processing

### 6.1 Create EvidenceSpan class

- [x] Add class EvidenceSpan to evidence.h
- [x] Add size_t start_pos member
- [x] Add size_t end_pos member
- [x] Add std::string content member
- [x] Add double confidence member
- [x] Add void merge(const EvidenceSpan& other) method

### 6.2 Add evidence to orbits

- [x] Add `std::vector<EvidenceSpan>` evidence to Orbit base class
- [x] Add void add_evidence(const EvidenceSpan& e) method
- [x] Add EvidenceSpan* get_evidence(size_t index) method
- [x] Add size_t evidence_count() const method

### 6.3 Extract evidence between anchors

- [x] Add method extract_evidence(size_t start, size_t end)
- [x] Find all non-anchor chars between positions
- [x] Create EvidenceSpan for continuous runs
- [x] Add evidence to current orbit

## Phase 7: Concurrent Speculation

### 7.1 Create SpeculativeMatch class

- [x] Add class SpeculativeMatch to speculation.h
- [x] Add size_t match_length member
- [x] Add double confidence member
- [x] Add std::string pattern_name member
- [x] Add OrbitFragment result member

### 7.2 Add speculation to combinators

- [x] Add `std::vector<SpeculativeMatch>` matches to RBCursiveScanner
- [x] Add void speculate(const std::string& text) method
- [x] Try all patterns in parallel
- [x] Store matches with confidence scores
- [x] Sort by match_length descending

### 7.3 Implement longest match selection

- [x] Add SpeculativeMatch* get_best_match() method
- [x] Return match with longest length
- [x] Break ties by confidence score
- [x] Return nullptr if no matches

## Phase 8: CPP2 Emission

### 8.1 Create CPP2Emitter class

- [x] Add class CPP2Emitter to cpp2_emitter.h
- [x] Add void emit(OrbitIterator& iter, std::ostream& out) method
- [x] Add void emit_fragment(const OrbitFragment& f, std::ostream& out)
- [x] Add void emit_orbit(const ConfixOrbit& o, std::ostream& out)

### 8.2 Implement direct emission

- [x] Reset orbit iterator to beginning
- [x] While orbit = iter.next() is not null
- [x] For each fragment in orbit
- [x] Write fragment.cpp2_text to output
- [x] Add spacing based on confix depth

### 8.3 Handle special CPP2 constructs

- [x] Detect function declarations (name: (...) -> type)
- [x] Detect parameter passing (in, out, inout, move, forward)
- [x] Detect contracts (pre<{}>, post<{}>)
- [x] Preserve inspect expressions
- [x] Preserve is/as expressions

## Phase 9: Build System

- [x] CMake build system already implemented (per code standards)
- [x] All source files automatically included via GLOB
- [x] Static library and executable targets configured
- [x] Dependencies properly linked

## Phase 10: Testing Infrastructure

### 10.1 Create test runner

- [x] Add test_runner.cpp
- [x] Add main() function
- [x] Add run_test(const std::string& file) function
- [x] Add bool compare_output(expected, actual) function

### 10.2 Test orbit depth tracking

- [x] Create test_confix_depth.cpp
- [x] Test nested parentheses
- [x] Test nested braces
- [x] Test mixed confix pairs
- [x] Verify balance checking

### 10.3 Test fragment correlation

- [x] Create test_correlation.cpp
- [x] Test CPP2 function to C/CPP
- [x] Test C function to CPP2
- [x] Test CPP template to CPP2

## Phase 11: Self-Bootstrap Preparation (Superseded by Lean Path)

### 11.1 Transpile scanner components

- [ ] Run stage0_cpp2 on orbit_ring.cpp
- [ ] Save output as orbit_ring.cpp2
- [ ] Run stage0_cpp2 on wide_scanner.cpp
- [ ] Save output as wide_scanner.cpp2
- [ ] Run stage0_cpp2 on rbcursive.cpp
- [ ] Save output as rbcursive.cpp2

### 11.2 Transpile pattern components

- [ ] Run stage0_cpp2 on pattern_loader.cpp
- [ ] Save output as pattern_loader.cpp2
- [ ] Run stage0_cpp2 on correlator.cpp
- [ ] Save output as correlator.cpp2

### 11.3 Transpile emitter

- [ ] Run stage0_cpp2 on cpp2_emitter.cpp
- [ ] Save output as cpp2_emitter.cpp2
- [ ] Verify CPP2 syntax correctness

## Phase 12: Self-Hosting Validation (Superseded by Lean Path)

### 12.1 Compile CPP2 versions

- [ ] Use stage0 to compile orbit_ring.cpp2
- [ ] Use stage0 to compile wide_scanner.cpp2
- [ ] Use stage0 to compile rbcursive.cpp2
- [ ] Link to create stage0_cpp2_selfhosted

### 12.2 Compare outputs

- [ ] Run original stage0_cpp2 on test file
- [ ] Run stage0_cpp2_selfhosted on same file
- [ ] Compare byte-for-byte output
- [ ] Log any differences

### 12.3 Regression validation

- [ ] Run on pure2-hello.cpp2
- [ ] Run on pure2-bounds-safety-span.cpp2
- [ ] Run on mixed-function-expression.cpp2
- [ ] Verify all produce valid CPP2

## Phase 13: Minimal Self-Hosting (Superseded by Lean Path)

### 13.1 Trim to essential files only

- [ ] Delete all test files except pure2-hello.cpp2
- [ ] Keep only: orbit_iterator, wide_scanner, pattern_loader
- [ ] Delete all unused headers
- [ ] Delete all debug/logging code
- [ ] Strip comments from all files

### 13.2 Minimal transpiler binary

- [ ] Single main.cpp with embedded patterns
- [ ] Statically link all orbits
- [ ] No dynamic allocation after startup
- [ ] Fixed-size packrat cache (64KB)
- [ ] Output directly to stdout

### 13.3 Self-host verification

- [ ] Transpile wide_scanner.cpp → wide_scanner.cpp2
- [ ] Transpile orbit_iterator.cpp → orbit_iterator.cpp2
- [ ] Transpile main.cpp → main.cpp2
- [ ] Compile cpp2 versions with cppfront
- [ ] Binary size < 100KB

## Current Self-Hosting Status

- [x] OrbitIterator + ConfixOrbit pipeline
- [x] PatternLoader (hardcoded patterns for clean room)
- [x] CPP2Emitter implemented
- [x] Transpile command in main.cpp
- [x] Stage0 unit tests: 6/6 passing (100%)
- [ ] Regression tests: 0/192 compiling (0%)
  - All unit tests passing: 6/6 (100%)
  - Function signatures transform: `int main()`, `std::string name()`
  - Variable declarations inside functions transform: `std::string s = "world";`
  - Remaining blockers:
    - Parameter transformations: `inout s: std::string` → needs `std::string& s`
    - Include generation: missing `#include <string>`, `#include <iostream>`
    - Forward declarations: functions called before defined
  - Pattern status: Function + variable patterns both enabled, line-by-line processing working

## Future Work (Not Implemented)

### Bidirectional Pattern Matching

- [ ] C++ → CPP2 transformation (currently CPP2 → C++ only)
- [ ] Reverse anchor behavior (join instead of split)
- [ ] Round-trip validation (CPP2→C++→CPP2)

### Recursive Pattern Application

- [ ] Nested pattern matching (apply patterns to extracted segments)
- [ ] Inside-out transformation (deepest matches first)
- [ ] Depth-limited recursion to prevent infinite loops

### Advanced Features

- [ ] Parameter transformation (inout/in/out/move/forward)
- [ ] Include generation based on type usage
- [ ] Forward declarations for function ordering
- [ ] Template specialization handling
