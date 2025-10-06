# TODO: Self-Hosting Path

code standards:
cmake, 1 ninja per dir, no shell scripts, no python, no makefiles.

this project current enforces the orbits holding the recursive combinators to stress the purpose built design simplicity of the data driven orbit tree configuration.

## Phase 1: Core Orbit Infrastructure

### 1.1 Create Orbit base class with grammar children

- [x] Add class Orbit to orbit_ring.h
- [x] Add OrbitType enum (Confix, Keyword, Operator, Identifier, Literal)
- [x] Add std::vector<EvidenceSpan> evidence member
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

### 2.3 SIMD autovectorized orbit fanout

- [x] Add __attribute__((vector)) to scan loop
- [x] Process 16/32/64 bytes simultaneously
- [x] Fan out to multiple orbits in parallel
- [x] Each orbit checks its pattern concurrently
- [x] Winner takes the parse result

## Phase 3: Combinator Integration

### 3.1 Add combinator to ConfixOrbit

- [x] Add forward declaration class RBCursiveScanner
- [x] Add RBCursiveScanner* combinator member
- [x] Add void set_combinator(RBCursiveScanner* c) method
- [x] Add RBCursiveScanner* get_combinator() const method
- [x] Initialize combinator to nullptr in constructor

### 3.2 Create CombinatorPool class

- [x] Add class CombinatorPool to rbcursive.h
- [x] Add std::vector<RBCursiveScanner> pool member
- [x] Add RBCursiveScanner* allocate() method
- [x] Add void release(RBCursiveScanner* c) method
- [x] Add size_t available() const method
- [x] Add constructor with initial size parameter

### 3.3 Connect combinators to orbits

- [x] Add CombinatorPool pool to OrbitIterator
- [x] In OrbitIterator constructor, allocate combinators
- [x] In OrbitIterator::next(), assign combinator to orbit
- [x] In orbit destructor, release combinator back to pool

## Phase 4: Pattern Loading

### 4.1 Create PatternLoader class

- [x] Add class PatternLoader to pattern_loader.h
- [x] Add bool load_yaml(const std::string& path) method
- [x] Add std::vector<PatternData> patterns member
- [x] Add struct PatternData with name, regex, category fields
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
- [x] Create RBCursiveScanner for pattern.regex
- [x] Assign combinator to orbit
- [x] Add orbit to OrbitIterator

## Phase 5: Fragment Correlation

### 5.1 Create FragmentCorrelator class

- [x] Add class FragmentCorrelator to correlator.h
- [x] Add void correlate(OrbitFragment& f) method
- [x] Add bool is_cpp2_syntax(const std::string& text) const
- [x] Add bool is_cpp_syntax(const std::string& text) const
- [x] Add bool is_c_syntax(const std::string& text) const

### 5.2 Implement syntax detection

- [ ] Check for ':' after identifier (CPP2)
- [ ] Check for '->' before type (CPP2)
- [ ] Check for 'template' keyword (CPP)
- [ ] Check for 'class' keyword (CPP)
- [ ] Check for 'typedef' keyword (C)
- [ ] Check for 'struct' without class (C)

### 5.3 Fragment transformation methods

- [ ] Add std::string cpp2_to_cpp(const std::string& cpp2)
- [ ] Add std::string cpp2_to_c(const std::string& cpp2)
- [ ] Add std::string cpp_to_cpp2(const std::string& cpp)
- [ ] Add std::string cpp_to_c(const std::string& cpp)
- [ ] Add std::string c_to_cpp2(const std::string& c)
- [ ] Add std::string c_to_cpp(const std::string& c)

### 5.4 Apply correlation to fragments

- [ ] In scanAnchorsWithOrbits, create FragmentCorrelator
- [ ] For each fragment, call correlator.correlate()
- [ ] Set fragment.c_text from correlation
- [ ] Set fragment.cpp_text from correlation
- [ ] Set fragment.cpp2_text from correlation

## Phase 6: Evidence Processing

### 6.1 Create EvidenceSpan class

- [ ] Add class EvidenceSpan to evidence.h
- [x] Add size_t start_pos member
- [x] Add size_t end_pos member
- [ ] Add std::string content member
- [x] Add double confidence member
- [ ] Add void merge(const EvidenceSpan& other) method

### 6.2 Add evidence to orbits

- [ ] Add std::vector<EvidenceSpan> evidence to ConfixOrbit
- [ ] Add void add_evidence(const EvidenceSpan& e) method
- [ ] Add EvidenceSpan* get_evidence(size_t index) method
- [ ] Add size_t evidence_count() const method

### 6.3 Extract evidence between anchors

- [ ] Add method extract_evidence(size_t start, size_t end)
- [ ] Find all non-anchor chars between positions
- [ ] Create EvidenceSpan for continuous runs
- [ ] Add evidence to current orbit

## Phase 7: Concurrent Speculation

### 7.1 Create SpeculativeMatch class

- [ ] Add class SpeculativeMatch to speculation.h
- [ ] Add size_t match_length member
- [x] Add double confidence member
- [ ] Add std::string pattern_name member
- [ ] Add OrbitFragment result member

### 7.2 Add speculation to combinators

- [ ] Add std::vector<SpeculativeMatch> matches to RBCursiveScanner
- [ ] Add void speculate(const std::string& text) method
- [ ] Try all patterns in parallel
- [ ] Store matches with confidence scores
- [ ] Sort by match_length descending

### 7.3 Implement longest match selection

- [ ] Add SpeculativeMatch* get_best_match() method
- [ ] Return match with longest length
- [ ] Break ties by confidence score
- [ ] Return nullptr if no matches

## Phase 8: CPP2 Emission

### 8.1 Create CPP2Emitter class

- [ ] Add class CPP2Emitter to cpp2_emitter.h
- [ ] Add void emit(OrbitIterator& iter, std::ostream& out) method
- [ ] Add void emit_fragment(const OrbitFragment& f, std::ostream& out)
- [ ] Add void emit_orbit(const ConfixOrbit& o, std::ostream& out)

### 8.2 Implement direct emission

- [ ] Reset orbit iterator to beginning
- [ ] While orbit = iter.next() is not null
- [ ] For each fragment in orbit
- [ ] Write fragment.cpp2_text to output
- [ ] Add spacing based on confix depth

### 8.3 Handle special CPP2 constructs

- [ ] Detect function declarations (name: (...) -> type)
- [ ] Detect parameter passing (in, out, inout, move, forward)
- [ ] Detect contracts (pre<{}>, post<{}>)
- [ ] Preserve inspect expressions
- [ ] Preserve is/as expressions

## Phase 9: Build System

### 9.1 Create Makefile

- [ ] Add CXX = g++
- [ ] Add CXXFLAGS = -std=c++20 -O2
- [ ] Add SRCS listing all .cpp files
- [ ] Add OBJS from SRCS
- [ ] Add stage0_cpp2 target

### 9.2 Compile orbit components

- [ ] Add orbit_ring.o target
- [ ] Add wide_scanner.o target
- [ ] Add rbcursive.o target
- [ ] Add pattern_loader.o target
- [ ] Add correlator.o target
- [ ] Add cpp2_emitter.o target

### 9.3 Link stage0_cpp2

- [ ] Link all object files
- [ ] Generate stage0_cpp2 executable
- [ ] Add clean target
- [ ] Add test target

## Phase 10: Testing Infrastructure

### 10.1 Create test runner

- [ ] Add test_runner.cpp
- [ ] Add main() function
- [ ] Add run_test(const std::string& file) function
- [ ] Add bool compare_output(expected, actual) function

### 10.2 Test orbit depth tracking

- [ ] Create test_confix_depth.cpp
- [ ] Test nested parentheses
- [ ] Test nested braces
- [ ] Test mixed confix pairs
- [ ] Verify balance checking

### 10.3 Test fragment correlation

- [ ] Create test_correlation.cpp
- [ ] Test CPP2 function to C/CPP
- [ ] Test C function to CPP2
- [ ] Test CPP template to CPP2

## Phase 11: Self-Bootstrap Preparation

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

## Phase 12: Self-Hosting Validation

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

## Phase 13: Minimal Self-Hosting

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

## Lean Self-Hosting Path (Replaces Phases 11-13)

- [x] Stage0 emits orbit fragments using `OrbitIterator` + `ConfixOrbit` pipeline
- [x] PatternLoader drives orbit creation from YAML without bespoke scripts
- [ ] Single cppfront validation pass feeding both stage0 variants
- [ ] Compare outputs via existing regression harness only
- [ ] Gate on binary size + confidence thresholds rather than file churn
