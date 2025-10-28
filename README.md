# Cppfort: C++2 Transpiler

Cppfort is a pattern-driven transpiler that converts C++2 (cpp2) syntax into standard C++20 code. C++2 is a cleaner, more expressive variant of C++ designed for safer and more maintainable software development.

## Technical Features

### Pattern-Driven Transformation

- **YAML Pattern Definitions**: Declarative patterns in `patterns/bnfc_cpp2_complete.yaml` define syntax transformations
- **Alternating Anchor Matching**: Precise segment extraction using literal anchors and evidence types
- **Bidirectional Support**: Planned support for C++ ↔ C++2 round-trip conversion
- **Grammar Classification**: Automatic detection of C, C++, and C++2 syntax variants
- **N-Way Transpilation**: Three-target support (C/C++/C++2) with shared TableGen infrastructure
- **Semantic Isomorphisms**: Common pattern bones enabling inheritance and cross-language transformations
- **TableGen Integration**: `.td` files define isomorphic patterns for Sea of Nodes intermediate representation

### Orbit System

- **Structural Analysis**: Orbit-based parsing tracks code structure through nesting depth
- **Confix Detection**: Identifies balanced constructs (parentheses, braces, brackets)
- **Lattice Classification**: Byte-level categorization with confidence scoring
- **Multi-Grammar Support**: Unified handling of C/C++/C++2 dialects

### Language Features Supported

- **Function Declarations**: `main: () -> int = { ... }` → `int main() { ... }`
- **Parameter Modes**: `inout`, `out`, `move`, `forward` parameter passing
- **Type System**: Auto deduction (`x := 42` → `auto x = 42`), templates, type aliases
- **Contracts**: Pre/post-condition syntax (planned)
- **Inspect Expressions**: Pattern matching and type inspection (planned)
- **Include Generation**: Automatic `#include` directive insertion

### Architecture Components

#### Core Libraries

- **orbit_scanner**: Wide scanning with SIMD anchor generation and orbit detection
- **cpp2_emitter**: Pattern-driven C++ output generation
- **pattern_loader**: YAML pattern parsing and validation
- **unified_pattern_matcher**: Alternating anchor-based matching engine
- **tblgen_loader**: TableGen JSON pattern loading for n-way transformations
- **multi_grammar_loader**: Loads C/C++/CPP2 grammar patterns from YAML

#### Build System

- **CMake + Ninja**: Cross-platform build configuration
- **CTest Integration**: Unit testing framework
- **Compile Commands**: Generated for IDE integration

#### Pattern Infrastructure

- **Anchor-Based Extraction**: Precise segment identification between literal markers
- **Evidence Validation**: Type-aware content validation (identifiers, expressions, parameters)
- **Substitution Templates**: Configurable output formatting with placeholder replacement
- **Confidence Scoring**: Match quality assessment for pattern selection
- **Orbit Masking**: Structural analysis with masking rings for semantic classification
- **Sea of Nodes**: Intermediate representation for optimization and code generation

## Current Status

- **Pattern Matching**: Core infrastructure functional
- **Basic Functions**: Simple function transpilation working
- **Include Detection**: Automatic header inclusion
- **Parameter Handling**: Basic parameter transformation
- **Test Coverage**: 192 regression tests defined, 0 currently passing
- **Self-Hosting**: Long-term goal for complete bootstrap

## Building and Running

```bash
# Configure with Ninja
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Run tests
cd build && ctest

# Transpile a file
./src/stage0/stage0 transpile input.cpp2 output.cpp
```

## Project Structure

```text
cppfort/
├── CMakeLists.txt          # Root build configuration
├── patterns/               # YAML pattern definitions
│   ├── bnfc_cpp2_complete.yaml
│   └── semantic_units.td
├── src/stage0/             # Core transpiler implementation
│   ├── cpp2_emitter.*      # Output generation
│   ├── pattern_loader.*    # YAML parsing
│   ├── orbit_scanner.*     # Structural analysis
│   └── unified_pattern_matcher.* # Pattern matching
├── regression-tests/       # Test suite (192 files)
├── build/                  # Generated build artifacts
└── README.md              # This file
```

## Coding Standards

### Build System Standards

- **CMake with Ninja**: All builds use CMake configuration with Ninja generator
- **No Shell Scripts**: All automation through CMake targets and CTest
- **Clean Builds**: No intermediate files or build artifacts in source directories

### Code Quality

- **Professional Standards**: Production-ready code with comprehensive error handling
- **No Litter**: Clean source tree, no temporary files or debug artifacts
- **Final Deliverable Expectations**: Code ready for enterprise deployment
- **Documentation**: Technical accuracy over optimism, honest progress reporting

### Development Practices

- **Pattern-Driven**: All transformations defined declaratively, not hardcoded
- **Incremental Progress**: One feature fully working before adding complexity
- **Honest Metrics**: Accurate test counts and capability assessments
- **Self-Hosting Path**: Architecture designed for eventual bootstrap compilation

## Technical Debt and Known Issues

- **Test Suite**: All 192 regression tests currently fail due to incomplete pattern implementations
- **Parameter Transformation**: Complex parameter modes (inout/out/move/forward) partially implemented
- **Recursive Processing**: Nested pattern application uses workarounds instead of proper orbit recursion
- **Bidirectional Patterns**: C++ to C++2 conversion not yet implemented
- **Performance**: Current implementation prioritizes correctness over optimization

## Future Directions

- Complete pattern implementations for all C++2 features
- Full bidirectional transpilation support
- Self-hosting achievement
- Performance optimizations and SIMD enhancements
- Extended language feature support (contracts, inspect, UFCS)

## License

[To be determined - project under active development]
