# N-Way Compiler - Quick Start Guide

## 5-Minute Setup

```bash
# Clone and build
git clone <repo> && cd cppfort
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target nway_compiler

# First transpilation
./build/nway_compiler examples/hello.c -o hello.cpp2

# Verify it works
./build/nway_compiler hello.cpp2 -o hello.cpp
g++ -std=c++20 hello.cpp -o hello && ./hello
```

## Command Cheat Sheet

### Basic Usage
```bash
# C to CPP2
nway_compiler input.c -o output.cpp2

# CPP2 to C++
nway_compiler input.cpp2 -o output.cpp

# With optimization
nway_compiler input.c -O3 -o output.cpp
```

### Attestation
```bash
# Enable (default)
nway_compiler input.c -o output.cpp --key mykey.pem

# Verify binary
nway_compiler --verify mybinary

# Disable
nway_compiler input.c -o output.cpp --no-attestation
```

### Testing
```bash
# Round-trip test
nway_compiler input.c --round-trip

# Deterministic check
SOURCE_DATE_EPOCH=1000000 nway_compiler input.c -o out1.cpp
SOURCE_DATE_EPOCH=2000000 nway_compiler input.c -o out2.cpp
diff out1.cpp out2.cpp  # Should be identical

# IR dump
nway_compiler input.c --ir-dump -o debug.ir
```

## Common Workflows

### C → CPP2 → Binary
```bash
nway_compiler legacy.c -o modern.cpp2
nway_compiler modern.cpp2 -o program
```

### Verify Compilation Chain
```bash
# Compile with attestation
nway_compiler -v --key dev.pem source.c -o output.cpp

# Check attestation record
nway_compiler --verify output.bin
```

### Debug Transpilation
```bash
# Verbose mode
nway_compiler -v input.c -o output.cpp

# IR inspection
nway_compiler --ir-dump input.c -o debug.ir
cat debug.ir
```

## Error Analysis (Fixed!)

```bash
# Run full error analysis
bash regression-tests/analyze_errors_simple.sh

# Current stats: 27/189 tests passing (14%)
# Priority fixes identified in output
```

## File Structure

```
cppfort/
├── src/
│   ├── nway_compiler.cpp       # Main driver
│   ├── ir/                     # Unified IR
│   │   ├── ir.h
│   │   └── ir.cpp
│   ├── parsers/                # Language parsers
│   │   ├── c_parser.h
│   │   └── c_parser.cpp
│   └── attestation/            # Crypto & anti-cheat
│       ├── attestation.h
│       └── attestation.cpp
├── tests/
│   ├── test_roundtrip.cpp      # Round-trip tests
│   └── test_deterministic.sh   # Deterministic tests
├── regression-tests/
│   └── analyze_errors_simple.sh # Error analysis (FIXED)
├── NWAY_README.md              # Full documentation
├── BUILD_REPORT.md             # Implementation report
└── QUICKSTART.md               # This file
```

## IR Quick Reference

### Types
```cpp
// Primitives
TypeKind::Void, Bool, Int32, Float64

// Composite
TypeKind::Pointer, Reference, Array, Function

// Language-specific
TypeKind::Auto      // C++ auto
TypeKind::Contract  // CPP2 contracts
```

### Expressions
```cpp
// Literals
ExprKind::IntLiteral, StringLiteral

// Operators
ExprKind::BinaryOp, UnaryOp, TernaryOp

// Language-specific
ExprKind::Lambda    // C++ lambda
ExprKind::Inspect   // CPP2 pattern match
```

### Statements
```cpp
// Control flow
StmtKind::If, While, For, Switch

// Language-specific
StmtKind::Try       // C++ exceptions
StmtKind::Defer     // CPP2 defer
```

## Attestation API

```cpp
// Initialize
AttestationSystem& sys = AttestationSystem::getInstance();
sys.initialize();

// Compile with attestation
sys.beginCompilation("source.c");
sys.attestIR(ir_data, size);
sys.attestOutput("output.cpp");
CompilationRecord rec = sys.finalizeCompilation();

// Verify
bool ok = sys.verifyCompilation(rec);
```

## Anti-Cheat API

```cpp
// Self-check
bool ok = AntiCheat::verifySelf(expected_hash);

// Detect injection
bool injected = AntiCheat::detectInjection();

// AST tampering
bool tampered = AntiCheat::detectASTTampering(ast, hash);
```

## Testing Commands

```bash
# All tests
cd build && ctest -V

# Specific suites
./nway_test --test roundtrip
./nway_test --test attestation
./nway_test --test perf

# Fuzzing (if AFL++ installed)
afl-fuzz -i testcases -o findings ./nway_fuzz @@
```

## Performance Tips

### Optimization Levels
- `-O0`: Debug, no optimization
- `-O1`: Basic optimization
- `-O2`: Standard (recommended)
- `-O3`: Maximum optimization

### Deterministic Builds
```bash
# Set for reproducibility
export SOURCE_DATE_EPOCH=1000000000
export BUILD_PATH_PREFIX_MAP="/old=/new"

# Compile
nway_compiler --deterministic input.c -o output.cpp
```

### Parallel Compilation
```bash
# Build in parallel
cmake --build build -j$(nproc)
```

## Troubleshooting

### Build Fails
```bash
# Clean build
rm -rf build && cmake -B build -S .
cmake --build build
```

### Attestation Fails
```bash
# Check key
ls -la mykey.pem

# Disable for debugging
nway_compiler --no-attestation input.c -o output.cpp
```

### Test Failures
```bash
# Debug mode
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run with sanitizers
./build/nway_compiler input.c -o output.cpp
```

### Error Analysis Not Running
```bash
# Use fixed version
bash regression-tests/analyze_errors_simple.sh

# Check stage1_cli exists
ls -la build/stage1_cli
```

## Development Workflow

### Add New Feature
1. Update IR schema if needed (`src/ir/ir.h`)
2. Implement parser changes
3. Update emitters
4. Add tests
5. Update documentation
6. Ensure deterministic builds

### Debug Transpilation
1. Run with `-v` flag
2. Dump IR with `--ir-dump`
3. Inspect intermediate output
4. Check error analysis results

### Submit Changes
1. Format code: `make format`
2. Run static analysis: `make analyze`
3. Run all tests: `ctest -V`
4. Verify determinism
5. Update attestation chain

## Key Files to Know

### Core Implementation
- `src/nway_compiler.cpp` - Main driver (600 lines)
- `src/ir/ir.cpp` - IR implementation (400 lines)
- `src/parsers/c_parser.cpp` - C parser (750 lines)
- `src/attestation/attestation.cpp` - Attestation (550 lines)

### Tests
- `tests/test_roundtrip.cpp` - Round-trip tests
- `tests/test_deterministic.sh` - Deterministic tests
- `regression-tests/analyze_errors_simple.sh` - Error analysis

### Documentation
- `NWAY_README.md` - Full documentation
- `BUILD_REPORT.md` - Implementation details
- `QUICKSTART.md` - This guide

## Next Steps

1. Read `NWAY_README.md` for full documentation
2. Review `BUILD_REPORT.md` for implementation details
3. Run tests to verify installation
4. Try transpiling simple programs
5. Explore IR with `--ir-dump`
6. Test attestation system
7. Contribute improvements!

## Support

- Security issues: security@example.com
- Bug reports: GitHub issues
- Questions: Stack Overflow tag `nway-compiler`

---

**Happy Compiling! 🚀**