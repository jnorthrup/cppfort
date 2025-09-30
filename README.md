# cppfort – Three‑Stage Compilation Pipeline

## Overview

The project implements a three‑stage pipeline for the **cpp2** language:

| Stage | Purpose | Main Target |
|------|---------|-------------|
| **0** | Core C++ infrastructure (AST, Emitter, documentation support). | `src/stage0` |
| **1** | **cpp2 → C++ transpiler**. Reads a `.cpp2` file, builds a minimal `TranslationUnit`, and emits C++ using the Stage 0 emitter. | `src/stage1` |
| **2** | **Anti‑cheat attestation**. Disassembles a compiled binary, hashes the output with SHA‑256, and returns a verifiable attestation. | `src/stage2` |

## Building the Project

```bash
# From the repository root
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

The above will build:

* `stage0_cli` – emits C++ from a `.cpp2` file.
* `stage1_cli` – transpiles a `.cpp2` file to C++ (wrapper around Stage 0).
* `anticheat` – command‑line helper executing the attestation (`src/stage2/anticheat_cli`).

## Running the Transpiler (Stage 1)

```bash
./stage1_cli <input_file.cpp2> <output_file.cpp>
```

Example:

```bash
./stage1_cli src/stage1/main.cpp2 stage1_output.cpp
```

## Running the Anticheat (Stage 2)

```bash
./anticheat <path_to_binary>
```

The tool prints a SHA‑256 hash of the binary's `objdump -d` disassembly.

## Regression Test Harness

### Triple Induction Testing Framework

The project implements a **triple induction** feedback loop where each stage improves the others:

- **Stage 2 → Stage 1**: Attestation validates transpilation produces deterministic binaries
- **Stage 1 → Stage 0**: Error analysis guides AST and emitter improvements
- **Stage 0 → Stage 2**: Roundtrip validation ensures semantic correctness

Run the complete triple induction test suite:

```bash
chmod +x regression-tests/run_triple_induction.sh
regression-tests/run_triple_induction.sh
```

### Individual Test Suites

**Basic Regression Tests** (Stage 0/1 comparison):
```bash
regression-tests/run_tests.sh
```

**Attestation Tests** (Stage 2→1 feedback):
```bash
regression-tests/run_attestation_tests.sh
```

**Error Analysis** (Stage 1→0 feedback):
```bash
regression-tests/run_error_analysis.sh
```

The framework provides detailed feedback on which components need improvement and prioritizes fixes by impact.

## Extending the Pipeline

* Add new `.cpp2` samples to `regression-tests/` and expand the test script as needed.
* Implement a full parser in Stage 1 to replace the current stub that builds a minimal `TranslationUnit`.
* Enhance the anticheat module to support additional verification mechanisms (e.g., signed attestations).

---

*© 2025 cppfort – Trustless cpp2 Compiler with Betanet Transport*