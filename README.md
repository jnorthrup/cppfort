# cppfort – Three‑Stage Compilation Pipeline

## Overview

The project implements a three‑stage pipeline for the **cpp2** language:

| Stage | Purpose | Main Target |
|------|---------|-------------|
| **0** | Core C++ infrastructure (AST, Emitter, documentation support). | `src/stage0` |
| **1** | **cpp2 → C++ transpiler**. Reads a `.cpp2` file, builds a minimal `TranslationUnit`, and emits C++ using the Stage 0 emitter. | `src/stage1` |
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
* `stage1_cli` – transpiles a `.cpp2` file to C++ (wrapper around Stage 0).
* `anticheat` – command‑line helper executing the attestation (`src/stage2/anticheat_cli`).

## Running the Transpiler (Stage 1)

```bash
./stage1_cli <input_file.cpp2> <output_file.cpp>
```

Example:

```bash
./stage1_cli src/stage1/main.cpp2 stage1_output.cpp
```

## Running the Anticheat (Stage 2)

```bash
./anticheat <path_to_binary>
```

The tool prints a SHA‑256 hash of the binary’s `objdump -d` disassembly.

## Regression Test Harness

A simple regression test is provided under `regression-tests/run_tests.sh`. It:

1. Builds Stage 0 and Stage 1.
2. Uses a sample `.cpp2` test file.
3. Emits C++ via Stage 0 and via Stage 1.
4. Diffs the two outputs; any differences indicate a regression.

Run the test:

```bash
chmod +x regression-tests/run_tests.sh
regression-tests/run_tests.sh
```

If the script reports *“Regression test passed – outputs match.”* the pipeline is working as expected.

## Extending the Pipeline

* Add new `.cpp2` samples to `regression-tests/` and expand the test script as needed.
* Implement a full parser in Stage 1 to replace the current stub that builds a minimal `TranslationUnit`.
* Enhance the anticheat module to support additional verification mechanisms (e.g., signed attestations).

--- 

*© 2025 cppfort – Trustless cpp2 Compiler with Betanet Transport*