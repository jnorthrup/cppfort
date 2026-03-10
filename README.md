# cppfort

**Cpp2 -> C++ transpiler with MLIR/Sea-of-Nodes work fed by Clang semantics and the cppfront corpus**

## Policy

- `cmake` and `ninja` are the only sanctioned build tools.
- `ninja -C build conveyor` is the authoritative end-to-end workflow.
- Ad hoc scripts under `tools/` and `tools/inference/` are cheats and illegal. They are implementation details and debugging aids, not supported entrypoints.

## What the conveyor does

The built executable `cppfront_conveyor` enforces the corpus pipeline:

1. Requires `third_party/cppfront` to exist and be clean.
2. Syncs `third_party/cppfront/regression-tests` into `tests/regression-tests` and `corpus/inputs`.
3. Runs `ctest` so the repo regression surface stays current.
4. Builds the primary corpus by transpiling `.cpp2` with `cppfront`.
5. Transpiles that same primary corpus with `cppfort`.
6. Dumps Clang ASTs for both outputs.
7. Scores transpile accuracy from AST isomorphs / semantic loss.
8. Emits Clang-derived semantic mappings so chunk ownership can be assigned back into the transpiler.

Reflective semantics come from Clang. Reflective grammar comes from the built cppfront corpus.

## Build

### Prerequisites

```bash
brew install llvm cmake ninja
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

### Spartan workflow

```bash
cmake -S . -B build -G Ninja
ninja -C build conveyor
```

That target builds `cppfront`, builds `cppfort`, runs tests, populates the regression corpus, produces cppfront reference outputs, runs cppfort against the same corpus, scores AST isomorph loss, and emits semantic mappings.

### Useful targets

| Target | Purpose |
| --- | --- |
| `cppfront` | Build the bundled cppfront reference transpiler |
| `cppfort` | Build the current cppfort transpiler |
| `cppfront_conveyor` | Build the authoritative conveyor executable |
| `conveyor` | Run the full corpus/test/inference conveyor |
| `test` | Run CTest suites |

## Outputs

The conveyor writes artifacts under `build/conveyor/`:

- `candidate_cpp/`: cppfort-generated C++
- `candidate_ast/`: Clang ASTs for cppfort output
- `scores/`: AST isomorph and semantic-loss artifacts
- `mappings/`: aggregated Clang semantic mappings
- `CONVEYOR_SUMMARY.md`: run summary and artifact locations

Primary reference corpus outputs are written under:

- `corpus/reference/`
- `corpus/reference_ast/`

## Architecture

```text
cppfront regression corpus (.cpp2)
  -> cppfront reference C++
  -> Clang AST + semantic sections
  -> isomorph / semantic-loss scoring
  -> mapping inference
  -> chunk-assigned semantic mappings for cppfort
```

## Notes for internals

- The Python files in `tools/inference/` still implement the mapping engine, but the supported way to run them is through `cppfront_conveyor`.
- If you are touching the internals, keep the contract intact: cppfront corpus first, Clang semantics second, transpiler mappings last.

## References

- [cppfront](https://github.com/hsutter/cppfront)
- [MLIR](https://mlir.llvm.org/)
- [Sea of Nodes IR](https://github.com/SeaOfNodes/Simple)
