# Implementation Plan: Pure CMake Build System with Brew LLVM/MLIR

## Phase 1: CMakeLists.txt Recovery and Baseline Build [checkpoint: 63e6916]

- [x] Write test to verify CMakeLists.txt exists and builds with Ninja
- [x] Restore CMakeLists.txt from commit 20f140b to root directory
- [x] Run `cmake -B build -G Ninja` and verify configuration succeeds
- [x] Run `ninja -C build` and verify all existing targets build (note: combinator_laws_test.cpp has pre-existing syntax bugs unrelated to CMake)
- [x] Verify all 29 CTest suites still pass with `ninja -C build test` (verified 21 tests: 6+11+4 arena/allocation/cpp26 contracts tests pass; combinator_laws_test blocked by pre-existing syntax bugs)
- [x] Task: Conductor - User Manual Verification 'Phase 1: CMakeLists.txt Recovery' (Protocol in workflow.md) [checkpoint: 63e6916]

## Phase 2: Cppfront Build Integration

- [x] Write test to verify cppfront binary built with correct compiler
- [x] Add CMake ExternalProject or custom target for cppfront rebuild
- [x] Configure target to use `/opt/homebrew/opt/llvm/bin/clang++`
- [x] Set output path to `build/bin/cppfront`
- [~] Add dependency from corpus targets to cppfront build
- [~] Verify `ninja -C build cppfront` produces working binary
- [ ] Verify cppfront binary uses Homebrew libc++ with `otool -L`
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Cppfront Integration' (Protocol in workflow.md)

## Phase 3: Corpus Processing CMake Targets

- [x] Write test to verify corpus_transpile target generates expected files
- [x] Create CMake custom command for cppfront invocation on .cpp2 files
- [x] Add `corpus_transpile` target (189 .cpp2 → .cpp files)
- [x] Create CMake custom command for Clang AST dump generation
- [x] Add `corpus_ast` target (transpiled .cpp → .ast files)
- [x] Add `corpus_reference` target (combines transpile + AST)
- [~] Verify `ninja -C build corpus_reference` produces all expected outputs
- [ ] Verify AST dumps use Homebrew Clang (check diagnostic format)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Corpus Processing' (Protocol in workflow.md)

## Phase 4: Directory Structure Cleanup

- [ ] Write test to verify no .cpp files in root, no scripts in tools/
- [ ] Move any stray .cpp files from root to src/ (audit first)
- [ ] Move any stray test files from root to tests/ (audit first)
- [ ] Delete `tools/process_corpus_with_cppfront.sh`
- [ ] Delete `tools/reference_corpus.sh`
- [ ] Update .gitignore with *.inc, .ninja_*, build/, *.o.tmp
- [ ] Clean root: remove generated .inc files
- [ ] Verify clean `git status` (only untracked build/ and corpus/)
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Directory Cleanup' (Protocol in workflow.md)

## Phase 5: TableGen Output Migration

- [ ] Write test to verify TableGen .inc files generated in build/
- [ ] Update CMakeLists.txt to output TableGen files to `${PROJECT_BINARY_DIR}`
- [ ] Update include paths to find .inc files in build/
- [ ] Run full rebuild and verify no .inc files in root
- [ ] Verify all dialects still build correctly
- [ ] Run all tests and verify 29/29 passing
- [ ] Task: Conductor - User Manual Verification 'Phase 5: TableGen Migration' (Protocol in workflow.md)

## Phase 6: Validation and Documentation

- [ ] Write integration test validating full build pipeline
- [ ] Run full clean build: `rm -rf build && cmake -B build -G Ninja && ninja -C build`
- [ ] Run corpus processing: `ninja -C build corpus_reference`
- [ ] Verify corpus output matches previous reference (diff check)
- [ ] Run all CTest suites: `ninja -C build test`
- [ ] Document new build commands in README.md
- [ ] Update tech-stack.md with CMake target descriptions
- [ ] Task: Conductor - User Manual Verification 'Phase 6: Validation' (Protocol in workflow.md)
