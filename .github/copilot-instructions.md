# Copilot / AI Agent Instructions — cppfort

Purpose: give an AI code assistant the immediate, focused context needed to be productive in this repo.

1) Big picture (read before editing)
- This repo implements a three-stage cpp2 → C++ pipeline: `src/stage0` (AST + emitter),
  `src/stage1` (transpiler / parser front), and `src/stage2` (attestation/anticheat).
- The project is evolving toward a Sea‑of‑Nodes IR (meta-transformer patterns in JSONL).
  Key architecture overview is in `README.md` and `docs/architecture.md` — read those first.

2) Key places to look / edit
- Language/core: `src/stage0/` — AST, emitter, canonical node classes.
- Transpiler front: `src/stage1/` — parser, translation-unit construction, stage1_cli.
- Attestation: `src/stage2/` — `anticheat` helper and attestation pipeline.
- Patterns & meta-layer: `src/*/meta` and `docs/Simple/` + any `*.jsonl` files (pattern language).
- Public API/headers: `include/cpp2.h`, `include/cpp2_impl.h`.

3) Build / test / debug — exact commands
- Standard build (out-of-source CMake):
  mkdir -p build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Debug
  cmake --build .

- Run built CLIs (from `build/`):
  ./stage1_cli <in.cpp2> <out.cpp>
  ./stage0_cli <in.cpp2>  # emits C++ via stage0 emitter
  ./anticheat <path_to_binary>

- Regression suites (use these after code changes):
  regression-tests/run_tests.sh            # basic regression tests
  regression-tests/run_attestation_tests.sh
  chmod +x regression-tests/run_triple_induction.sh && regression-tests/run_triple_induction.sh

4) Project-specific patterns & conventions (do not assume defaults)
- The repo follows an incremental 'Simple chapters' progression — changes should preserve earlier chapter behavior.
- Pattern files are JSONL-driven for meta-transformations. If you add/remove patterns update any loader in
  `src/*/meta` and run the small smoke test that exercises pattern loading.
- TableGen (where present) is used to generate JSONL → check `tools/` or `src/*/meta` integration tests.

5) Testing & acceptance criteria for AI-made changes
- Small code changes: run `cmake --build .` then `regression-tests/run_tests.sh`.
- Any change to emitters, AST or patterns must not break regression harness. If you touch stage0/stage1,
  run the triple induction script to ensure the end-to-end feedback loop still passes.
- Use `build/compile_commands.json` to find translation units for quick navigation and LSP actions.

6) Where automated/hand-authored docs live (use them as canonical sources)
- `README.md` (top-level) — high-level architecture and exact build/test commands.
- `docs/architecture.md` and `docs/Simple/` — design rationale, Simple-chapter mapping, and the Sea‑of‑Nodes plan.
- `AGENTS.md`, `.clinerules/` and `.cursor/rules/` — repo-specific agent rules and personas (useful when an AI agent
  must obey repo policies).

7) Useful examples (copy-paste friendly)
- Transpile a sample file with the stage1 CLI (from repository root):
  build/stage1_cli src/stage1/main.cpp2 stage1_output.cpp

- Run the attestation on a binary (produces SHA‑256 of disassembly):
  build/anticheat ./test_binary

8) Safe-edit guidance / quick checks
- Prefer small, atomic commits. Run the regression harness before pushing.
- Do not remove or reformat `docs/Simple/` material — it maps to the Simple chapter progression used by the meta-layer.
- When changing patterns or TableGen outputs, check-in both source pattern files and a short note in the commit message
  that indicates which regression tests you ran.

9) Where to leave pointers for human reviewers
- If the change affects regression behavior, add a short summary in the commit message and in the PR description:
  which stage was touched, which regression script was run, and the exit status of the tests.

10) If you are an AI agent following persona rules
- Consult `AGENTS.md` and `.clinerules/*` for local agent constraints (owner/edit permissions, which tasks require human elicitations).
- If a task asks for interactive elicitation, follow the repository’s 1–9 options convention found in `.bmad-core/tasks/create-doc.md` (see AGENTS.md references).

If anything in these notes is unclear or you want an expanded section (examples for pattern edits, or a short checklist for emitter changes), say which area and I will expand it.
---

## Pattern-edit checklist (concrete)
When editing JSONL pattern files or TableGen outputs, follow this checklist and run the smoke test below.

- File globs to inspect/modify:
  - Pattern sources: `src/**/meta/**/*.jsonl`
  - Pattern loaders: `src/**/meta/*` (look for `load_patterns`, `jsonl::` usage)
  - TableGen inputs / generated: `tools/**`, `src/**/meta/*.td`, `src/**/meta/*_generated.*`
  - Tests/examples: `docs/Simple/**`, `regression-tests/**` (new sample files belong here)

- Checklist before committing:
  1. Update or add JSONL under `src/*/meta/` and commit the source JSONL (not only generated code).
  2. If TableGen (.td) changed, check-in both `.td` and the generated `.jsonl`/sources or add a short note in the commit.
  3. Run the local pattern smoke test (below) to ensure loader can parse and register patterns.
  4. Run `cmake --build .` and `regression-tests/run_tests.sh`.
  5. If patterns affect emitter behavior, run `regression-tests/run_triple_induction.sh`.

### Pattern smoke-test (small, fast)
Create a temporary build and run the pattern loader smoke test. From repo root:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j2
# Run a tiny binary that exercises pattern loading if available, else run stage0_cli on a minimal sample
if [ -x stage0_cli ]; then
  ./stage0_cli ../regression-tests/mixed-bounds-check.cpp2 >/dev/null || true
else
  echo "stage0_cli not built; run full build and then the smoke test"
fi
```

Adjust the sample path to a minimal `.cpp2` test that touches the patterns you changed.

## Emitter / AST change checklist
Emitter and AST edits are sensitive — follow this focused checklist.

- Files & areas to inspect:
  - AST & nodes: `src/stage0/ast*`, `src/stage0/node*`, `include/cpp2_impl.h`
  - Emitter: `src/stage0/emitter*`, `src/stage0/emit_*` (search for `emit_` helpers)
  - Tests: `regression-tests/` and `tests/` that exercise emitted C++/roundtrip

- Quick checklist before PR:
  1. Run compile for the project (prefer in `build/`): `cmake --build .`.
  2. Use `build/compile_commands.json` to identify translation units touching stage0 AST/emitter and run them via the compiler for fast feedback. Example targets to run from compile_commands.json:
     - `src/stage0/main.cpp` (or whichever TU declares the emitter)
     - `src/stage0/emitter.cpp` / `src/stage0/ast.cpp` (look for exact filenames in compile_commands.json)
  3. Run unit/regression tests: `regression-tests/run_tests.sh` and `regression-tests/run_triple_induction.sh` when changing semantics.
  4. If you change public headers (`include/cpp2.h`, `include/cpp2_impl.h`) update docs and run a full build.

Notes on using compile_commands.json:
- Open `build/compile_commands.json` (generated by CMake). Search for source files under `src/stage0` to find exact TU file names. You can compile a single translation unit manually with:

```bash
g++ -c <path/to/tu.cpp> -Iinclude -I<other include paths from compile_commands.json> -o /tmp/tu.o
```

This gives quicker compile feedback than a full project rebuild.

## Selected policy excerpts merged from `AGENTS.md`
The repo contains a detailed `AGENTS.md` with agent rules. AI assistants should obey these highlights:

- Use `AGENTS.md` and `.clinerules/*` to discover agent-specific constraints (which agent role owns which sections, elicitation formats, and when to halt for human inputs).
- When templates/tasks indicate `elicit: true`, follow the repository's 1–9 numeric elicitation protocol (see `.bmad-core/tasks/create-doc.md`).
- Changes to documentation or templates must preserve the repository's required interaction patterns (do not remove elicitation hooks).

If you want the full `AGENTS.md` policy inserted or a tailored agent permission matrix for this repo, I can expand this section.
