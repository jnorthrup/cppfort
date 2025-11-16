Regression Runner (Git history)
================================

This executable extracts `regression-tests/*.cpp2` files from the git history and runs them through the transpile/compile/execute pipeline.

Usage:

  regression_runner_git <repo_root> <patterns> <stage0_cli> [--limit N] [filter]

Options:
- `repo_root` - path to root of repo where git commands are run
- `patterns`  - path to bnfc/semantic patterns used for transpilation
- `stage0_cli` - path to the stage0 CLI binary (unused but accepted to remain compatible)
- `--limit N` - optional limit of commits to scan (default: all)
- `filter` - optional substring to match in filenames

Examples:

  # Scan the last 200 commits of the repo, running all historical regression tests
  cmake --build build && cmake --build build --target regression_tests_git

Notes:
- The command creates a `regression_git_log.txt` in the working directory with details.
- This is primarily for historical analysis; it may be slow when scanning many commits.

Using cppfront as a reference compiler
--------------------------------------

The project contains a copy of Herb Sutter's `cppfront` in `third_party/cppfront` (cloned from `hsutter/cppfront`). It includes a large suite of regression tests in `third_party/cppfront/regression-tests`.

You can copy these tests into your local `regression-tests` tree and build a `cppfront` binary for comparison using the provided scripts:

 - `scripts/grab_cppfront_regression_tests.sh` — copies the `cppfront` regression tests into the local `regression-tests/cppfront/` folder (non-destructive by default)
 - `scripts/build_cppfront.sh` — builds a local cppfront binary at `build/third_party/cppfront/cppfront` using your local C++ compiler
 - `scripts/run_cppfront_regression_tests.sh` — builds cppfront (if necessary), and runs the `cppfront` repo's `run-tests.sh` harness using your local compiler to validate expected outputs

If you want to add `cppfront` as a reference compiler to our `regression_runner` logic, we can either:

 1) Add a `--cppfront <path>` argument to `regression_runner` and `regression_runner_git` that uses the `cppfront` binary to transpile tests into C++1, then compare our transpilation outputs against cppfront's outputs. This lets you track behavioral differences–I can implement this next.
 2) Or use the included `third_party/cppfront/regression-tests/run-tests.sh` harness which already validates the generated outputs vs `test-results/` references included with `cppfront`.

On macOS/arm64 you may need a specific compiler and link flags to build `cppfront` successfully. The `build_cppfront.sh` script attempts a default build and will retry with `-stdlib=libc++` and `-lc++abi` if the default fails. If the binary fails to link, please try building with a matching clang++/libc++ toolchain or adjusting the build flags as required.

