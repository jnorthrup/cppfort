# Debugging Stage0 with Sanitizers

This document explains how to build and run the Stage0 test suite with AddressSanitizer and Undefined Behavior Sanitizer enabled, and how to collect useful diagnostics.

## Configure & Build with Sanitizers

Run the provided script in the project root:

```bash
./scripts/run_stage0_with_sanitizers.sh <build-dir>
```

By default the script uses `build_san` as the build directory; `STAGE0_ENABLE_SANITIZERS=ON` is passed to CMake.

## Running a specific test

To run the failing test with sanitizers enabled, call the script (or invoke the built test directly):

```bash
./scripts/run_stage0_with_sanitizers.sh build_san
# or run the test directly
ctest --test-dir build_san/src/stage0 --output-on-failure -R test_orbit_pipeline_pattern_selection
```

## Watchdog & Signal Handlers

The Stage0 tests include a small backtrace and watchdog helper:

- Signal handlers print a backtrace for common signals (SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL) and re-raise them to produce core dumps.
- A watchdog thread monitors the environment variable `DEBUG_WATCHDOG_SECONDS`. If set and the timeout elapses, the process will abort.

You can configure it for a test by adding a CTest test property ENVIRONMENT or by passing env variables at the CLI:

```bash
env DEBUG_WATCHDOG_SECONDS=10 ctest --test-dir build_san/src/stage0 --output-on-failure -R test_orbit_pipeline_pattern_selection
```

## Interpreting Sanitizer Output

- AddressSanitizer will print a stack trace when it detects use-after-free, double-free, stack overflow, out-of-bounds reads/writes, or global buffer overflow.
- Undefined Behavior Sanitizer will list UB incidents such as misaligned accesses, signed integer overflow, invalid `memcpy` sizes, etc.

Use the trace to find the originating code location; you can run under a debugger with the sanitizer-enabled binary as well to get a human-friendly trace (gdb/lldb).

---
This document is intended to help Stage0 developers reproduce and debug memory errors and undefined behavior using sanitizers and the included debug helpers.
