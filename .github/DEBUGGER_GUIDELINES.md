# Avoid leaving debuggers or debugging constructs in commits

This document describes a short checklist and guardrails to avoid leaving interactive debuggers or debug-only constructs (infinite loops, manual pauses, lldb/gdb commands, etc.) in the source tree, especially in CI or production builds.

## Why this matters

- Interactive debugger calls and manual breakpoints can leave tests, builds, or CI runs stalled or aborted. This slows down development and may mask bugs.

- Sanitizers and test harnesses are preferred for automated diagnostics — they provide structured and reproducible output.

## Developer responsibilities

- Remove or gate any temporary interactive debugger usage before committing. Examples include:

  - `lldb`, `gdb`, or `pdb.set_trace()` invocations in source and scripts.

  - Input waits: `std::cin`, `scanf`, `getchar`, `system("read")`, `system("pause")` in paths executed by tests/CI.

  - Sleep/wait loops like `sleep()`, `usleep()`, `std::this_thread::sleep_for()` used to block tests.

  - Infinite loops or `while(true)` used to halt execution during debugging.

  - Calls intentionally aborting the program (`__builtin_trap()`, `abort()`) used for temporary breakpoints.

## Local debugging best practices

- Prefer short, gated debug helpers instead of interactive breakpoints. Use compile-time or environmental guards for debug features:

  - Example (compile-time):

    ```cpp
    #ifdef ENABLE_DEBUG_HELPERS
    debug::install_signal_handlers();
    debug::start_watchdog_from_env();
    #endif
    ```

  - Example (runtime):

    ```cpp
    if (std::getenv("ENABLE_DEBUG_HELPERS")) {
        debug::install_signal_handlers();
        debug::start_watchdog_from_env();
    }
    ```

- Prefer adding deterministic tests or sanitizer/ASAN runs instead of interactive steps for verification.

## Scripted checks and CI enforcement

- The `scripts/check_for_debuggers.sh` script scans `src/` and `tests/` for common debugging patterns (e.g., `lldb`, `gdb`, `sleep`, `std::cin`, `getchar`). It fails if suspicious patterns are found, except for files listed in the script's allowlist.

- The CI job (`.github/workflows/ci.yml`) runs `scripts/check_for_debuggers.sh` as an early step — PRs that introduce disallowed debug constructs will fail this CI step.

## How to run the check locally

```bash
bash scripts/check_for_debuggers.sh
```

## If the check finds potential issues

- If the use is allowed by design (e.g., `src/stage0/test_timeout_verification.cpp` intentionally sleeps to verify test timeouts), add an `ALLOW_DEBUG` comment or list the file in the script's allowlist.

- Otherwise, gate the debug code behind a compile-time option or remove it. When gating, provide a brief comment about why the gating exists and ensure the behavior is covered by a deterministic test.

## Adding a new exception

- If you believe a new pattern is valid and should be whitelisted: open a PR, explain the justification and add the file/pattern to the allowlist in `scripts/check_for_debuggers.sh`.

## Example guidance

- Allowed: `src/stage0/test_timeout_verification.cpp` (explicitly whitelisted) — this test intentionally sleeps to verify test harness timeout behavior.

- Not allowed: including an interactive `lldb` invocation or `while(true) {}` loops in `src/` files. Convert to gated test or remove before committing.

## References

- CI configuration: `.github/workflows/ci.yml`
- Debug helpers: `src/stage0/debug_helpers.h` and `src/stage0/debug_helpers.cpp`

Want a pre-commit hook for this check? I can add `scripts/install-git-hooks.sh` and a minimal `husky`-style hook script to run the check automatically on the developer's machine.
# Avoid leaving debuggers or debugging constructs in commits

This document describes a short checklist and guardrails to avoid leaving interactive debuggers, manual pausing, or debug-only constructs (infinite loops, lldb/gdb commands, etc.) in the source tree, especially in CI or production builds.

## Why this matters

- Interactive debugger calls and manual breakpoints can leave tests, builds, or CI runs stalled or aborted. This slows down development and may mask bugs.

- Sanitizers, signal handlers, and test harnesses are preferred for automated diagnostics — they provide structured and reproducible output and are better suited for CI.

## Developer responsibilities

- Remove or gate any temporary interactive debugger usage before committing. Examples include:

  - `lldb`, `gdb`, or `pdb.set_trace()` invocations in source and scripts.

  - Input waits: `std::cin`, `scanf`, `getchar`, `system("read")`, `system("pause")` in paths executed by tests/CI.

  - Sleep/wait loops like `sleep()`, `usleep()`, `std::this_thread::sleep_for()` used to block tests.

  - Infinite loops or `while(true)` used to halt execution during debugging.

  - Calls intentionally aborting the program (`__builtin_trap()`, `abort()`) used for temporary breakpoints.

## Local debugging best practices

- Prefer short, gated debug helpers instead of interactive breakpoints. Use compile-time or environmental guards for debug features:

  - Example (compile-time):

    ```cpp
    #ifdef ENABLE_DEBUG_HELPERS
    debug::install_signal_handlers();
    debug::start_watchdog_from_env();
    #endif
    ```

  - Example (runtime):

    ```cpp
    if (std::getenv("ENABLE_DEBUG_HELPERS")) {
        debug::install_signal_handlers();
        debug::start_watchdog_from_env();
    }
    ```

- Prefer adding deterministic tests and sanitizer/ASAN runs instead of interactive steps for verification.

## Scripted checks and CI enforcement

- The `scripts/check_for_debuggers.sh` script scans `src/` and `tests/` for common debugging patterns (e.g., `lldb`, `gdb`, `sleep`, `std::cin`, `getchar`). It fails if suspicious patterns are found, except for files listed in the script's allowlist.

- The CI job (`.github/workflows/ci.yml`) runs `scripts/check_for_debuggers.sh` as an early step — PRs that introduce disallowed debug constructs will fail this CI step.

## How to run the check locally

```bash
bash scripts/check_for_debuggers.sh
```

## If the check finds potential issues

- If the use is allowed by design (e.g., `src/stage0/test_timeout_verification.cpp` intentionally sleeps to verify test timeouts), add an `ALLOW_DEBUG` comment or list the file in the script's allowlist.

- Otherwise, gate the debug code behind a compile-time option or remove it. When gating, provide a brief comment about why the gating exists and ensure the behavior is covered by a deterministic test.

## Adding a new exception

- If you believe a new pattern is valid and should be whitelisted: open a PR, explain the justification and add the file/pattern to the allowlist in `scripts/check_for_debuggers.sh`.

## Example guidance

- Allowed: `src/stage0/test_timeout_verification.cpp` (explicitly whitelisted) — this test intentionally sleeps to verify test harness timeout behavior.

- Not allowed: including an interactive `lldb` invocation or `while(true) {}` loops in `src/` files. Convert to gated test or remove before committing.

## References

- CI configuration: `.github/workflows/ci.yml`
- Debug helpers: `src/stage0/debug_helpers.h` and `src/stage0/debug_helpers.cpp`

If you want me to add a pre-commit Git hook to run the check automatically or add a CI gating step, tell me and I can insert a minimal pre-commit script `scripts/install-git-hooks.sh`.
# Avoid leaving debuggers or debugging constructs in commits

This document describes a short checklist and guardrails to avoid leaving interactive debuggers, manual pausing, or debug-only constructs (infinite loops, lldb/gdb commands, etc.) in the source tree, especially in CI or production builds.

## Why this matters

- Interactive debugger calls and manual breakpoints can leave tests, builds, or CI runs stalled or aborted. This slows down development and may mask bugs.

- Sanitizers, signal handlers, and test harnesses are preferred for automated diagnostics — they provide structured and reproducible output and are better suited for CI.

## Developer responsibilities

- Remove or gate any temporary interactive debugger usage before committing. Examples include:

  - `lldb`, `gdb`, or `pdb.set_trace()` invocations in source and scripts.
  - Input waits: `std::cin`, `scanf`, `getchar`, `system("read")`, `system("pause")` in paths executed by tests/CI.
  - Sleep/wait loops like `sleep()`, `usleep()`, `std::this_thread::sleep_for()` used to block tests.
  - Infinite loops or `while(true)` used to halt execution during debugging.
  - Calls intentionally aborting the program (`__builtin_trap()`, `abort()`) used for temporary breakpoints.

## Local debugging best practices

- Prefer short, gated debug helpers instead of interactive breakpoints. Use compile-time or environmental guards for debug features:

  - Example (compile-time):

    ```cpp
    #ifdef ENABLE_DEBUG_HELPERS
    debug::install_signal_handlers();
    debug::start_watchdog_from_env();
    #endif
    ```

  - Example (runtime):

    ```cpp
    if (std::getenv("ENABLE_DEBUG_HELPERS")) {
        debug::install_signal_handlers();
        debug::start_watchdog_from_env();
    }
    ```

- Prefer adding deterministic tests and sanitizer/ASAN runs instead of interactive steps for verification.

## Scripted checks and CI enforcement

- The `scripts/check_for_debuggers.sh` script scans `src/` and `tests/` for common debugging patterns (e.g., `lldb`, `gdb`, `sleep`, `std::cin`, `getchar`). It fails if suspicious patterns are found, except for files listed in the script's allowlist (for example `test_timeout_verification.cpp` which intentionally sleeps to verify CTest TIMEOUT handling).

- The CI job (`.github/workflows/ci.yml`) runs `scripts/check_for_debuggers.sh` as an early step — PRs that introduce disallowed debug constructs will fail this CI step.

## How to run the check locally

```bash
bash scripts/check_for_debuggers.sh
```

If the script finds a suspicious pattern and you think it is legitimate, either:
- Add the file to the allowlist inside `scripts/check_for_debuggers.sh` (with justification), or
- Gate the behavior behind `ENABLE_DEBUG_HELPERS` or a similar mechanism so it doesn't affect CI runs.

## Adding a new allowlist entry

- If you need to add an allowlist for a test that intentionally exercises timeouts or sleeps, document the justification in the PR and add the file path to the `ALLOWLIST` in `scripts/check_for_debuggers.sh`.

## Example guidance

- Allowed (explicitly whitelisted): `src/stage0/test_timeout_verification.cpp` — this test intentionally sleeps to verify test harness timeout behavior.

- Not allowed: committing an `lldb` invocation or `while(true) {}` loop within `src/` files. Convert to gated code or remove prior to committing.

## References

- CI workflow: `.github/workflows/ci.yml`
- Debug helpers: `src/stage0/debug_helpers.h` and `src/stage0/debug_helpers.cpp`

Want a pre-commit hook for this check? I can add `scripts/install-git-hooks.sh` and a minimal `husky`-style hook script to run the check automatically on the developer's machine.
# Avoid leaving debuggers or debugging constructs in commits

This document describes a short checklist and guardrails to avoid leaving interactive debuggers or debug-only constructs (infinite loops, manual pauses, lldb/gdb commands, etc.) in the source tree, especially in CI or production builds.

## Why this matters

- Interactive debugger calls and manual breakpoints can leave tests, builds, or CI runs stalled or aborted. This slows down development and may mask bugs.

- Sanitizers and test harnesses are preferred for automated diagnostics — they provide structured and reproducible output.

## Guidelines (developer responsibilities)

- Remove or gate any temporary interactive debugger usage before committing:

  - `lldb`, `gdb`, `pdb`, `import pdb` or similar CLI debugger commands in scripts.

  - Calls to `std::cin`, `scanf`, `getchar`, `system("read")`, `system("pause")`, `sleep`/`usleep`, `std::this_thread::sleep_for()` in code paths exercised by tests or CI.

  - Manual infinite loops or `while(true) {}` used to halt execution.

  - Use of `__builtin_trap()` or intentionally causing SIGABRT outside of error handling.

  - Any file commented with `// DEBUG` or `// TODO: debug` should either be gated or removed before committing.

## Patterns to avoid and replace

- If you need an interactive run for local debugging, use compile-time or environment guards:

  - Wrap `install_signal_handlers()` and watchdog starts behind `#ifdef ENABLE_DEBUG_HELPERS` or the `ENABLE_DEBUG_HELPERS` CMake option.

  - Prefer `if (getenv("DEBUG_HELPERS_ENABLED")) { ... }` so CI remains stable unless explicitly enabled.

- Prefer a logging macro (e.g., `LOG_DBG(...)`) which can be compiled out in non-debug builds (or set `LOG_LEVEL`). Do not scatter raw `std::cout` debug prints in libraries.

## Pre-commit and CI checks (mandatory)

- Run `scripts/check_for_debuggers.sh` locally before committing or opening a PR. It scans C/C++ sources and scripts for suspicious patterns and fails if any are found, except in an allowed list.

- CI: The repository runs `scripts/check_for_debuggers.sh` and fails the job if it finds suspicious debug constructs.

## How to run the local check

```bash
bash scripts/check_for_debuggers.sh
```

## If the check finds potential issues

- If the use is allowed by design (e.g., `test_timeout_verification.cpp` intentionally sleeps to verify timeout semantics), add an `ALLOW_DEBUG` comment or list the file in the script's allowlist.

- Otherwise, gate the debug code behind a compile-time option or remove it. When gating, add an explanation comment and consider adding a test that mirrors the interactive behavior in a non-blocking, deterministic way.

## Adding a new exception

- If you believe a new pattern is valid and should be whitelisted: open a PR, explain the justification and add the file/pattern to the allowlist in `scripts/check_for_debuggers.sh`.

## Example canonical examples

- Allowed: `tests/test_timeout_verification.cpp` contains an intentional sleep to verify that test timeouts are recognized by CTest — keep it whitelisted.

- Not allowed: committing `lldb` invocation lines or a manual `while(true) {}` in `src/` files. Convert to a gated test or remove the loop.

## References

- CI workflow file: `.github/workflows/ci.yml` — ensure the script runs at the start of the job.

- Debug helpers: `src/stage0/debug_helpers.h/cpp` — use those safely and gate them.

If you want me to add a pre-commit Git hook to run the check automatically or add a CI gating step, tell me and I can insert a minimal `.github/workflows/ci.yml` change and/or `scripts/install-git-hooks.sh`.
# Avoid leaving debuggers or debugging constructs in commits

This document describes a short checklist and guardrails to avoid leaving interactive debuggers or debug-only constructs (infinite loops, manual pauses, lldb/gdb commands, etc.) in the source tree, especially in CI or production builds.

Why this matters
- Interactive debugger calls and manual breakpoints can leave tests, builds, or CI runs stalled or aborted. This slows down development and may mask bugs.
- Sanitizers and test harnesses are preferred for automated diagnostics — they provide structured and reproducible output.

Guidelines (developer responsibilities)
- Remove or gate any temporary interactive debugger usage before committing:
  - `lldb`, `gdb`, `pdb`, `import pdb` or similar CLI debugger commands in scripts.
  - Calls to `std::cin`, `scanf`, `getchar`, `system("read")`, `system("pause")`, `sleep`/`usleep`, `std::this_thread::sleep_for()` in code paths exercised by tests or CI.
  - Manual infinite loops or `while(true) {}` used to halt execution.
  - Use of `__builtin_trap()` or intentionally causing SIGABRT outside of error handling.
  - Any file commented with `// DEBUG` or `// TODO: debug` should either be gated or removed before committing.

Patterns to avoid and replace
- If you need an interactive run for local debugging, use compile-time or environment guards:
  - Wrap `install_signal_handlers()` and watchdog starts behind `#ifdef ENABLE_DEBUG_HELPERS` or the `ENABLE_DEBUG_HELPERS` CMake option.
  - Prefer `if (getenv("DEBUG_HELPERS_ENABLED")) { ... }` so CI remains stable unless explicitly enabled.
- Prefer a logging macro (e.g., `LOG_DBG(...)`) which can be compiled out in non-debug builds (or set `LOG_LEVEL`). Do not scatter raw `std::cout` debug prints in libraries.

Pre-commit and CI checks (mandatory)
- Run `scripts/check_for_debuggers.sh` locally before committing or opening a PR. It scans C/C++ sources and scripts for suspicious patterns and fails if any are found, except in an allowed list.
- CI: The repository must run `scripts/check_for_debuggers.sh` and fail the job if it finds suspicious debug constructs.

How to run the local check
```bash
bash scripts/check_for_debuggers.sh
```

If the check finds potential issues
- If the use is allowed by design (e.g., `test_timeout_verification.cpp` intentionally sleeps to verify timeout semantics), add an `ALLOW_DEBUG` comment or list the file in the script's allowlist.
- Otherwise, gate the debug code behind a compile-time option or remove it. When gating, add an explanation comment and consider adding a test that mirrors the interactive behavior in a non-blocking, deterministic way.

Adding a new exception
- If you believe a new pattern is valid and should be whitelisted: open a PR, explain the justification and add the file/pattern to the allowlist in `scripts/check_for_debuggers.sh`.

Example canonical examples
- Allowed: `tests/test_timeout_verification.cpp` contains an intentional sleep to verify that test timeouts are recognized by CTest — keep it whitelisted.
- Not allowed: committing `lldb` invocation lines or a manual `while(true) {}` in `src/` files. Convert to a gated test or remove the loop.

References
- CI workflow file: `.github/workflows/ci.yml` — ensure the script runs at the start of the job.
- Debug helpers: `src/stage0/debug_helpers.h/cpp` — use those safely and gate them.

If you want me to add a pre-commit Git hook to run the check automatically or add a CI gating step, tell me and I can insert a minimal `.github/workflows/ci.yml` change and/or `scripts/install-git-hooks.sh`.
