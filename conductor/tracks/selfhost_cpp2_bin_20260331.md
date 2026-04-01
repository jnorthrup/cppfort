# Track: Self-Hosted cpp2_bin

*Status: ACTIVE*
*Started: 2026-03-31*

## Objective
Build `cpp2_bin` — a self-hosted cpp2 transpiler that reads .h2/.cpp2 files, scans -> folds -> lexes -> parses, then emits region graph + token stream + original source.

## Architecture
- `trikeshed.h2` — join<A,B>, series<T>, strview, char_series (pure C++ headers, cppfront passes through)
- `cpp2.h2` — bitmap scanner: pixel, region, scan(), fold() (pure C++ headers)
- `bbcursive.h2` — lexer: tok, kind, lex(), reader with combinators (chlit, strlit, confix, opt, rep, and_, not_), parse() (pure C++ headers)
- `cpp2_main.cpp2` — main entry: reads file, runs scan/fold/lex/parse pipeline, emits diagnostics (cpp2, transpiled by cppfront)

## Key Design Decisions
1. .h2 files are pure C++ — cppfront passes them through verbatim as #include
2. cppfront rewrites `#include "X.h2"` to `#include "X.h"` — we transpile .h2 -> .h separately in CMake
3. cpp2's `as` does not support enum->int conversion — added `to_int()` helper in cpp2.h2
4. `join<A,B>` uses deleted copy + move semantics (cppfront @struct pattern) — pure C++ avoids this

## Build
```bash
cmake --build build --target cpp2_bin
./build/src/selfhost/cpp2_bin src/selfhost/cpp2_main.cpp2
```

## Verified (2026-03-31)
- `cpp2_bin` builds and runs
- Scan produces correct region graph (100 regions from cpp2_main.cpp2)
- Passthrough emits original source
- All existing tests pass: `ctest --test-dir build -R selfhost` → 1/1 PASS

## Remaining Work
- `lex()` is stub (returns empty series)
- `parse()` is stub (returns nullopt)
- Reader combinators exist but untested against real token stream
- No code generation yet (passthrough only)
