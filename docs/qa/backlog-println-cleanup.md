# Println/Debug Output Cleanup - Backlog

**Status:** Deferred
**Priority:** Low
**Category:** Code Hygiene
**Estimated Effort:** 1-2 hours

## Overview

Production code contains debug print statements that create noise and should be gated behind verbosity flags or removed.

## Source Files Ranked by Print Volume

### Test Files (Keep as-is)
- `test_chapter2.cpp` - 31 prints ✓ Expected
- `test_chapter3.cpp` - 30 prints ✓ Expected
- `test_chapter1.cpp` - 20 prints ✓ Expected
- `test_band1.cpp` - 6 prints ✓ Expected

### Production Code (Needs cleanup)

#### High Priority: Remove or flag-gate

**src/stage1/transpiler.cpp** (10 prints)
```
Line 51: std::cout << "Parsed AST with " << ast.functions.size() << " functions\n";
Line 54: std::cout << "Function: " << fn.name << "\n";
Line 57: std::cout << "  param " << p.name << " kind=" << k << " type='" << p.type << "'\n";
Line 66: std::cout << "Emitted C++ code, length: " << transformed.size() << "\n";
Line 80: std::cout << "Wrote transformed C++ to " << output_path << "\n";
```

**Action:** Add `--verbose` flag and gate all progress output behind it.

**src/stage0/main.cpp** (9 prints)
```
Line 59:  std::cout << "Wrote " << output_path << "\n";
Line 107: std::cout << "Parsed " << successes << " of " << attempted ...
Line 110: std::cout << "(Additional " << (failures - 5) << " failures not shown.)"
```

**Action:** Add `--quiet` flag and suppress progress output when enabled. Keep error output on stderr.

#### Low Priority: Keep (acceptable)

**src/stage0/gcm.h** (2 prints in debug method)
```
Line 170-172: debugLateSchedule() method
```
**Rationale:** Explicit debug API, not noise in normal execution.

**src/stage0/emitter.cpp** (4 references - NOT actual prints)
**src/stage0/bidirectional.cpp** (2 references - NOT actual prints)
**Rationale:** String pattern matching for UFCS transforms (fprintf, etc), not actual debug output.

## Proposed Implementation

### Phase 1: Add Verbosity Control
```cpp
// Add to main.cpp command line parsing
bool opt_verbose = false;
bool opt_quiet = false;

// In transpiler.cpp and main.cpp:
if (opt_verbose) {
    std::cout << "Parsed AST with " << ast.functions.size() << " functions\n";
}
```

### Phase 2: Standardize Output
- Progress/info → stdout (when not --quiet)
- Errors → stderr (always)
- Debug → optional (only with --verbose)

## Acceptance Criteria
- [ ] Default run produces minimal output (only errors and final result)
- [ ] `--verbose` flag enables detailed progress logging
- [ ] `--quiet` flag suppresses all non-error output
- [ ] Test files unchanged
- [ ] Error messages always visible on stderr

## Net Impact
- **Production prints removed:** 7-10 statements
- **User experience:** Cleaner default output, optional verbosity
- **No functionality change:** Only output control

## Related Files
- `src/stage1/transpiler.cpp`
- `src/stage0/main.cpp`

## Notes
This is polish work that improves UX but doesn't affect core functionality. Defer until post-Band 4 or when command-line interface is being refined.
