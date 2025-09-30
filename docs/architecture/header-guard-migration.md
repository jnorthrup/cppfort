# Header Guard Migration Status

## Standard: `#pragma once`

As of 2025-09-30, the cppfort project has adopted `#pragma once` as the mandatory standard for header guards (see [coding-standards.md](coding-standards.md)).

## Files Requiring Migration

The following files still use legacy sentinel-style header guards and should be migrated:

### stage0
- `src/stage0/type.h` - `#ifndef CPPFORT_TYPE_H`
- `src/stage0/node.h` - `#ifndef CPPFORT_NODE_H`
- `src/stage0/son_parser.h` - `#ifndef CPPFORT_SON_PARSER_H`
- `src/stage0/iterpeeps.h` - `#ifndef CPPFORT_ITERPEEPS_H`
- `src/stage0/gcm.h` - `#ifndef CPPFORT_GCM_H`

### stage1
_(Check for additional files)_

### stage2
_(Check for additional files)_

### utils
_(Check for additional files)_

## Migration Process

For each file:

1. **Remove** the opening guard:
   ```cpp
   #ifndef CPPFORT_FILENAME_H
   #define CPPFORT_FILENAME_H
   ```

2. **Replace** with:
   ```cpp
   #pragma once
   ```

3. **Remove** the closing guard:
   ```cpp
   #endif // CPPFORT_FILENAME_H
   ```

4. **Verify** compilation succeeds after migration

5. **Commit** with message: `style: Migrate FILENAME.h to #pragma once`

## Priority

**Low** - Migrate opportunistically during other edits to these files. No urgent need to migrate all at once, but all new headers MUST use `#pragma once`.

## Tracking

- [ ] `src/stage0/type.h`
- [ ] `src/stage0/node.h`
- [ ] `src/stage0/son_parser.h`
- [ ] `src/stage0/iterpeeps.h`
- [ ] `src/stage0/gcm.h`
- [ ] Additional files (TBD after full scan)

## Notes

All modern compilers support `#pragma once`:
- GCC 3.4+ (2004)
- Clang 2.6+ (2009)
- MSVC 7.1+ (2003)
- Intel C++ 11.0+ (2008)

No compatibility concerns for this project's target compilers.
