# RESCINDED: MultiIndex2D Container

**Date:** 2025-09-30
**Status:** To Be Removed
**Replacement:** Subsumption Engine (see ADR-001)

## Issue

The file `src/utils/multi_index.h` contains a `MultiIndex2D<T, Rows, Cols>` class that is:
1. **Misnamed** - "Multi-index" has a specific meaning in this project (subsumption queries)
2. **Misleading** - Suggests Boost-style multi-index containers
3. **Wrong abstraction** - Simple 2D array doesn't relate to compiler needs

## Architectural Clarification

The term **"multi-index"** in cppfort refers to:
- Multi-criteria subsumption queries (type + CFG + data flow)
- Hierarchical rule matching for MLIR/graph projections
- N-way graph transformations via feature projection

It does NOT refer to:
- Generic container libraries
- 2D/3D array indexing
- Boost multi-index containers

## Action Required

### Remove from `src/utils/multi_index.h`:
```cpp
// DELETE THIS:
template <typename T, std::size_t Rows, std::size_t Cols>
requires (Rows > 0 && Cols > 0)
class MultiIndex2D {
    // ... 119 lines of 2D array code
};
```

### Keep (if needed):
```cpp
// EnumBase CRTP helper is fine (unrelated to multi-index concept)
template <typename Enum>
struct EnumBase {
    // ... enum utilities
};
```

### Replace with:
Design and implement the **Subsumption Engine** per:
- [ADR-001: Multi-Index Subsumption Engine](decisions/ADR-001-multi-index-subsumption-engine.md)
- [Subsumption Engine Architecture](subsumption-engine.md)

## If 2D Array is Actually Needed

If a 2D array container is genuinely needed somewhere, either:
1. Use `std::vector<std::vector<T>>` or `std::array<std::array<T, Cols>, Rows>`
2. Create `src/utils/array2d.h` with appropriate naming
3. Inline the 2D indexing math where used (often clearest)

But do NOT call it "multi-index" as that has a specific subsumption meaning.

## Migration Path

1. **Audit usages** - Check if MultiIndex2D is used anywhere
2. **Remove or rename** - Delete class or move to properly-named file
3. **Document intent** - Clear comments about subsumption engine
4. **Implement foundation** - Hash-based primary index first

## Verified Requirements

From the architect (Jim):
> "i know 100% we need hash based lookup as intended, and THEN we need it as a
> multi-index that will support hierarchical subsumption rules which will guide
> mlir and n-way graph conversions through projection against node features"

This confirms:
- ✅ Hash-based lookup (O(1) primary index)
- ✅ Hierarchical subsumption rules (type/CFG/data lattices)
- ✅ MLIR integration (pattern matching, dialect conversions)
- ✅ N-way graph projections (Sea of Nodes ↔ MLIR ↔ other IRs)
- ✅ Node feature queries (multi-criteria filtering)

This does NOT confirm:
- ❌ Need for 2D array containers
- ❌ Generic indexing utilities
- ❌ Boost-style multi-index containers

## Status Tracking

- [ ] Audit `MultiIndex2D` usages
- [ ] Remove or relocate class
- [ ] Clean up `multi_index.h` file
- [ ] Begin subsumption engine design
- [ ] Document intended API surface

## Related Documents

- [ADR-001: Multi-Index Subsumption Engine](decisions/ADR-001-multi-index-subsumption-engine.md)
- [Subsumption Engine Architecture](subsumption-engine.md)
- [Coding Standards](coding-standards.md)
