# ADR-002: cpp2.h Distribution Strategy

## Context

The Stage 1 transpiler generates C++ code that depends on the `cpp2.h` header file. This creates a dependency that users must satisfy by having the header in their include path. We need to decide on the best strategy for distributing this dependency.

## Decision

We will implement a hybrid approach that provides multiple options for handling the cpp2.h dependency:

1. **Inline cpp2.h contents** - Embed the minimal required definitions directly in generated code
2. **Bundle cpp2.h with output** - Prepend cpp2.h contents to generated code  
3. **Installation script** - Provide scripts to install cpp2.h to standard locations

## Status

Accepted

## Consequences

### Positive
- Users can choose the approach that best fits their workflow
- Generated code can be compiled without external dependencies when using inline/bundle options
- Installation script provides a clean solution for development environments
- Maintains backward compatibility with existing workflows

### Negative
- Increased complexity in the emitter implementation
- Larger generated code size when inlining or bundling
- Need to maintain multiple distribution mechanisms

## Options Considered

### Option A: Inline cpp2.h Contents
- **Pros**: Standalone output, no external dependencies
- **Cons**: Larger generated code size, maintenance overhead

### Option B: Bundle cpp2.h with Output
- **Pros**: Standalone output, clear separation of generated code and dependencies
- **Cons**: Larger output files, potential duplication

### Option C: Installation Script
- **Pros**: Clean separation, smaller generated code
- **Cons**: Additional installation step required

### Option D: Keep as External Dependency
- **Pros**: Minimal generated code size
- **Cons**: Users must manage dependency, -I flags required

## Selected Approach

We chose to implement all three active options (A, B, C) to provide maximum flexibility to users while maintaining backward compatibility.