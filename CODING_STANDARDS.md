# Coding Standards - cppfort

## ZERO TOLERANCE VIOLATIONS

### 1. NO REGEX
- No `#include <regex>`
- No `std::regex`, `std::regex_search`, `std::regex_replace`
- Use RBCursive combinators or direct string operations (`find()`, `substr()`)

### 2. NO MAKEFILES
- No Makefile, GNUmakefile, makefile
- CMake and Ninja only

### 3. NO SHELL SCRIPTS
- No `.sh` files
- No heredocs (`cat > script.sh << 'EOF'`)
- No bash loops in scripts
- Direct bash commands only, one at a time
- Or write C++ programs for complex operations

## REQUIRED PRACTICES

### Clean Room Implementation
- No external dependencies except build tools
- No Boost libraries
- No parser generators
- Standard library only: string, vector, map, optional

### Anchor-Based Pattern Matching
- Use direct string search: `text.find(anchor)`
- Extract segments: `text.substr(pos, len)`
- No regex patterns

### Orbit Architecture
- Maintain orbit infrastructure as training-bias firewall
- Do not bypass with simple text replacement
- Speculation and confidence weighting required

### Build System
- CMake only
- Ninja build files
- One build directory structure

## ENFORCEMENT

Violations of standards 1-3 result in immediate rollback.
