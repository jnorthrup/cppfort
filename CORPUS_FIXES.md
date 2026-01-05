# Corpus Fix Requirements

## Status
- **172/190 passing** (90.5%)
- **10 error tests** (correctly fail)
- **8 files require fixes**

## The 8 Failing Files

### 1. mixed-is-as-variant.cpp2 (91 lines)
**Issue**: C++1 function-try-catch blocks not handled in passthrough
**Fix Applied**: Modified `cpp1_passthrough_declaration()` in parser.cpp (lines 5786-5846) to capture catch blocks after function body
**Syntax**:
```cpp
template <std::size_t I>
void set_to_valueless_by_exception(auto& v) try {
    v.template emplace<I>(std::runtime_error("make valueless"));
} catch (...) {}

auto expect_no_throw(auto&& l) -> std::string try {
    if constexpr ( requires { { l() } -> std::convertible_to<std::string>; }) {
        return l();
    }
} catch (std::exception const& e) {
    return e.what();
}
```
**Files to modify**: `src/parser.cpp`
**Status**: Fix implemented, needs rebuild

### 2. pure2-raw-string-literal-and-interpolation.cpp2 (34 lines)
**Issue**: Lexer doesn't support raw string literals (`R"..."`) or raw string interpolation (`$R"..."`)
**Syntax**:
```cpp
raw_str : std::string = R"string(raw string without interpolation)string";
raw_str_inter : std::string = $R"test(text (i)$ R"(more)$)test";
```
**Required changes**:
1. Add `RawStringLiteral` token type to lexer.hpp
2. Implement `scan_raw_string_literal()` in lexer.cpp
3. Handle `$R"..."` pattern for interpolation
**Files to modify**: `include/lexer.hpp`, `src/lexer.cpp` (MISSING)
**Status**: lexer.cpp missing - needs restoration

### 3. pure2-for-loop-range-with-lambda.cpp2 (30 lines)
**Issue**: `next` clauses and lambda expressions in for-range
**Syntax**:
```cpp
for ints.first(:(forward x) = x) do (i) { ... }
for :( ) = args$;() do (i) _ = i;
for :(forward x) = x;(args) next _ = :( ) = args$;() do (j) _ = j;
```
**Required changes**:
1. Add support for `next` keyword in for-loop grammar
2. Handle lambda expressions as range expressions: `:(params) = body`
3. Support empty parameter lists in lambdas
**Files to modify**: `src/parser.cpp` (function_declaration, for_statement)
**Status**: Needs implementation

### 4. mixed-bugfix-for-ufcs-non-local.cpp2 (50 lines)
**Issue**: UFCS (Uniform Function Call Syntax) with preconditions
**Syntax**:
```cpp
template<typename T, typename U>
auto s = x.template f<T>(u)
    pre(s.sz() != 0)
    pre<bounds_safety, testing_enabled>(0 < s.ssize() < 100);
```
**Required changes**:
1. Support `pre(...)` precondition clauses in function declarations
2. Handle template UFCS calls: `.template f<T>(args)`
3. Support multiple preconditions with qualifiers
**Files to modify**: `src/parser.cpp`
**Status**: Needs implementation

### 5. pure2-bugfix-for-non-local-function-expression.cpp2 (13 lines)
**Issue**: Lambdas in non-local contexts (concepts, type aliases, base classes)
**Syntax**:
```cpp
v: <T> concept = :() -> bool = true;();
u: type == decltype(:() = {});
t: @struct type = {
    this: decltype(:() = {});
}
```
**Required changes**:
1. Allow function expressions (`:() = body`) in concept initializers
2. Support function expressions in decltype() for type aliases
3. Handle function expressions as base class specifiers
**Files to modify**: `src/parser.cpp`
**Status**: Needs implementation

### 6. pure2-bugfix-for-unbraced-function-expression.cpp2 (15 lines)
**Issue**: Unbraced function expression syntax
**Syntax**:
```cpp
(x := t()) { x[:() -> _ = 0]; }
assert(!(:() = 0; is int));
```
**Required changes**:
1. Support unbraced function expressions in array subscript
2. Handle function expressions as test expressions
**Files to modify**: `src/parser.cpp`
**Status**: Needs implementation

### 7. pure2-print.cpp2 (109 lines)
**Issue**: @print metafunction, namespace aliases, type_of()
**Syntax**:
```cpp
outer: @print type = {
    namespace_alias: namespace == ::std;
    type_alias: type == array<int,10>;
    x: (_: type_of(0)) = { }
}
```
**Required changes**:
1. Implement @print decorator handling
2. Support namespace aliases: `name: namespace == value`
3. Support type aliases: `name: type == type`
4. Implement `type_of()` expression
**Files to modify**: `src/parser.cpp`, `include/parser.hpp`
**Status**: Complex - multiple features needed

### 8. pure2-last-use.cpp2 (1044 lines)
**Issue**: Complex last-use move semantics
**Required changes**:
1. Track variable usage
2. Detect last use
3. Insert move/move-forward at last use point
**Files to modify**: `src/semantic_analyzer.cpp`, `src/codegen.cpp`
**Status**: Complex semantic analysis feature

## Missing Source Files
- `src/lexer.cpp` - Contains Lexer implementation (scan_string, scan_number, etc.)
- Possibly other source files

## Next Steps
1. Restore missing source files from git history
2. Ensure all source files are from same version
3. Apply fixes in documented order
4. Rebuild transpiler
5. Test all 8 files
