# Local Tests for Cppfort

## Test Categories

### Working Tests (PASS) - 19 tests

These tests parse correctly and demonstrate features that work:

**Parameter Semantics**:
- `002_inout_parameter.cpp2` - `inout` parameter → `T&`
- `003_out_parameter.cpp2` - `out` parameter → `T&`
- `004_in_parameter.cpp2` - `in` parameter → `const T&`
- `005_move_parameter.cpp2` - `move` parameter → `T&&`
- `006_forward_parameter.cpp2` - `forward` parameter (template)
- `032_this_parameter.cpp2` - `this` as parameter name (non-mutating method)

**Types and Declarations**:
- `001_basic_type.cpp2` - Basic type definition
- `012_namespace_with_equals.cpp2` - Namespace with optional `=`
- `013_template_defaults.cpp2` - Template parameter default values
- `014_nested_templates.cpp2` - Nested template arguments `std::plus<>`
- `025_type_alias.cpp2` - Type alias with `type ==`
- `034_type_qualifier.cpp2` - Type qualifiers (`@final`) before `type`

**Mixed-Mode**:
- `010_mixed_mode_cpp1_function.cpp2` - Pure C++1 file passthrough
- `011_mixed_mode_combined.cpp2` - Cpp2 + C++1 in same file (parse with errors)
- `033_pointer_type.cpp2` - C++1 pointer type `*const char`

**Control Flow**:
- `026_contracts.cpp2` - Contract annotations `assert<name>(condition)`
- `030_for_loop.cpp2` - While loop with `next` clause
- `031_if_expression.cpp2` - If-expression `if cond { a } else { b }`

**Templates**:
- `021_variadic_int_pack.cpp2` - Variadic int pack expansion

### Known Failures (REVEAL BUGS)

These tests parse but fail semantic analysis, revealing current limitations:

- `020_variadic_type_pack.cpp2` - **VARIADIC TYPE PACK**
  - Issue: `<Ts...>` not recognized in type parameter list
  - Expected: Should expand types like `std::tuple<Ts...>`

- `022_enum_basic.cpp2` - **ENUM BARE IDENTIFIER**
  - Issue: Enum members without explicit initializer not supported
  - Expected: Sequential values like C++ enum (0, 1, 2...)
  - Workaround: Use explicit `:= value` for each member

- `023_enum_with_method.cpp2` - **ENUM WITH METHOD**
  - Same bare identifier issue as 022
  - Also tests enum methods with `inout this`

- `024_decorator_with_template.cpp2` - **DECORATOR TEMPLATE PARAMETER**
  - Issue: Template parameter in decorator `<T:type>` not resolved
  - Expected: `@struct <T:type>` should declare generic struct

### Mixed-Mode Limitation

**Issue**: Parser cannot switch from Cpp2 to C++1 within same file.

When a file contains Cpp2 declarations first, then C++1 code like `auto main() -> int`,
the parser fails to recognize the C++1 syntax. Files that are entirely C++1 work fine.

**Impact**: 50 mixed-mode tests from cppfront corpus.

**Example that fails**:
```cpp2
// Cpp2 function
name: () -> std::string = { return "world"; }

// C++1 function - NOT RECOGNIZED after Cpp2
auto main() -> int {
    return 0;
}
```

## Running Tests

```bash
./tests/run_local_tests.sh
```

## Test File Naming

- `001-099`: Basic features (types, parameters, functions)
- `100-199`: Templates
- `200-299`: Mixed-mode C++1 passthrough
- `300-399`: Contracts and safety
- `400-499`: Advanced features (inspect, UFCS, etc.)
