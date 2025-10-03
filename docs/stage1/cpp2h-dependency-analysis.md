# cpp2.h Dependency Analysis

## Overview

The Stage 1 transpiler generates C++ code that requires the `cpp2.h` header file. This creates a dependency that users must satisfy by having the header in their include path.

## Symbols Used in Generated Code

The generated C++ code may use the following symbols from `cpp2.h`:

### Type Aliases
- `cpp2::i8`, `cpp2::i16`, `cpp2::i32`, `cpp2::i64` - Signed integer types
- `cpp2::u8`, `cpp2::u16`, `cpp2::u32`, `cpp2::u64` - Unsigned integer types

### Template Type Aliases in `cpp2::impl`
- `cpp2::impl::in<T>` - Const lvalue reference (`const T&`)
- `cpp2::impl::copy<T>` - Value parameter (`T`)
- `cpp2::impl::move<T>` - Move parameter (`T`)
- `cpp2::impl::out<T>` - Non-const lvalue reference (`T&`)
- `cpp2::impl::forward<T>` - Forwarding reference (`T&&`)

### Utility Functions in `cpp2::impl`
- `assert_not_null(P p)` - Null pointer checking
- `deref(P p)` - Dereference with null checking
- `unchecked_narrow<To, From>(From v)` - Narrowing cast
- `unchecked_cast<To, From>(From v)` - Reinterpret cast
- `as_<To, From>(From&& v)` - Type conversion
- `is<T, From>(From const&)` - Type checking (stub)

### Global Using Declarations
- `unchecked_narrow` - Global alias for `cpp2::impl::unchecked_narrow`
- `unchecked_cast` - Global alias for `cpp2::impl::unchecked_cast`
- `assert_not_null` - Global alias for `cpp2::impl::assert_not_null`
- `as_` - Global alias for `cpp2::impl::as_`
- `is` - Global alias for `cpp2::impl::is`

## Usage Analysis

### Always Included
The emitter always includes `cpp2.h` in generated code, even if no symbols are used.

### Actual Usage
Not all generated files use cpp2.h symbols. For example, simple functions without Cpp2-specific parameter passing may not use any of these symbols.

## Dependency Options

1. **Inline cpp2.h contents** - Embed minimal required definitions directly in generated code
2. **Bundle cpp2.h with output** - Prepend cpp2.h contents to generated code
3. **Installation script** - Provide script to install cpp2.h to standard locations
4. **Keep as external dependency** - Document requirement and provide clear instructions

## Recommendation

Inline the cpp2.h contents with only the actually-used symbols to create a standalone output that doesn't require external dependencies.