# Band 4: Type System Expansion (Chapters 12-15)

## Overview

Band 4 extends the Sea of Nodes type system from basic integers to a comprehensive type system including floating point, references, narrow integer types, and arrays. This expansion enables the stage0 meta-transpiler to handle real-world code patterns across C, C++, CPP2, and other target languages.

## Type System Architecture

The type system maintains the lattice structure introduced in Band 1:
- **TOP (⊤)**: Unknown value at compile time
- **Constants**: Specific known values
- **Ranges**: Value ranges for optimization
- **BOTTOM (⊥)**: Non-constant runtime value

## Chapter 12: Floating Point Types

### TypeFloat Design

Floating point types mirror the integer type lattice:

```cpp
class TypeFloat : public Type {
    enum Precision { F32, F64 };  // IEEE 754 single/double
private:
    Precision _precision;
    double _value;
    bool _is_constant;
};
```

### Key Features

- **Constant folding**: float arithmetic at compile time
- **Precision tracking**: f32 vs f64 explicit
- **IEEE 754 semantics**: NaN, Inf handling (future)
- **Cross-language mapping**:
  - C: float/double
  - C++: float/double
  - CPP2: f32/f64
  - MLIR: f32/f64 types

### Operations

- Arithmetic: FAdd, FSub, FMul, FDiv nodes
- Comparisons: float-aware comparison operators
- Conversions: I2F, F2I cast nodes

## Chapter 13: Reference Types

### TypePointer Design

References with nullable/non-nullable distinction:

```cpp
class TypePointer : public Type {
private:
    std::string _target_name;  // Struct/type name
    bool _nullable;            // Struct vs Struct?
    bool _is_null;             // null constant
};
```

### Nullability Analysis

The Simple language uses `?` to denote nullable references:
- `Foo` - non-nullable reference, never null
- `Foo?` - nullable reference, may be null
- `null` - null constant, meets with Foo?

### Null Safety

- **Flow-sensitive**: Null checks refine type
- **Meet semantics**: Foo meet Foo? = Foo?
- **Null pointer elimination**: Dead null checks removed

### Cross-Language Mapping

- **C**: Raw pointers (no null safety)
- **C++**: T* vs T& (partial null safety)
- **CPP2**: T vs T? (full null safety)
- **MLIR**: !llvm.ptr with attributes

## Chapter 14: Narrow Integer Types

### TypeNarrow Design

Sub-word integer types with explicit widths:

```cpp
class TypeNarrow : public Type {
    enum Width {
        I8, I16, I32, I64,  // Signed
        U8, U16, U32        // Unsigned
    };
private:
    Width _width;
    long _lo, _hi;  // Value range
};
```

### Widening and Narrowing

```
i8 ---widen--> i32 ---narrow--> i8
    (sign ext)         (truncate)
```

- **Widening**: Always safe, preserves value
- **Narrowing**: May lose data, explicit cast required
- **Range analysis**: Tracks value ranges for optimization

### Coercion Rules

Following Simple's semantics:
1. Explicit casts required for narrowing
2. Widening happens implicitly in some contexts
3. Mixed-width arithmetic requires explicit conversions

### Cross-Language Mapping

| Simple | C | C++ | CPP2 | MLIR |
|--------|---|-----|------|------|
| i8 | int8_t | int8_t | i8 | i8 |
| i16 | int16_t | int16_t | i16 | i16 |
| i32 | int32_t | int32_t | i32 | i32 |
| i64 | int64_t | int64_t | i64 | i64 |
| u8 | uint8_t | uint8_t | u8 | i8 (unsigned) |

## Chapter 15: Array Types

### TypeArray Design

Fixed and dynamic arrays with element type tracking:

```cpp
class TypeArray : public Type {
private:
    Type* _element_type;
    long _length;         // -1 for dynamic
    bool _nullable;       // Array ref can be null?
};
```

### Array Operations

- **NewArray[T](length)**: Allocate array
- **ALoad(mem, array, index)**: Load element
- **AStore(mem, array, index, value)**: Store element
- **ArrayLength(array)**: Get length (# operator)

### Bounds Checking

- **Static**: Fixed-size arrays, compile-time index check
- **Dynamic**: Runtime bounds check insertion
- **Optimization**: Eliminate redundant checks via range analysis

### Cross-Language Mapping

| Simple | C | C++ | CPP2 | MLIR |
|--------|---|-----|------|------|
| int[] | int* | std::vector<int> | int[] | memref<?xi32> |
| int[10] | int[10] | std::array<int,10> | int[10] | memref<10xi32> |

## Type Conversion Nodes

### CastNode Operations

```cpp
enum CastType {
    INT_TO_FLOAT,     // int -> float
    FLOAT_TO_INT,     // float -> int (truncate)
    NARROW_TO_WIDE,   // i8 -> i32
    WIDE_TO_NARROW,   // i32 -> i8
    FLOAT32_TO_64,    // f32 -> f64
    FLOAT64_TO_32,    // f64 -> f32
    INT_TO_PTR,       // Unsafe cast
    PTR_TO_INT        // Unsafe cast
};
```

### Cast Peepholes

Optimization patterns for cast nodes:
- **Cast composition**: `widen(narrow(x))` → identity if same width
- **Constant folding**: `i2f(5)` → `5.0f` at compile time
- **Dead cast elimination**: Remove no-op casts

## Integration with Sea of Nodes

### Scheduling Impact

Band 4 types integrate with Band 3's GCM:
- Float operations schedulable like integer ops
- Array bounds checks must not be hoisted across stores
- Null checks can be hoisted out of loops

### Memory Model

Arrays and references extend Band 2's memory model:
- Arrays use alias analysis for disambiguation
- Reference loads/stores tracked by alias class
- Null dereference becomes unreachable control

### Peephole Opportunities

Type expansion enables new optimizations:
- **Strength reduction**: `x * 2.0` → `x + x` for integers
- **Algebraic simplification**: `(f32)x + (f32)y` → `(f32)(x + y)`
- **Range-based DCE**: Dead branches eliminated via type ranges

## Implementation Status

### Completed

- ✓ TypeFloat with f32/f64 precision
- ✓ TypePointer with nullable/non-nullable
- ✓ TypeNarrow with all integer widths
- ✓ TypeArray with fixed/dynamic lengths
- ✓ CastNode for all conversions
- ✓ Array operation nodes (NewArray, ALoad, AStore, ArrayLength)
- ✓ Type lattice meet operations

### Future Work

- IEEE 754 special values (NaN, Inf)
- Overflow checking for narrow types
- SIMD vector types (Band 5+)
- Struct field types (already in Band 2, formalize here)
- Function pointer types (Band 5+)

## Testing Strategy

Band 4 requires comprehensive type system testing:

1. **Constant folding**: Verify compile-time evaluation
2. **Range analysis**: Check value range propagation
3. **Null flow analysis**: Validate null safety
4. **Bounds checking**: Ensure array safety
5. **Cross-type operations**: Test all cast combinations

## Conclusion

Band 4 transforms stage0 from a toy compiler into a production-capable meta-transpiler foundation. The expanded type system enables:

- **Real code compilation**: Handle production C/C++ patterns
- **Cross-language consistency**: Unified type semantics
- **Optimization power**: Rich type info enables aggressive transforms
- **Safety**: Null and bounds checking from source

With Bands 1-4 complete, stage0 has:
- Control flow (Band 1)
- Memory model (Band 2)
- Scheduling (Band 3)
- Type system (Band 4)

Next: Band 5 will add advanced optimizations and full code generation.
