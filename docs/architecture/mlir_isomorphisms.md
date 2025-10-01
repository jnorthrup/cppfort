# MLIR Dialect Isomorphism Analysis

This document maps operations from standard MLIR dialects to their equivalent constructs in C/C++, as part of the effort to integrate MLIR concepts more deeply into the `cpp2` compiler.

## `arith` Dialect (Arithmetic)

The `arith` dialect provides operations for integer and floating-point arithmetic.

| `arith` Operation | C/C++ Equivalent | Notes |
|---|---|---|
| `arith.constant` | `123`, `3.14f` | Represents a constant value. |
| `arith.addi` | `a + b` (integers) | Integer addition. |
| `arith.addf` | `a + b` (floats) | Floating-point addition. |
| `arith.subi` | `a - b` (integers) | Integer subtraction. |
| `arith.subf` | `a - b` (floats) | Floating-point subtraction. |
| `arith.muli` | `a * b` (integers) | Integer multiplication. |
| `arith.mulf` | `a * b` (floats) | Floating-point multiplication. |
| `arith.divsi` | `a / b` (signed) | Signed integer division. |
| `arith.divui` | `a / b` (unsigned) | Unsigned integer division. |
| `arith.divf` | `a / b` (floats) | Floating-point division. |
| `arith.remsi` | `a % b` (signed) | Signed integer remainder. |
| `arith.remui` | `a % b` (unsigned) | Unsigned integer remainder. |
| `arith.remf` | `fmod(a, b)` | Floating-point remainder. |
| `arith.andi` | `a & b` | Bitwise AND. |
| `arith.ori` | `a \| b` | Bitwise OR. |
| `arith.xori` | `a ^ b` | Bitwise XOR. |
| `arith.shli` | `a << b` | Shift left. |
| `arith.shrsi` | `a >> b` (signed) | Arithmetic (signed) shift right. |
| `arith.shrui` | `a >> b` (unsigned) | Logical (unsigned) shift right. |
| `arith.cmpi` | `a == b`, `a != b`, `a < b`, etc. | Integer comparison. The predicate (`eq`, `ne`, `slt`, `ult`, etc.) determines the comparison type. |
| `arith.cmpf` | `a == b`, `a < b`, etc. (floats) | Floating-point comparison. |
| `arith.extsi` | `(int64_t)my_int32` | Sign-extend an integer. |
| `arith.extui` | `(uint64_t)my_uint32` | Zero-extend an integer. |
| `arith.trunci` | `(int32_t)my_int64` | Truncate an integer. |
| `arith.sitofp` | `(double)my_signed_int` | Signed integer to float conversion. |
| `arith.uitofp` | `(double)my_unsigned_int` | Unsigned integer to float conversion. |
| `arith.fptosi` | `(int)my_float` | Float to signed integer conversion. |
| `arith.fptoui` | `(unsigned int)my_float` | Float to unsigned integer conversion. |

## `cf` Dialect (Control Flow)

The `cf` dialect provides unstructured control flow operations.

| `cf` Operation | C/C++ Equivalent | Notes |
|---|---|---|
| `cf.br` | `goto label;` | Unconditional branch. Corresponds to jumping to a basic block. |
| `cf.cond_br` | `if (cond) { goto true_label; } else { goto false_label; }` | Conditional branch. The core of `if`, `for`, `while` statements. |
| `cf.switch` | `switch (value) { case 0: ...; case 1: ...; default: ...; }` | Multi-way branch. |
| `cf.assert` | `assert(cond);` | Runtime assertion. |
