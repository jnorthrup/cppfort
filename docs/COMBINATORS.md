# Cppfort Combinators Library

A zero-copy, lazy-evaluation combinator library for Cpp2.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Types](#core-types)
- [Structural Combinators](#structural-combinators)
- [Transformation Combinators](#transformation-combinators)
- [Reduction Combinators](#reduction-combinators)
- [Parsing Combinators](#parsing-combinators)
- [Recipes](#recipes)
- [Performance Characteristics](#performance-characteristics)
- [Zero-Copy Invariants](#zero-copy-invariants)

---

## Overview

This library provides compositional, orthogonal combinators for working with sequences and data in Cpp2. Key features:

- **Zero-copy**: Operations return views, never copy underlying data
- **Lazy evaluation**: No work done until terminal operation
- **Composable**: Pipeline operator `|>` for left-to-right composition
- **Type-safe**: Full C++ template type deduction

---

## Quick Start

```cpp2
#include <cpp2_combinators.hpp>

main: () -> int = {
    // Create a ByteBuffer from data
    data: std::array<char, 11> = ('H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd');
    buf: cpp2::ByteBuffer = (data.data(), data.size());

    // Take first 5 bytes
    hello := buf |> cpp2::combinators::curried::take(5);
    std::cout << "First 5: " << std::string(hello.data(), hello.size()) << "\n";

    // Skip "Hello ", take "World"
    world := buf |> cpp2::combinators::curried::skip(6) |> cpp2::combinators::curried::take(5);
    std::cout << "World: " << std::string(world.data(), world.size()) << "\n";

    return 0;
}
```

---

## Core Types

### ByteBuffer

A non-owning view into a contiguous byte sequence.

```cpp2
namespace cpp2 {

class ByteBuffer {
    // Create empty buffer
    constexpr ByteBuffer();

    // Create view from pointer and length
    constexpr ByteBuffer(const char* p, size_t l);

    // Accessors
    constexpr const char* data() const;
    constexpr size_t size() const;
    constexpr bool empty() const;

    // Slicing - zero copy
    constexpr ByteBuffer slice(size_t start, size_t end) const;

    // Iterator support
    using iterator = const char*;
    constexpr iterator begin() const;
    constexpr iterator end() const;
};

}
```

**Key properties:**
- Zero-copy slicing: `slice()` returns new view without copying data
- Const propagation: read-only access to underlying data
- STL-compatible: works with range-based for loops

### StrView

A non-owning view into UTF-8 string data with Unicode awareness.

```cpp2
namespace cpp2 {

class StrView {
    constexpr StrView();
    constexpr StrView(const char* p, size_t l);

    constexpr const char* data() const;
    constexpr size_t size() const;
    constexpr bool empty() const;

    // Trim whitespace
    constexpr StrView trim() const;

    // UTF-8 codepoint iteration
    constexpr CharsRange chars() const;
};

}
```

---

## Structural Combinators

Located in: `include/combinators/structural.hpp`
Namespace: `cpp2::combinators`

### take(n)

Returns first `n` elements of a sequence.

```cpp2
take: <Seq>(seq: Seq, n: size_t) -> TakeRange<Seq>
```

**Example:**
```cpp2
data: std::array<char, 10> = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
buf: cpp2::ByteBuffer = (data.data(), data.size());

first_three := cpp2::combinators::take(buf, 3);
// first_three.size() == 3
// first_three.data() points to same memory as buf (no copy)
```

**Curried version:**
```cpp2
result := buf |> cpp2::combinators::curried::take(3);
```

**Complexity:** O(1)

---

### skip(n)

Returns sequence after first `n` elements.

```cpp2
skip: <Seq>(seq: Seq, n: size_t) -> SkipRange<Seq>
```

**Example:**
```cpp2
data: std::array<char, 10> = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
buf: cpp2::ByteBuffer = (data.data(), data.size());

last_seven := cpp2::combinators::skip(buf, 3);
// last_seven.size() == 7
// View starts at offset 3
```

**Complexity:** O(1)

---

### slice(start, end)

Combines `skip` and `take` for range selection.

```cpp2
slice: <Seq>(seq: Seq, start: size_t, end: size_t) -> SliceRange<Seq>
```

**Example:**
```cpp2
data: std::array<char, 10> = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
buf: cpp2::ByteBuffer = (data.data(), data.size());

middle := cpp2::combinators::slice(buf, 3, 7);
// middle contains [3, 4, 5, 6]
```

**Complexity:** O(1)

---

### split(delimiter)

Divides sequence at delimiter boundaries.

```cpp2
split: <Seq>(seq: Seq, delimiter: typename Seq::value_type) -> SplitRange<Seq>
```

**Example:**
```cpp2
data: std::array<char, 11> = ('a', ',', 'b', ',', 'c');
buf: cpp2::ByteBuffer = (data.data(), data.size());

parts := cpp2::combinators::split(buf, ',');
// Lazy iterator over: ['a'], ['b'], ['c']
```

**Complexity:** O(number of chunks)

---

### chunk(size)

Divides sequence into fixed-size blocks.

```cpp2
chunk: <Seq>(seq: Seq, size: size_t) -> ChunkRange<Seq>
```

**Example:**
```cpp2
data: std::array<char, 10> = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
buf: cpp2::ByteBuffer = (data.data(), data.size());

chunks := cpp2::combinators::chunk(buf, 3);
// Yields: [0,1,2], [3,4,5], [6,7,8], [9]
```

**Complexity:** O(number of chunks)

---

### window(size)

Returns sliding windows of `n` elements.

```cpp2
window: <Seq>(seq: Seq, size: size_t) -> WindowRange<Seq>
```

**Example:**
```cpp2
data: std::array<char, 5> = (1, 2, 3, 4, 5);
buf: cpp2::ByteBuffer = (data.data(), data.size());

windows := cpp2::combinators::window(buf, 3);
// Yields: [1,2,3], [2,3,4], [3,4,5]
```

**Complexity:** O(number of windows)

---

## Transformation Combinators

Located in: `include/combinators/transformation.hpp`
Namespace: `cpp2::combinators`

### map(f)

Applies function `f` to each element.

```cpp2
map: <Seq, F>(seq: Seq, f: F) -> MapRange<Seq, F>
```

**Example:**
```cpp2
data: std::array<int, 5> = (1, 2, 3, 4, 5);
doubled := data |> cpp2::combinators::map(: (x: int) -> int = x * 2);
// Yields: 2, 4, 6, 8, 10
```

**Complexity:** O(n) during iteration

---

### filter(pred)

Keeps elements matching predicate.

```cpp2
filter: <Seq, P>(seq: Seq, pred: P) -> FilterRange<Seq, P>
```

**Example:**
```cpp2
data: std::array<int, 5> = (1, 2, 3, 4, 5);
evens := data |> cpp2::combinators::filter(: (x: int) -> bool = x % 2 == 0);
// Yields: 2, 4
```

**Complexity:** O(n) during iteration

---

### enumerate()

Pairs elements with their indices.

```cpp2
enumerate: <Seq>(seq: Seq) -> EnumerateRange<Seq>
```

**Example:**
```cpp2
data: std::array<int, 3> = ('a', 'b', 'c');
with_index := cpp2::combinators::enumerate(data);
// Yields: (0,'a'), (1,'b'), (2,'c')
```

**Complexity:** O(1)

---

### zip(other)

Combines two sequences element-wise.

```cpp2
zip: <Seq1, Seq2>(seq1: Seq1, seq2: Seq2) -> ZipRange<Seq1, Seq2>
```

**Example:**
```cpp2
names: std::array<std::string, 3> = ("Alice", "Bob", "Carol");
ages: std::array<int, 3> = (25, 30, 35);

pairs := cpp2::combinators::zip(names, ages);
// Yields: ("Alice", 25), ("Bob", 30), ("Carol", 35)
```

**Complexity:** O(min(n, m))

---

## Reduction Combinators

Located in: `include/combinators/reduction.hpp`
Namespace: `cpp2::combinators`

### fold(init, f)

Left-fold with initial value.

```cpp2
fold: <Seq, T, F>(seq: Seq, init: T, f: F) -> T
```

**Example:**
```cpp2
data: std::array<int, 5> = (1, 2, 3, 4, 5);
sum := cpp2::combinators::fold(data, 0, :(acc: int, x: int) -> int = acc + x);
// sum == 15
```

**Complexity:** O(n)

---

### reduce(f)

Fold using first element as initial value.

```cpp2
reduce: <Seq, F>(seq: Seq, f: F) -> std::optional<typename Seq::value_type>
```

**Example:**
```cpp2
data: std::array<int, 5> = (1, 2, 3, 4, 5);
sum := cpp2::combinators::reduce(data, :(a: int, b: int) -> int = a + b);
// sum == 15

empty: std::array<int, 0> = ();
result := cpp2::combinators::reduce(empty, :(a: int, b: int) -> int = a + b);
// result == std::nullopt
```

**Complexity:** O(n)

---

### find(pred)

Returns first element matching predicate.

```cpp2
find: <Seq, P>(seq: Seq, pred: P) -> std::optional<typename Seq::value_type>
```

**Example:**
```cpp2
data: std::array<int, 5> = (1, 2, 3, 4, 5);
result := cpp2::combinators::find(data, :(x: int) -> bool = x > 3);
// result == 4
```

**Complexity:** O(n) worst, O(1) best (short-circuits)

---

### all(pred) / any(pred)

Tests if all/any elements match predicate.

```cpp2
all: <Seq, P>(seq: Seq, pred: P) -> bool
any: <Seq, P>(seq: Seq, pred: P) -> bool
```

**Example:**
```cpp2
data: std::array<int, 5> = (2, 4, 6, 8, 10);
all_even := cpp2::combinators::all(data, :(x: int) -> bool = x % 2 == 0);
// all_even == true

has_large := cpp2::combinators::any(data, :(x: int) -> bool = x > 100);
// has_large == false
```

**Complexity:** O(n) worst, short-circuits

---

## Parsing Combinators

Located in: `include/combinators/parsing.hpp`
Namespace: `cpp2::combinators`

### byte()

Consumes single byte, returns `std::optional<char>`.

```cpp2
data: std::array<char, 3> = ('A', 'B', 'C');
buf: cpp2::ByteBuffer = (data.data(), data.size());

result := cpp2::combinators::byte(buf);
// result == 'A'
// Remaining buffer has 2 bytes
```

---

### bytes(n)

Consumes `n` bytes if available.

```cpp2
data: std::array<char, 5> = (0, 1, 2, 3, 4);
buf: cpp2::ByteBuffer = (data.data(), data.size());

result := cpp2::combinators::bytes(buf, 3);
// result == ByteBuffer with [0,1,2]
```

---

### until(delimiter)

Consumes until delimiter found.

```cpp2
data: std::array<char, 10> = ('H', 'e', 'l', 'l', 'o', '\n', 'W', 'o', 'r', 'l');
buf: cpp2::ByteBuffer = (data.data(), data.size());

line := cpp2::combinators::until(buf, '\n');
// line == "Hello"
```

---

### le_i16() / be_i16()

Parse little/big-endian 16-bit integer.

```cpp2
data: std::array<char, 2> = (0x01, 0x02);  // Little-endian 0x0201 = 513
buf: cpp2::ByteBuffer = (data.data(), data.size());

value := cpp2::combinators::le_i16(buf);
// value == 513
```

---

## Recipes

### Parse HTTP Header

```cpp2
parse_header: (buf: cpp2::ByteBuffer) -> std::map<std::string, std::string> = {
    lines := buf |> cpp2::combinators::split('\n');
    // Process each line...
}
```

### Read Null-Terminated String

```cpp2
data: std::array<char, 10> = ('H', 'i', '\0', 'x', 'x', 'x', 'x', 'x', 'x', 'x');
buf: cpp2::ByteBuffer = (data.data(), data.size());

str := cpp2::combinators::c_str(buf);
// str ByteBuffer with "Hi" (stops at \0)
```

### Sliding Window Sum

```cpp2
data: std::array<int, 10> = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
buf: cpp2::ByteBuffer = (reinterpret_cast<char*>(data.data()), data.size() * 4);

windows := buf |> cpp2::combinators::window(3);
sums := windows |> cpp2::combinators::map(: (w: auto) -> int = {
    return cpp2::combinators::fold(w, 0, :(acc: int, x: int) -> int = acc + x);
});
// sums: 6, 9, 12, 15, 18, 21, 24, 27
```

---

## Performance Characteristics

### Complexity Summary

| Combinator | Time | Space | Lazy |
|------------|------|-------|------|
| take | O(1) | O(1) | ✓ |
| skip | O(1) | O(1) | ✓ |
| slice | O(1) | O(1) | ✓ |
| split | O(chunks) | O(chunks) | ✓ |
| chunk | O(chunks) | O(chunks) | ✓ |
| window | O(windows) | O(windows) | ✓ |
| map | O(n) | O(1) | ✓ |
| filter | O(n) | O(1) | ✓ |
| enumerate | O(1) | O(1) | ✓ |
| zip | O(min(n,m)) | O(1) | ✓ |
| fold | O(n) | O(1) | ✗ |
| reduce | O(n) | O(1) | ✗ |
| find | O(n) | O(1) | ✗ |
| all/any | O(n) | O(1) | ✗ |

### Overhead

Combinators typically add <5% overhead compared to hand-written loops when compiled with `-O2` or higher.

---

## Zero-Copy Invariants

### Guaranteed Invariants

1. **No allocation in structural ops**
   - `take`, `skip`, `slice` return views only
   - Pointer arithmetic only, no `new`/`malloc`

2. **Lazy until terminal**
   - Transformation combinators defer work
   - No intermediate allocations
   - Only `fold`, `reduce`, `collect` materialize data

3. **View lifetime**
   - Views don't extend lifetime of underlying data
   - Ensure source data outlives views
   - Typical pattern: views are temporary within expression

### Verification

```cpp2
// Verify zero-copy: same pointer
data: std::array<char, 10> = (/* ... */);
buf: cpp2::ByteBuffer = (data.data(), data.size());

sliced := buf |> cpp2::combinators::curried::slice(2, 5);
assert(sliced.data() == buf.data() + 2);  // Same pointer base
```

---

## Pipeline Operator

All combinators support the pipeline operator `|>` for left-to-right composition:

```cpp2
result := source
    |> combinator1(args)
    |> combinator2(args)
    |> combinator3(args);
```

Desugars to:
```cpp2
result := combinator3(
    combinator2(
        combinator1(source, args),
        args
    ),
    args
);
```

---

## Namespace Reference

```cpp2
// Core types
cpp2::ByteBuffer
cpp2::StrView

// Structural combinators
cpp2::combinators::take
cpp2::combinators::skip
cpp2::combinators::slice
cpp2::combinators::split
cpp2::combinators::chunk
cpp2::combinators::window

// Transformation combinators
cpp2::combinators::map
cpp2::combinators::filter
cpp2::combinators::flat_map
cpp2::combinators::enumerate
cpp2::combinators::zip
cpp2::combinators::intersperse

// Reduction combinators
cpp2::combinators::fold
cpp2::combinators::reduce
cpp2::combinators::scan
cpp2::combinators::find
cpp2::combinators::find_index
cpp2::combinators::all
cpp2::combinators::any
cpp2::combinators::count

// Parsing combinators
cpp2::combinators::byte
cpp2::combinators::bytes
cpp2::combinators::until
cpp2::combinators::while_pred
cpp2::combinators::le_i16 / be_i16
cpp2::combinators::le_i32 / be_i32
cpp2::combinators::le_i64 / be_i64
cpp2::combinators::c_str
cpp2::combinators::pascal_string

// Curried versions (for pipeline operator)
cpp2::combinators::curried::take
cpp2::combinators::curried::skip
// etc.
```
