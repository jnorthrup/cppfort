# Compositional Orthogonal Combinator Basics for ByteBuffers and StrViews

**Date**: 2026-01-02
**Objective**: Design and implement a library of orthogonal combinators for ByteBuffer and strview manipulation with zero-copy semantics and compositional patterns

## Motivation

### Current Gaps
- **Fragmented APIs**: ByteBuffer and strview operations scattered across different interfaces
- **No compositional patterns**: Each operation reimplements traversal/parsing logic
- **Copy overhead**: Common patterns require intermediate allocations
- **Limited chaining**: No natural way to combine operations fluently

### Combinator Design Principles

1. **Orthogonality**: Each combinator does one thing well, independent of others
2. **Composability**: Any combinator can be combined with any other
3. **Zero-copy**: Operations return views/slices, never allocate unless necessary
4. **Type-safe**: Compile-time guarantees for buffer safety
5. **Lazy evaluation**: Composition builds a pipeline, executed only when consumed

## Core Type Definitions

### ByteBuffer
```cpp2
namespace cpp2 {

// Immutable contiguous byte range
class ByteBuffer {
    pointer: const char*;
    length: size_t;

    // Zero-copy slicing
    func slice(self, start: size_t, end: size_t) -> ByteBuffer {
        // Returns view without copying
    }

    func split(self, delimiter: char) -> Iterator<ByteBuffer> {
        // Lazy iterator over substrings
    }
}

}
```

### StrView
```cpp2
namespace cpp2 {

// UTF-8 aware string view
class StrView {
    data: const char*;
    byte_length: size_t;

    // Code point aware operations
    func chars(self) -> Iterator<char32_t> {
        // Lazy decode of UTF-8 codepoints
    }

    func lines(self) -> Iterator<StrView> {
        // Split by newlines, zero-copy
    }

    func trim(self) -> StrView {
        // Returns trimmed view
    }
}

}
```

## Orthogonal Combinator Categories

### 1. Structural Combinators

**Purpose**: Modify buffer/view structure without interpreting content

```cpp2
// Take first N bytes/chars
func take<T>(n: size_t) -> (Sequence<T>) -> Sequence<T>

// Skip first N bytes/chars
func skip<T>(n: size_t) -> (Sequence<T>) -> Sequence<T>

// Take while predicate holds
func take_while<T>(pred: (T) -> bool) -> (Sequence<T>) -> Sequence<T>

// Drop while predicate holds
func drop_while<T>(pred: (T) -> bool) -> (Sequence<T>) -> Sequence<T>

// Slice from start to end
func slice<T>(start: size_t, end: size_t) -> (Sequence<T>) -> Sequence<T>

// Split by delimiter
func split<T>(delim: T) -> (Sequence<T>) -> Iterator<Sequence<T>>

// Chunk into fixed-size blocks
func chunk<T>(size: size_t) -> (Sequence<T>) -> Iterator<Sequence<T>>

// Window sliding view
func window<T>(size: size_t) -> (Sequence<T>) -> Iterator<Sequence<T>>
```

### 2. Transformation Combinators

**Purpose**: Apply function to each element

```cpp2
// Map function over elements
func map<T, U>(f: (T) -> U) -> (Sequence<T>) -> Sequence<U>

// FlatMap for nested sequences
func flat_map<T, U>(f: (T) -> Sequence<U>) -> (Sequence<T>) -> Sequence<U>

// Filter elements by predicate
func filter<T>(pred: (T) -> bool) -> (Sequence<T>) -> Sequence<T>

// Enumerate with indices
func enumerate<T>() -> (Sequence<T>) -> Sequence<(size_t, T)>

// Zip two sequences together
func zip<T, U>(other: Sequence<U>) -> (Sequence<T>) -> Sequence<(T, U)>

// Intersperse element between items
func intersperse<T>(sep: T) -> (Sequence<T>) -> Sequence<T>
```

### 3. Reduction Combinators

**Purpose**: Aggregate sequence into single value

```cpp2
// Fold left with accumulator
func fold<T, U>(init: U, f: (U, T) -> U) -> (Sequence<T>) -> U

// Reduce with first element as init
func reduce<T>(f: (T, T) -> T) -> (Sequence<T>) -> T

// Scan (prefix sums)
func scan<T, U>(init: U, f: (U, T) -> U) -> (Sequence<T>) -> Sequence<U>

// Find first matching element
func find<T>(pred: (T) -> bool) -> (Sequence<T>) -> std::optional<T>

// Find index of element
func find_index<T>(pred: (T) -> bool) -> (Sequence<T>) -> std::optional<size_t>

// Check if all elements match
func all<T>(pred: (T) -> bool) -> (Sequence<T>) -> bool

// Check if any element matches
func any<T>(pred: (T) -> bool) -> (Sequence<T>) -> bool

// Count matching elements
func count<T>(pred: (T) -> bool) -> (Sequence<T>) -> size_t

// Get first/last element
func first<T>() -> (Sequence<T>) -> std::optional<T>
func last<T>() -> (Sequence<T>) -> std::optional<T>
```

### 4. Parsing Combinators

**Purpose**: Parse structured data from byte streams

```cpp2
// Parse single byte/char
func byte() -> (ByteBuffer) -> std::optional<uint8_t>

// Parse N bytes
func bytes(n: size_t) -> (ByteBuffer) -> std::optional<ByteBuffer>

// Parse until delimiter
func until(delim: char) -> (ByteBuffer) -> std::optional<ByteBuffer>

// Parse while predicate holds
func while_pred(pred: (char) -> bool) -> (ByteBuffer) -> std::optional<ByteBuffer>

// Parse integer (LE/BE)
func le_i16() -> (ByteBuffer) -> std::optional<int16_t>
func be_i16() -> (ByteBuffer) -> std::optional<int16_t>
func le_i32() -> (ByteBuffer) -> std::optional<int32_t>
func be_i32() -> (ByteBuffer) -> std::optional<int32_t>
func le_i64() -> (ByteBuffer) -> std::optional<int64_t>
func be_i64() -> (ByteBuffer) -> std::optional<int64_t>

// Parse null-terminated string
func c_str() -> (ByteBuffer) -> std::optional<StrView>

// Parse length-prefixed string
func pascal_string() -> (ByteBuffer) -> std::optional<StrView>
```

### 5. Validation Combinators

**Purpose**: Assert properties of sequences

```cpp2
// Ensure exact length
func length_eq<T>(n: size_t) -> (Sequence<T>) -> bool

// Ensure min/max bounds
func length_between<T>(min: size_t, max: size_t) -> (Sequence<T>) -> bool

// Ensure all elements unique
func is_unique<T>() -> (Sequence<T>) -> bool where T: EqualityComparable

// Ensure sorted order
func is_sorted<T>() -> (Sequence<T>) -> bool where T: Comparable

// Ensure element is in set
func contains<T>(val: T) -> (Sequence<T>) -> bool where T: EqualityComparable

// Ensure starts/ends with prefix/suffix
func starts_with<T>(prefix: Sequence<T>) -> (Sequence<T>) -> bool
func ends_with<T>(suffix: Sequence<T>) -> (Sequence<T>) -> bool
```

## Composition Examples

### Example 1: Parse HTTP Header
```cpp2
func parse_header(header: ByteBuffer) -> std::optional<(StrView, StrView)> {
    // "Content-Type: application/json"
    return header
        |> split(':')
        |> map(|s| s.trim())
        |> collect<Vec<StrView>>()
        |> bind(|parts| {
            if parts.size() == 2 {
                return some((parts[0], parts[1]))
            } else {
                return none
            }
        })
}
```

### Example 2: Extract URLs from Text
```cpp2
func extract_urls(text: StrView) -> Vec<StrView> {
    return text
        |> split(' ')
        |> map(|s| s.trim())
        |> filter(|s| s.starts_with("http://") || s.starts_with("https://"))
        |> filter(|s| s.contains('.'))
        |> collect<Vec<StrView>>()
}
```

### Example 3: Parse Binary Protocol
```cpp2
struct Packet {
    magic: uint32_t = 0xDEADBEEF
    length: uint16_t
    payload: ByteBuffer
}

func parse_packet(buf: ByteBuffer) -> std::optional<Packet> {
    return bind(|b| {
        let m = b |> bytes(4) |> flat_map(|x| x |> be_u32())
        let l = b |> skip(4) |> bytes(2) |> flat_map(|x| x |> be_u16())
        let p = b |> skip(6) |> take(l)

        return zip(m, l, p)
            |> filter(|(magic, len, _)| magic == 0xDEADBEEF)
            |> map(|(magic, len, payload)| Packet{magic, len, payload})
            |> first()
    })(buf)
}
```

### Example 4: Word Frequency Count
```cpp2
func word_frequencies(text: StrView) -> HashMap<StrView, size_t> {
    return text
        |> to_lower()
        |> split(|c| c == ' ' || c == '\n' || c == '\t')
        |> filter(|w| w.length() > 0)
        |> filter(|w| !is_stop_word(w))
        |> map(|w| (w, 1))
        |> fold(HashMap<StrView, size_t>{}, |map, (word, count)| {
            map[word] = map.get(word).unwrap_or(0) + count
            return map
        })
}
```

## Implementation Strategy

### Phase 1: Core Types
- ByteBuffer with zero-copy slicing
- StrView with UTF-8 awareness
- Iterator<T> lazy evaluation framework

### Phase 2: Structural Combinators
- take, skip, slice, split, chunk, window
- Unit tests for zero-copy guarantees

### Phase 3: Transformation Combinators
- map, filter, flat_map, enumerate, zip
- Chaining operator `|>` implementation

### Phase 4: Reduction Combinators
- fold, reduce, scan, find, count, all, any
- Short-circuit evaluation semantics

### Phase 5: Parsing and Validation
- byte-oriented parsing combinators
- integer endianness parsers
- Validation predicates

## Integration with Cpp2

### Syntax Extensions
```cpp2
// Pipeline operator for composition
result := buffer
    |> take(10)
    |> map(|b| b * 2)
    |> filter(|b| b > 5)
    |> collect<Vec<uint8_t>>()

// Lambda shorthand
result := buffer
    |> take(10)          // First 10 bytes
    |> map(*2)           // Double each (shorthand for |b| b*2)
    |> filter(>5)        // Keep if >5 (shorthand)
    |> collect()
```

### Cpp2 Type System Integration
```cpp2
// Concepts for combinator constraints
concept Sequence<T> {
    func iterator() -> Iterator<T>
    func size() -> size_t
}

concept LazySequence<T> : Sequence<T> {
    // Lazy evaluation guarantees
}
```

## Performance Characteristics

### Zero-Copy Invariants
- `slice`: O(1), allocates view only
- `take/skip`: O(1), defer iteration
- `split`: O(1), lazy iterator
- `map/filter`: O(n) but deferred until collection

### Memory Guarantees
- No intermediate allocations in pipeline
- Single allocation on `collect()` or similar
- Stack allocation for combinators where possible

### Benchmarks (Targets)
- `slice(0, 100)`: <1ns, 0 allocations
- `map(|x| x*2) |> collect(1000)`: <500ns, 1 allocation
- `split(' ') |> filter(!empty)`: <100ns overhead over raw loop

## Testing Strategy

### Unit Tests
- Each combinator tested in isolation
- Zero-copy property verification
- Edge cases: empty, single element, bounds

### Property-Based Tests
- `take(n).size() == min(n, input.size())`
- `map(f).map(g) == map(f ∘ g)`
- `filter(p).size() <= input.size()`
- `fold(op) == reduce(op)` for non-empty

### Integration Tests
- Multi-stage compositions
- Real-world parsing scenarios
- Performance regression tests

## Expected Outcomes

### API Surface
- 20-25 core combinators across 5 categories
- Fluent `|>` chaining syntax
- Complete documentation with examples

### Performance
- Zero-copy for all structural operations
- Lazy evaluation for map/filter chains
- <5% overhead vs hand-written loops

### Code Examples
- HTTP header parsing
- Binary protocol parsing
- Text processing workflows
- Data validation pipelines

## References

- Haskell `Pipes` and `Conduit` libraries
- Rust `Iterator` combinators
- C++20 `Ranges` library
- F# computation expressions
- Parser combinator theory (Hutton/Meijer)
