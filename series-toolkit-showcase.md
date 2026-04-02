# TrikeShed Algebraic Data Types Showcase

## Overview
TrikeShed is a comprehensive algebraic data type system implemented in Cpp2 with cppfront transpilation. It provides foundational types for the self-hosting compiler through a layered architecture: cpp2 source files (`.cpp2`) define the surface APIs, cpp2 headers (`.h2`) provide the core abstractions, and cppfront transpiles everything to standard C++ for compilation.

## Key Features

### Algebraic Foundation
- **Product Types**: `join<A,B>` for pairs, `twin<T>` for same-type pairs
- **Sum Types**: `either<L,R>` for discriminated unions
- **Lazy Collections**: `series<T>` for indexed sequences with deferred evaluation
- **Index Selectors**: `span` and `range` for slice operations
- **Character Processing**: `strview` and `char_series` for text handling

### Functional Programming Surface
- **Projection Operator**: `operator>>` for compact transformations
- **Indexing Overloads**: Support for spans, ranges, series<int>, vectors, and initializer lists
- **Lambda Integration**: First-class support for function objects and lambdas
- **Type Safety**: Compile-time verification of function signatures

### Minimal Dependencies
- Avoids std:: containers except where necessary
- Self-contained core with no external library dependencies
- Designed for compiler bootstrap and embedded use cases

### Compiler Integration
- Maps to SoN (Sea of Nodes) dialect operations
- Supports semantic objects and dense lowered views
- Enables manifold/coordinate systems for advanced transformations

## Architecture

The Series Toolkit is built around six core components in [`src/selfhost/trikeshed.h2`](../src/selfhost/trikeshed.h2):

### `join<A,B>` - Product Pair Payloads
```cpp
// From src/selfhost/trikeshed.h2 lines 27-37
template<typename A, typename B>
struct join {
    A a {};
    B b {};
    join() = default;
    join(A a_, B b_) : a(std::move(a_)), b(std::move(b_)) {}
    join(join&& o) noexcept : a(std::move(o.a)), b(std::move(o.b)) {}
    join& operator=(join&& o) noexcept { a = std::move(o.a); b = std::move(o.b); return *this; }
    join(const join&) = delete;
    join& operator=(const join&) = delete;
};
```

### `twin<T>` - Same-Type Pair (alias for `join<T,T>`)
```cpp
// From src/selfhost/trikeshed.h2 line 43
template<typename T> using twin = join<T, T>;
```

### `span` and `range` - Index Selectors
```cpp
// From src/selfhost/trikeshed.h2 lines 51-81
struct span {
    int lo = 0;
    int hi = 0;

    span() = default;
    span(int lo_, int hi_) : lo(lo_), hi(hi_) {}

    [[nodiscard]] auto size() const -> int { return hi - lo; }
};

struct range {
    int lo = 0;
    int hi = 0;
    int step = 1;

    range() = default;
    range(int lo_, int hi_, int step_ = 1) : lo(lo_), hi(hi_), step(step_) {}

    [[nodiscard]] auto size() const -> int {
        if (step == 0) return 0;
        if ((step > 0 && lo >= hi) || (step < 0 && lo <= hi)) return 0;
        int delta = hi - lo;
        int abs_delta = delta < 0 ? -delta : delta;
        int abs_step = step < 0 ? -step : step;
        return (abs_delta + abs_step - 1) / abs_step;
    }

    [[nodiscard]] auto at(int i) const -> int {
        return lo + (i * step);
    }
};
```

### `series<T>` - Lazy Indexed Collections
```cpp
// From src/selfhost/trikeshed.h2 lines 89-105
template<typename T>
struct series {
    int a {};
    std::function<T(int)> b {};

    series() = default;
    series(int n, std::function<T(int)> f) : a(n), b(std::move(f)) {}
    series(const series&) = default;
    series(series&&) noexcept = default;
    series& operator=(const series&) = default;
    series& operator=(series&&) noexcept = default;

    [[nodiscard]] auto size() const -> int { return a; }
    [[nodiscard]] auto operator[](int i) const -> T { return b(i); }
```

Indexing operators support spans, ranges, series<int> selectors, initializer_lists, and vectors:

```cpp
// From src/selfhost/trikeshed.h2 lines 107-136
    [[nodiscard]] auto operator[](span sp) const -> series<T> {
        return series<T>(sp.size(), [src = *this, sp](int i) -> T {
            return src.b(sp.lo + i);
        });
    }

    [[nodiscard]] auto operator[](range rg) const -> series<T> {
        return series<T>(rg.size(), [src = *this, rg](int i) -> T {
            return src.b(rg.at(i));
        });
    }

    [[nodiscard]] auto operator[](series<int> idx) const -> series<T> {
        return series<T>(idx.a, [src = *this, idx](int i) -> T {
            return src.b(idx.b(i));
        });
    }

    [[nodiscard]] auto operator[](std::initializer_list<int> picks) const -> series<T> {
        std::vector<int> idx(picks);
        return series<T>(static_cast<int>(idx.size()), [src = *this, idx = std::move(idx)](int i) -> T {
            return src.b(idx[static_cast<std::size_t>(i)]);
        });
    }

    [[nodiscard]] auto operator[](std::vector<int> picks) const -> series<T> {
        return series<T>(static_cast<int>(picks.size()), [src = *this, picks = std::move(picks)](int i) -> T {
            return src.b(picks[static_cast<std::size_t>(i)]);
        });
    }
```

Projection with `operator>>` (supports three function signatures):

```cpp
// From src/selfhost/trikeshed.h2 lines 138-157
    template<typename F>
    [[nodiscard]] auto operator>>(F f) const {
        if constexpr (std::is_invocable_v<F, int, T>) {
            using T2 = std::invoke_result_t<F, int, T>;
            return series<T2>(a, [src = *this, f = std::move(f)](int i) mutable -> T2 {
                return f(i, src.b(i));
            });
        } else if constexpr (std::is_invocable_v<F, int>) {
            using T2 = std::invoke_result_t<F, int>;
            return series<T2>(a, [f = std::move(f)](int i) mutable -> T2 {
                return f(i);
            });
        } else {
            static_assert(std::is_invocable_v<F, T>, "series >> fn expects fn(int), fn(T), or fn(int,T)");
            using T2 = std::invoke_result_t<F, T>;
            return series<T2>(a, [src = *this, f = std::move(f)](int i) mutable -> T2 {
                return f(src.b(i));
            });
        }
    }

    template<typename F>
    [[nodiscard]] auto view(F f) const {
        return (*this) >> std::move(f);
    }
```

### `strview` - Character Series (alias for `series<char>`)
```cpp
// From src/selfhost/trikeshed.h2 line 196
using strview = series<char>;
```

### `char_series` - Character Cursor with State
```cpp
// From src/selfhost/trikeshed.h2 lines 203-214
struct char_series {
    strview buf {};
    int pos  = 0;
    int limit = 0;
    int mark  = -1;

    char_series() = default;
    char_series(strview b) : buf(std::move(b)), pos(0), limit(buf.a), mark(-1) {}
    char_series(const char_series&) = default;
    char_series(char_series&&) noexcept = default;
    char_series& operator=(const char_series&) = default;
    char_series& operator=(char_series&&) noexcept = default;
```

Cursor operations:

```cpp
// From src/selfhost/trikeshed.h2 lines 216-221
    char get()       { return buf[pos]; }
    char peek(int i) { return buf[pos + i]; }
    bool has_next()  { return pos < limit; }
    char advance()   { return buf[pos++]; }
    void mark_pos()  { mark = pos; }
    void reset()     { if (mark >= 0) pos = mark; }
```

Slice operations:

```cpp
// From src/selfhost/trikeshed.h2 lines 224-246
    char_series slice(int lo, int hi) {
        return char_series(buf[span{lo, hi}]);
    }

    char_series slice(span sp) {
        return char_series(buf[sp]);
    }

    char_series slice(range rg) {
        return char_series(buf[rg]);
    }

    char_series slice(series<int> idx) {
        return char_series(buf[idx]);
    }

    char_series slice(std::initializer_list<int> picks) {
        return char_series(buf[picks]);
    }

    char_series slice(std::vector<int> picks) {
        return char_series(buf[std::move(picks)]);
    }
```

Projection:

```cpp
// From src/selfhost/trikeshed.h2 lines 248-251
    template<typename F>
    auto operator>>(F f) const {
        return buf >> std::move(f);
    }
```

## Usage Examples

### Basic Series Reading
```cpp
#include "series_toolkit.h"

// Memory-map a large data file
auto mapped_file = mmap_series("large_dataset.bin");

// Create a lazy series reader
auto reader = series<char>(mapped_file);

// Process data lazily
for (auto& chunk : reader) {
    // Process chunk without loading entire file
    process_data(chunk);
}
```

### Advanced Pipeline Processing
```cpp
// Chain operations on series data
auto processed = series_pipeline(mapped_file)
    .filter(is_valid_data)
    .transform(normalize_values)
    .aggregate(sum_by_category);
```
 welcome! Please follow the project's coding standards and ensure all changes are tested with `cargo check --workspace` before submission.