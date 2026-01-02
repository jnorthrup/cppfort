# Implementation Plan: Compositional Orthogonal Combinator Basics

## Phase 1: Core Type Infrastructure [checkpoint: c407ac9]

### Tasks

- [x] Design ByteBuffer class in `include/bytebuffer.hpp`
  - `pointer: const char*`, `length: size_t` fields
  - `slice(start, end)` method returning new ByteBuffer
  - `data()`, `size()`, `empty()` accessors
  - Iterator support for `char` traversal
- [x] Design StrView class in `include/strview.hpp`
  - UTF-8 codepoint boundaries validation
  - `chars()` iterator returning `char32_t`
  - `lines()` iterator for newline splitting
  - `trim()`, `trim_left()`, `trim_right()` views
- [x] Implement LazyIterator<T> framework in `include/lazy_iterator.hpp`
  - Deferred evaluation base class
  - `map()`, `filter()` iterator adapters
  - `collect<Container>()` terminal operation
- [x] Add ByteBuffer and StrView unit tests in `tests/bytebuffer_test.cpp2`
  - Test zero-copy property (same pointer base)
  - Test slice invariants
  - Test UTF-8 decoding edge cases

### Success Criteria

- ByteBuffer slice creates view without copy (verified via pointer equality)
- StrView correctly handles multi-byte UTF-8 sequences
- LazyIterator defers evaluation until `collect()`

---

## Phase 2: Structural Combinators [checkpoint: 78442e4]

### Tasks

- [x] Implement `take(n)` combinator in `include/combinators/structural.hpp`
  - Returns lazy sequence of first N elements
  - Handles N > size gracefully
- [x] Implement `skip(n)` combinator
  - Returns lazy sequence after N elements
- [x] Implement `slice(start, end)` combinator
  - Combines take/skip semantics
- [x] Implement `split(delimiter)` combinator
  - Returns lazy iterator of sub-ranges
  - Handles consecutive delimiters
- [x] Implement `chunk(size)` combinator
  - Divides sequence into fixed-size blocks
  - Last chunk may be smaller
- [x] Implement `window(size)` combinator
  - Returns sliding windows of size N
  - Overlapping views
- [x] Unit tests for all structural combinators
  - Verify zero-copy for split/chunk/window
  - Edge cases: empty input, single element

### Success Criteria

- All structural ops are O(1) or O(number of output views)
- No allocations until iteration
- 100% test coverage

---

## Phase 3: Transformation Combinators [checkpoint: 2521c41]

### Tasks

- [x] Implement `map(f)` combinator in `include/combinators/transformation.hpp`
  - Lazy application of f to each element
  - Type transformation support (T -> U)
- [x] Implement `filter(pred)` combinator
  - Lazy predicate application
  - Short-circuit iteration
- [x] Implement `flat_map(f)` combinator
  - Flattens nested sequences
  - Lazy inner sequence evaluation
- [x] Implement `enumerate()` combinator
  - Returns (index, element) pairs
- [x] Implement `zip(other)` combinator
  - Combines two sequences element-wise
  - Stops at shorter sequence
- [x] Implement `intersperse(sep)` combinator
  - Inserts separator between elements
- [ ] Implement pipeline operator `|>` in Cpp2 grammar
  - Parser extension for left-to-right composition
  - AST node for pipeline expressions
  - Codegen for method call chaining
- [x] Unit tests for all transformation combinators
  - Verify lazy evaluation (no work until iteration)
  - Test chaining: map |> filter |> map

### Success Criteria

- `map` and `filter` chains don't allocate intermediate storage
- `|>` syntax compiles and generates equivalent code
- Laws: `map(id) == id`, `filter(true) == id`

---

## Phase 4: Reduction Combinators [checkpoint: ce2015f]

### Tasks

- [x] Implement `fold(init, f)` in `include/combinators/reduction.hpp`
  - Left fold over sequence
  - Returns accumulator value
- [x] Implement `reduce(f)` combinator
  - Uses first element as initial value
  - Returns `std::optional<T>` (empty case)
- [x] Implement `scan(init, f)` combinator
  - Prefix sum / running total
  - Returns sequence of intermediate values
- [x] Implement `find(pred)` combinator
  - Short-circuit search
  - Returns `std::optional<T>`
- [x] Implement `find_index(pred)` combinator
  - Returns position of first match
- [x] Implement `all(pred)` and `any(pred)` combinators
  - Short-circuit boolean logic
- [x] Implement `count(pred)` combinator
  - Counts matching elements
- [x] Implement `first()` and `last()` combinators
  - Access first/last element
- [x] Unit tests for all reduction combinators
  - Short-circuit behavior verified
  - Empty sequence handling

### Success Criteria

- `all`/`any`/`find` stop on first match
- `reduce` returns none for empty
- `scan` produces correct intermediate values

---

## Phase 5: Parsing and Validation Combinators [checkpoint: d8952bb]

### Tasks

- [x] Implement `byte()` parser in `include/combinators/parsing.hpp`
  - Consumes single byte, returns optional
- [x] Implement `bytes(n)` parser
  - Consumes N bytes if available
- [x] Implement `until(delimiter)` parser
  - Consumes until delimiter found
- [x] Implement `while_pred(pred)` parser
  - Consumes while predicate holds
- [x] Implement endian integer parsers
  - `le_i16()`, `be_i16()`
  - `le_i32()`, `be_i32()`
  - `le_i64()`, `be_i64()`
- [x] Implement `c_str()` and `pascal_string()` parsers
  - Null-terminated and length-prefixed strings
- [x] Implement validation predicates
  - `length_eq(n)`, `length_between(min, max)`
  - `starts_with(prefix)`, `ends_with(suffix)`
  - `contains(element)`
  - `is_unique()`, `is_sorted()`
- [x] Unit tests for all parsing combinators
  - Malformed input handling
  - Bounds checking

### Success Criteria

- All parsers return none on invalid input
- No buffer overflow possible
- Correct endianness handling

---

## Integration Tasks

### Parser Extension for Pipeline Operator [checkpoint: 2a607ee]

- [x] Extend lexer in `src/lexer.cpp` to recognize `|>` token
- [x] Add grammar rule for pipeline expressions in `src/parser.cpp`
- [x] Generate AST node `PipelineExpr` with left-to-right chaining
- [x] Implement code generation in `src/code_generator.cpp`
  - Convert `a |> f |> g` to `g(f(a))`
  - Handle lambda arguments: `a |> :(x: int) -> int = x * 2`
- [x] Add comprehensive tests in `tests/pipeline_operator_test.cpp2`
  - 25 tests: basic, chained, lambdas, curried, combinators, validation, edge cases

**Tests:** 

- [x] Simple pipeline: `x |> f` → `f(x)`
- [x] Chained pipeline: `x |> f |> g` → `g(f(x))`
- [x] Pipeline with arithmetic: `(3 + 4) |> f` → `f(3 + 4)`
- [x] Inline lambdas: `5 |> :(x: int) -> int = x * 2`
- [x] Curried combinators: `buf |> curried::take(3)`
- [x] Chained combinators: `buf |> skip(1) |> take(3)`
- [x] Type transformations: `42 |> stringify |> parse_int`

### Standard Library Integration

- [ ] Add `std::cpp2::bytebuffer` to `include/cpp2_runtime.h`
- [ ] Add `std::cpp2::strview` to runtime
- [ ] Export all combinators in `std::cpp2::combinators` namespace
- [ ] Add Cpp2 syntax sugar: `buffer.take(10)` vs `take(10)(buffer)`

---

## Documentation Tasks

- [ ] Write tutorial for combinator usage
- [ ] Document each combinator with examples
- [ ] Create "Recipes" section for common patterns
- [ ] Add performance characteristics documentation
- [ ] Document zero-copy invariants

---

## Verification Tasks

- [ ] Property-based tests for combinator laws
- [ ] Benchmark suite vs hand-written loops
- [ ] Zero-copy verification (valgrind, ASAN)
- [ ] Integration tests with real parsers (HTTP, binary protocols)
- [ ] Corpus tests for combinator compositions

---

## Track Completion Checklist

- [x] All 5 phases implemented and tested
- [x] Pipeline operator `|>` working in Cpp2 syntax
- [ ] Zero-copy properties verified for all structural ops
- [ ] Benchmark targets met (<5% overhead)
- [ ] Documentation complete
- [ ] Integration tests passing
- [ ] No regressions in existing corpus tests
