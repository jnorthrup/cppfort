# C++ Style: Algebraic Golf

Compositional code golf. Expressions dominate Y-axis. Line-by-line fluency.

## Core

```
density > verbosity
operators > keywords
expressions > statements
ternary > if/else
switch > chains
enums > constants
arrays > maps
lambdas > functions
```

## Line Shape

Each line: one complete thought, readable left-to-right.
```cpp
auto result = input | transform(f) | filter(p) | fold(init, op);
```

Not:
```cpp
auto result = std::accumulate(
  std::begin(filtered),
  std::end(filtered),
  init,
  op
);
```

## Expression Dominance

```cpp
// yes: expression
auto x = cond ? a : b;
auto y = table[static_cast<int>(e)];
auto z = ops[idx](arg);

// no: statement soup
int x;
if (cond) { x = a; } else { x = b; }
```

## Ternary Scope

Ternary chains variables, not calls.

```cpp
// yes: variable selection
auto val = ready ? cached : fallback;
auto idx = found ? pos : npos;
auto cat = x < 0 ? neg : x > 0 ? pos : zero;

// no: nested calls in ternary
auto r = valid ? parse(transform(input)) : handle(error(state));

// instead: prepare then select
auto ok = parse(transform(input));
auto fail = handle(error(state));
auto r = valid ? ok : fail;

// or: procedural with single exit
auto r = Result{};
if (valid) r = parse(transform(input));
else r = handle(error(state));
return r;
```

## Associative Dispatch

```cpp
// yes: table-driven
constexpr auto ops = std::array{op_add, op_sub, op_mul, op_div};
return ops[static_cast<size_t>(token.type)](lhs, rhs);

// no: switch cascade
switch (token.type) {
  case Add: return lhs + rhs;
  case Sub: return lhs - rhs;
  // ...
}
```

## Enum Arrays

```cpp
enum class Op : uint8_t { Add, Sub, Mul, Div, Count };
constexpr auto names = std::array<std::string_view, static_cast<size_t>(Op::Count)>{
  "add", "sub", "mul", "div"
};
```

## Lambda Over Function

```cpp
// yes: inline, scoped
auto parse = [&](auto&& tok) { return tok.type == expected; };

// no: separate declaration
bool matches(Token t, TokenType e);
```

## Single Exit via Expression

```cpp
// yes: ternary
auto parse() -> Result { return valid() ? do_parse() : Result::Error; }

// yes: switch preparing result
auto dispatch(Op op, int a, int b) -> int {
  auto result = 0;
  switch (op) {
  case Op::Add: result = a + b; break;
  case Op::Sub: result = a - b; break;
  case Op::Mul: result = a * b; break;
  case Op::Div: result = a / b; break;
  }
  return result;
}

// no: scattered returns
auto parse() -> Result {
  if (!valid()) return Result::Error;
  return do_parse();
}
```

## Brace Elimination

```cpp
// yes
if (x) return y;
for (auto& e : v) process(e);
while (p()) advance();

// no
if (x) {
  return y;
}
```

## Chained Expressions

```cpp
// yes: semicolon golf
auto a = f(), b = g(), c = h();

// operator chaining
auto result = src
  | views::filter(pred)
  | views::transform(func)
  | to<vector>();
```

## Flow Control

```cpp
// yes: prepare result, single exit
auto classify(int x) -> Category {
  auto cat = Category::None;
  if (x < 0) cat = Category::Negative;
  else if (x == 0) cat = Category::Zero;
  else cat = Category::Positive;
  return cat;
}

// yes: ternary chain of enums (no calls)
auto sign(int x) -> Sign { return x < 0 ? Sign::Neg : x > 0 ? Sign::Pos : Sign::Zero; }

// best: table dispatch
constexpr auto cats = std::array{Category::Negative, Category::Zero, Category::Positive};
auto classify(int x) -> Category { return cats[1 + (x > 0) - (x < 0)]; }
```

## Forbidden

- scattered returns throughout function body
- `else` after `return`
- braces around single statements
- named functions for one-shot logic
- `const int X = 1;` when `enum` works
- if-else chains when ternary fits
- switch when array dispatch fits
- comments explaining what (code should say it)
- early return guards (prepare result instead)
- nested calls in ternary branches
- ternary at procedural junctures
