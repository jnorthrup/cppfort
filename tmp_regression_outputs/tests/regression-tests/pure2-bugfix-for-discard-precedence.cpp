#include "cpp2_inline.h"
auto quantity: type = {
  number: i32;
  operator=(out this, i32 x) { return number    = x;
  operator+: (inout this, that) -> quantity = quantity(number + that.number);
}; }

int main(args) { quantity x = (1729);
  _           = x + x; // Not `(void) x + x`; would attempt to add a `void` to `x`.
  _           = args;  // Not `void(args)`; would attempt to declare `args` with `void` type. }
