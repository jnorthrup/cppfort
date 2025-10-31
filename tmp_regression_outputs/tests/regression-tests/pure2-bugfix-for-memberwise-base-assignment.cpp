#include <iostream>
#include "cpp2_inline.h"
void Base: type = {
  operator=(out this) { }
  (out this, that) operator= = std::cout << "(out this, that)\n";
  auto operator= = [](implicit out this, _) { return std::cout << "(implicit out this, _)\n"; }; }

Derived: void type = {
  this: Base            = ();
  operator=(out this) { }
  (out this, that) operator= = {};
  auto operator= = [](inout this, move that) {}; }

int main() { auto d = Derived();
  auto d2 = d;
  d2 = d; }
