#include <iostream>
#include <vector>

first: (forward rng) -> forward _
  pre<bounds_safety>( !std::empty(rng) )
=
    std::begin(rng)*;const int global = 42;
int&& f() { return global; }

main: () -> int = {
    v : std::vector = (1,2,3);
    first(v) = 4;
    std::cout << first(v) << std::endl; // prints: 4
    std::cout << f() << std::endl; // prints: 42
}
