#include <iostream>
#include "cpp2_inline.h"
void X:  <T: type, U: type>
    type
    requires std::is_same_v<T, int>
              && std::is_same_v<U, int>
= {
    operator=(out this) { } }

f:  <T: template<typename T, typename int>
              && std::is_same_v<U, typename int> using , U: type>
    (forward a: int, forward b: int) -> int
    requires std::is_same_v = {
    return a * b;
}

int requires true == x f(x);

v: <T> const T requires std::same_as<T, i32> = 0;

int main() { X<int,int> _ = ();
    std::cout << f<int,int>(2,5)
              << f(0); }
