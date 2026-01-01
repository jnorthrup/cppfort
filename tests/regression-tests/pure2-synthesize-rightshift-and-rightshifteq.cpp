#include "cpp2util.h"

[[nodiscard]] auto f(int a, int b) -> int;

[[nodiscard]] auto f(int a, int b) -> int {
    auto x = a;
    x ?op? b * 2;
    return x >> 1;
}

auto main() -> int {
    std::cout << f(32, 1) << "\n";
}

