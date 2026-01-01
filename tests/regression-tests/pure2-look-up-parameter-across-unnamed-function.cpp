#include "cpp2util.h"


auto f() -> void;

auto g() -> void;

auto pred = default;

auto main() -> int {
    std::cout << f() + g() << "\n";
}

