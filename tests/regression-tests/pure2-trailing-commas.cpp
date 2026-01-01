#include "cpp2util.h"

auto f(auto a, auto b) -> void;
auto g(T a, U b) -> void;

auto f(auto a, auto b) -> void {
    a + b;
}

template<typename T, typename U>
auto g(T a, U b) -> void {
    a + b;
}

auto doubler(int a) -> void;

struct vals {
    int i = default;
};

auto main() -> void {
    /* expression kind 18 */;
}

std::vector grouping = {0, 1, 2};

std::array array = {0, 1, 2};

