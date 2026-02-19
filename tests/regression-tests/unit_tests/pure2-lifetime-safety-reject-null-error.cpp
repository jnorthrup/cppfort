#include "cpp2util.h"

auto print_and_decorate(auto x) -> void;

auto main() -> int {
    std::vector words = {"decorated", "hello", "world"};
    *int p = nullptr;
    print_and_decorate(*p);
}

auto print_and_decorate(auto x) -> void {
    std::cout << ">> " << x << "\n";
}

