#include "cpp2util.h"

[[nodiscard]] auto decorate(auto& x) -> int;
auto print_it(_ x, _ len) -> void;

auto main() -> int {
    std::vector vec = {"hello", "2022"};
for (auto str : vec)     {
        auto len = decorate(str);
        print_it(str, len);
    }
}

[[nodiscard]] auto decorate(auto& x) -> int {
    x = "[" + x + "]";
    return x.ssize();
}

auto print_it(_ x, _ len) -> void {
    std::cout << ">> " << x << " - length " << len << "\n";
}

