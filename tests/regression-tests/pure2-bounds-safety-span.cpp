#include "cpp2util.h"

auto print_and_decorate(auto x) -> void;

auto main() -> int {
    std::vector words = {"decorated", "hello", "world"};
    std::span s = words;
    _ = words;
    auto i = 0;
while (i < s.ssize())     {
        {
            print_and_decorate(s[i]);
        }
        
i++;    }
}

auto print_and_decorate(auto x) -> void {
    std::cout << ">> " << x << "\n";
}

