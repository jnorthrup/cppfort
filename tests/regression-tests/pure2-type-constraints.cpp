#include "cpp2util.h"

auto print(auto _) -> void;

auto print(_ r) -> void;

auto print(auto _) -> void {
    std::cout << "fallback\n";
}

struct irregular {
};

auto main() -> void {
    print(42);
    print(irregular());
    _ ok = default;
}

