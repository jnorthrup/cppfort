#include "cpp2util.h"

[[nodiscard]] auto name() -> std::string;
auto decorate(std::string& s) -> void;

auto main() -> int {
    std::cout << "Hello " << name() << "\n";
}

[[nodiscard]] auto name() -> std::string {
    std::string s = "world";
    decorate(s);
    return s;
}

auto decorate(std::string& s) -> void {
    s = "[" + s + "]";
}

