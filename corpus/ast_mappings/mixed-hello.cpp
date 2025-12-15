#include <iostream>
#include <string>

// Forward declaration
auto decorate(std::string& s) -> void;

// Cpp2 function signature: name: () -> std::string
auto name() -> std::string {
    // Cpp2: s: std::string = "world";
    std::string s = "world";

    // Cpp2: decorate(s);  (UFCS call becomes regular call)
    decorate(s);

    return s;
}

// Cpp2 function signature: decorate: (inout s: std::string)
auto decorate(std::string& s) -> void {
    // Cpp2: s = "[" + s + "]";
    s = "[" + s + "]";
}

auto main() -> int {
    std::cout << "Hello " << name() << "\n";
}
