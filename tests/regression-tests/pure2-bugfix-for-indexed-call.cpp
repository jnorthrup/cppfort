#include "cpp2util.h"

auto f(int _) -> void;

auto f(int _) -> void {
}

auto main() -> void {
    std::array array_of_functions = {f, f};
    auto index = 0;
    int arguments = 0;
    array_of_functions[index](arguments);
    _ = array_of_functions;
    _ = index;
    _ = arguments;
}

