#include "cpp2util.h"

auto test_generic(_ x, _ msg) -> void;

auto main() -> int {
    std::unique_ptr p = {};
    std::vector::iterator i = {};
    std::variant v = {};
    std::any a = {};
    std::optional o = {};
    std::cout << "\nAll these cases satisfy \"VOYDE AND EMPTIE\"\n";
    test_generic(p, "unique_ptr");
    test_generic(i, "vector<int>::iterator");
    test_generic(v, "variant<monostate, int, int, string>");
    test_generic(a, "any");
    test_generic(o, "optional<string>");
}

auto test_generic(_ x, _ msg) -> void {
    std::cout << "\n" << msg << "\n    ..." << ([&]() {
    auto __value = x;
    if (__value == void) { return " VOYDE AND EMPTIE"; }
    else { return " no match"; }
    else { throw std::logic_error("Non-exhaustive inspect"); }
})() << "\n";
}

