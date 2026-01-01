#include "cpp2util.h"

auto print_an_int(_ x) -> void;

auto main() -> int {
    print_an_int("syzygy");
    print_an_int(1);
    print_an_int(1.1);
}

auto print_an_int(_ x) -> void {
    std::cout << std::setw(30) << cpp2::to_string(x) << " value is " << ([&]() {
    auto __value = x;
    if (__value == int) { return std::to_string(/* expression kind 10 */); }
    else { return "not an int"; }
    else { throw std::logic_error("Non-exhaustive inspect"); }
})() << "\n";
}

