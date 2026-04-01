#include "src/selfhost/cpp2.h2"
#include <iostream>
#include <string_view>

int main() {
    for (auto src : {std::string_view{"a :=\nb"}, std::string_view{"f: (a,\nb) = { }"}, std::string_view{"f: () ->\nint = { }"}, std::string_view{"a := 1 +\n2"}}) {
        auto marks = cpp2::scan(src);
        std::cout << "SRC:" << src << "\n";
        for (auto const& m : marks) std::cout << m.pos << ":" << static_cast<int>(m.k) << "\n";
        auto regs = cpp2::fold(src, marks);
        for (auto const& r : regs) std::cout << "REG " << static_cast<int>(r.k) << " [" << r.lo << "," << r.hi << ") => " << src.substr(r.lo, r.hi-r.lo) << "\n";
        std::cout << "---\n";
    }
}
