#include <iostream>
#include <string>
#include <filesystem>
#include <iostream>
#include <typeinfo>

namespace my {
    using u16 = float;
}

void test(x) { std::cout
        << std::is_floating_point_v<CPP2_TYPEOF(x)> as std::string
        << "\n"; }

mytype: @struct <T> type = {
    myvalue: <U> bool == true;
}

int main(args) { my::u16 y = 42;
    test(y);

    u16 z = 42;
    test(z);

    for args do (arg)
        std::cout << std::filesystem::path(arg).filename() << "\n";

    std::cout << "(mytype<int>::myvalue<int>)$\n"; }
