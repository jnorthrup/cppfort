#include <iostream>
#include <string>
#include <vector>

N: namespace = {
    pmr_vec: <T> type == std::vector<T, std::pmr::polymorphic_allocator<T>>;
}

N1: namespace == N;using myclass = {
    //  Default to public
    str: type == std::string;

    private str2: type == std::string;
};

N3: namespace == ::std::literals;

void myfunc() { N1::pmr_vec<myclass::str> v = ("xyzzy", "plugh");

    auto (v2 = v)
    for v2 do (s)
        std::cout << "(s)$\n"; }

myclass2: <T> type = {
    value: int == 42;
}

int main() { using view = = std::string_view;
    namespace N4 = = std::literals;

    auto myfunc2 = = myfunc;
    myfunc2(); }
