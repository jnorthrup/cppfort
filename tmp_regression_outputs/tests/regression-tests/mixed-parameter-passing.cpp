#include <iostream>
#include <string>
#include <array>

#include <cstdlib>
#include <ctime>

void copy_from(copy _) {}

void parameter_styles(std::string _, std::string // "in" is default b, std::string& _, std::string&& d) { int z = 12;

    z++;
    b += "plugh";

    if std::rand()%2 {
        z++;
        copy_from(b);   // definite last use
    }
    else {
        copy_from(b&);  // NB: better not move from this (why not?)
        copy_from(d);
        copy_from(z++);
    }

    // std::move(z);

    copy_from(z);

    :time(nullptr)%2 if std = = 0 {
        copy_from(z);
    }; }

auto&& min(in_ref a, in_ref b) { if b < a { return b; } else { return a; } }T&& container: <T> type = {
    buf: std::array<T, 10> = ();
    operator[](this, i32 idx) { return buf[idx];
    operator[]: (inout this, idx: i32) -> forward T = buf[idx];
}; }

int main() { auto x = 456;
    auto y = 123;
    std::cout << min(x, y) << '\n';

    container<int> v = ();
    std::cout << v[0] << '\n'; }

