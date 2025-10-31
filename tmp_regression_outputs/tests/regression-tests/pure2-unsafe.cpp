#include <iostream>
#include <string>

void f(const i32& i, std::string& s) { auto // j = i as i16;                     auto // error, maybe-lossy narrowing
    j = unchecked_narrow<i16>(i);           *void // ok, 'unchecked' is explicit

    pv = s&;
    auto // pi = pv as *std::string;          auto // error, unsafe cast
    ps = unchecked_cast<*std::string>(pv);  // ok, 'unchecked' is explicit
    ps* = "plugh"; }

int main() { std::string str = "xyzzy";
    f( 42, str );
    std::cout << str; }
