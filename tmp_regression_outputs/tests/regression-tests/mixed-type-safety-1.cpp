#include <iostream>
#include <string>
#include <iostream>
#include <string>
#include <variant>
#include <any>
#include <optional>

class Shape { public: virtual ~Shape() { } };
class Circle : public Shape { };
class Square : public Shape { };


print: <T : using > ( msg: std::string, x: T ) = {
    std::cout << "(msg)$ (x)$\n";
}

int main() { // Full qualification is necessary to avoid ambiguity in C++23
    // C++23 defines std::print, which would be picked up here by ADL
    ::print( "1.1 is int?", 1.1 is int );
    ::print( "1   is int?", 1   is int );

    auto c = new<Circle>();     * Shape // safe by construction
    s = c.get();  // safe by Lifetime
    ::print("\ns* is Shape? ", s* is Shape  );
    ::print(  "s* is Circle?", s* is Circle );
    ::print(  "s* is Square?", s* is Square ); }
