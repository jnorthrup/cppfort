#include <iostream>
#include <string>
#include <variant>
#include <any>
#include <optional>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


class Shape { public: virtual ~Shape() { } };
class Circle : public Shape { };
class Square : public Shape { };

template<typename T> auto print(cpp2::impl::in<std::string> msg, T const& x) -> void;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


template<typename T> auto print(cpp2::impl::in<std::string> msg, T const& x) -> void{
    std::cout << "" + cpp2::to_string(msg) + " " + cpp2::to_string(x) + "\n";
}

[[nodiscard]] auto main() -> int
{
    // Full qualification is necessary to avoid ambiguity in C++23
    // C++23 defines std::print, which would be picked up here by ADL
    ::print("1.1 is int?", cpp2::impl::is<int>(1.1));
    ::print( "1   is int?", cpp2::impl::is<int>(1));

    auto c {cpp2_new<Circle>()}; // safe by construction
    Shape* s {CPP2_UFCS(get)(cpp2::move(c))}; // safe by Lifetime
    ::print("\ns* is Shape? ", cpp2::impl::is<Shape>(*cpp2::impl::assert_not_null(s)));
    ::print(  "s* is Circle?", cpp2::impl::is<Circle>(*cpp2::impl::assert_not_null(s)));
    ::print(  "s* is Square?", cpp2::impl::is<Square>(*cpp2::impl::assert_not_null(cpp2::move(s))));
}

