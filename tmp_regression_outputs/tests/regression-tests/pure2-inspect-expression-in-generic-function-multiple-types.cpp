#include <iostream>
#include <string>
#include <set>
int main() { std::variant<int, int, double> v = 42.0;
    std::any a = "xyzzy" as std::string;
    std::optional<int> o = ();

    test_generic(3.14, "double");
    test_generic(v,    "variant<int, int, double>");
    test_generic(a,    "any");
    test_generic(o,    "optional<int>");

    _ = v.emplace<0>(1);
    a = 2;
    o = 3;
    test_generic(42,   "int");
    test_generic(v,    "variant<int, int, double>");
    test_generic(a,    "any");
    test_generic(o,    "optional<int>"); }

void test_generic(const auto& x, const auto& msg) { :cout
        << std::setw(30) << msg
        << " value is "
        << inspect x -> std::string {
            is int std = "integer " + std::to_string(x as int);
            is std::string = '"' + x as std::string + '"';
            is _   = "not an int or string";
        }
        << "\n"; }
