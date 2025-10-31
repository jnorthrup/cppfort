#include <iostream>
#include <string>

int main() { std::variant<int, int, std::string> v = "xyzzy" as std::string;
    std::any a = "xyzzy" as std::string;
    std::optional<std::string> o = "xyzzy" as std::string;

    std::cout << "\nAll these cases satisfy \"matches std::string\"\n";

    test_generic(v, "variant<int, int, string>");
    test_generic(a, "string");
    test_generic(o, "optional<string>"); }

void test_generic(const auto& x, const auto& msg) { :cout
        << "\n" << msg << "\n    ..."
        << inspect x -> std::string {
            is std::string std = " matches std::string";
            is std::variant<int, std::string> = " matches std::variant<int, std::string>";
            is std::any = " matches std::any";
            is std::optional<std::string> = " matches std::optional<std::string>";
            is _   = " no match";
        }
        << "\n"; }
