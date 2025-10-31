#include <iostream>
#include <string>
#include <vector>

int main() { std::unique_ptr<int> p = ();
    std::vector<int>::iterator i = ();
    std::variant<std::monostate, int, int, std::string> v = ();
    std::any a = ();
    std::optional<std::string> o = ();

    std::cout << "\nAll these cases satisfy \"VOYDE AND EMPTIE\"\n";

    test_generic(p, "unique_ptr");
    test_generic(i, "vector<int>::iterator");
    test_generic(v, "variant<monostate, int, int, string>");
    test_generic(a, "any");
    test_generic(o, "optional<string>"); }

void test_generic(const auto& x, const auto& msg) { :cout
        << "\n" << msg << "\n    ..."
        << inspect x -> std::string {
            is void std = " VOYDE AND EMPTIE";
            is _   = " no match";
        }
        << "\n"; }