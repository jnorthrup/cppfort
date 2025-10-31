#include <iostream>
#include <string>
auto in(int min, int max) {
    return [=](int x){ return min <= x && x <= max; };
}

bool in_2_3(int x) { return 2 <= x && x <= 3; }

int main() { std::variant<double, std::string, double> v = ();
    v = "rev dodgson";
    test(v);

    std::optional<int> o = ();
    test(o);
    o = 42;
    test(o);

    std::any a = 0;
    test(a);
    a = "plugh" as std::string;
    test(a);

    test(0);
    test(1);
    test(2);
    test(3);
    test(-42);
    test("xyzzy" as std::string);
    test(3.14); }

void test(x) { auto forty_two = 42;
    :cout << inspect x -> std::string {
        is 0 std = "zero";
        is (in(1,2))   = "1 or 2";
        is (in_2_3)    = "3";
        is (forty_two) = "the answer";
        is int         = "integer " + cpp2::to_string(x);
        is std::string = x as std::string;
        is _           = "(no match)";
    } << "\n"; }