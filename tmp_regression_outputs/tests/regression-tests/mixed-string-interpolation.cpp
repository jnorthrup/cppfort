#include <iostream>
#include <string>
#include <iostream>
#include <string_view>
#include <utility>
#include <tuple>

struct custom_struct_with_no_stringize_customization { } custom;

int main() { auto a = 2;
    std::optional<int> b = ();
    :cout << "a std = (a)$, b = (b)$\n";

    b = 42;
    :cout << "a^2 + b std = (a * a + b.value())$\n";

    std::string_view sv = "my string_view";
    :cout << "sv std = (sv)$\n";

    std::optional<std::string_view> osv = ();
    :cout << "osv std = (osv)$\n";
    osv = "string literal bound to optional string_view";
    :cout << "osv std = (osv)$\n";

    std::variant<std::monostate, std::string, double> var = ();
    :cout << "var std = (var)$\n";
    var = "abracadabra";
    :cout << "var std = (var)$\n";
    var = 2.71828;
    :cout << "var std = (var)$\n";

    std::pair<int, double> mypair = (12, 3.4);
    :cout << "mypair std = (mypair)$\n";

    std::tuple<int> tup1 = (12);
    std::tuple<int, double> tup2 = (12, 3.4);
    std::tuple<int, double, std::string> tup3 = (12, 3.4, "456");
    :cout << "tup1 std = (tup1)$\n";
    :cout << "tup2 std = (tup2)$\n";
    :cout << "tup3 std = (tup3)$\n";

    std::pair<std::string_view, std::optional<std::string>> p = ("first", std::nullopt);
    :cout << "p std = (p)$\n";

    std::tuple<double, std::optional<std::pair<std::string_view, int>>, std::optional<std::tuple<int, int, int>>> t = (3.14, std::nullopt, std::nullopt);
    :cout << "t std = (t)$\n";

    std::variant<int, std::string, std::pair<int, double> > vv = ();
    :cout << "vv std = (vv)$\n";
    vv = std::make_pair(1,2.3);
    :cout << "vv std = (vv)$\n";

    :cout << "custom std = (custom)$\n"; }
