#include <iostream>
#include <string>
#include <map>
int main() { auto i = 42;
    std::map<std::string, int> m = ();
    m["one"] = 1;
    m["two"] = 2;

    std::string str = "this is a string";

    std::string raw_str = R"string(raw string without interpolation)string";

    std::string raw_str_multi = R"test(this is raw string literal

that can last for multiple

lines)test";

    std::string raw_str_inter = $R"test(this is raw string literal
that can last for multiple
lines
(i)$ R"(this can be added too)"
calculations like m["one"] + m["two"] = (m["one"] + m["two"])$ also works
("at the beginning of the line")$!!!)test";

    std::string raw_str_inter_multi = $R"(

    )" + $R"((i)$)" + $R"((i)$)";

    std::cout << str << std::endl;
    std::cout << raw_str << std::endl;
    std::cout << raw_str_multi << std::endl;
    std::cout << raw_str_inter << std::endl;
    std::cout << raw_str_inter_multi << std::endl;
    std::cout << ($R"((m["one"])$.)" + $R"((m["two"])$.)" + $R"((m["three"])$.)" + $R"((i)$)") << std::endl; }
