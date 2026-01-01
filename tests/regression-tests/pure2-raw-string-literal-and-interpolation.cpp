#include "cpp2util.h"


auto main() -> int {
    auto i = 42;
    std::map m = {};
    m["one"] = 1;
    m["two"] = 2;
    std::string str = "this is a string";
    std::string raw_str = "R"string(raw string without interpolation)string"";
    std::string raw_str_multi = "R"test(this is raw string literal

that can last for multiple

lines)test"";
    std::string raw_str_inter = /* null expression */;
    std::string raw_str_inter_multi = /* null expression */;
    std::cout << str << std::endl;
    std::cout << raw_str << std::endl;
    std::cout << raw_str_multi << std::endl;
    std::cout << raw_str_inter << std::endl;
    std::cout << raw_str_inter_multi << std::endl;
    std::cout << /* null expression */;
}

