#include "cpp2util.h"

auto fun(std::string&& s1, std::string&& s2, std::string&& s3) -> void;

auto fun(std::string&& s1, std::string&& s2, std::string&& s3) -> void {
    std::cout << s1 << s2 << s3 << std::endl;
}

auto main() -> void {
    std::string b = "b";
    std::string c = "c";
    fun(std::string("a"), b, c);
    b = "";
}

