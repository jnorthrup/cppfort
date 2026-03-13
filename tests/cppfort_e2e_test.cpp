#include "cppfort_parser.h"
#include <iostream>

int main() {
    cppfort::Parser parser;
    std::string_view source = "chart foo(x) { }";
    
    std::cout << "=== Testing: " << source << " ===" << std::endl;
    
    auto result = parser.parse(source);
    
    if (result.success()) {
        std::cout << "Parse successful!" << std::endl;
        return 0;
    } else {
        std::cerr << "Parse failed:" << std::endl;
        for (const auto& err : result.errors) {
            std::cerr << "  Line " << err.line << ", Col " << err.column << ": " << err.message << std::endl;
        }
        return 1;
    }
}
