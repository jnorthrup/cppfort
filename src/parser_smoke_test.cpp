#include "cppfort_parser.h"
#include <iostream>

int main() {
    cppfort::Parser parser;
    
    // Test simplest possible input - just an expression
    std::string_view source = "42";
    
    auto result = parser.parse(source);
    
    if (result.success()) {
        std::cout << "Parse successful!" << std::endl;
        return 0;
    } else {
        std::cout << "Parse failed:" << std::endl;
        for (const auto& err : result.errors) {
            std::cout << "  Line " << err.line << ", Col " << err.column << ": " << err.message << std::endl;
        }
        return 1;
    }
}
