#include "json_scanner.h"
#include <cassert>
#include <iostream>

int main() {
    const std::string json = R"json({"hello": "world", "answer": 42})json";
    cppfort::stage0::JsonScanner scanner(json);
    auto doc = scanner.scan();
    auto values = scanner.extract_value(static_cast<size_t>(doc.second.first.front()));
    assert(!values.empty());
    std::cout << "Parsed first value: " << values << "\n";
    std::cout << "Json scanner simple test passed\n";
    return 0;
}
