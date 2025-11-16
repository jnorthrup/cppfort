#include <iostream>
#include <cassert>
#include "json_pattern_loader.h"

int main() {
    std::cout << "Testing JsonPatternLoader with patterns/cppfort_core_patterns.json...\n\n";
    
    cppfort::stage0::JsonPatternLoader loader;
    auto patterns = loader.load_from_file("patterns/cppfort_core_patterns.json");
    
    if (patterns.empty()) {
        std::cerr << "Failed to load patterns: " << loader.get_last_error() << "\n";
        return 1;
    }
    
    std::cout << "Loaded " << patterns.size() << " patterns\n";
    
    // Verify some key patterns have templates
    bool found_function = false;
    bool found_parameter = false;
    bool found_variable = false;
    
    for (const auto& pattern : patterns) {
        if (pattern.name == "cpp2_function_with_return") {
            found_function = true;
            auto it = pattern.templates.find(2);
            assert(it != pattern.templates.end());
            assert(it->second == "$3 $1($2) { $4 }");
            std::cout << "✓ cpp2_function_with_return template correct: " << it->second << "\n";
        }
        if (pattern.name == "cpp2_parameter") {
            found_parameter = true;
            auto it = pattern.templates.find(2);
            assert(it != pattern.templates.end());
            assert(it->second == "$2 $1");
            std::cout << "✓ cpp2_parameter template correct: " << it->second << "\n";
        }
        if (pattern.name == "cpp2_typed_variable") {
            found_variable = true;
            auto it = pattern.templates.find(2);
            assert(it != pattern.templates.end());
            assert(it->second == "$2 $1 = $3;");
            std::cout << "✓ cpp2_typed_variable template correct: " << it->second << "\n";
        }
    }
    
    assert(found_function);
    assert(found_parameter);
    assert(found_variable);
    
    std::cout << "\n✅ All JsonPatternLoader tests passed!\n";
    return 0;
}