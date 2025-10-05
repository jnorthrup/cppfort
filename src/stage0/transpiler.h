#pragma once

#include <string>
#include "ast.h"

namespace cppfort::stage0 {

class Transpiler {
public:
    Transpiler() = default;
    
    // Parse cpp2 source code into AST
    TranslationUnit parse(const std::string& source, const std::string& filename);
    
private:
    // Helper methods for parsing
    std::string extractFunctionName(const std::string& line);
    std::string extractReturnType(const std::string& line);
    std::vector<Parameter> extractParameters(const std::string& line);
    
    // Transform cpp2 syntax to C++ syntax
    std::string transformCpp2ToCpp(const std::string& cpp2_body);
};

} // namespace cppfort::stage0