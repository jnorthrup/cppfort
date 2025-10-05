#pragma once

#include <string>
#include "ast.h"

namespace cppfort::stage0 {

class Emitter {
public:
    Emitter() = default;
    
    // Emit AST as C++ source code
    std::string emit(const TranslationUnit& unit, const EmitOptions& options);
    
private:
    // Helper methods for emission
    std::string emitFunction(const FunctionDecl& func);
    std::string emitParameter(const Parameter& param);
    std::string emitIncludes(const std::vector<std::string>& includes);
};

} // namespace cppfort::stage0