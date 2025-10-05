#include "emitter.h"
#include <sstream>

namespace cppfort::stage0 {

std::string Emitter::emit(const TranslationUnit& unit, const EmitOptions& options) {
    std::ostringstream out;
    
    // Emit includes
    for (const auto& include : unit.includes) {
        out << include.path << "\n";
    }
    
    if (!unit.includes.empty()) {
        out << "\n";
    }
    
    // Emit functions
    for (const auto& func : unit.functions) {
        out << emitFunction(func) << "\n";
    }
    
    return out.str();
}

std::string Emitter::emitFunction(const FunctionDecl& func) {
    std::ostringstream out;
    
    // Special case for main function
    if (func.name == "main") {
        out << "int main() ";
    } else {
        // Emit return type and function name
        if (func.return_type) {
            out << *func.return_type << " ";
        } else {
            out << "void ";
        }
        out << func.name << "(";
        
        // Emit parameters
        for (size_t i = 0; i < func.parameters.size(); ++i) {
            if (i > 0) out << ", ";
            out << emitParameter(func.parameters[i]);
        }
        out << ") ";
    }
    
    // Emit function body
    std::visit([&out, &func](const auto& body) {
        using T = std::decay_t<decltype(body)>;
        if constexpr (std::is_same_v<T, OrbitBlock>) {
            out << func.raw_body;
        } else if constexpr (std::is_same_v<T, OrbitExpressionBody>) {
            out << "{ return " << body.expression << "; }\n";
        }
    }, func.body);
    
    return out.str();
}

std::string Emitter::emitParameter(const Parameter& param) {
    return param.type + " " + param.name;
}

} // namespace cppfort::stage0