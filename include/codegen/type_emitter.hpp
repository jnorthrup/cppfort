// include/codegen/type_emitter.hpp - Type generation for C++ code
// Part of Phase 2: Code Generator Extraction
#pragma once

#include "codegen/emitter_context.hpp"
#include "ast.hpp"
#include <string>

namespace cpp2_transpiler {

// Forward declarations
class ExpressionEmitter;

/// Generates C++ type representations from Cpp2 AST types
class TypeEmitter : public EmitterBase {
public:
    explicit TypeEmitter(EmitterContext& ctx) : EmitterBase(ctx) {}

    /// Generate C++ type string from AST Type
    std::string generate_type(Type* type);
    
    /// Generate parameter type with qualifiers
    std::string generate_parameter_type(Type* type, const std::vector<ParameterQualifier>& qualifiers);
    
    /// Convert Cpp2 function type "(params) -> return_type" to C++ "return_type(params)"
    std::string convert_function_type_to_cpp(const std::string& func_type);

private:
    std::string generate_builtin_type(Type* type);
    std::string generate_user_type(Type* type);
    std::string generate_pointer_type(Type* type);
    std::string generate_reference_type(Type* type);
    std::string generate_array_type(Type* type);
    std::string generate_function_type(Type* type);
    std::string generate_template_type(Type* type);
};

} // namespace cpp2_transpiler
