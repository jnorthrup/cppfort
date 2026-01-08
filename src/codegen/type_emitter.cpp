// src/codegen/type_emitter.cpp - Type generation implementation
// Part of Phase 2: Code Generator Extraction
#include "codegen/type_emitter.hpp"
#include <sstream>

namespace cpp2_transpiler {

std::string TypeEmitter::generate_type(Type* type) {
    if (!type) return "void";

    // Map common Cpp2-style builtin names to C++ spellings.
    if (type->kind == Type::Kind::Builtin) {
        if (type->name == "i32" || type->name == "int32") return "int";
        if (type->name == "u32" || type->name == "uint32") return "unsigned int";
        if (type->name == "string") return "std::string";
        if (type->name == "string_view") return "std::string_view";
    }

    switch (type->kind) {
        case Type::Kind::Builtin:
        case Type::Kind::UserDefined:
            return type->name;
        case Type::Kind::Template: {
            std::string result = type->name;
            if (!type->template_args.empty()) {
                result += "<";
                for (size_t i = 0; i < type->template_args.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += generate_type(type->template_args[i].get());
                }
                result += ">";
            }
            return result;
        }
        case Type::Kind::Pointer: {
            // Special case: pointer to function type needs special C++ syntax
            if (type->pointee && type->pointee->kind == Type::Kind::FunctionType) {
                std::string func_cpp = convert_function_type_to_cpp(type->pointee->name);
                auto paren_pos = func_cpp.find('(');
                if (paren_pos != std::string::npos) {
                    std::string result = func_cpp.substr(0, paren_pos) + "(*)" + func_cpp.substr(paren_pos);
                    if (type->is_const) {
                        result += " const";
                    }
                    return result;
                }
                return func_cpp + "*";
            }

            std::string pointee_type = generate_type(type->pointee.get());

            if (type->pointee && type->pointee->is_const) {
                pointee_type += " const";
            }

            std::string result = pointee_type + "*";

            if (type->is_const) {
                result += " const";
            }

            return result;
        }
        case Type::Kind::Reference:
            return generate_type(type->pointee.get()) + "&";
        case Type::Kind::Auto:
            return "auto";
        case Type::Kind::Deduced:
            if (type->name.starts_with("_ is ")) {
                std::string constraint = type->name.substr(5);
                return constraint;
            }
            return "decltype(auto)";
        case Type::Kind::FunctionType:
            return convert_function_type_to_cpp(type->name);
        default:
            return type->name;
    }
}

std::string TypeEmitter::generate_parameter_type(Type* type, const std::vector<ParameterQualifier>& qualifiers) {
    std::string base_type;
    
    if (!type) {
        base_type = "auto";
    } else if (type->kind == Type::Kind::Deduced) {
        base_type = "auto";
    } else {
        base_type = generate_type(type);
    }

    for (const auto& qual : qualifiers) {
        switch (qual) {
            case ParameterQualifier::In:
                return "const " + base_type + "&";
            case ParameterQualifier::InOut:
            case ParameterQualifier::Out:
                return base_type + "&";
            case ParameterQualifier::Move:
            case ParameterQualifier::Forward:
                return base_type + "&&";
            default:
                break;
        }
    }

    return base_type;
}

std::string TypeEmitter::convert_function_type_to_cpp(const std::string& func_type) {
    // Parse "(params) -> return_type" format
    // Find the -> separator
    auto arrow_pos = func_type.find(") ->");
    if (arrow_pos == std::string::npos) {
        return func_type;  // Fallback
    }
    
    // Extract params (skip opening paren)
    std::string params = func_type.substr(1, arrow_pos - 1);
    
    // Extract return type (skip ") -> ")
    std::string return_type = func_type.substr(arrow_pos + 4);
    
    // Trim whitespace from return type
    size_t start = return_type.find_first_not_of(" ");
    if (start != std::string::npos) {
        return_type = return_type.substr(start);
    }
    
    // Handle "forward" return kind - just strip it for C++ output
    if (return_type.starts_with("forward ")) {
        return_type = return_type.substr(8);
    }
    
    // Convert parameter list from Cpp2 to C++
    // Cpp2: "x:int, y:int" -> C++: "int, int"
    std::string cpp_params;
    std::istringstream param_stream(params);
    std::string param;
    bool first = true;
    
    while (std::getline(param_stream, param, ',')) {
        // Trim whitespace
        size_t pstart = param.find_first_not_of(" ");
        size_t pend = param.find_last_not_of(" ");
        if (pstart == std::string::npos) continue;
        param = param.substr(pstart, pend - pstart + 1);
        
        // Find colon separator: "name: type" or "name:type"
        auto colon_pos = param.find(':');
        if (colon_pos != std::string::npos) {
            // Extract type after colon
            std::string type_part = param.substr(colon_pos + 1);
            // Trim whitespace from type
            size_t tstart = type_part.find_first_not_of(" ");
            if (tstart != std::string::npos) {
                type_part = type_part.substr(tstart);
            }
            
            if (!first) cpp_params += ", ";
            cpp_params += type_part;
            first = false;
        }
    }
    
    // Generate C++ function type: return_type(params)
    if (return_type == "void" || return_type.empty()) {
        return "void(" + cpp_params + ")";
    }
    return return_type + "(" + cpp_params + ")";
}

// Private methods - placeholders for complete implementation
std::string TypeEmitter::generate_builtin_type(Type* type) {
    return type ? type->name : "void";
}

std::string TypeEmitter::generate_user_type(Type* type) {
    return type ? type->name : "";
}

std::string TypeEmitter::generate_pointer_type(Type* type) {
    if (!type || !type->pointee) return "void*";
    return generate_type(type->pointee.get()) + "*";
}

std::string TypeEmitter::generate_reference_type(Type* type) {
    if (!type || !type->pointee) return "void&";
    return generate_type(type->pointee.get()) + "&";
}

std::string TypeEmitter::generate_array_type(Type* type) {
    // TODO: Full array type generation
    return type ? type->name : "void[]";
}

std::string TypeEmitter::generate_function_type(Type* type) {
    return convert_function_type_to_cpp(type ? type->name : "() -> void");
}

std::string TypeEmitter::generate_template_type(Type* type) {
    if (!type) return "";
    std::string result = type->name;
    if (!type->template_args.empty()) {
        result += "<";
        for (size_t i = 0; i < type->template_args.size(); ++i) {
            if (i > 0) result += ", ";
            result += generate_type(type->template_args[i].get());
        }
        result += ">";
    }
    return result;
}

} // namespace cpp2_transpiler
