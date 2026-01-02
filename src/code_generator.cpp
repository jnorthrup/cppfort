#include "code_generator.hpp"
#include <iostream>
#include <format>
#include <algorithm>
#include <unordered_set>
#include <regex>

namespace cpp2_transpiler {

// Helper to generate a template parameter declaration
// Handles variadic packs: "Ts..." becomes "typename... Ts"
// Handles non-type packs: "Ts...: int" becomes "int... Ts" (conceptually)
// Regular type params: "T" becomes "typename T"
static std::string generate_template_param(const std::string& param) {
    // Check if this is a variadic parameter (ends with ...)
    if (param.size() >= 3 && param.substr(param.size() - 3) == "...") {
        std::string name = param.substr(0, param.size() - 3);
        return "typename... " + name;
    }
    return "typename " + param;
}

// Helper to process Cpp2 string interpolation: (expr)$
// Transforms "(x)$" into proper C++ string concatenation with type-aware conversion
// This handles the Cpp2 string interpolation syntax
static std::string process_string_interpolation(const std::string& str) {
    // If string doesn't contain )$, no interpolation needed
    if (str.find(")$") == std::string::npos) {
        return str;
    }

    std::string result;
    size_t i = 0;
    bool in_string = false;
    size_t string_start = 0;

    // Track if we're processing content inside the string quotes
    // Look for (expr)$ patterns
    while (i < str.length()) {
        if (str[i] == '"' && (i == 0 || str[i-1] != '\\')) {
            if (!in_string) {
                in_string = true;
                string_start = i;
                result += '"';
            } else {
                in_string = false;
                result += '"';
            }
            i++;
        } else if (in_string && str[i] == '(' && i > string_start) {
            // Potential interpolation start - look for matching )$
            size_t paren_count = 1;
            size_t j = i + 1;
            while (j < str.length() && paren_count > 0) {
                if (str[j] == '(') paren_count++;
                else if (str[j] == ')') paren_count--;
                j++;
            }
            // Check if ) is followed by $
            if (j < str.length() && str[j] == '$') {
                // Found interpolation (expr)$
                std::string expr = str.substr(i + 1, j - i - 2);
                // Close current string, add type-aware conversion, reopen string
                // Use streaming with std::ostringstream for universal type support
                result += "\" + (([&]() { std::ostringstream __oss; __oss << " + expr + "; return __oss.str(); })()) + \"";
                i = j + 1;  // Skip past the $
            } else {
                result += str[i];
                i++;
            }
        } else {
            result += str[i];
            i++;
        }
    }
    return result;
}

// Helper to check if a token type is a comparison operator
static bool is_comparison_operator(TokenType op) {
    return op == TokenType::LessThan ||
           op == TokenType::GreaterThan ||
           op == TokenType::LessThanOrEqual ||
           op == TokenType::GreaterThanOrEqual ||
           op == TokenType::DoubleEqual ||
           op == TokenType::NotEqual;
}

// Helper to get the string representation of a comparison operator
static std::string get_comparison_operator_string(TokenType op) {
    switch (op) {
        case TokenType::LessThan: return "<";
        case TokenType::GreaterThan: return ">";
        case TokenType::LessThanOrEqual: return "<=";
        case TokenType::GreaterThanOrEqual: return ">=";
        case TokenType::DoubleEqual: return "==";
        case TokenType::NotEqual: return "!=";
        default: return "/* unknown */";
    }
}

CodeGenerator::CodeGenerator() : indent_level(0), needs_semicolon(true), output_mode_(OutputMode::Inline) {}

CodeGenerator::CodeGenerator(OutputMode mode) : indent_level(0), needs_semicolon(true), output_mode_(mode) {}

std::string CodeGenerator::generate(AST& ast) {
    output.str("");
    output.clear();

    write_includes();

    // Pass 0: Generate C++1 passthrough declarations that should come first
    // (includes, struct definitions, etc.) - these need to appear before forward declarations
    for (auto& decl : ast.declarations) {
        if (decl->kind == Declaration::Kind::Cpp1Passthrough) {
            auto* cpp1 = static_cast<Cpp1PassthroughDeclaration*>(decl.get());
            // Check if this is an include directive or struct/class definition
            bool is_early = (cpp1->raw_code.find("#include") == 0 ||
                             cpp1->raw_code.find("#define") == 0 ||
                             cpp1->raw_code.find("#pragma") == 0 ||
                             cpp1->raw_code.find("struct ") == 0 ||
                             cpp1->raw_code.find("class ") == 0 ||
                             cpp1->raw_code.find("union ") == 0 ||
                             cpp1->raw_code.find("enum ") == 0 ||
                             cpp1->raw_code.find("template") == 0);
            if (is_early) {
                generate_cpp1_passthrough_declaration(cpp1);
                write_line("");
            }
        }
    }

    // First pass: Generate forward declarations for all functions (except main)
    for (auto& decl : ast.declarations) {
        if (decl->kind == Declaration::Kind::Function) {
            auto* func = static_cast<FunctionDeclaration*>(decl.get());
            // Don't forward declare main(), and only forward declare if there's a body
            if (func->name != "main" && func->body) {
                generate_function_forward_declaration(func);
            }
        }
    }

    write_line("");

    // Second pass: Generate full definitions for all declarations
    // Skip C++1 passthrough that was already emitted in pass 0
    for (auto& decl : ast.declarations) {
        if (decl->kind == Declaration::Kind::Cpp1Passthrough) {
            auto* cpp1 = static_cast<Cpp1PassthroughDeclaration*>(decl.get());
            bool is_early = (cpp1->raw_code.find("#include") == 0 ||
                             cpp1->raw_code.find("#define") == 0 ||
                             cpp1->raw_code.find("#pragma") == 0 ||
                             cpp1->raw_code.find("struct ") == 0 ||
                             cpp1->raw_code.find("class ") == 0 ||
                             cpp1->raw_code.find("union ") == 0 ||
                             cpp1->raw_code.find("enum ") == 0 ||
                             cpp1->raw_code.find("template") == 0);
            if (is_early) {
                continue;  // Already emitted
            }
        }
        generate_declaration(decl.get());
        write_line("");
    }

    // Third pass: Generate C++20 module stubs for markdown blocks
    for (auto& decl : ast.declarations) {
        if (!decl->markdown_blocks.empty()) {
            generate_markdown_module_stubs(decl->markdown_blocks);
        }
    }

    return output.str();
}

void CodeGenerator::write_line(const std::string& line) {
    output << get_indent() << line << "\n";
}

void CodeGenerator::write(const std::string& text) {
    output << text;
}

void CodeGenerator::indent() {
    indent_level++;
}

void CodeGenerator::dedent() {
    indent_level--;
}

std::string CodeGenerator::get_indent() const {
    return std::string(indent_level * 4, ' ');
}

void CodeGenerator::write_includes() {
    switch (output_mode_) {
        case OutputMode::PCH:
            // Fastest: precompiled header provides everything
            // User compiles with: clang++ -include-pch cpp2_pch.h.pch
            // No #include needed in generated code
            write_line("// Requires: -include-pch cpp2_pch.h.pch");
            write_line("");
            return;
            
        case OutputMode::Header:
            // Smaller output: use runtime header
            write_line("#include <iostream>");
            write_line("#include <string>");
            write_line("#include <sstream>");
            write_line("#include <vector>");
            write_line("#include <memory>");
            write_line("#include <optional>");
            write_line("#include <variant>");
            write_line("#include <functional>");
            write_line("#include <utility>");
            write_line("#include <type_traits>");
            write_line("#include <stdexcept>");
            write_line("#include <cstdint>");
            write_line("#include <iterator>");  // For std::ssize
            write_line("#include <cpp2_runtime.h>");
            write_line("");
            return;
            
        case OutputMode::Inline:
        default:
            // Self-contained: inline everything
            break;
    }
    
    // Inline mode: embed all headers and runtime
    write_line("#include <iostream>");
    write_line("#include <string>");
    write_line("#include <sstream>");
    write_line("#include <vector>");
    write_line("#include <memory>");
    write_line("#include <optional>");
    write_line("#include <variant>");
    write_line("#include <functional>");
    write_line("#include <utility>");
    write_line("#include <type_traits>");
    write_line("#include <stdexcept>");
    write_line("#include <cstdint>");
    write_line("#include <iterator>");  // For std::ssize
    write_line("#include <filesystem>"); // For std::filesystem
    write_line("");
    // Inline the minimal cpp2 namespace (matches cpp2_runtime.h API)
    write_line("namespace cpp2 {");
    write_line("    template<typename T> auto to_string(T const& x) -> std::string {");
    write_line("        if constexpr (std::is_same_v<T, std::string>) { return x; }");
    write_line("        else if constexpr (std::is_same_v<T, const char*>) { return std::string(x); }");
    write_line("        else if constexpr (std::is_same_v<T, char>) { return std::string(1, x); }");
    write_line("        else if constexpr (std::is_same_v<T, bool>) { return x ? \"true\" : \"false\"; }");
    write_line("        else if constexpr (std::is_arithmetic_v<T>) { return std::to_string(x); }");
    write_line("        else { std::ostringstream oss; oss << x; return oss.str(); }");
    write_line("    }");
    write_line("    template<typename T, typename U> constexpr auto is(U const& x) -> bool {");
    write_line("        if constexpr (std::is_same_v<T, U> || std::is_base_of_v<T, U>) { return true; }");
    write_line("        else if constexpr (std::is_polymorphic_v<U>) { return dynamic_cast<T const*>(&x) != nullptr; }");
    write_line("        else { return false; }");
    write_line("    }");
    write_line("    template<typename T, typename U> constexpr auto as(U const& x) -> T {");
    write_line("        if constexpr (std::is_same_v<T, U>) { return x; }");
    write_line("        else if constexpr (std::is_base_of_v<T, U>) { return static_cast<T const&>(x); }");
    write_line("        else if constexpr (std::is_polymorphic_v<U>) { return dynamic_cast<T const&>(x); }");
    write_line("        else { return static_cast<T>(x); }");
    write_line("    }");
    write_line("    namespace impl {");
    write_line("        template<typename T, typename U> constexpr auto is_(U const& x) -> bool { return is<T>(x); }");
    write_line("        template<typename T, typename U> constexpr auto as_(U const& x) -> T { return as<T>(x); }");
    write_line("    }");
    write_line("    // main(args) support");
    write_line("    struct args_t { int argc; char const* const* argv; };");
    write_line("    inline auto make_args(int argc, char** argv) -> args_t { return { argc, const_cast<char const* const*>(argv) }; }");
    write_line("} // namespace cpp2");
    write_line("");
}

void CodeGenerator::generate_markdown_module_stubs(const std::vector<MarkdownBlockAttr>& blocks) {
    for (const auto& block : blocks) {
        generate_markdown_module_stub(block);
    }
}

void CodeGenerator::generate_markdown_module_stub(const MarkdownBlockAttr& block) {
    // Module name: use block name or generate __cas_<first 16 chars of hash>
    std::string module_name;
    if (!block.name.empty()) {
        module_name = block.name;
    } else {
        module_name = "__cas_" + block.sha256.substr(0, 16);
    }

    write_line("export module " + module_name + ";");
    write_line("");
    write_line("inline constexpr char cas_sha256[] = \"" + block.sha256 + "\";");
    write_line("");
}

void CodeGenerator::generate_declaration(Declaration* decl) {
    if (!decl) return;

    switch (decl->kind) {
        case Declaration::Kind::Variable:
            generate_variable_declaration(static_cast<VariableDeclaration*>(decl));
            break;
        case Declaration::Kind::Function:
            generate_function_declaration(static_cast<FunctionDeclaration*>(decl));
            break;
        case Declaration::Kind::Type:
            generate_type_declaration(static_cast<TypeDeclaration*>(decl));
            break;
        case Declaration::Kind::Namespace:
            generate_namespace_declaration(static_cast<NamespaceDeclaration*>(decl));
            break;
        case Declaration::Kind::Operator:
            generate_operator_declaration(static_cast<OperatorDeclaration*>(decl));
            break;
        case Declaration::Kind::Using:
            generate_using_declaration(static_cast<UsingDeclaration*>(decl));
            break;
        case Declaration::Kind::Import:
            generate_import_declaration(static_cast<ImportDeclaration*>(decl));
            break;
        case Declaration::Kind::Cpp1Passthrough:
            generate_cpp1_passthrough_declaration(static_cast<Cpp1PassthroughDeclaration*>(decl));
            break;
        default:
            break;
    }
}

void CodeGenerator::generate_variable_declaration(VariableDeclaration* decl) {
    if (!decl) return;

    std::string type_str = decl->type ? generate_type(decl->type.get()) : "auto";

    // Check if we're in a @regex type and this is a regex member
    for (const auto& metafunc : current_type_metafunctions) {
        if (metafunc == "regex") {
            // Transform members named "regex" or "regex_*" to std::regex
            if (decl->name == "regex" || decl->name.find("regex_") == 0) {
                type_str = "std::regex";
            }
            break;
        }
    }

    if (decl->is_const) {
        write("const ");
    }

    write_line(type_str + " " + decl->name + " = " +
              (decl->initializer ? generate_expression_to_string(decl->initializer.get()) : "default") + ";");
}

void CodeGenerator::generate_function_forward_declaration(FunctionDeclaration* decl) {
    if (!decl) return;

    std::string return_type = decl->return_type ? generate_type(decl->return_type.get()) : "void";

    // main() must return int in C++
    if (decl->name == "main" && return_type == "void") {
        return_type = "int";
    }

    // Detect if this is a member function (has 'this' parameter)
    bool is_member_function = false;
    bool is_const_member = false;
    bool is_rvalue_member = false;
    size_t this_param_index = SIZE_MAX;
    
    for (size_t i = 0; i < decl->parameters.size(); ++i) {
        if (decl->parameters[i].name == "this") {
            is_member_function = true;
            this_param_index = i;
            for (const auto& qual : decl->parameters[i].qualifiers) {
                if (qual == ParameterQualifier::In) is_const_member = true;
                if (qual == ParameterQualifier::Move) is_rvalue_member = true;
            }
            break;
        }
    }

    // Generate template header if this is a template function
    if (!decl->template_parameters.empty()) {
        write("template<");
        for (size_t i = 0; i < decl->template_parameters.size(); ++i) {
            if (i > 0) write(", ");
            write(generate_template_param(decl->template_parameters[i]));
        }
        write_line(">");
    }

    // [[nodiscard]] goes before the return type for widest compatibility
    if (needs_nodiscard(decl)) {
        write("[[nodiscard]] ");
    }

    // Use auto return_type syntax for cppfront compatibility
    write("auto " + decl->name + "(");

    // Parameters - skip 'this' parameter
    bool first_param = true;
    for (size_t i = 0; i < decl->parameters.size(); ++i) {
        if (i == this_param_index) continue;
        if (!first_param) write(", ");
        first_param = false;
        const auto& param = decl->parameters[i];
        write(generate_parameter_type(param.type.get(), param.qualifiers) + " " + param.name);
    }

    write(")");
    
    // Add const/&& qualifiers
    if (is_const_member) write(" const");
    if (is_rvalue_member) write(" &&");
    
    write(" -> " + return_type);

    write_line(";");
}

void CodeGenerator::generate_function_declaration(FunctionDeclaration* decl) {
    if (!decl) return;

    // Check for special member functions (operator=: patterns)
    // These need different treatment in C++
    if (decl->name == "operator=:" || decl->name == "=:") {
        generate_special_member_function(decl);
        return;
    }

    // Generate template header if this is a template function
    if (!decl->template_parameters.empty()) {
        write("template<");
        for (size_t i = 0; i < decl->template_parameters.size(); ++i) {
            if (i > 0) write(", ");
            write(generate_template_param(decl->template_parameters[i]));
        }
        write_line(">");
    }

    std::string return_type = decl->return_type ? generate_type(decl->return_type.get()) : "void";
    
    // main() must return int per C++ standard
    if (decl->name == "main" && return_type == "void") {
        return_type = "int";
    }

    // [[nodiscard]] goes before the return type for widest compatibility
    if (needs_nodiscard(decl)) {
        write("[[nodiscard]] ");
    }

    // Check for main(args) pattern - Cpp2 idiom for command line arguments
    // main: (args) = { ... } becomes:
    //   auto main(int argc_, char** argv_) -> int {
    //       auto const args = cpp2::make_args(argc_, argv_);
    //       ...
    //   }
    bool is_main_with_args = false;
    std::string args_param_name;
    if (decl->name == "main" && decl->parameters.size() == 1) {
        const auto& param = decl->parameters[0];
        // If parameter type is unspecified (auto) and named something like "args"
        if (!param.type || generate_type(param.type.get()) == "auto") {
            is_main_with_args = true;
            args_param_name = param.name;
        }
    }

    // Detect if this is a member function (has 'this' parameter)
    // In Cpp2: print: (this) = { ... } or f: (this, x: int) = { ... }
    // In C++: auto print() -> void { ... } or auto f(int x) -> void { ... }
    bool is_member_function = false;
    bool is_const_member = false;
    bool is_rvalue_member = false;  // move this
    size_t this_param_index = SIZE_MAX;
    
    for (size_t i = 0; i < decl->parameters.size(); ++i) {
        if (decl->parameters[i].name == "this") {
            is_member_function = true;
            this_param_index = i;
            for (const auto& qual : decl->parameters[i].qualifiers) {
                if (qual == ParameterQualifier::In) is_const_member = true;
                if (qual == ParameterQualifier::Move) is_rvalue_member = true;
            }
            break;
        }
    }

    // Use auto return_type syntax for cppfront compatibility
    write("auto " + decl->name + "(");

    // Special case: main(args) -> main(int argc_, char** argv_)
    if (is_main_with_args) {
        write("int argc_, char** argv_");
    } else {
        // Parameters - skip 'this' parameter as it's implicit in C++
        bool first_param = true;
        for (size_t i = 0; i < decl->parameters.size(); ++i) {
            if (i == this_param_index) continue;  // Skip 'this' parameter
            if (!first_param) write(", ");
            first_param = false;
            const auto& param = decl->parameters[i];
            write(generate_parameter_type(param.type.get(), param.qualifiers) + " " + param.name);
        }
    }

    write(")");
    
    // Add const/&& qualifiers for member functions
    if (is_const_member) {
        write(" const");
    }
    if (is_rvalue_member) {
        write(" &&");
    }
    
    write(" -> " + return_type);

    if (decl->body) {
        // Check if body is already a BlockStatement - if so, generate it directly
        // to avoid adding extra nested braces
        if (decl->body->kind == Statement::Kind::Block) {
            write(" {\n");
            indent();
            // For main(args), inject the args initialization
            if (is_main_with_args) {
                write_line("auto const " + args_param_name + " = cpp2::make_args(argc_, argv_);");
            }
            // Generate block contents without the outer braces
            auto* block = static_cast<BlockStatement*>(decl->body.get());
            for (auto& stmt : block->statements) {
                generate_statement(stmt.get());
            }
            dedent();
            write_line("}");
        } else {
            write(" {\n");
            indent();
            // For main(args), inject the args initialization
            if (is_main_with_args) {
                write_line("auto const " + args_param_name + " = cpp2::make_args(argc_, argv_);");
            }
            generate_statement(decl->body.get());
            dedent();
            write_line("}");
        }
    } else {
        write_line(";");
    }
}

void CodeGenerator::generate_special_member_function(FunctionDeclaration* decl) {
    // Handle Cpp2 operator=: patterns which map to C++ special member functions:
    // - operator=: (out this) -> default constructor
    // - operator=: (out this, x: T) -> converting constructor
    // - operator=: (inout this, that) -> copy/move assignment
    // - operator=: (move this) -> destructor
    
    if (!decl || current_class_name.empty()) return;
    
    // Analyze parameters to determine what kind of special member function this is
    bool has_out_this = false;
    bool has_inout_this = false;
    bool has_move_this = false;
    bool has_implicit_this = false;
    std::vector<FunctionDeclaration::Parameter*> non_this_params;
    
    for (const auto& param : decl->parameters) {
        if (param.name == "this") {
            for (const auto& qual : param.qualifiers) {
                if (qual == ParameterQualifier::Out) has_out_this = true;
                if (qual == ParameterQualifier::InOut) has_inout_this = true;
                if (qual == ParameterQualifier::Move) has_move_this = true;
                if (qual == ParameterQualifier::Implicit) has_implicit_this = true;
            }
        } else {
            non_this_params.push_back(const_cast<FunctionDeclaration::Parameter*>(&param));
        }
    }
    
    if (has_move_this && non_this_params.empty()) {
        // Destructor: operator=: (move this)
        write_line("~" + current_class_name + "() {");
        indent();
        if (decl->body) generate_statement(decl->body.get());
        dedent();
        write_line("}");
    } else if (has_out_this) {
        // Constructor: operator=: (out this, ...)
        // Could be default, copy, move, or converting constructor
        
        if (non_this_params.empty()) {
            // Default constructor
            write_line(current_class_name + "() {");
        } else if (non_this_params.size() == 1 && non_this_params[0]->name == "that") {
            // Copy or move constructor based on 'that' qualifier
            bool is_move = false;
            for (const auto& qual : non_this_params[0]->qualifiers) {
                if (qual == ParameterQualifier::Move) is_move = true;
            }
            if (is_move) {
                write_line(current_class_name + "(" + current_class_name + "&& that) {");
            } else {
                write_line(current_class_name + "(const " + current_class_name + "& that) {");
            }
        } else {
            // Converting constructor
            std::string explicit_kw = has_implicit_this ? "" : "explicit ";
            write(explicit_kw + current_class_name + "(");
            for (size_t i = 0; i < non_this_params.size(); ++i) {
                if (i > 0) write(", ");
                auto* param = non_this_params[i];
                std::string ptype = param->type ? generate_type(param->type.get()) : "auto";
                write("const " + ptype + "& " + param->name);
            }
            write_line(") {");
        }
        indent();
        if (decl->body) generate_statement(decl->body.get());
        dedent();
        write_line("}");
    } else if (has_inout_this) {
        // Assignment operator: operator=: (inout this, that)
        if (non_this_params.size() == 1 && non_this_params[0]->name == "that") {
            bool is_move = false;
            for (const auto& qual : non_this_params[0]->qualifiers) {
                if (qual == ParameterQualifier::Move) is_move = true;
            }
            if (is_move) {
                write_line(current_class_name + "& operator=(" + current_class_name + "&& that) {");
            } else {
                write_line(current_class_name + "& operator=(const " + current_class_name + "& that) {");
            }
        } else {
            // General assignment from other types
            write(current_class_name + "& operator=(");
            for (size_t i = 0; i < non_this_params.size(); ++i) {
                if (i > 0) write(", ");
                auto* param = non_this_params[i];
                std::string ptype = param->type ? generate_type(param->type.get()) : "auto";
                write("const " + ptype + "& " + param->name);
            }
            write_line(") {");
        }
        indent();
        if (decl->body) generate_statement(decl->body.get());
        write_line("return *this;");
        dedent();
        write_line("}");
    } else {
        // Fallback - just generate as regular function
        write_line("// WARNING: Unknown operator=: pattern");
        write("auto operator_assign(");
        for (size_t i = 0; i < decl->parameters.size(); ++i) {
            if (i > 0) write(", ");
            write(generate_parameter_type(decl->parameters[i].type.get(), decl->parameters[i].qualifiers) + " " + decl->parameters[i].name);
        }
        write_line(") {");
        indent();
        if (decl->body) generate_statement(decl->body.get());
        dedent();
        write_line("}");
    }
}

void CodeGenerator::generate_operator_declaration(OperatorDeclaration* decl) {
    if (!decl) return;

    // Cpp2 operator=: special member functions
    // operator=: (out this) -> default constructor
    // operator=: (out this, x: T) -> converting constructor
    // operator=: (inout this, that) -> copy/move assignment
    // operator=: (move this) -> destructor
    // The parser stores the name as "=:" not "operator=:"
    
    if (decl->name == "=:") {
        // Handle as special member function
        generate_operator_eq_colon(decl);
        return;
    }

    std::string return_type = decl->return_type ? generate_type(decl->return_type.get()) : "void";

    // Get the class name from the context (outer type)
    std::string class_name = current_class_name.empty() ? "ClassName" : current_class_name;

    // operator name
    std::string op_name = "operator" + decl->name;

    // Return type: usually ClassName& for assignment operators
    write(class_name + "& " + op_name + "(");

    // Parameters
    for (size_t i = 0; i < decl->parameters.size(); ++i) {
        if (i > 0) write(", ");
        const auto& param = decl->parameters[i];

        // Skip 'this' parameter - it's implicit in C++
        if (param->name == "this") {
            continue;
        }

        std::string param_type = param->type ? generate_type(param->type.get()) : "auto";

        // Handle qualifiers
        bool is_move = false;
        bool is_const = false;
        for (const auto& qual : param->qualifiers) {
            if (qual == ParameterQualifier::Move) {
                is_move = true;
            } else if (qual == ParameterQualifier::Out) {
                // out parameter is passed by non-const reference
            }
        }

        if (is_move) {
            write(param_type + "&&");
        } else {
            write("const " + param_type + "&");
        }

        write(" " + param->name);
    }

    write(")");

    // Function body
    if (decl->body) {
        if (auto* block = dynamic_cast<BlockStatement*>(decl->body.get())) {
            write_line(" {");
            indent();
            generate_statement(block);
            write_line("return *this;");
            dedent();
            write_line("}");
        } else {
            // Expression body
            write_line(" {");
            indent();
            generate_statement(decl->body.get());
            write_line("return *this;");
            dedent();
            write_line("}");
        }
    } else {
        write_line(" {");
        indent();
        write_line("return *this;");
        dedent();
        write_line("}");
    }
}

void CodeGenerator::generate_operator_eq_colon(OperatorDeclaration* decl) {
    // Handle Cpp2 operator=: patterns which map to C++ special member functions
    if (!decl || current_class_name.empty()) return;
    
    // Analyze parameters to determine what kind of special member function this is
    bool has_out_this = false;
    bool has_inout_this = false;
    bool has_move_this = false;
    bool has_implicit_this = false;
    std::vector<FunctionDeclaration::Parameter*> non_this_params;
    
    for (const auto& param : decl->parameters) {
        if (param->name == "this") {
            for (const auto& qual : param->qualifiers) {
                if (qual == ParameterQualifier::Out) has_out_this = true;
                if (qual == ParameterQualifier::InOut) has_inout_this = true;
                if (qual == ParameterQualifier::Move) has_move_this = true;
                if (qual == ParameterQualifier::Implicit) has_implicit_this = true;
            }
        } else {
            non_this_params.push_back(param.get());
        }
    }
    
    if (has_move_this && non_this_params.empty()) {
        // Destructor: operator=: (move this)
        write_line("~" + current_class_name + "() {");
        indent();
        if (decl->body) generate_statement(decl->body.get());
        dedent();
        write_line("}");
    } else if (has_out_this) {
        // Constructor: operator=: (out this, ...)
        if (non_this_params.empty()) {
            // Default constructor
            write_line(current_class_name + "() {");
        } else if (non_this_params.size() == 1 && non_this_params[0]->name == "that") {
            // Copy or move constructor based on 'that' qualifier
            bool is_move = false;
            for (const auto& qual : non_this_params[0]->qualifiers) {
                if (qual == ParameterQualifier::Move) is_move = true;
            }
            if (is_move) {
                write_line(current_class_name + "(" + current_class_name + "&& that) {");
            } else {
                write_line(current_class_name + "(const " + current_class_name + "& that) {");
            }
        } else {
            // Converting constructor
            std::string explicit_kw = has_implicit_this ? "" : "explicit ";
            write(explicit_kw + current_class_name + "(");
            for (size_t i = 0; i < non_this_params.size(); ++i) {
                if (i > 0) write(", ");
                auto* param = non_this_params[i];
                std::string ptype = param->type ? generate_type(param->type.get()) : "auto";
                write("const " + ptype + "& " + param->name);
            }
            write_line(") {");
        }
        indent();
        if (decl->body) generate_statement(decl->body.get());
        dedent();
        write_line("}");
    } else if (has_inout_this) {
        // Assignment operator: operator=: (inout this, that)
        if (non_this_params.size() == 1 && non_this_params[0]->name == "that") {
            bool is_move = false;
            for (const auto& qual : non_this_params[0]->qualifiers) {
                if (qual == ParameterQualifier::Move) is_move = true;
            }
            if (is_move) {
                write_line(current_class_name + "& operator=(" + current_class_name + "&& that) {");
            } else {
                write_line(current_class_name + "& operator=(const " + current_class_name + "& that) {");
            }
        } else {
            // General assignment from other types
            write(current_class_name + "& operator=(");
            for (size_t i = 0; i < non_this_params.size(); ++i) {
                if (i > 0) write(", ");
                auto* param = non_this_params[i];
                std::string ptype = param->type ? generate_type(param->type.get()) : "auto";
                write("const " + ptype + "& " + param->name);
            }
            write_line(") {");
        }
        indent();
        if (decl->body) generate_statement(decl->body.get());
        write_line("return *this;");
        dedent();
        write_line("}");
    } else {
        // Fallback - unknown pattern
        write_line("// WARNING: Unknown operator=: pattern");
        write("void unknown_operator(");
        for (size_t i = 0; i < decl->parameters.size(); ++i) {
            if (i > 0) write(", ");
            auto& param = decl->parameters[i];
            write(generate_parameter_type(param->type.get(), param->qualifiers) + " " + param->name);
        }
        write_line(") {");
        indent();
        if (decl->body) generate_statement(decl->body.get());
        dedent();
        write_line("}");
    }
}

void CodeGenerator::generate_type_declaration(TypeDeclaration* decl) {
    if (!decl) return;

    // Generate template header if this is a template type
    // Skip for Alias types as they handle their own template header
    if (!decl->template_parameters.empty() && decl->type_kind != TypeDeclaration::TypeKind::Alias) {
        write("template<");
        for (size_t i = 0; i < decl->template_parameters.size(); ++i) {
            if (i > 0) write(", ");
            write(generate_template_param(decl->template_parameters[i]));
        }
        write_line(">");
    }

    // Generate requires clause if present
    if (decl->requires_clause) {
        write("requires ");
        write_line(generate_expression_to_string(decl->requires_clause.get()));
    }

    // Check for @interface and @union metafunctions which need special handling
    bool is_interface = false;
    bool is_union = false;
    for (const auto& metafunc : decl->metafunctions) {
        if (metafunc == "interface") {
            is_interface = true;
        } else if (metafunc == "union") {
            is_union = true;
        }
    }

    // Track metafunctions for this type (for context-sensitive generation like @regex)
    current_type_metafunctions = decl->metafunctions;

    switch (decl->type_kind) {
        case TypeDeclaration::TypeKind::Struct: {
            // Use union if @union metafunction is present
            if (is_union) {
                write_line("union " + decl->name + " {");
            } else {
                write_line("struct " + decl->name + " {");
            }
            indent();

            // For @interface, make member functions pure virtual
            if (is_interface) {
                write_line("// @interface metafunction: pure interface");
                write_line("virtual ~" + decl->name + "() = default;");
                write_line("");
                write_line("// Delete copy and move");
                write_line(decl->name + "(const " + decl->name + "&) = delete;");
                write_line(decl->name + "(" + decl->name + "&&) = delete;");
                write_line(decl->name + "& operator=(const " + decl->name + "&) = delete;");
                write_line(decl->name + "& operator=(" + decl->name + "&&) = delete;");
                write_line("");
            }

            // Track class name for operator declarations
            std::string prev_class_name = current_class_name;
            current_class_name = decl->name;

            for (auto& member : decl->members) {
                generate_declaration(member.get());
            }

            current_class_name = prev_class_name;

            // Generate metafunction code for @value, @ordered, etc.
            for (const auto& metafunc : decl->metafunctions) {
                if (metafunc == "interface") {
                    // Already handled above
                    continue;
                }
                if (metafunc == "value") {
                    // @value: Generate value semantics (defaulted special members)
                    write_line("");
                    write_line("// @value metafunction: value semantics");
                    write_line(decl->name + "(const " + decl->name + "&) = default;");
                    write_line(decl->name + "(" + decl->name + "&&) = default;");
                    write_line(decl->name + "& operator=(const " + decl->name + "&) = default;");
                    write_line(decl->name + "& operator=(" + decl->name + "&&) = default;");
                    write_line("");
                    write_line("bool operator==(const " + decl->name + "& other) const = default;");
                    write_line("bool operator!=(const " + decl->name + "& other) const = default;");
                } else if (metafunc == "ordered") {
                    // @ordered: Generate ordering operators
                    write_line("");
                    write_line("// @ordered metafunction: ordering operators");
                    write_line("auto operator<=>(const " + decl->name + "& other) const = default;");
                } else if (metafunc == "weakly_ordered" || metafunc == "weakly_ordered_value") {
                    // @weakly_ordered: Generate weak ordering operators
                    write_line("");
                    write_line("// @weakly_ordered metafunction: weak ordering operators");
                    write_line("std::weak_ordering operator<=>(const " + decl->name + "& other) const = default;");
                    write_line("bool operator==(const " + decl->name + "& other) const = default;");
                } else if (metafunc == "partially_ordered" || metafunc == "partially_ordered_value") {
                    // @partially_ordered: Generate partial ordering operators
                    write_line("");
                    write_line("// @partially_ordered metafunction: partial ordering operators");
                    write_line("std::partial_ordering operator<=>(const " + decl->name + "& other) const = default;");
                    write_line("bool operator==(const " + decl->name + "& other) const = default;");
                } else if (metafunc == "polymorphic_base") {
                    // @polymorphic_base: Generate virtual destructor
                    write_line("");
                    write_line("// @polymorphic_base metafunction: virtual destructor");
                    write_line("virtual ~" + decl->name + "() = default;");
                } else if (metafunc == "copyable") {
                    // @copyable: Explicitly defaulted copy operations
                    write_line("");
                    write_line("// @copyable metafunction: copy semantics");
                    write_line(decl->name + "(const " + decl->name + "&) = default;");
                    write_line(decl->name + "& operator=(const " + decl->name + "&) = default;");
                } else if (metafunc == "movable") {
                    // @movable: Explicitly defaulted move operations
                    write_line("");
                    write_line("// @movable metafunction: move semantics");
                    write_line(decl->name + "(" + decl->name + "&&) = default;");
                    write_line(decl->name + "& operator=(" + decl->name + "&&) = default;");
                } else if (metafunc == "struct") {
                    // @struct: Just a marker, struct is the default
                    // No additional code needed
                } else if (metafunc == "print") {
                    // @print: Generate reflection/print functions
                    write_line("");
                    write_line("// @print metafunction: to_string() for debugging");
                    write_line("std::string to_string() const {");
                    indent();
                    write_line("std::string result = \"" + decl->name + "{ \";");
                    // Add members to the string
                    bool first = true;
                    for (const auto& member : decl->members) {
                        if (auto* var_member = dynamic_cast<VariableDeclaration*>(member.get())) {
                            if (!first) {
                                write_line("result += \", \";");
                            }
                            write_line("result += \"" + var_member->name + " = \" + std::to_string(" + var_member->name + ");");
                            first = false;
                        }
                    }
                    write_line("result += \" }\";");
                    write_line("return result;");
                    dedent();
                    write_line("}");
                } else if (metafunc == "regex") {
                    // @regex: Transform regex_ members into std::regex objects
                    write_line("");
                    write_line("// @regex metafunction: compile-time regex validation");
                    write_line("// Note: regex members are compiled at construction");
                } else if (metafunc.find("autodiff") == 0) {
                    // @autodiff: Generate automatic differentiation support
                    write_line("");
                    write_line("// @autodiff metafunction: automatic differentiation");
                    write_line("// Derivative methods with _d suffix for forward mode");

                    // Generate derivative versions of member functions
                    for (const auto& member : decl->members) {
                        if (auto* func = dynamic_cast<FunctionDeclaration*>(member.get())) {
                            // Generate derivative version of the function
                            write_line("");
                            write_line("// Derivative of " + func->name);

                            // Build derivative function signature
                            std::string deriv_name = func->name + "_d";
                            std::string params_str;

                            for (size_t i = 0; i < func->parameters.size(); ++i) {
                                if (i > 0) params_str += ", ";
                                std::string param_type = func->parameters[i].type ? generate_type(func->parameters[i].type.get()) : "auto";
                                params_str += param_type + " " + func->parameters[i].name;
                                params_str += ", " + param_type + " " + func->parameters[i].name + "_d";
                            }

                            std::string return_type = func->return_type ? generate_type(func->return_type.get()) : "void";
                            if (return_type != "void") {
                                write_line(return_type + " " + deriv_name + "(" + params_str + ") const {");
                                indent();
                                write_line("// Derivative computation placeholder");
                                write_line("return " + return_type + "{};");
                                dedent();
                                write_line("}");
                            }
                        }
                    }
                } else if (metafunc == "sample_traverser") {
                    // @sample_traverser: Generate visitor pattern for traversing members
                    write_line("");
                    write_line("// @sample_traverser metafunction: visitor pattern");
                    write_line("template<typename Visitor>");
                    write_line("void traverse(Visitor&& visitor) {");
                    indent();

                    // Visit each member
                    for (const auto& member : decl->members) {
                        if (auto* var_member = dynamic_cast<VariableDeclaration*>(member.get())) {
                            write_line("visitor(\"" + var_member->name + "\", " + var_member->name + ");");
                        } else if (auto* func = dynamic_cast<FunctionDeclaration*>(member.get())) {
                            write_line("// Function: " + func->name);
                        }
                    }

                    dedent();
                    write_line("}");

                    write_line("");
                    write_line("template<typename Visitor>");
                    write_line("void traverse(Visitor&& visitor) const {");
                    indent();

                    // Visit each member (const version)
                    for (const auto& member : decl->members) {
                        if (auto* var_member = dynamic_cast<VariableDeclaration*>(member.get())) {
                            write_line("visitor(\"" + var_member->name + "\", " + var_member->name + ");");
                        }
                    }

                    dedent();
                    write_line("}");
                } else if (metafunc == "enum") {
                    // @enum is handled separately in TypeKind::Enum
                    // But if used with struct, just mark it
                } else if (metafunc == "union") {
                    // @union: Mark as union instead of struct
                    // This should be handled in the type kind, but we can note it
                }
            }

            dedent();
            write_line("};");

            // Generate std::hash specialization for @hashable
            for (const auto& metafunc : decl->metafunctions) {
                if (metafunc == "hashable") {
                    write_line("");
                    write_line("// @hashable metafunction: std::hash specialization");
                    write_line("namespace std {");
                    indent();
                    write_line("template<>");
                    write_line("struct hash<" + decl->name + "> {");
                    indent();
                    write_line("size_t operator()(const " + decl->name + "& value) const {");
                    indent();
                    write_line("// TODO: Implement actual hash combination");
                    write_line("return 0;");
                    dedent();
                    write_line("}");
                    dedent();
                    write_line("};");
                    dedent();
                    write_line("}");
                }
            }
        }

            break;

        case TypeDeclaration::TypeKind::Class:
            write_line("class " + decl->name + " {");
            indent();
            write_line("public:");
            for (auto& member : decl->members) {
                generate_declaration(member.get());
            }

            // Generate metafunction code for classes too
            for (const auto& metafunc : decl->metafunctions) {
                if (metafunc == "interface") {
                    continue; // Already handled
                } else if (metafunc == "value") {
                    write_line("");
                    write_line("// @value metafunction: value semantics");
                    write_line(decl->name + "(const " + decl->name + "&) = default;");
                    write_line(decl->name + "(" + decl->name + "&&) = default;");
                    write_line(decl->name + "& operator=(const " + decl->name + "&) = default;");
                    write_line(decl->name + "& operator=(" + decl->name + "&&) = default;");
                    write_line("");
                    write_line("bool operator==(const " + decl->name + "& other) const = default;");
                    write_line("bool operator!=(const " + decl->name + "& other) const = default;");
                } else if (metafunc == "ordered") {
                    write_line("");
                    write_line("// @ordered metafunction: ordering operators");
                    write_line("auto operator<=>(const " + decl->name + "& other) const = default;");
                } else if (metafunc == "weakly_ordered" || metafunc == "weakly_ordered_value") {
                    write_line("");
                    write_line("// @weakly_ordered metafunction: weak ordering operators");
                    write_line("std::weak_ordering operator<=>(const " + decl->name + "& other) const = default;");
                    write_line("bool operator==(const " + decl->name + "& other) const = default;");
                } else if (metafunc == "partially_ordered" || metafunc == "partially_ordered_value") {
                    write_line("");
                    write_line("// @partially_ordered metafunction: partial ordering operators");
                    write_line("std::partial_ordering operator<=>(const " + decl->name + "& other) const = default;");
                    write_line("bool operator==(const " + decl->name + "& other) const = default;");
                } else if (metafunc == "polymorphic_base") {
                    write_line("");
                    write_line("// @polymorphic_base metafunction: virtual destructor");
                    write_line("virtual ~" + decl->name + "() = default;");
                } else if (metafunc == "copyable") {
                    write_line("");
                    write_line("// @copyable metafunction: copy semantics");
                    write_line(decl->name + "(const " + decl->name + "&) = default;");
                    write_line(decl->name + "& operator=(const " + decl->name + "&) = default;");
                } else if (metafunc == "movable") {
                    write_line("");
                    write_line("// @movable metafunction: move semantics");
                    write_line(decl->name + "(" + decl->name + "&&) = default;");
                    write_line(decl->name + "& operator=(" + decl->name + "&&) = default;");
                }
            }

            dedent();
            write_line("};");

            // Generate std::hash specialization for @hashable
            for (const auto& metafunc : decl->metafunctions) {
                if (metafunc == "hashable") {
                    write_line("");
                    write_line("// @hashable metafunction: std::hash specialization");
                    write_line("namespace std {");
                    indent();
                    write_line("template<>");
                    write_line("struct hash<" + decl->name + "> {");
                    indent();
                    write_line("size_t operator()(const " + decl->name + "& value) const {");
                    indent();
                    write_line("// TODO: Implement actual hash combination");
                    write_line("return 0;");
                    dedent();
                    write_line("}");
                    dedent();
                    write_line("};");
                    dedent();
                    write_line("}");
                }
            }

            break;

        case TypeDeclaration::TypeKind::Enum: {
            // Check if it's a flag_enum
            bool is_flag_enum = false;
            std::string underlying_type = "int";
            for (const auto& metafunc : decl->metafunctions) {
                if (metafunc.find("flag_enum") == 0) {
                    is_flag_enum = true;
                    // Extract underlying type if specified: @flag_enum<u8>
                    size_t lt = metafunc.find('<');
                    if (lt != std::string::npos) {
                        size_t gt = metafunc.find('>');
                        if (gt != std::string::npos) {
                            underlying_type = metafunc.substr(lt + 1, gt - lt - 1);
                        }
                    }
                    break;
                }
            }

            write_line("enum class " + decl->name + " : " + underlying_type + " {");
            indent();

            // Generate enum values from members
            for (size_t i = 0; i < decl->members.size(); ++i) {
                if (auto* var_member = dynamic_cast<VariableDeclaration*>(decl->members[i].get())) {
                    write(var_member->name);
                    if (var_member->initializer) {
                        write(" = " + generate_expression_to_string(var_member->initializer.get()));
                    }
                    if (i < decl->members.size() - 1) {
                        write(",");
                    }
                    write_line("");
                }
            }

            dedent();
            write_line("};");

            // For flag_enum, generate bitwise operators
            if (is_flag_enum) {
                write_line("");
                write_line("// @flag_enum: bitwise operators");
                write_line("constexpr " + decl->name + " operator|(" + decl->name + " a, " + decl->name + " b) {");
                indent();
                write_line("return static_cast<" + decl->name + ">(static_cast<" + underlying_type + ">(a) | static_cast<" + underlying_type + ">(b));");
                dedent();
                write_line("}");

                write_line("constexpr " + decl->name + " operator&(" + decl->name + " a, " + decl->name + " b) {");
                indent();
                write_line("return static_cast<" + decl->name + ">(static_cast<" + underlying_type + ">(a) & static_cast<" + underlying_type + ">(b));");
                dedent();
                write_line("}");

                write_line("constexpr " + decl->name + " operator^(" + decl->name + " a, " + decl->name + " b) {");
                indent();
                write_line("return static_cast<" + decl->name + ">(static_cast<" + underlying_type + ">(a) ^ static_cast<" + underlying_type + ">(b));");
                dedent();
                write_line("}");

                write_line("constexpr " + decl->name + " operator~(" + decl->name + " a) {");
                indent();
                write_line("return static_cast<" + decl->name + ">(~static_cast<" + underlying_type + ">(a));");
                dedent();
                write_line("}");
            }

            break;
        }

        case TypeDeclaration::TypeKind::Union:
            write_line("union " + decl->name + " {");
            indent();
            for (auto& member : decl->members) {
                generate_declaration(member.get());
            }
            dedent();
            write_line("};");
            break;

        case TypeDeclaration::TypeKind::Alias: {
            // Check if this is a concept definition
            bool is_concept = std::find(decl->metafunctions.begin(), decl->metafunctions.end(), "concept") != decl->metafunctions.end();
            
            if (is_concept && decl->underlying_type) {
                // Generate concept definition: template<typename T> concept name = constraint;
                std::string template_str = "template<";
                for (size_t i = 0; i < decl->template_parameters.size(); ++i) {
                    if (i > 0) template_str += ", ";
                    std::string param = decl->template_parameters[i];
                    // Remove trailing ... for variadic (handle separately)
                    if (param.size() > 3 && param.substr(param.size() - 3) == "...") {
                        param = param.substr(0, param.size() - 3);
                        template_str += "typename... " + param;
                    } else {
                        template_str += "typename " + param;
                    }
                }
                template_str += ">";
                write_line(template_str);
                write_line("concept " + decl->name + " = " + generate_type(decl->underlying_type.get()) + ";");
            } else if (decl->underlying_type) {
                // Regular type alias
                // Add template parameters if present
                if (!decl->template_parameters.empty()) {
                    std::string template_str = "template<";
                    for (size_t i = 0; i < decl->template_parameters.size(); ++i) {
                        if (i > 0) template_str += ", ";
                        std::string param = decl->template_parameters[i];
                        if (param.size() > 3 && param.substr(param.size() - 3) == "...") {
                            param = param.substr(0, param.size() - 3);
                            template_str += "typename... " + param;
                        } else {
                            template_str += "typename " + param;
                        }
                    }
                    template_str += ">";
                    write_line(template_str);
                }
                write_line("using " + decl->name + " = " + generate_type(decl->underlying_type.get()) + ";");
            }
            break;
        }

        default:
            break;
    }

    // Clear metafunction context
    current_type_metafunctions.clear();
}

void CodeGenerator::generate_namespace_declaration(NamespaceDeclaration* decl) {
    if (!decl) return;

    // Check if this is a namespace alias
    if (!decl->alias_target.empty()) {
        write_line("namespace " + decl->name + " = " + decl->alias_target + ";");
        return;
    }

    write_line("namespace " + decl->name + " {");
    indent();
    for (auto& member : decl->members) {
        generate_declaration(member.get());
    }
    dedent();
    write_line("}");
}

void CodeGenerator::generate_using_declaration(UsingDeclaration* decl) {
    if (!decl) return;
    // Alias form: 'using name = target;'
    if (!decl->name.empty()) {
        write_line("using " + decl->name + " = " + decl->target + ";");
        return;
    }

    // Non-alias using: either 'using qualified::name;' or wildcard shorthand 'qualified::_'
    if (decl->target.size() >= 3 && decl->target.compare(decl->target.size() - 3, 3, "::_") == 0) {
        // Wildcard shorthand; emit 'using namespace QUAL;' where QUAL is target without '::_'
        std::string qual = decl->target.substr(0, decl->target.size() - 3);
        write_line("using namespace " + qual + ";");
    } else {
        // Emit a direct using-declaration (bring name into scope)
        write_line("using " + decl->target + ";");
    }
}

void CodeGenerator::generate_import_declaration(ImportDeclaration* decl) {
    if (!decl) return;
    write_line("// import " + decl->module_name);
}

void CodeGenerator::generate_cpp1_passthrough_declaration(Cpp1PassthroughDeclaration* decl) {
    if (!decl) return;
    // Emit the raw C++1 code as-is
    write_line(decl->raw_code);
}

void CodeGenerator::generate_statement(Statement* stmt) {
    if (!stmt) return;

    switch (stmt->kind) {
        case Statement::Kind::Expression: {
            auto expr_stmt = static_cast<ExpressionStatement*>(stmt);
            write_line(generate_expression_to_string(expr_stmt->expr.get()) + ";");
            break;
        }
        case Statement::Kind::Declaration: {
            auto decl_stmt = static_cast<DeclarationStatement*>(stmt);
            generate_declaration(decl_stmt->declaration.get());
            break;
        }
        case Statement::Kind::Block:
            generate_block_statement(static_cast<BlockStatement*>(stmt));
            break;
        case Statement::Kind::If:
            generate_if_statement(static_cast<IfStatement*>(stmt));
            break;
        case Statement::Kind::While:
            generate_while_statement(static_cast<WhileStatement*>(stmt));
            break;
        case Statement::Kind::DoWhile:
            generate_do_while_statement(static_cast<DoWhileStatement*>(stmt));
            break;
        case Statement::Kind::For:
            generate_for_statement(static_cast<ForStatement*>(stmt));
            break;
        case Statement::Kind::ForRange:
            generate_for_range_statement(static_cast<ForRangeStatement*>(stmt));
            break;
        case Statement::Kind::Return:
            generate_return_statement(static_cast<ReturnStatement*>(stmt));
            break;
        case Statement::Kind::Break:
            generate_break_statement(static_cast<BreakStatement*>(stmt));
            break;
        case Statement::Kind::Continue:
            generate_continue_statement(static_cast<ContinueStatement*>(stmt));
            break;
        case Statement::Kind::Contract: {
            auto contract_stmt = static_cast<ContractStatement*>(stmt);
            if (contract_stmt->contract && contract_stmt->contract->condition) {
                auto cond = generate_expression_to_string(contract_stmt->contract->condition.get());
                if (cond.size() >= 2 && cond.front() == '(' && cond.back() == ')') {
                    cond = cond.substr(1, cond.size() - 2);
                }
                write_line("assert(" + cond + ");");
            }
            break;
        }
        // ============================================================================
        // Concurrency Statements (Kotlin-style structured concurrency)
        // ============================================================================
        case Statement::Kind::CoroutineScope:
            generate_coroutine_scope_statement(static_cast<CoroutineScopeStatement*>(stmt));
            break;
        case Statement::Kind::ParallelFor:
            generate_parallel_for_statement(static_cast<ParallelForStatement*>(stmt));
            break;
        case Statement::Kind::ChannelDecl:
            generate_channel_declaration(static_cast<ChannelDeclarationStatement*>(stmt));
            break;
        case Statement::Kind::ScopeBlock:
            generate_scope_block_statement(static_cast<ScopeBlockStatement*>(stmt));
            break;
        default:
            break;
    }
}

void CodeGenerator::generate_block_statement(BlockStatement* stmt) {
    if (!stmt) return;

    write_line("{");
    indent();
    for (auto& s : stmt->statements) {
        generate_statement(s.get());
    }
    dedent();
    write_line("}");
}

void CodeGenerator::generate_scope_block_statement(ScopeBlockStatement* stmt) {
    if (!stmt) return;

    // Generate: { initializers; body }
    write_line("{");
    indent();

    // Generate all initializers as local variables
    for (const auto& init : stmt->initializers) {
        std::string init_str = generate_expression_to_string(init->initializer.get());
        std::string type_str = "auto";
        if (init->type && init->type->kind != Type::Kind::Auto) {
            type_str = generate_type(init->type.get());
        }
        write_line(type_str + " " + init->name + " = " + init_str + ";");
    }

    // Generate the body
    if (stmt->body) {
        if (stmt->body->kind == Statement::Kind::Block) {
            // If the body is a block, generate its contents directly (avoid double braces)
            auto block = static_cast<BlockStatement*>(stmt->body.get());
            for (auto& s : block->statements) {
                generate_statement(s.get());
            }
        } else {
            generate_statement(stmt->body.get());
        }
    }

    dedent();
    write_line("}");
}

void CodeGenerator::generate_if_statement(IfStatement* stmt) {
    if (!stmt) return;

    write("if (" + generate_expression_to_string(stmt->condition.get()) + ") ");
    generate_statement(stmt->then_stmt.get());

    if (stmt->else_stmt) {
        write(" else ");
        generate_statement(stmt->else_stmt.get());
    }
}

void CodeGenerator::generate_while_statement(WhileStatement* stmt) {
    if (!stmt) return;

    bool is_labeled = !stmt->label.empty();
    bool has_increment = stmt->increment != nullptr;
    bool has_loop_inits = !stmt->loop_inits.empty();
    std::string cond_str = generate_expression_to_string(stmt->condition.get());

    // If there's a loop initializer, wrap everything in a scope
    if (has_loop_inits) {
        write_line("{");
        indent();
        // Generate all loop initializer variables
        for (const auto& loop_init : stmt->loop_inits) {
            std::string type_str = "auto";
            if (loop_init->type && loop_init->type->kind != Type::Kind::Auto) {
                type_str = generate_type(loop_init->type.get());
            }
            std::string init_str = generate_expression_to_string(loop_init->initializer.get());
            write_line(type_str + " " + loop_init->name + " = " + init_str + ";");
        }
    }

    if (is_labeled) {
        // Labeled while loop - need to use goto pattern for continue to work
        // Cpp2: label: while (cond) next inc { body }
        // Generate as:
        // label: while (true) {
        //     if (!(cond)) goto label_end;
        //     body;
        //     label_continue:;  // for continue statements
        //     inc;  // if has_increment
        // }
        // label_end:;

        write_line(stmt->label + ":;");
        write("while (true) ");
        write_line("{");
        indent();

        // Condition check with goto to end if false
        write("if (!(" + cond_str + ")) goto " + stmt->label + "_end;");
        write_line("");

        write_line("");

        // Body
        generate_statement(stmt->body.get());

        // Continue label (after body, before increment)
        write_line(stmt->label + "_continue:;");

        // Increment if present
        if (has_increment) {
            write_line(generate_expression_to_string(stmt->increment.get()) + ";");
        }

        dedent();
        write_line("}");
        write_line(stmt->label + "_end:;");
    } else {
        // Non-labeled while - generate normally
        write("while (" + cond_str + ") ");

        if (!has_increment) {
            generate_statement(stmt->body.get());
        } else {
            write_line("{");
            indent();
            generate_statement(stmt->body.get());
            write_line("");
            write(generate_expression_to_string(stmt->increment.get()) + ";");
            dedent();
            write_line("}");
        }
    }

    // Close loop initializer scope if present
    if (has_loop_inits) {
        dedent();
        write_line("}");
    }
}

void CodeGenerator::generate_do_while_statement(DoWhileStatement* stmt) {
    if (!stmt) return;

    // Cpp2 do-while: do { body } next increment while condition
    // Generate as: do { body; increment; } while (condition);

    // Handle labeled do-while - emit continue label before loop when no increment
    if (!stmt->label.empty() && !stmt->increment) {
        write_line(stmt->label + "_continue:;");
    }

    write("do ");

    if (stmt->increment) {
        write_line("{");
        indent();
        generate_statement(stmt->body.get());

        // Continue label - for continue to jump to (before increment)
        if (!stmt->label.empty()) {
            write_line(stmt->label + "_continue:;");
        }

        write(generate_expression_to_string(stmt->increment.get()) + ";");
        dedent();
        write("} ");
    } else {
        generate_statement(stmt->body.get());
    }

    write_line("while (" + generate_expression_to_string(stmt->condition.get()) + ");");

    if (!stmt->label.empty()) {
        write_line(stmt->label + "_end:;");
    }
}

void CodeGenerator::generate_for_statement(ForStatement* stmt) {
    if (!stmt) return;

    bool is_labeled = !stmt->label.empty();
    std::string init_str, cond_str, inc_str;

    // Generate init
    if (stmt->init) {
        if (auto var_decl = dynamic_cast<VariableDeclaration*>(stmt->init.get())) {
            auto type_str = var_decl->type ? generate_type(var_decl->type.get()) : "auto";
            init_str = type_str + " " + var_decl->name;
            if (var_decl->initializer) {
                init_str += " = " + generate_expression_to_string(var_decl->initializer.get());
            }
        }
    }

    // Generate condition
    if (stmt->condition) {
        cond_str = generate_expression_to_string(stmt->condition.get());
    }

    // Generate increment
    if (stmt->increment) {
        inc_str = generate_expression_to_string(stmt->increment.get());
    }

    if (is_labeled) {
        // Labeled for loop - convert to while(true) pattern for continue to work
        // label: for (init; cond; inc) { body; }
        // Generates:
        // label:;
        // init;
        // while (true) {
        //     if (!(cond)) goto label_end;
        //     body;
        //     label_continue:;
        //     inc;
        // }
        // label_end:;

        write_line(stmt->label + ":;");

        if (!init_str.empty()) {
            write_line(init_str + ";");
        }

        write_line("while (true) {");
        indent();

        if (!cond_str.empty()) {
            write("if (!(" + cond_str + ")) goto " + stmt->label + "_end;");
            write_line("");
            write_line("");
        }

        generate_statement(stmt->body.get());

        write_line(stmt->label + "_continue:;");

        if (!inc_str.empty()) {
            write_line(inc_str + ";");
        }

        dedent();
        write_line("}");
        write_line(stmt->label + "_end:;");
    } else {
        // Non-labeled for - generate normally
        write("for (");
        if (!init_str.empty()) {
            write(init_str);
        }
        write("; ");
        if (!cond_str.empty()) {
            write(cond_str);
        }
        write("; ");
        if (!inc_str.empty()) {
            write(inc_str);
        }
        write(") ");
        generate_statement(stmt->body.get());
    }
}

void CodeGenerator::generate_for_range_statement(ForRangeStatement* stmt) {
    if (!stmt) return;

    bool is_labeled = !stmt->label.empty();
    bool has_loop_inits = !stmt->loop_inits.empty();
    std::string var_type = stmt->var_type ? generate_type(stmt->var_type.get()) : "auto";
    std::string range_str = generate_expression_to_string(stmt->range.get());
    
    // Apply qualifier to variable type
    std::string var_ref = "";
    if (stmt->var_qualifier == "inout" || stmt->var_qualifier == "out") {
        var_ref = "&";
    } else if (stmt->var_qualifier == "move" || stmt->var_qualifier == "forward") {
        var_ref = "&&";
    } else if (stmt->var_qualifier == "in") {
        // const reference
        var_type = "const " + var_type;
        var_ref = "&";
    }
    // copy or empty = pass by value (no ref)

    // If there's a loop initializer, wrap everything in a scope
    if (has_loop_inits) {
        write_line("{");
        indent();
        // Generate all loop initializer variables
        for (const auto& loop_init : stmt->loop_inits) {
            std::string type_str = "auto";
            if (loop_init->type && loop_init->type->kind != Type::Kind::Auto) {
                type_str = generate_type(loop_init->type.get());
            }
            std::string init_str = generate_expression_to_string(loop_init->initializer.get());
            write_line(type_str + " " + loop_init->name + " = " + init_str + ";");
        }
    }

    if (is_labeled) {
        // Labeled for-range loop - convert to while(true) with explicit iterators
        // label: for (auto x : range) { body; }
        // Generates:
        // label:;
        // {
        //     auto&& __range = range;
        //     auto __begin = __range.begin();
        //     auto __end = __range.end();
        //     while (true) {
        //         if (!(__begin != __end)) goto label_end;
        //         auto x = *__begin;
        //         body;
        //         label_continue:;
        //         ++__begin;
        //     }
        //     label_end:;
        // }

        write_line(stmt->label + ":;");
        write_line("{");
        indent();

        write_line("auto&& __range = " + range_str + ";");
        write_line("auto __begin = __range.begin();");
        write_line("auto __end = __range.end();");

        write_line("while (true) {");
        indent();

        write("if (!(__begin != __end)) goto " + stmt->label + "_end;");
        write_line("");
        write_line("");

        write_line(var_type + var_ref + " " + stmt->variable + " = *__begin;");
        write_line("++__begin;");

        generate_statement(stmt->body.get());

        write_line(stmt->label + "_continue:;");

        dedent();
        write_line("}");
        write_line(stmt->label + "_end:;");

        dedent();
        write_line("}");
    } else {
        // Non-labeled for-range - generate normally
        write("for (" + var_type + var_ref + " " + stmt->variable + " : " + range_str + ") ");
        generate_statement(stmt->body.get());
    }

    // Close loop initializer scope if present
    if (has_loop_inits) {
        dedent();
        write_line("}");
    }
}

void CodeGenerator::generate_return_statement(ReturnStatement* stmt) {
    if (!stmt) return;

    if (stmt->value) {
        write_line("return " + generate_expression_to_string(stmt->value.get()) + ";");
    } else {
        write_line("return;");
    }
}

void CodeGenerator::generate_break_statement(BreakStatement* stmt) {
    if (!stmt) return;

    if (stmt->label.empty()) {
        write_line("break;");
    } else {
        // C++ doesn't have labeled break - use goto
        // The label should have been emitted at the loop start
        // We need to jump past the loop
        write_line("goto " + stmt->label + "_end;");
    }
}

void CodeGenerator::generate_continue_statement(ContinueStatement* stmt) {
    if (!stmt) return;

    if (stmt->label.empty()) {
        write_line("continue;");
    } else {
        // C++ doesn't have labeled continue - use goto
        // The label should have been emitted at the loop start
        write_line("goto " + stmt->label + "_continue;");
    }
}

// ============================================================================
// Concurrency Statement Generators (Kotlin-style structured concurrency)
// ============================================================================

void CodeGenerator::generate_coroutine_scope_statement(CoroutineScopeStatement* stmt) {
    if (!stmt) return;

    // Structured concurrency scope - for now generate a simple block
    // TODO: Implement proper coroutine runtime when needed
    write_line("{");
    indent();
    write_line("// Coroutine scope (structured concurrency)");
    
    // Generate the body - handle both Block and single statement
    if (auto block = dynamic_cast<BlockStatement*>(stmt->body.get())) {
        for (auto& s : block->statements) {
            generate_statement(s.get());
        }
    } else {
        generate_statement(stmt->body.get());
    }
    
    dedent();
    write_line("}");
}

void CodeGenerator::generate_parallel_for_statement(ParallelForStatement* stmt) {
    if (!stmt) return;

    // Generate parallel for using std::async or thread pool
    // For simplicity, use std::async with a vector of futures
    write_line("{");
    indent();
    
    std::string lower = generate_expression_to_string(stmt->lower_bound.get());
    std::string upper = generate_expression_to_string(stmt->upper_bound.get());
    std::string step = stmt->step ? generate_expression_to_string(stmt->step.get()) : "1";
    
    write_line("std::vector<std::future<void>> __parallel_tasks;");
    write_line("for (auto " + stmt->loop_variable + " = " + lower + "; " + 
               stmt->loop_variable + " < " + upper + "; " + 
               stmt->loop_variable + " += " + step + ") {");
    indent();
    write_line("__parallel_tasks.push_back(std::async(std::launch::async, [=]() {");
    indent();
    
    // Generate the loop body
    generate_statement(stmt->body.get());
    
    dedent();
    write_line("}));");
    dedent();
    write_line("}");
    
    // Wait for all tasks to complete
    write_line("for (auto& __task : __parallel_tasks) { __task.wait(); }");
    
    dedent();
    write_line("}");
}

void CodeGenerator::generate_channel_declaration(ChannelDeclarationStatement* stmt) {
    if (!stmt) return;

    // TODO: Implement proper channel runtime when needed
    // For now, generate a placeholder comment
    std::string type_str = stmt->element_type ? generate_type(stmt->element_type.get()) : "void";
    std::string capacity_str = std::to_string(stmt->buffer_size);
    
    write_line("// Channel<" + type_str + "> " + stmt->name + " (capacity: " + capacity_str + ") - not yet supported");
}

std::string CodeGenerator::generate_expression_to_string(Expression* expr) {
    if (!expr) return "/* null expression */";

    std::ostringstream expr_output;

    switch (expr->kind) {
        case Expression::Kind::Literal: {
            auto lit = static_cast<LiteralExpression*>(expr);
            if (std::holds_alternative<int64_t>(lit->value)) {
                expr_output << std::get<int64_t>(lit->value);
            } else if (std::holds_alternative<double>(lit->value)) {
                expr_output << std::get<double>(lit->value);
            } else if (std::holds_alternative<bool>(lit->value)) {
                expr_output << (std::get<bool>(lit->value) ? "true" : "false");
            } else if (std::holds_alternative<std::string>(lit->value)) {
                const std::string& str_val = std::get<std::string>(lit->value);
                // Check if this is a raw string literal that already includes quotes
                // (e.g., from adjacent string concatenation or prefix strings)
                bool already_quoted = (!str_val.empty() && str_val[0] == '"') ||
                    (str_val.size() >= 2 && (
                        (str_val[0] == 'u' && str_val[1] == '"') ||
                        (str_val[0] == 'U' && str_val[1] == '"') ||
                        (str_val[0] == 'L' && str_val[1] == '"') ||
                        (str_val[0] == 'R' && str_val[1] == '"')
                    )) ||
                    (str_val.size() >= 3 && (
                        (str_val[0] == 'u' && str_val[1] == '8' && str_val[2] == '"') ||
                        (str_val[0] == 'L' && str_val[1] == 'R' && str_val[2] == '"') ||
                        (str_val[0] == 'u' && str_val[1] == 'R' && str_val[2] == '"') ||
                        (str_val[0] == 'U' && str_val[1] == 'R' && str_val[2] == '"')
                    )) ||
                    (str_val.size() >= 4 && str_val[0] == 'u' && str_val[1] == '8' && str_val[2] == 'R' && str_val[3] == '"');
                
                if (already_quoted) {
                    // Process string interpolation (expr)$
                    expr_output << process_string_interpolation(str_val);
                } else {
                    expr_output << "\"" << str_val << "\"";  // Regular string, add quotes
                }
            } else if (std::holds_alternative<char>(lit->value)) {
                expr_output << "'" << std::get<char>(lit->value) << "'";
            }
            break;
        }
        case Expression::Kind::Identifier: {
            auto id = static_cast<IdentifierExpression*>(expr);
            // Convert cpp2 library functions to std equivalents
            if (id->name == "cpp2::to_string") {
                expr_output << "std::to_string";
            } else {
                expr_output << id->name;
            }
            break;
        }
        case Expression::Kind::Binary: {
            auto binary = static_cast<BinaryExpression*>(expr);

            // Check for chained comparison: a <= b <= c
            // Parsed as (a <= b) <= c, needs to become a <= b && b <= c
            if (is_comparison_operator(binary->op)) {
                if (binary->left->kind == Expression::Kind::Binary) {
                    auto* left_binary = static_cast<BinaryExpression*>(binary->left.get());
                    if (is_comparison_operator(left_binary->op)) {
                        // This is a chained comparison - expand it
                        // (a <= b) <= c becomes a <= b && b <= c
                        // Extract the middle expression (b)
                        std::string left_str = generate_expression_to_string(left_binary->left.get());
                        std::string middle_str = generate_expression_to_string(left_binary->right.get());
                        std::string right_str = generate_expression_to_string(binary->right.get());

                        // Get the operators
                        std::string op1 = get_comparison_operator_string(left_binary->op);
                        std::string op2 = get_comparison_operator_string(binary->op);

                        expr_output << "(" << left_str << " " << op1 << " " << middle_str
                                   << " && " << middle_str << " " << op2 << " " << right_str << ")";
                        break;
                    }
                }
            }

            expr_output << generate_expression_to_string(binary->left.get());

            switch (binary->op) {
                case TokenType::Plus: expr_output << " + "; break;
                case TokenType::Minus: expr_output << " - "; break;
                case TokenType::Asterisk: expr_output << " * "; break;
                case TokenType::Slash: expr_output << " / "; break;
                case TokenType::Percent: expr_output << " % "; break;
                case TokenType::Equal: expr_output << " = "; break;
                case TokenType::PlusEqual: expr_output << " += "; break;
                case TokenType::MinusEqual: expr_output << " -= "; break;
                case TokenType::AsteriskEqual: expr_output << " *= "; break;
                case TokenType::SlashEqual: expr_output << " /= "; break;
                case TokenType::PercentEqual: expr_output << " %= "; break;
                case TokenType::DoubleEqual: expr_output << " == "; break;
                case TokenType::NotEqual: expr_output << " != "; break;
                case TokenType::LessThan: expr_output << " < "; break;
                case TokenType::GreaterThan: expr_output << " > "; break;
                case TokenType::LessThanOrEqual: expr_output << " <= "; break;
                case TokenType::GreaterThanOrEqual: expr_output << " >= "; break;
                case TokenType::Spaceship: expr_output << " <=> "; break;
                case TokenType::LeftShift: expr_output << " << "; break;
                case TokenType::RightShift: expr_output << " >> "; break;
                case TokenType::LeftShiftEqual: expr_output << " <<= "; break;
                case TokenType::RightShiftEqual: expr_output << " >>= "; break;
                case TokenType::Ampersand: expr_output << " & "; break;
                case TokenType::DoubleAmpersand: expr_output << " && "; break;
                case TokenType::Pipe: expr_output << " | "; break;
                case TokenType::DoublePipe: expr_output << " || "; break;
                case TokenType::Caret: expr_output << " ^ "; break;
                case TokenType::AmpersandEqual: expr_output << " &= "; break;
                case TokenType::PipeEqual: expr_output << " |= "; break;
                case TokenType::CaretEqual: expr_output << " ^= "; break;
                default: expr_output << " /* unknown op */ "; break;
            }

            expr_output << generate_expression_to_string(binary->right.get());
            break;
        }
        case Expression::Kind::Call: {
            auto call = static_cast<CallExpression*>(expr);

            // Handle constructor-style call expression (null callee) - e.g., (out y)
            // This is used for initialization with qualified arguments
            if (!call->callee) {
                // Just output the arguments as a parenthesized list
                expr_output << "(";
                if (!call->arguments.empty()) {
                    for (size_t i = 0; i < call->arguments.size(); ++i) {
                        if (i > 0) expr_output << ", ";
                        if (call->arguments[i].qualifier == ParameterQualifier::Move) {
                            expr_output << "std::move(";
                            expr_output << generate_expression_to_string(call->arguments[i].expr.get());
                            expr_output << ")";
                        } else {
                            expr_output << generate_expression_to_string(call->arguments[i].expr.get());
                        }
                    }
                }
                expr_output << ")";
                break;
            }

            // Check for UFCS pattern: x.func() where func is a known free function
            // Transform x.ssize() -> std::ssize(x), x.size() -> std::size(x), etc.
            if (call->callee->kind == Expression::Kind::MemberAccess) {
                auto member = static_cast<MemberAccessExpression*>(call->callee.get());
                static const std::unordered_set<std::string> std_free_functions = {
                    "ssize", "size", "data", "empty", "begin", "end", "cbegin", "cend",
                    "rbegin", "rend", "crbegin", "crend"
                };
                if (std_free_functions.count(member->member)) {
                    // Transform x.ssize() -> std::ssize(x)
                    expr_output << "std::" << member->member << "(";
                    expr_output << generate_expression_to_string(member->object.get());
                    // Add any additional arguments from call
                    for (size_t i = 0; i < call->arguments.size(); ++i) {
                        expr_output << ", ";
                        expr_output << generate_expression_to_string(call->arguments[i].expr.get());
                    }
                    for (size_t i = 0; i < call->args.size(); ++i) {
                        expr_output << ", ";
                        expr_output << generate_expression_to_string(call->args[i].get());
                    }
                    expr_output << ")";
                    break;
                }
            }

            // Note: Cpp2 library functions (assert, assume, unchecked_cast, etc.)
            // are handled without a cpp2:: prefix - we generate direct C++ equivalents
            // or rely on standard library facilities
            expr_output << generate_expression_to_string(call->callee.get()) << "(";

            // Use new arguments structure if populated, otherwise fall back to legacy args
            if (!call->arguments.empty()) {
                for (size_t i = 0; i < call->arguments.size(); ++i) {
                    if (i > 0) expr_output << ", ";
                    // For out/inout params, just pass the expression (C++ uses reference)
                    // No special prefix needed - the parameter declaration handles it
                    if (call->arguments[i].qualifier == ParameterQualifier::Move) {
                        expr_output << "std::move(";
                        expr_output << generate_expression_to_string(call->arguments[i].expr.get());
                        expr_output << ")";
                    } else {
                        expr_output << generate_expression_to_string(call->arguments[i].expr.get());
                    }
                }
            } else {
                for (size_t i = 0; i < call->args.size(); ++i) {
                    if (i > 0) expr_output << ", ";
                    expr_output << generate_expression_to_string(call->args[i].get());
                }
            }

            expr_output << ")";
            break;
        }
        case Expression::Kind::MemberAccess: {
            auto member = static_cast<MemberAccessExpression*>(expr);
            std::string object_str = generate_expression_to_string(member->object.get());

            // Convert cpp2 library functions to std equivalents
            if (object_str == "cpp2") {
                if (member->member == "to_string") {
                    expr_output << "std::to_string";
                    break;
                }
                // Add more cpp2:: function conversions here as needed
            }

            // In Cpp2, 'this.x' means member access, but in C++ 'this' is a pointer
            // So 'this.x' should become 'this->x' or just 'x' (since we're in member context)
            if (object_str == "this") {
                // Just use the member name directly - we're inside a member function
                expr_output << member->member;
            } else {
                expr_output << object_str << "." << member->member;
            }
            break;
        }
        case Expression::Kind::Subscript: {
            auto sub = static_cast<SubscriptExpression*>(expr);
            expr_output << generate_expression_to_string(sub->array.get()) << "["
                       << generate_expression_to_string(sub->index.get()) << "]";
            break;
        }
        case Expression::Kind::Unary: {
            auto unary = static_cast<UnaryExpression*>(expr);
            if (unary->is_postfix) {
                // Cpp2 has some postfix operators (e.g., `p*`, `x&`) that need
                // to become prefix operators in C++.
                if (unary->op == TokenType::Asterisk || unary->op == TokenType::Ampersand) {
                    expr_output << (unary->op == TokenType::Asterisk ? "*" : "&");
                    expr_output << generate_expression_to_string(unary->operand.get());
                } else {
                    expr_output << generate_expression_to_string(unary->operand.get());
                    switch (unary->op) {
                        case TokenType::PlusPlus: expr_output << "++"; break;
                        case TokenType::MinusMinus: expr_output << "--"; break;
                        default: break;
                    }
                }
            } else {
                switch (unary->op) {
                    case TokenType::Minus: expr_output << "-"; break;
                    case TokenType::Exclamation: expr_output << "!"; break;
                    case TokenType::Tilde: expr_output << "~"; break;
                    case TokenType::Asterisk: expr_output << "*"; break;
                    case TokenType::Ampersand: expr_output << "&"; break;
                    default: break;
                }
                expr_output << generate_expression_to_string(unary->operand.get());
            }
            break;
        }
        case Expression::Kind::InspectExpr: {
            auto inspect_expr = static_cast<InspectExpression*>(expr);
            // Generate as immediately invoked lambda for expression context
            expr_output << "([&]() {\n";
            expr_output << "    auto __value = " << generate_expression_to_string(inspect_expr->value.get()) << ";\n";

            // Check if there's a wildcard pattern (exhaustive match)
            bool has_wildcard = false;
            for (const auto& arm : inspect_expr->arms) {
                if (arm.pattern_kind == InspectExpression::Arm::PatternKind::Wildcard) {
                    has_wildcard = true;
                    break;
                }
            }

            // Generate if-else chain
            for (size_t i = 0; i < inspect_expr->arms.size(); ++i) {
                const auto& arm = inspect_expr->arms[i];

                if (i > 0) expr_output << "    else ";
                else expr_output << "    ";

                if (arm.pattern_kind == InspectExpression::Arm::PatternKind::Wildcard) {
                    // Wildcard always matches - make it the final else
                    expr_output << "{ return " << generate_expression_to_string(arm.result_value.get()) << "; }\n";
                } else if (arm.pattern_kind == InspectExpression::Arm::PatternKind::Type) {
                    // Type pattern: use std::get_if for variant types
                    std::string type_str = generate_type(arm.pattern_type.get());
                    // Check if this is a variant type - use std::get_if for variants
                    if (type_str.find("std::variant") == 0 || type_str.find("variant") == 0) {
                        expr_output << "if (std::get_if<" << type_str << ">(&__value)) ";
                    } else {
                        // For non-variant types, use dynamic_cast or direct comparison
                        expr_output << "if (typeid(__value) == typeid(" << type_str << ")) ";
                    }
                    expr_output << "{ return " << generate_expression_to_string(arm.result_value.get()) << "; }\n";
                } else {
                    // Value pattern - use std::get_if for variant comparisons
                    if (arm.pattern_value) {
                        expr_output << "if (__value == " << generate_expression_to_string(arm.pattern_value.get()) << ") ";
                        expr_output << "{ return " << generate_expression_to_string(arm.result_value.get()) << "; }\n";
                    }
                }
            }

            // Only add throw if no wildcard (non-exhaustive inspect)
            if (!has_wildcard) {
                expr_output << "    else { throw std::logic_error(\"Non-exhaustive inspect\"); }\n";
            }
            expr_output << "})()";
            break;
        }
        case Expression::Kind::List: {
            auto list = static_cast<ListExpression*>(expr);
            // Generate tuple/initializer list as { elem1, elem2, ... }
            expr_output << "{";
            for (size_t i = 0; i < list->elements.size(); ++i) {
                if (i > 0) expr_output << ", ";
                expr_output << generate_expression_to_string(list->elements[i].get());
            }
            expr_output << "}";
            break;
        }
        case Expression::Kind::As: {
            auto as_expr = static_cast<AsExpression*>(expr);
            // Generate: std::get<Type>(value) for variant types
            std::string type_str = generate_type(as_expr->type.get());
            if (type_str.find("std::variant") == 0 || type_str.find("variant") == 0 ||
                type_str.find("std::optional") == 0 || type_str.find("optional") == 0) {
                // For variant/optional types, use std::get
                expr_output << "std::get<" << type_str << ">(";
                expr_output << generate_expression_to_string(as_expr->expr.get()) << ")";
            } else {
                // For other types, use static_cast
                expr_output << "static_cast<" << type_str << ">(";
                expr_output << generate_expression_to_string(as_expr->expr.get()) << ")";
            }
            break;
        }
        case Expression::Kind::Is: {
            auto is_expr = static_cast<IsExpression*>(expr);
            // Generate: std::get_if<Type>(&value) != nullptr for variant types
            std::string type_str = generate_type(is_expr->type.get());
            if (type_str.find("std::variant") == 0 || type_str.find("variant") == 0 ||
                type_str.find("std::optional") == 0 || type_str.find("optional") == 0) {
                expr_output << "(std::get_if<" << type_str << ">(";
                expr_output << "&" << generate_expression_to_string(is_expr->expr.get()) << ") != nullptr)";
            } else {
                // For other types, use typeid
                expr_output << "(typeid(" << generate_expression_to_string(is_expr->expr.get()) << ") == typeid(" << type_str << "))";
            }
            break;
        }
        case Expression::Kind::Move: {
            auto move_expr = static_cast<MoveExpression*>(expr);
            // Generate: std::move(value) or std::forward<T>(value)
            if (move_expr->op == TokenType::Move) {
                expr_output << "std::move(" << generate_expression_to_string(move_expr->operand.get()) << ")";
            } else if (move_expr->op == TokenType::Forward) {
                expr_output << "std::forward(" << generate_expression_to_string(move_expr->operand.get()) << ")";
            } else {
                // Copy - just generate the operand
                expr_output << generate_expression_to_string(move_expr->operand.get());
            }
            break;
        }
        case Expression::Kind::StringInterpolation: {
            auto interp = static_cast<StringInterpolationExpression*>(expr);
            // Generate string concatenation using std::to_string()
            // For "hello $world" generate: "hello " + std::to_string(world)
            for (size_t i = 0; i < interp->parts.size(); ++i) {
                if (i > 0) expr_output << " + ";
                if (std::holds_alternative<std::string>(interp->parts[i])) {
                    // String literal part
                    expr_output << "\"" << std::get<std::string>(interp->parts[i]) << "\"";
                } else {
                    // Expression part - wrap with std::to_string for arithmetic types
                    expr_output << "(([&]() -> std::string { ";
                    expr_output << "auto __val = " << generate_expression_to_string(std::get<std::unique_ptr<Expression>>(interp->parts[i]).get()) << "; ";
                    expr_output << "if constexpr (std::is_arithmetic_v<decltype(__val)>) { ";
                    expr_output << "return std::to_string(__val); ";
                    expr_output << "} else { ";
                    expr_output << "return __val; ";
                    expr_output << "} ";
                    expr_output << "})())";
                }
            }
            break;
        }
        case Expression::Kind::Lambda: {
            auto lambda = static_cast<LambdaExpression*>(expr);
            // Cpp2 lambda: generate [&](params) -> type { body }
            expr_output << "[&](";
            for (size_t i = 0; i < lambda->parameters.size(); ++i) {
                if (i > 0) expr_output << ", ";
                const auto& param = lambda->parameters[i];
                if (param.type) {
                    expr_output << generate_type(param.type.get()) << " ";
                } else {
                    expr_output << "auto ";
                }
                expr_output << param.name;
            }
            expr_output << ")";
            if (lambda->return_type) {
                expr_output << " -> " << generate_type(lambda->return_type.get());
            }
            expr_output << " { ";
            // Generate body - for simple lambdas, assume single expression return
            for (const auto& stmt : lambda->body) {
                // Simplified - full implementation would call generate_statement
                if (auto ret_stmt = dynamic_cast<ReturnStatement*>(stmt.get())) {
                    expr_output << "return " << generate_expression_to_string(ret_stmt->value.get()) << "; ";
                }
            }
            expr_output << "}";
            break;
        }
        // ============================================================================
        // Concurrency Expressions (Kotlin-style)
        // ============================================================================
        case Expression::Kind::Await: {
            auto await_expr = static_cast<AwaitExpression*>(expr);
            // Generate: co_await <value>
            expr_output << "co_await " << generate_expression_to_string(await_expr->value.get());
            break;
        }
        case Expression::Kind::Spawn: {
            auto spawn_expr = static_cast<SpawnExpression*>(expr);
            // Generate async task launch using std::async
            expr_output << "std::async(std::launch::async, [&]() { ";
            expr_output << generate_expression_to_string(spawn_expr->task.get());
            expr_output << "; })";
            break;
        }
        case Expression::Kind::ChannelSend: {
            auto send_expr = static_cast<ChannelSendExpression*>(expr);
            // Generate: <channel>.send(<value>)
            expr_output << send_expr->channel << ".send(";
            expr_output << generate_expression_to_string(send_expr->value.get()) << ")";
            break;
        }
        case Expression::Kind::ChannelRecv: {
            auto recv_expr = static_cast<ChannelRecvExpression*>(expr);
            // Generate: <channel>.receive() or <channel>.tryReceive()
            expr_output << recv_expr->channel;
            if (recv_expr->non_blocking) {
                expr_output << ".tryReceive()";
            } else {
                expr_output << ".receive()";
            }
            break;
        }
        case Expression::Kind::ChannelSelect: {
            auto select_expr = static_cast<ChannelSelectExpression*>(expr);
            // Generate select as a lambda that polls channels
            // This is a simplified implementation - full version would use condition_variable
            expr_output << "[&]() {\n";
            for (size_t i = 0; i < select_expr->cases.size(); ++i) {
                const auto& case_ = select_expr->cases[i];
                expr_output << "    if (auto __val = " << case_.channel;
                if (case_.kind == ChannelSelectExpression::SelectCase::Kind::Recv) {
                    expr_output << ".tryReceive(); __val) {\n";
                    expr_output << "        return ";
                    expr_output << generate_expression_to_string(case_.action.get());
                    expr_output << ";\n    }\n";
                } else {
                    expr_output << ".trySend(";
                    expr_output << generate_expression_to_string(case_.value.get());
                    expr_output << ")) {\n";
                    expr_output << "        return ";
                    expr_output << generate_expression_to_string(case_.action.get());
                    expr_output << ";\n    }\n";
                }
            }
            if (select_expr->default_case) {
                expr_output << "    return ";
                expr_output << generate_expression_to_string(select_expr->default_case.get());
                expr_output << ";\n";
            }
            expr_output << "}()";
            break;
        }
        case Expression::Kind::Cpp1Lambda: {
            auto lambda = static_cast<Cpp1LambdaExpression*>(expr);
            // C++1 lambda: pass through directly
            expr_output << "[";
            for (size_t i = 0; i < lambda->captures.size(); ++i) {
                if (i > 0) expr_output << ", ";
                const auto& cap = lambda->captures[i];
                switch (cap.mode) {
                    case Cpp1LambdaExpression::Capture::Mode::DefaultCopy:
                        expr_output << "=";
                        break;
                    case Cpp1LambdaExpression::Capture::Mode::DefaultRef:
                        expr_output << "&";
                        break;
                    case Cpp1LambdaExpression::Capture::Mode::ByCopy:
                        expr_output << cap.name;
                        break;
                    case Cpp1LambdaExpression::Capture::Mode::ByRef:
                        expr_output << "&" << cap.name;
                        break;
                    case Cpp1LambdaExpression::Capture::Mode::This:
                        expr_output << "this";
                        break;
                }
            }
            expr_output << "](";
            for (size_t i = 0; i < lambda->parameters.size(); ++i) {
                if (i > 0) expr_output << ", ";
                const auto& param = lambda->parameters[i];
                expr_output << param.type_str << " " << param.name;
            }
            expr_output << ")";
            if (lambda->return_type) {
                expr_output << " -> " << generate_type(lambda->return_type.get());
            }
            expr_output << " { ";
            // Generate body - simplified
            for (const auto& stmt : lambda->body) {
                if (auto ret_stmt = dynamic_cast<ReturnStatement*>(stmt.get())) {
                    expr_output << "return " << generate_expression_to_string(ret_stmt->value.get()) << "; ";
                } else if (auto expr_stmt = dynamic_cast<ExpressionStatement*>(stmt.get())) {
                    expr_output << generate_expression_to_string(expr_stmt->expr.get()) << "; ";
                }
            }
            expr_output << "}";
            break;
        }
        default:
            expr_output << "/* expression kind " << static_cast<int>(expr->kind) << " */";
            break;
    }

    return expr_output.str();
}

// Convert Cpp2 function type "(params) -> return_type" to C++ "return_type(params)"
std::string CodeGenerator::convert_function_type_to_cpp(const std::string& func_type) {
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

std::string CodeGenerator::generate_type(Type* type) {
    if (!type) return "void";

    // Map common Cpp2-style builtin names to C++ spellings.
    // Keep this minimal and test-driven.
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
            // Generate template type with arguments: name<arg1, arg2, ...>
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
        case Type::Kind::Pointer:
            // Special case: pointer to function type needs special C++ syntax
            if (type->pointee && type->pointee->kind == Type::Kind::FunctionType) {
                // Convert function type pointer: need "return_type(*)(params)"
                std::string func_cpp = convert_function_type_to_cpp(type->pointee->name);
                // func_cpp is "return_type(params)", need to insert (*) 
                auto paren_pos = func_cpp.find('(');
                if (paren_pos != std::string::npos) {
                    return func_cpp.substr(0, paren_pos) + "(*)" + func_cpp.substr(paren_pos);
                }
                return func_cpp + "*";
            }
            return generate_type(type->pointee.get()) + "*";
        case Type::Kind::Reference:
            return generate_type(type->pointee.get()) + "&";
        case Type::Kind::Auto:
            return "auto";
        case Type::Kind::Deduced:
            // _ as type means decltype(auto) or auto in return position
            // Handle constrained deduced types: _ is constraint
            if (type->name.starts_with("_ is ")) {
                // Extract constraint name (after "_ is ")
                std::string constraint = type->name.substr(5);
                // Generate as C++20 abbreviated template with constraint
                return constraint;  // constraint auto param
            }
            return "decltype(auto)";
        case Type::Kind::FunctionType:
            // Function type: (params) -> return_type
            // Convert to C++ function type for use in std::function<> etc.
            // The name contains the Cpp2 form which we need to convert
            return convert_function_type_to_cpp(type->name);
        default:
            return type->name;
    }
}

std::string CodeGenerator::generate_parameter_type(Type* type, const std::vector<ParameterQualifier>& qualifiers) {
    std::string base_type;
    
    if (!type) {
        base_type = "auto";
    } else if (type->kind == Type::Kind::Deduced) {
        // In parameter context, decltype(auto) is invalid - use auto instead
        base_type = "auto";
    } else {
        base_type = generate_type(type);
    }

    // Check for parameter qualifiers
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

    // No qualifier - pass by value
    return base_type;
}

bool CodeGenerator::needs_nodiscard(FunctionDeclaration* func) {
    // Non-void functions should have [[nodiscard]], except for main()
    // which is the program entry point and cannot have [[nodiscard]]
    if (func->name == "main") {
        return false;
    }
    return func->return_type && func->return_type->name != "void";
}

} // namespace cpp2_transpiler