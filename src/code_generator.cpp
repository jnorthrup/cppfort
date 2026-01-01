#include "code_generator.hpp"
#include <iostream>
#include <format>
#include <algorithm>

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

CodeGenerator::CodeGenerator() : indent_level(0), needs_semicolon(true) {}

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
    write_line("#include \"cpp2util.h\"");
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

    // [[nodiscard]] goes before the return type for widest compatibility
    if (needs_nodiscard(decl)) {
        write("[[nodiscard]] ");
    }

    // Use auto return_type syntax for cppfront compatibility
    write("auto " + decl->name + "(");

    // Parameters
    for (size_t i = 0; i < decl->parameters.size(); ++i) {
        if (i > 0) write(", ");
        const auto& param = decl->parameters[i];
        write(generate_parameter_type(param.type.get(), param.qualifiers) + " " + param.name);
    }

    write(") -> " + return_type);

    write_line(";");
}

void CodeGenerator::generate_function_declaration(FunctionDeclaration* decl) {
    if (!decl) return;

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

    // [[nodiscard]] goes before the return type for widest compatibility
    if (needs_nodiscard(decl)) {
        write("[[nodiscard]] ");
    }

    // Use auto return_type syntax for cppfront compatibility
    write("auto " + decl->name + "(");

    // Parameters
    for (size_t i = 0; i < decl->parameters.size(); ++i) {
        if (i > 0) write(", ");
        const auto& param = decl->parameters[i];
        write(generate_parameter_type(param.type.get(), param.qualifiers) + " " + param.name);
    }

    write(") -> " + return_type);

    if (decl->body) {
        // Check if body is already a BlockStatement - if so, generate it directly
        // to avoid adding extra nested braces
        if (decl->body->kind == Statement::Kind::Block) {
            write(" ");
            generate_statement(decl->body.get());
        } else {
            write(" {\n");
            indent();
            generate_statement(decl->body.get());
            dedent();
            write_line("}");
        }
    } else {
        write_line(";");
    }
}

void CodeGenerator::generate_operator_declaration(OperatorDeclaration* decl) {
    if (!decl) return;

    // Cpp2 operator=: is the assignment operator
    // operator=: (out this, that) = { body }
    // Transpiles to: ClassName& operator=(ClassName that) { body }

    std::string return_type = decl->return_type ? generate_type(decl->return_type.get()) : "void";

    // Get the class name from the context (outer type)
    std::string class_name = current_class_name.empty() ? "ClassName" : current_class_name;

    // operator=: becomes operator=
    std::string op_name = decl->name;
    if (op_name == "operator=:") {
        op_name = "operator=";
    }

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
    write_line("using " + decl->name + " = " + decl->target + ";");
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
    std::string cond_str = generate_expression_to_string(stmt->condition.get());

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
    std::string var_type = stmt->var_type ? generate_type(stmt->var_type.get()) : "auto";
    std::string range_str = generate_expression_to_string(stmt->range.get());

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

        write_line(var_type + " " + stmt->variable + " = *__begin;");
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
        write("for (" + var_type + " " + stmt->variable + " : " + range_str + ") ");
        generate_statement(stmt->body.get());
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

    // Generate: cpp2::coroutineScope([&](cpp2::CoroutineScope& scope) { ... })
    // This ensures all launched tasks complete before the scope exits
    write_line("{");
    indent();
    write_line("cpp2::CoroutineScope __scope;");
    
    // Generate the body - handle both Block and single statement
    if (auto block = dynamic_cast<BlockStatement*>(stmt->body.get())) {
        for (auto& s : block->statements) {
            generate_statement(s.get());
        }
    } else {
        generate_statement(stmt->body.get());
    }
    
    // CoroutineScope destructor will joinAll() automatically
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

    // Generate: cpp2::Channel<T> name(capacity);
    std::string type_str = stmt->element_type ? generate_type(stmt->element_type.get()) : "void";
    std::string capacity_str = std::to_string(stmt->buffer_size);
    
    write_line("cpp2::Channel<" + type_str + "> " + stmt->name + "(" + capacity_str + ");");
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
                // Check if this is a string with prefix (u", U", u8", L", R", LR", uR", UR", u8R")
                // These already include their prefix and quotes in the lexeme
                bool has_prefix = (str_val.size() >= 2) && (
                    (str_val[0] == 'u' && str_val[1] == '"') ||
                    (str_val[0] == 'U' && str_val[1] == '"') ||
                    (str_val.size() >= 3 && str_val[0] == 'u' && str_val[1] == '8' && str_val[2] == '"') ||
                    (str_val[0] == 'L' && str_val[1] == '"') ||
                    (str_val[0] == 'R' && str_val[1] == '"') ||
                    (str_val.size() >= 3 && str_val[0] == 'L' && str_val[1] == 'R' && str_val[2] == '"') ||
                    (str_val.size() >= 3 && str_val[0] == 'u' && str_val[1] == 'R' && str_val[2] == '"') ||
                    (str_val.size() >= 3 && str_val[0] == 'U' && str_val[1] == 'R' && str_val[2] == '"') ||
                    (str_val.size() >= 4 && str_val[0] == 'u' && str_val[1] == '8' && str_val[2] == 'R' && str_val[3] == '"')
                );
                if (has_prefix) {
                    expr_output << str_val;  // Already has prefix and quotes
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
            expr_output << id->name;
            break;
        }
        case Expression::Kind::Binary: {
            auto binary = static_cast<BinaryExpression*>(expr);
            expr_output << generate_expression_to_string(binary->left.get());

            switch (binary->op) {
                case TokenType::Plus: expr_output << " + "; break;
                case TokenType::Minus: expr_output << " - "; break;
                case TokenType::Asterisk: expr_output << " * "; break;
                case TokenType::Slash: expr_output << " / "; break;
                case TokenType::Equal: expr_output << " = "; break;
                case TokenType::DoubleEqual: expr_output << " == "; break;
                case TokenType::NotEqual: expr_output << " != "; break;
                case TokenType::LessThan: expr_output << " < "; break;
                case TokenType::GreaterThan: expr_output << " > "; break;
                case TokenType::LessThanOrEqual: expr_output << " <= "; break;
                case TokenType::GreaterThanOrEqual: expr_output << " >= "; break;
                case TokenType::LeftShift: expr_output << " << "; break;
                case TokenType::RightShift: expr_output << " >> "; break;
                default: expr_output << " ?op? "; break;
            }

            expr_output << generate_expression_to_string(binary->right.get());
            break;
        }
        case Expression::Kind::Call: {
            auto call = static_cast<CallExpression*>(expr);

            // Check for Cpp2 library functions that need cpp2:: prefix
            bool needs_cpp2_prefix = false;
            if (call->callee->kind == Expression::Kind::Identifier) {
                auto id = static_cast<IdentifierExpression*>(call->callee.get());
                if (id->name == "unchecked_cast" || id->name == "unchecked_narrow" ||
                    id->name == "unsafe_cast" || id->name == "unsafe_narrow" ||
                    id->name == "assert" || id->name == "assume") {
                    needs_cpp2_prefix = true;
                }
            }

            if (needs_cpp2_prefix) {
                expr_output << "cpp2::";
            }
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
            expr_output << generate_expression_to_string(member->object.get()) << "." << member->member;
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

            // Generate if-else chain
            for (size_t i = 0; i < inspect_expr->arms.size(); ++i) {
                const auto& arm = inspect_expr->arms[i];

                if (i > 0) expr_output << "    else ";
                else expr_output << "    ";

                if (arm.pattern_kind == InspectExpression::Arm::PatternKind::Wildcard) {
                    // Wildcard always matches - make it the final else
                    expr_output << "{ return " << generate_expression_to_string(arm.result_value.get()) << "; }\n";
                } else if (arm.pattern_kind == InspectExpression::Arm::PatternKind::Type) {
                    // Type pattern: check if value is of the given type
                    expr_output << "if (cpp2::impl::is_<" << generate_type(arm.pattern_type.get()) << ">(__value)) ";
                    expr_output << "{ return " << generate_expression_to_string(arm.result_value.get()) << "; }\n";
                } else {
                    // Value pattern
                    expr_output << "if (__value == " << generate_expression_to_string(arm.pattern_value.get()) << ") ";
                    expr_output << "{ return " << generate_expression_to_string(arm.result_value.get()) << "; }\n";
                }
            }

            expr_output << "    else { throw std::logic_error(\"Non-exhaustive inspect\"); }\n";
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
            // Generate: cpp2::impl::as_<Type>(value)
            expr_output << "cpp2::impl::as_<" << generate_type(as_expr->type.get()) << ">(";
            expr_output << generate_expression_to_string(as_expr->expr.get()) << ")";
            break;
        }
        case Expression::Kind::Is: {
            auto is_expr = static_cast<IsExpression*>(expr);
            // Generate: cpp2::impl::is_<Type>(value)
            expr_output << "cpp2::impl::is_<" << generate_type(is_expr->type.get()) << ">(";
            expr_output << generate_expression_to_string(is_expr->expr.get()) << ")";
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
            // Generate string concatenation using cpp2::to_string()
            // For "hello $world" generate: "hello " + cpp2::to_string(world)
            for (size_t i = 0; i < interp->parts.size(); ++i) {
                if (i > 0) expr_output << " + ";
                if (std::holds_alternative<std::string>(interp->parts[i])) {
                    // String literal part
                    expr_output << "\"" << std::get<std::string>(interp->parts[i]) << "\"";
                } else {
                    // Expression part - wrap with cpp2::to_string
                    expr_output << "cpp2::to_string(";
                    expr_output << generate_expression_to_string(std::get<std::unique_ptr<Expression>>(interp->parts[i]).get());
                    expr_output << ")";
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
            // Generate: cpp2::spawn([&]() -> cpp2::Task<void> { <task>; co_return; })
            // For fire-and-forget coroutine launch
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
            return generate_type(type->pointee.get()) + "*";
        case Type::Kind::Reference:
            return generate_type(type->pointee.get()) + "&";
        case Type::Kind::Auto:
            return "auto";
        case Type::Kind::Deduced:
            // _ as type means decltype(auto) or auto in return position
            return "decltype(auto)";
        default:
            return type->name;
    }
}

std::string CodeGenerator::generate_parameter_type(Type* type, const std::vector<ParameterQualifier>& qualifiers) {
    std::string base_type = type ? generate_type(type) : "auto";

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