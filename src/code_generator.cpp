#include "code_generator.hpp"
#include <iostream>
#include <format>

namespace cpp2_transpiler {

CodeGenerator::CodeGenerator() : indent_level(0), needs_semicolon(true) {}

std::string CodeGenerator::generate(AST& ast) {
    output.str("");
    output.clear();

    write_includes();

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
    for (auto& decl : ast.declarations) {
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
    write_line("#include <cassert>");
    write_line("#include <iostream>");
    write_line("#include <string>");
    write_line("#include <string_view>");
    write_line("#include <vector>");
    write_line("#include <span>");
    write_line("#include <format>");
    write_line("#include <ranges>");
    write_line("#include <memory>");
    write_line("#include <optional>");
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
        case Declaration::Kind::Using:
            generate_using_declaration(static_cast<UsingDeclaration*>(decl));
            break;
        case Declaration::Kind::Import:
            generate_import_declaration(static_cast<ImportDeclaration*>(decl));
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

    write(return_type + " " + decl->name + "(");

    // Parameters
    for (size_t i = 0; i < decl->parameters.size(); ++i) {
        if (i > 0) write(", ");
        const auto& param = decl->parameters[i];
        write(generate_parameter_type(param.type.get(), param.qualifiers) + " " + param.name);
    }

    write_line(");");
}

void CodeGenerator::generate_function_declaration(FunctionDeclaration* decl) {
    if (!decl) return;

    // Generate template header if this is a template function
    if (!decl->template_parameters.empty()) {
        write("template<");
        for (size_t i = 0; i < decl->template_parameters.size(); ++i) {
            if (i > 0) write(", ");
            write("typename " + decl->template_parameters[i]);
        }
        write_line(">");
    }

    std::string return_type = decl->return_type ? generate_type(decl->return_type.get()) : "void";

    // [[nodiscard]] goes before the return type for widest compatibility
    if (needs_nodiscard(decl)) {
        write("[[nodiscard]] ");
    }

    write(return_type + " " + decl->name + "(");

    // Parameters
    for (size_t i = 0; i < decl->parameters.size(); ++i) {
        if (i > 0) write(", ");
        const auto& param = decl->parameters[i];
        write(generate_parameter_type(param.type.get(), param.qualifiers) + " " + param.name);
    }

    write(")");

    if (decl->body) {
        write(" {\n");
        indent();
        generate_statement(decl->body.get());
        dedent();
        write_line("}");
    } else {
        write_line(";");
    }
}

void CodeGenerator::generate_type_declaration(TypeDeclaration* decl) {
    if (!decl) return;

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
        case TypeDeclaration::TypeKind::Struct:
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

            for (auto& member : decl->members) {
                generate_declaration(member.get());
            }

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

        case TypeDeclaration::TypeKind::Alias:
            if (decl->underlying_type) {
                write_line("using " + decl->name + " = " + generate_type(decl->underlying_type.get()) + ";");
            }
            break;

        default:
            break;
    }

    // Clear metafunction context
    current_type_metafunctions.clear();
}

void CodeGenerator::generate_namespace_declaration(NamespaceDeclaration* decl) {
    if (!decl) return;

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
        case Statement::Kind::For:
            generate_for_statement(static_cast<ForStatement*>(stmt));
            break;
        case Statement::Kind::ForRange:
            generate_for_range_statement(static_cast<ForRangeStatement*>(stmt));
            break;
        case Statement::Kind::Return:
            generate_return_statement(static_cast<ReturnStatement*>(stmt));
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

    write("while (" + generate_expression_to_string(stmt->condition.get()) + ") ");

    // Check if there's an increment clause (Cpp2 'next' syntax)
    bool has_increment = stmt->increment != nullptr;

    if (!has_increment) {
        // Simple case: no increment clause
        generate_statement(stmt->body.get());
    } else {
        // Has increment clause: while(cond) next inc { body }
        // Generate as: while(cond) { body; inc; }
        write_line("{");
        indent();
        generate_statement(stmt->body.get());
        write_line("");
        write(generate_expression_to_string(stmt->increment.get()) + ";");
        dedent();
        write_line("}");
    }
}

void CodeGenerator::generate_for_statement(ForStatement* stmt) {
    if (!stmt) return;

    write("for (");
    if (stmt->init) {
        // Generate init without semicolon
        if (auto var_decl = dynamic_cast<VariableDeclaration*>(stmt->init.get())) {
            auto type_str = var_decl->type ? generate_type(var_decl->type.get()) : "auto";
            write(type_str + " " + var_decl->name);
            if (var_decl->initializer) {
                write(" = " + generate_expression_to_string(var_decl->initializer.get()));
            }
        }
    }
    write("; ");
    if (stmt->condition) {
        write(generate_expression_to_string(stmt->condition.get()));
    }
    write("; ");
    if (stmt->increment) {
        write(generate_expression_to_string(stmt->increment.get()));
    }
    write(") ");
    generate_statement(stmt->body.get());
}

void CodeGenerator::generate_for_range_statement(ForRangeStatement* stmt) {
    if (!stmt) return;

    std::string var_type = stmt->var_type ? generate_type(stmt->var_type.get()) : "auto";
    write("for (" + var_type + " " + stmt->variable + " : " +
          generate_expression_to_string(stmt->range.get()) + ") ");
    generate_statement(stmt->body.get());
}

void CodeGenerator::generate_return_statement(ReturnStatement* stmt) {
    if (!stmt) return;

    if (stmt->value) {
        write_line("return " + generate_expression_to_string(stmt->value.get()) + ";");
    } else {
        write_line("return;");
    }
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
                expr_output << "\"" << std::get<std::string>(lit->value) << "\"";
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
            expr_output << "(" << generate_expression_to_string(binary->left.get());

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

            expr_output << generate_expression_to_string(binary->right.get()) << ")";
            break;
        }
        case Expression::Kind::Call: {
            auto call = static_cast<CallExpression*>(expr);
            expr_output << generate_expression_to_string(call->callee.get()) << "(";

            for (size_t i = 0; i < call->args.size(); ++i) {
                if (i > 0) expr_output << ", ";
                expr_output << generate_expression_to_string(call->args[i].get());
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
        case Type::Kind::Pointer:
            return generate_type(type->pointee.get()) + "*";
        case Type::Kind::Reference:
            return generate_type(type->pointee.get()) + "&";
        case Type::Kind::Auto:
            return "auto";
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