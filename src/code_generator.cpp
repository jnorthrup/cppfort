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
        write((param.type ? generate_type(param.type.get()) : "auto") + " " + param.name);
    }

    write_line(");");
}

void CodeGenerator::generate_function_declaration(FunctionDeclaration* decl) {
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
        write((param.type ? generate_type(param.type.get()) : "auto") + " " + param.name);
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

    switch (decl->type_kind) {
        case TypeDeclaration::TypeKind::Struct:
            write_line("struct " + decl->name + " {");
            indent();
            for (auto& member : decl->members) {
                generate_declaration(member.get());
            }
            dedent();
            write_line("};");
            break;

        case TypeDeclaration::TypeKind::Class:
            write_line("class " + decl->name + " {");
            indent();
            write_line("public:");
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
    generate_statement(stmt->body.get());
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

bool CodeGenerator::needs_nodiscard(FunctionDeclaration* func) {
    // Non-void functions should have [[nodiscard]], except for main()
    // which is the program entry point and cannot have [[nodiscard]]
    if (func->name == "main") {
        return false;
    }
    return func->return_type && func->return_type->name != "void";
}

} // namespace cpp2_transpiler