#pragma once
// ============================================================================
// ParseTree to AST Converter
// ============================================================================
// Bridges the slim ParseTree from the combinator parser to the rich AST
// expected by semantic analyzer, code generator, etc.
// ============================================================================

#include "slim_ast.hpp"
#include "ast.hpp"
#include <memory>

namespace cpp2::parser {

class ParseTreeConverter {
    const ast::ParseTree& tree_;
    std::span<const cpp2_transpiler::Token> tokens_;

public:
    explicit ParseTreeConverter(const ast::ParseTree& tree)
        : tree_(tree), tokens_(tree.tokens) {}

    std::unique_ptr<cpp2_transpiler::AST> convert() {
        auto result = std::make_unique<cpp2_transpiler::AST>();

        if (tree_.nodes.empty()) return result;

        const auto& root = tree_[tree_.root];
        if (root.kind != ast::NodeKind::TranslationUnit) return result;

        for (const auto& child : tree_.children(root)) {
            if (child.kind == ast::NodeKind::Declaration) {
                if (auto decl = convert_declaration(child)) {
                    result->declarations.push_back(std::move(decl));
                }
            }
        }

        return result;
    }

private:
    // Get line number for a node
    std::size_t line_of(const ast::Node& n) const {
        if (n.token_start < tokens_.size()) {
            return tokens_[n.token_start].line;
        }
        return 1;
    }

    // Get lexeme for single-token node
    std::string_view lexeme_of(const ast::Node& n) const {
        return tree_.lexeme(n);
    }

    // Find child of specific kind
    const ast::Node* find_child(const ast::Node& n, ast::NodeKind kind) const {
        for (const auto& child : tree_.children(n)) {
            if (child.kind == kind) return &child;
        }
        return nullptr;
    }

    // ========================================================================
    // Declaration Conversion
    // ========================================================================

    std::unique_ptr<cpp2_transpiler::Declaration> convert_declaration(const ast::Node& n) {
        auto children = tree_.children(n);

        // Find the unified declaration node
        const ast::Node* unified = find_child(n, ast::NodeKind::UnifiedDeclaration);
        if (!unified) return nullptr;

        // Get name from first identifier-like child
        std::string name;
        auto unified_children = tree_.children(*unified);
        if (!unified_children.empty()) {
            const auto& first = unified_children[0];
            if (first.kind == ast::NodeKind::Identifier) {
                name = std::string(lexeme_of(first));
            }
        }

        // Check what kind of declaration based on suffix
        if (find_child(*unified, ast::NodeKind::FunctionSuffix)) {
            return convert_function_decl(*unified, name);
        }
        if (find_child(*unified, ast::NodeKind::TypeSuffix)) {
            return convert_type_decl(*unified, name);
        }
        if (find_child(*unified, ast::NodeKind::NamespaceSuffix)) {
            return convert_namespace_decl(*unified, name);
        }
        if (find_child(*unified, ast::NodeKind::VariableSuffix)) {
            return convert_variable_decl(*unified, name);
        }

        // Check for := style declaration (type-deduced variable)
        for (const auto& child : unified_children) {
            if (child.kind == ast::NodeKind::Expression) {
                // This is a := declaration
                return convert_variable_decl(*unified, name);
            }
        }

        return nullptr;
    }

    std::unique_ptr<cpp2_transpiler::FunctionDeclaration> convert_function_decl(
            const ast::Node& n, const std::string& name) {
        auto decl = std::make_unique<cpp2_transpiler::FunctionDeclaration>(name, line_of(n));

        const ast::Node* suffix = find_child(n, ast::NodeKind::FunctionSuffix);
        if (!suffix) return decl;

        // Parse parameter list
        if (const ast::Node* params = find_child(*suffix, ast::NodeKind::ParamList)) {
            for (const auto& param : tree_.children(*params)) {
                if (param.kind == ast::NodeKind::Parameter) {
                    decl->parameters.push_back(convert_parameter(param));
                }
            }
        }

        // Parse return spec
        if (const ast::Node* ret = find_child(*suffix, ast::NodeKind::ReturnSpec)) {
            if (const ast::Node* type = find_child(*ret, ast::NodeKind::TypeSpecifier)) {
                decl->return_type = convert_type(*type);
            }
        }

        // Parse body
        if (const ast::Node* body = find_child(*suffix, ast::NodeKind::FunctionBody)) {
            // Check for block statement or expression body
            if (const ast::Node* block = find_child(*body, ast::NodeKind::BlockStatement)) {
                decl->body = convert_block_stmt(*block);
            } else if (const ast::Node* expr = find_child(*body, ast::NodeKind::Expression)) {
                // = expr; form - convert to return statement
                auto ret_stmt = std::make_unique<cpp2_transpiler::ReturnStatement>(
                    convert_expression(*expr), line_of(*body));
                auto block_stmt = std::make_unique<cpp2_transpiler::BlockStatement>(line_of(*body));
                block_stmt->statements.push_back(std::move(ret_stmt));
                decl->body = std::move(block_stmt);
            }
        }

        return decl;
    }

    std::unique_ptr<cpp2_transpiler::VariableDeclaration> convert_variable_decl(
            const ast::Node& n, const std::string& name) {
        auto decl = std::make_unique<cpp2_transpiler::VariableDeclaration>(name, line_of(n));

        // Check for suffix-style: name: type = init;
        if (const ast::Node* suffix = find_child(n, ast::NodeKind::VariableSuffix)) {
            if (const ast::Node* type = find_child(*suffix, ast::NodeKind::TypeSpecifier)) {
                decl->type = convert_type(*type);
            }
            if (const ast::Node* init = find_child(*suffix, ast::NodeKind::Expression)) {
                decl->initializer = convert_expression(*init);
            }
        } else {
            // := style: name := expr;
            auto children = tree_.children(n);
            for (const auto& child : children) {
                if (child.kind == ast::NodeKind::Expression) {
                    decl->initializer = convert_expression(child);
                    // Type is auto-deduced
                    decl->type = std::make_unique<cpp2_transpiler::Type>(
                        cpp2_transpiler::Type::Kind::Auto);
                    break;
                }
            }
        }

        return decl;
    }

    std::unique_ptr<cpp2_transpiler::TypeDeclaration> convert_type_decl(
            const ast::Node& n, const std::string& name) {
        auto decl = std::make_unique<cpp2_transpiler::TypeDeclaration>(
            name, cpp2_transpiler::TypeDeclaration::TypeKind::Struct, line_of(n));

        if (const ast::Node* suffix = find_child(n, ast::NodeKind::TypeSuffix)) {
            if (const ast::Node* body = find_child(*suffix, ast::NodeKind::TypeBody)) {
                for (const auto& member : tree_.children(*body)) {
                    if (member.kind == ast::NodeKind::Declaration) {
                        if (auto mem_decl = convert_declaration(member)) {
                            decl->members.push_back(std::move(mem_decl));
                        }
                    }
                }
            }
        }

        return decl;
    }

    std::unique_ptr<cpp2_transpiler::NamespaceDeclaration> convert_namespace_decl(
            const ast::Node& n, const std::string& name) {
        auto decl = std::make_unique<cpp2_transpiler::NamespaceDeclaration>(name, line_of(n));

        if (const ast::Node* suffix = find_child(n, ast::NodeKind::NamespaceSuffix)) {
            if (const ast::Node* body = find_child(*suffix, ast::NodeKind::NamespaceBody)) {
                for (const auto& member : tree_.children(*body)) {
                    if (member.kind == ast::NodeKind::Declaration) {
                        if (auto mem_decl = convert_declaration(member)) {
                            decl->members.push_back(std::move(mem_decl));
                        }
                    }
                }
            }
        }

        return decl;
    }

    cpp2_transpiler::FunctionDeclaration::Parameter convert_parameter(const ast::Node& n) {
        cpp2_transpiler::FunctionDeclaration::Parameter param;

        auto children = tree_.children(n);
        for (const auto& child : children) {
            if (child.kind == ast::NodeKind::Identifier) {
                param.name = std::string(lexeme_of(child));
            } else if (child.kind == ast::NodeKind::TypeSpecifier) {
                param.type = convert_type(child);
            } else if (child.kind == ast::NodeKind::Expression) {
                param.default_value = convert_expression(child);
            } else if (child.kind == ast::NodeKind::ParamQualifier) {
                // Convert qualifier
                auto lex = lexeme_of(child);
                if (lex == "in") param.qualifiers.push_back(cpp2_transpiler::ParameterQualifier::In);
                else if (lex == "out") param.qualifiers.push_back(cpp2_transpiler::ParameterQualifier::Out);
                else if (lex == "inout") param.qualifiers.push_back(cpp2_transpiler::ParameterQualifier::InOut);
                else if (lex == "move") param.qualifiers.push_back(cpp2_transpiler::ParameterQualifier::Move);
                else if (lex == "forward") param.qualifiers.push_back(cpp2_transpiler::ParameterQualifier::Forward);
            }
        }

        return param;
    }

    // ========================================================================
    // Type Conversion
    // ========================================================================

    std::unique_ptr<cpp2_transpiler::Type> convert_type(const ast::Node& n) {
        auto type = std::make_unique<cpp2_transpiler::Type>(cpp2_transpiler::Type::Kind::UserDefined);

        // Collect type name from tokens
        std::string type_name;
        for (std::size_t i = n.token_start; i < n.token_end && i < tokens_.size(); ++i) {
            if (!type_name.empty() && tokens_[i].lexeme != "*" && tokens_[i].lexeme != "&") {
                type_name += " ";
            }
            type_name += tokens_[i].lexeme;
        }

        // Check for builtin types
        if (type_name == "auto" || type_name == "_") {
            type->kind = cpp2_transpiler::Type::Kind::Auto;
        } else if (type_name == "int" || type_name == "i32" || type_name == "i64" ||
                   type_name == "uint" || type_name == "u32" || type_name == "u64" ||
                   type_name == "float" || type_name == "double" || type_name == "bool" ||
                   type_name == "char" || type_name == "void") {
            type->kind = cpp2_transpiler::Type::Kind::Builtin;
        }

        type->name = std::move(type_name);
        return type;
    }

    // ========================================================================
    // Statement Conversion
    // ========================================================================

    std::unique_ptr<cpp2_transpiler::Statement> convert_statement(const ast::Node& n) {
        switch (n.kind) {
            case ast::NodeKind::BlockStatement:
                return convert_block_stmt(n);
            case ast::NodeKind::ReturnStatement:
                return convert_return_stmt(n);
            case ast::NodeKind::IfStatement:
                return convert_if_stmt(n);
            case ast::NodeKind::WhileStatement:
                return convert_while_stmt(n);
            case ast::NodeKind::ForStatement:
                return convert_for_stmt(n);
            case ast::NodeKind::ExpressionStatement:
                return convert_expr_stmt(n);
            case ast::NodeKind::Statement:
                // Generic statement - check children
                for (const auto& child : tree_.children(n)) {
                    if (auto stmt = convert_statement(child)) {
                        return stmt;
                    }
                }
                return nullptr;
            default:
                return nullptr;
        }
    }

    std::unique_ptr<cpp2_transpiler::BlockStatement> convert_block_stmt(const ast::Node& n) {
        auto block = std::make_unique<cpp2_transpiler::BlockStatement>(line_of(n));

        for (const auto& child : tree_.children(n)) {
            if (child.kind == ast::NodeKind::Statement) {
                if (auto stmt = convert_statement(child)) {
                    block->statements.push_back(std::move(stmt));
                }
            }
        }

        return block;
    }

    std::unique_ptr<cpp2_transpiler::ReturnStatement> convert_return_stmt(const ast::Node& n) {
        std::unique_ptr<cpp2_transpiler::Expression> value;

        if (const ast::Node* expr = find_child(n, ast::NodeKind::Expression)) {
            value = convert_expression(*expr);
        }

        return std::make_unique<cpp2_transpiler::ReturnStatement>(std::move(value), line_of(n));
    }

    std::unique_ptr<cpp2_transpiler::IfStatement> convert_if_stmt(const ast::Node& n) {
        auto children = tree_.children(n);

        std::unique_ptr<cpp2_transpiler::Expression> cond;
        std::unique_ptr<cpp2_transpiler::Statement> then_stmt;
        std::unique_ptr<cpp2_transpiler::Statement> else_stmt;

        bool found_cond = false;
        for (const auto& child : children) {
            if (child.kind == ast::NodeKind::Expression && !found_cond) {
                cond = convert_expression(child);
                found_cond = true;
            } else if (child.kind == ast::NodeKind::BlockStatement) {
                if (!then_stmt) {
                    then_stmt = convert_block_stmt(child);
                } else {
                    else_stmt = convert_block_stmt(child);
                }
            }
        }

        return std::make_unique<cpp2_transpiler::IfStatement>(
            std::move(cond), std::move(then_stmt), std::move(else_stmt), line_of(n));
    }

    std::unique_ptr<cpp2_transpiler::WhileStatement> convert_while_stmt(const ast::Node& n) {
        auto children = tree_.children(n);

        std::unique_ptr<cpp2_transpiler::Expression> cond;
        std::unique_ptr<cpp2_transpiler::Statement> body;

        for (const auto& child : children) {
            if (child.kind == ast::NodeKind::Expression && !cond) {
                cond = convert_expression(child);
            } else if (child.kind == ast::NodeKind::BlockStatement) {
                body = convert_block_stmt(child);
            }
        }

        return std::make_unique<cpp2_transpiler::WhileStatement>(
            std::move(cond), std::move(body), line_of(n));
    }

    std::unique_ptr<cpp2_transpiler::ForRangeStatement> convert_for_stmt(const ast::Node& n) {
        auto children = tree_.children(n);

        std::string var;
        std::unique_ptr<cpp2_transpiler::Expression> range;
        std::unique_ptr<cpp2_transpiler::Statement> body;

        for (const auto& child : children) {
            if (child.kind == ast::NodeKind::Identifier && var.empty()) {
                var = std::string(lexeme_of(child));
            } else if (child.kind == ast::NodeKind::Expression && !range) {
                range = convert_expression(child);
            } else if (child.kind == ast::NodeKind::BlockStatement) {
                body = convert_block_stmt(child);
            }
        }

        return std::make_unique<cpp2_transpiler::ForRangeStatement>(
            std::move(var), nullptr, std::move(range), std::move(body), line_of(n));
    }

    std::unique_ptr<cpp2_transpiler::ExpressionStatement> convert_expr_stmt(const ast::Node& n) {
        if (const ast::Node* expr = find_child(n, ast::NodeKind::Expression)) {
            return std::make_unique<cpp2_transpiler::ExpressionStatement>(
                convert_expression(*expr), line_of(n));
        }
        return nullptr;
    }

    // ========================================================================
    // Expression Conversion
    // ========================================================================

    std::unique_ptr<cpp2_transpiler::Expression> convert_expression(const ast::Node& n) {
        switch (n.kind) {
            case ast::NodeKind::Identifier:
                return convert_identifier(n);
            case ast::NodeKind::Literal:
                return convert_literal(n);
            case ast::NodeKind::PrimaryExpression:
                return convert_primary_expr(n);
            case ast::NodeKind::PostfixExpression:
                return convert_postfix_expr(n);
            case ast::NodeKind::PrefixExpression:
                return convert_prefix_expr(n);
            case ast::NodeKind::AssignmentExpression:
            case ast::NodeKind::LogicalOrExpression:
            case ast::NodeKind::LogicalAndExpression:
            case ast::NodeKind::BitwiseOrExpression:
            case ast::NodeKind::BitwiseXorExpression:
            case ast::NodeKind::BitwiseAndExpression:
            case ast::NodeKind::EqualityExpression:
            case ast::NodeKind::ComparisonExpression:
            case ast::NodeKind::ShiftExpression:
            case ast::NodeKind::AdditiveExpression:
            case ast::NodeKind::MultiplicativeExpression:
                return convert_binary_expr(n);
            case ast::NodeKind::TernaryExpression:
                return convert_ternary_expr(n);
            case ast::NodeKind::PipelineExpression:
                return convert_pipeline_expr(n);
            case ast::NodeKind::GroupedExpression:
                // Unwrap grouped expression
                for (const auto& child : tree_.children(n)) {
                    if (auto expr = convert_expression(child)) {
                        return expr;
                    }
                }
                return nullptr;
            case ast::NodeKind::Expression:
                // Generic expression - unwrap
                for (const auto& child : tree_.children(n)) {
                    if (auto expr = convert_expression(child)) {
                        return expr;
                    }
                }
                return nullptr;
            default:
                return nullptr;
        }
    }

    std::unique_ptr<cpp2_transpiler::IdentifierExpression> convert_identifier(const ast::Node& n) {
        return std::make_unique<cpp2_transpiler::IdentifierExpression>(
            std::string(lexeme_of(n)), line_of(n));
    }

    std::unique_ptr<cpp2_transpiler::LiteralExpression> convert_literal(const ast::Node& n) {
        auto lex = lexeme_of(n);

        // Check token type for literal classification
        if (n.token_start < tokens_.size()) {
            const auto& tok = tokens_[n.token_start];
            switch (tok.type) {
                case cpp2_transpiler::TokenType::IntegerLiteral:
                    return std::make_unique<cpp2_transpiler::LiteralExpression>(
                        static_cast<int64_t>(std::stoll(std::string(lex))), line_of(n));
                case cpp2_transpiler::TokenType::FloatLiteral:
                    return std::make_unique<cpp2_transpiler::LiteralExpression>(
                        std::stod(std::string(lex)), line_of(n));
                case cpp2_transpiler::TokenType::StringLiteral:
                    return std::make_unique<cpp2_transpiler::LiteralExpression>(
                        std::string(lex), line_of(n));
                case cpp2_transpiler::TokenType::CharacterLiteral:
                    return std::make_unique<cpp2_transpiler::LiteralExpression>(
                        lex.size() > 2 ? lex[1] : '\0', line_of(n));
                default:
                    break;
            }
        }

        // Default: treat as string
        if (lex == "true") {
            return std::make_unique<cpp2_transpiler::LiteralExpression>(true, line_of(n));
        } else if (lex == "false") {
            return std::make_unique<cpp2_transpiler::LiteralExpression>(false, line_of(n));
        }
        return std::make_unique<cpp2_transpiler::LiteralExpression>(std::string(lex), line_of(n));
    }

    std::unique_ptr<cpp2_transpiler::Expression> convert_primary_expr(const ast::Node& n) {
        auto children = tree_.children(n);
        if (children.empty()) {
            // Single token primary expression
            auto lex = lexeme_of(n);
            if (lex == "this" || lex == "that" || lex == "_") {
                return std::make_unique<cpp2_transpiler::IdentifierExpression>(
                    std::string(lex), line_of(n));
            }
            // Check if it's a literal
            if (n.token_start < tokens_.size()) {
                const auto& tok = tokens_[n.token_start];
                if (tok.type == cpp2_transpiler::TokenType::Identifier) {
                    return std::make_unique<cpp2_transpiler::IdentifierExpression>(
                        std::string(lex), line_of(n));
                }
            }
            // Try as literal
            return convert_literal(n);
        }

        // Has children - delegate
        for (const auto& child : children) {
            if (auto expr = convert_expression(child)) {
                return expr;
            }
        }
        return nullptr;
    }

    std::unique_ptr<cpp2_transpiler::Expression> convert_postfix_expr(const ast::Node& n) {
        auto children = tree_.children(n);
        if (children.empty()) return convert_primary_expr(n);

        // Start with primary expression
        std::unique_ptr<cpp2_transpiler::Expression> result;
        for (const auto& child : children) {
          if (child.kind == ast::NodeKind::PrimaryExpression || child.kind == ast::NodeKind::Identifier ||
              ast::NodeKind::Literal == child.kind) {
            result = convert_expression(child);
            break;
          }
        }

        if (!result) return nullptr;

        // Apply postfix operations
        for (const auto& child : children) {
            if (child.kind == ast::NodeKind::CallOp) {
                auto call = std::make_unique<cpp2_transpiler::CallExpression>(
                    std::move(result), line_of(child));
                // Parse arguments
                for (const auto& arg : tree_.children(child)) {
                    if (arg.kind == ast::NodeKind::Expression) {
                        call->args.push_back(convert_expression(arg));
                    }
                }
                result = std::move(call);
            } else if (child.kind == ast::NodeKind::MemberOp) {
                std::string member;
                for (const auto& m : tree_.children(child)) {
                    if (m.kind == ast::NodeKind::Identifier) {
                        member = std::string(lexeme_of(m));
                        break;
                    }
                }
                result = std::make_unique<cpp2_transpiler::MemberAccessExpression>(
                    std::move(result), std::move(member), line_of(child));
            } else if (child.kind == ast::NodeKind::SubscriptOp) {
                std::unique_ptr<cpp2_transpiler::Expression> index;
                for (const auto& i : tree_.children(child)) {
                    if (i.kind == ast::NodeKind::Expression) {
                        index = convert_expression(i);
                        break;
                    }
                }
                result = std::make_unique<cpp2_transpiler::SubscriptExpression>(
                    std::move(result), std::move(index), line_of(child));
            } else if (child.kind == ast::NodeKind::PostfixOp) {
                auto op_lex = lexeme_of(child);
                cpp2_transpiler::TokenType op = cpp2_transpiler::TokenType::Unknown;
                if (op_lex == "++") op = cpp2_transpiler::TokenType::PlusPlus;
                else if (op_lex == "--") op = cpp2_transpiler::TokenType::MinusMinus;
                result = std::make_unique<cpp2_transpiler::UnaryExpression>(
                    op, std::move(result), line_of(child), true);
            }
        }

        return result;
    }

    std::unique_ptr<cpp2_transpiler::Expression> convert_prefix_expr(const ast::Node& n) {
        auto children = tree_.children(n);

        // Collect prefix operators
        std::vector<cpp2_transpiler::TokenType> ops;
        std::unique_ptr<cpp2_transpiler::Expression> operand;

        for (const auto& child : children) {
            if (child.kind == ast::NodeKind::PrefixOp) {
                auto op_lex = lexeme_of(child);
                cpp2_transpiler::TokenType op = cpp2_transpiler::TokenType::Unknown;
                if (op_lex == "+") op = cpp2_transpiler::TokenType::Plus;
                else if (op_lex == "-") op = cpp2_transpiler::TokenType::Minus;
                else if (op_lex == "!") op = cpp2_transpiler::TokenType::Exclamation;
                else if (op_lex == "~") op = cpp2_transpiler::TokenType::Tilde;
                else if (op_lex == "++") op = cpp2_transpiler::TokenType::PlusPlus;
                else if (op_lex == "--") op = cpp2_transpiler::TokenType::MinusMinus;
                else if (op_lex == "&") op = cpp2_transpiler::TokenType::Ampersand;
                else if (op_lex == "*") op = cpp2_transpiler::TokenType::Asterisk;
                ops.push_back(op);
            } else if (child.kind == ast::NodeKind::PostfixExpression) {
                operand = convert_postfix_expr(child);
            }
        }

        if (!operand) {
            // No postfix expr found, try converting first non-prefix child
            for (const auto& child : children) {
                if (child.kind != ast::NodeKind::PrefixOp) {
                    operand = convert_expression(child);
                    if (operand) break;
                }
            }
        }

        if (!operand) return nullptr;

        // Apply prefix operators in reverse order
        for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
            operand = std::make_unique<cpp2_transpiler::UnaryExpression>(
                *it, std::move(operand), line_of(n), false);
        }

        return operand;
    }

    std::unique_ptr<cpp2_transpiler::Expression> convert_binary_expr(const ast::Node& n) {
        auto children = tree_.children(n);
        if (children.empty()) return nullptr;

        // Binary expression: left op right [op right...]
        std::unique_ptr<cpp2_transpiler::Expression> result;
        cpp2_transpiler::TokenType pending_op = cpp2_transpiler::TokenType::Unknown;

        for (const auto& child : children) {
            // Check if this is an operator token
            if (child.token_count() == 1) {
                auto lex = lexeme_of(child);
                cpp2_transpiler::TokenType op = token_type_from_lexeme(lex);
                if (op != cpp2_transpiler::TokenType::Unknown) {
                    pending_op = op;
                    continue;
                }
            }

            // Convert operand
            auto operand = convert_expression(child);
            if (!operand) continue;

            if (!result) {
                result = std::move(operand);
            } else if (pending_op != cpp2_transpiler::TokenType::Unknown) {
                result = std::make_unique<cpp2_transpiler::BinaryExpression>(
                    std::move(result), pending_op, std::move(operand), line_of(n));
                pending_op = cpp2_transpiler::TokenType::Unknown;
            }
        }

        return result;
    }

    std::unique_ptr<cpp2_transpiler::Expression> convert_ternary_expr(const ast::Node& n) {
        auto children = tree_.children(n);

        std::unique_ptr<cpp2_transpiler::Expression> cond;
        std::unique_ptr<cpp2_transpiler::Expression> then_expr;
        std::unique_ptr<cpp2_transpiler::Expression> else_expr;

        int expr_count = 0;
        for (const auto& child : children) {
            if (ast::meta::is_expression(child.kind)) {
                auto expr = convert_expression(child);
                if (!expr) continue;

                if (expr_count == 0) cond = std::move(expr);
                else if (expr_count == 1) then_expr = std::move(expr);
                else if (expr_count == 2) else_expr = std::move(expr);
                expr_count++;
            }
        }

        if (cond && then_expr && else_expr) {
            return std::make_unique<cpp2_transpiler::TernaryExpression>(
                std::move(cond), std::move(then_expr), std::move(else_expr), line_of(n));
        }

        // Not a full ternary, return condition
        return cond;
    }

    std::unique_ptr<cpp2_transpiler::Expression> convert_pipeline_expr(const ast::Node& n) {
        auto children = tree_.children(n);

        std::unique_ptr<cpp2_transpiler::Expression> result;

        for (const auto& child : children) {
            if (ast::meta::is_expression(child.kind)) {
                auto expr = convert_expression(child);
                if (!expr) continue;

                if (!result) {
                    result = std::move(expr);
                } else {
                    result = std::make_unique<cpp2_transpiler::PipelineExpression>(
                        std::move(result), std::move(expr), line_of(n));
                }
            }
        }

        return result;
    }

    // ========================================================================
    // Utility
    // ========================================================================

    static cpp2_transpiler::TokenType token_type_from_lexeme(std::string_view lex){
        // Binary operators
        if (lex == "+") return cpp2_transpiler::TokenType::Plus;
        if (lex == "-") return cpp2_transpiler::TokenType::Minus;
        if (lex == "*") return cpp2_transpiler::TokenType::Asterisk;
        if (lex == "/") return cpp2_transpiler::TokenType::Slash;
        if (lex == "%") return cpp2_transpiler::TokenType::Percent;
        if (lex == "==") return cpp2_transpiler::TokenType::DoubleEqual;
        if (lex == "!=") return cpp2_transpiler::TokenType::NotEqual;
        if (lex == "<") return cpp2_transpiler::TokenType::LessThan;
        if (lex == ">") return cpp2_transpiler::TokenType::GreaterThan;
        if (lex == "<=") return cpp2_transpiler::TokenType::LessThanOrEqual;
        if (lex == ">=") return cpp2_transpiler::TokenType::GreaterThanOrEqual;
        if (lex == "<=>") return cpp2_transpiler::TokenType::Spaceship;
        if (lex == "&&") return cpp2_transpiler::TokenType::DoubleAmpersand;
        if (lex == "||") return cpp2_transpiler::TokenType::DoublePipe;
        if (lex == "&") return cpp2_transpiler::TokenType::Ampersand;
        if (lex == "|") return cpp2_transpiler::TokenType::Pipe;
        if (lex == "^") return cpp2_transpiler::TokenType::Caret;
        if (lex == "<<") return cpp2_transpiler::TokenType::LeftShift;
        if (lex == ">>") return cpp2_transpiler::TokenType::RightShift;
        if (lex == "=") return cpp2_transpiler::TokenType::Equal;
        if (lex == "+=") return cpp2_transpiler::TokenType::PlusEqual;
        if (lex == "-=") return cpp2_transpiler::TokenType::MinusEqual;
        if (lex == "*=") return cpp2_transpiler::TokenType::AsteriskEqual;
        if (lex == "/=") return cpp2_transpiler::TokenType::SlashEqual;
        if (lex == "%=") return cpp2_transpiler::TokenType::PercentEqual;
        if (lex == "|>") return cpp2_transpiler::TokenType::Pipeline;
        return cpp2_transpiler::TokenType::Unknown;
    }
};

// Convenience function
inline std::unique_ptr<cpp2_transpiler::AST> convert_to_ast(const ast::ParseTree& tree) {
    return ParseTreeConverter(tree).convert();
}

} // namespace cpp2::parser
