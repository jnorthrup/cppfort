#include "semantic_hash.hpp"
#include "ast.hpp"

namespace cppfort::crdt {

// Visitor for computing semantic hashes from cpp2 AST
class Cpp2SemanticHashVisitor : public SemanticHashVisitor {
public:
    SHA256Hash visit(const void* node) override {
        if (!node) return SHA256Hash::compute("");

        // Use RTTI or kind field to determine node type
        const auto* expr = static_cast<const cpp2_transpiler::Expression*>(node);
        if (expr) {
            return visit_expression(expr);
        }

        const auto* stmt = static_cast<const cpp2_transpiler::Statement*>(node);
        if (stmt) {
            return visit_statement(stmt);
        }

        const auto* decl = static_cast<const cpp2_transpiler::Declaration*>(node);
        if (decl) {
            return visit_declaration(decl);
        }

        return SHA256Hash::compute("unknown");
    }

    std::optional<std::string> map_cpp2_to_clang(const std::string& cpp2_kind) override {
        auto mappings = SemanticMapping::get_builtin_mappings();
        auto it = mappings.find(cpp2_kind);
        if (it != mappings.end()) {
            return it->second.clang_kind;
        }
        return std::nullopt;
    }

    std::optional<std::string> map_clang_to_cpp2(const std::string& clang_kind) override {
        auto mappings = SemanticMapping::get_builtin_mappings();
        for (const auto& [cpp2_kind, mapping] : mappings) {
            if (mapping.clang_kind == clang_kind) {
                return cpp2_kind;
            }
        }
        return std::nullopt;
    }

private:
    SHA256Hash visit_expression(const cpp2_transpiler::Expression* expr) {
        SemanticContentBuilder builder;

        builder.add("Expression");
        builder.add(static_cast<int>(expr->kind));
        builder.add(expr->line);

        switch (expr->kind) {
            case cpp2_transpiler::Expression::Kind::Literal: {
                const auto* lit = static_cast<const cpp2_transpiler::LiteralExpression*>(expr);
                builder.add("Literal");
                std::visit([&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, int64_t>) {
                        builder.add("int64");
                        builder.add(value);
                    } else if constexpr (std::is_same_v<T, double>) {
                        builder.add("double");
                        builder.add(value);
                    } else if constexpr (std::is_same_v<T, bool>) {
                        builder.add("bool");
                        builder.add(static_cast<int>(value));
                    } else if constexpr (std::is_same_v<T, std::string>) {
                        builder.add("string");
                        builder.add(value);
                    } else if constexpr (std::is_same_v<T, char>) {
                        builder.add("char");
                        builder.add(static_cast<int>(value));
                    }
                }, lit->value);
                break;
            }

            case cpp2_transpiler::Expression::Kind::Identifier: {
                const auto* ident = static_cast<const cpp2_transpiler::IdentifierExpression*>(expr);
                builder.add("Identifier");
                builder.add(ident->name);
                break;
            }

            case cpp2_transpiler::Expression::Kind::Binary: {
                const auto* bin = static_cast<const cpp2_transpiler::BinaryExpression*>(expr);
                builder.add("Binary");
                builder.add(static_cast<int>(bin->op));
                // Child hashes would be added by the context
                break;
            }

            case cpp2_transpiler::Expression::Kind::Call: {
                const auto* call = static_cast<const cpp2_transpiler::CallExpression*>(expr);
                builder.add("Call");
                builder.add(call->is_ufcs ? "ufcs" : "regular");
                break;
            }

            case cpp2_transpiler::Expression::Kind::MemberAccess: {
                const auto* access = static_cast<const cpp2_transpiler::MemberAccessExpression*>(expr);
                builder.add("MemberAccess");
                builder.add(access->member);
                break;
            }

            default:
                builder.add("UnknownExpression");
                break;
        }

        return SHA256Hash::compute(builder.build());
    }

    SHA256Hash visit_statement(const cpp2_transpiler::Statement* stmt) {
        SemanticContentBuilder builder;

        builder.add("Statement");
        builder.add(static_cast<int>(stmt->kind));
        builder.add(stmt->line);

        switch (stmt->kind) {
            case cpp2_transpiler::Statement::Kind::Return: {
                const auto* ret = static_cast<const cpp2_transpiler::ReturnStatement*>(stmt);
                builder.add("Return");
                break;
            }

            case cpp2_transpiler::Statement::Kind::If: {
                const auto* if_stmt = static_cast<const cpp2_transpiler::IfStatement*>(stmt);
                builder.add("If");
                break;
            }

            case cpp2_transpiler::Statement::Kind::While: {
                const auto* while_stmt = static_cast<const cpp2_transpiler::WhileStatement*>(stmt);
                builder.add("While");
                break;
            }

            case cpp2_transpiler::Statement::Kind::For: {
                const auto* for_stmt = static_cast<const cpp2_transpiler::ForStatement*>(stmt);
                builder.add("For");
                break;
            }

            case cpp2_transpiler::Statement::Kind::ForRange: {
                const auto* for_range = static_cast<const cpp2_transpiler::ForRangeStatement*>(stmt);
                builder.add("ForRange");
                builder.add(for_range->variable);
                break;
            }

            case cpp2_transpiler::Statement::Kind::Block: {
                const auto* block = static_cast<const cpp2_transpiler::BlockStatement*>(stmt);
                builder.add("Block");
                break;
            }

            default:
                builder.add("UnknownStatement");
                break;
        }

        return SHA256Hash::compute(builder.build());
    }

    SHA256Hash visit_declaration(const cpp2_transpiler::Declaration* decl) {
        SemanticContentBuilder builder;

        builder.add("Declaration");
        builder.add(static_cast<int>(decl->kind));
        builder.add(decl->name);
        builder.add(decl->line);

        switch (decl->kind) {
            case cpp2_transpiler::Declaration::Kind::Function: {
                const auto* func = static_cast<const cpp2_transpiler::FunctionDeclaration*>(decl);
                builder.add("Function");
                builder.add(func->name);

                // Hash parameter types and qualifiers
                for (const auto& param : func->parameters) {
                    builder.add(param.name);
                    if (param.type) {
                        builder.add(serialize_type(param.type.get()));
                    }
                    for (auto qual : param.qualifiers) {
                        builder.add(static_cast<int>(qual));
                    }
                }

                // Hash return type
                if (func->return_type) {
                    builder.add(serialize_type(func->return_type.get()));
                }

                break;
            }

            case cpp2_transpiler::Declaration::Kind::Variable: {
                const auto* var = static_cast<const cpp2_transpiler::VariableDeclaration*>(decl);
                builder.add("Variable");
                builder.add(var->name);

                if (var->type) {
                    builder.add(serialize_type(var->type.get()));
                }

                builder.add(var->is_const ? "const" : "mut");
                builder.add(var->is_mut ? "mut" : "immutable");

                for (auto qual : var->qualifiers) {
                    builder.add(static_cast<int>(qual));
                }

                break;
            }

            case cpp2_transpiler::Declaration::Kind::Type: {
                const auto* type_decl = static_cast<const cpp2_transpiler::TypeDeclaration*>(decl);
                builder.add("Type");
                builder.add(type_decl->name);
                builder.add(static_cast<int>(type_decl->type_kind));
                break;
            }

            default:
                builder.add("UnknownDeclaration");
                break;
        }

        return SHA256Hash::compute(builder.build());
    }

    std::string serialize_type(const cpp2_transpiler::Type* type) {
        if (!type) return "null";

        SemanticContentBuilder builder;
        builder.add(static_cast<int>(type->kind));
        builder.add(type->name);
        builder.add(type->is_const ? "const" : "mutable");
        builder.add(type->is_mut ? "mut" : "immutable");

        switch (type->kind) {
            case cpp2_transpiler::Type::Kind::Pointer:
                if (type->pointee) {
                    builder.add(serialize_type(type->pointee.get()));
                }
                break;
            case cpp2_transpiler::Type::Kind::Template:
                for (const auto& arg : type->template_args) {
                    builder.add(serialize_type(arg.get()));
                }
                break;
            default:
                break;
        }

        return builder.build();
    }
};

// Factory function to create a cpp2 semantic hash visitor
std::unique_ptr<SemanticHashVisitor> create_cpp2_visitor() {
    return std::make_unique<Cpp2SemanticHashVisitor>();
}

} // namespace cppfort::crdt
