#include "clang_ast_reverse.hpp"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/ASTUnit.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <chrono>

namespace cppfort::crdt {

// ClangToCpp2Visitor implementation
ClangToCpp2Visitor::ClangToCpp2Visitor(clang::ASTContext& ctx)
    : context_(ctx) {}

std::string ClangToCpp2Visitor::get_type_string(const clang::QualType& qual_type) {
    if (qual_type.isNull()) return "auto";

    clang::PrintingPolicy policy(context_.getLangOpts());
    policy.SuppressScope = false;
    return qual_type.getAsString(policy);
}

std::unique_ptr<cpp2_transpiler::Type> ClangToCpp2Visitor::make_type(const clang::QualType& qual_type) {
    auto type = std::make_unique<cpp2_transpiler::Type>(cpp2_transpiler::Type::Kind::Builtin);
    type->name = get_type_string(qual_type);
    type->is_const = qual_type.isConstQualified();
    return type;
}

// Parameter qualifier inference implementation
ParameterQualifierInference::Qualifier
ParameterQualifierInference::infer_from_clang_type(const clang::QualType& clang_type,
                                                     bool is_template_context) {
    const clang::Type* type = clang_type.getTypePtr();

    // RValue reference (T&&) - could be move or forwarding
    if (type->isRValueReferenceType()) {
        return is_template_context ? Qualifier::Forward : Qualifier::Move;
    }

    // LValue reference (T&) - could be inout or out
    if (type->isLValueReferenceType()) {
        // Default to inout; out requires dataflow analysis
        return Qualifier::InOut;
    }

    // Non-reference - could be in (const) or regular value
    if (clang_type.isConstQualified()) {
        return Qualifier::In;
    }

    return Qualifier::None;
}

std::string ParameterQualifierInference::qualifier_to_cpp2(Qualifier q) {
    switch (q) {
        case Qualifier::InOut: return "inout";
        case Qualifier::Out: return "out";
        case Qualifier::Move: return "move";
        case Qualifier::Forward: return "forward";
        case Qualifier::In: return "in";
        case Qualifier::None: return "";
    }
    return "";
}

ReverseMappingResult ClangToCpp2Visitor::convert(const void* clang_node) {
    ReverseMappingResult result;

    if (!clang_node) return result;

    // Try Statement first
    if (auto stmt = static_cast<const clang::Stmt*>(clang_node)) {
        result.stmt = visit_statement(stmt);
    }
    // Then Expression (subclass of Stmt)
    else if (auto expr = static_cast<const clang::Expr*>(clang_node)) {
        result.expr = visit_expression(expr);
    }
    // Then Declaration
    else if (auto decl = static_cast<const clang::Decl*>(clang_node)) {
        result.decl = visit_declaration(decl);
    }

    // Generate inferred cpp2 syntax
    result.inferred_cpp2_syntax = generate_cpp2_syntax(result);

    // Compute semantic hash
    result.semantic_hash = compute_semantic_hash(clang_node);

    return result;
}

std::unique_ptr<cpp2_transpiler::Statement>
ClangToCpp2Visitor::visit_statement(const clang::Stmt* stmt) {
    if (!stmt) return nullptr;

    // Handle different statement types
    if (stmt->getStmtClass() == clang::Stmt::ReturnStmtClass) {
        return visit_return_statement(stmt);
    } else if (stmt->getStmtClass() == clang::Stmt::IfStmtClass) {
        return visit_if_statement(stmt);
    } else if (stmt->getStmtClass() == clang::Stmt::WhileStmtClass) {
        return visit_while_statement(stmt);
    } else if (stmt->getStmtClass() == clang::Stmt::ForStmtClass) {
        return visit_for_statement(stmt);
    } else if (stmt->getStmtClass() == clang::Stmt::CXXForRangeStmtClass) {
        return visit_for_range_statement(stmt);
    } else if (stmt->getStmtClass() == clang::Stmt::CompoundStmtClass) {
        return visit_compound_statement(stmt);
    }

    // Default: expression statement
    if (auto expr = clang::dyn_cast<const clang::Expr>(stmt)) {
        auto expr_result = visit_expression(expr);
        if (expr_result) {
            return std::make_unique<cpp2_transpiler::ExpressionStatement>(
                std::move(expr_result), stmt->getBeginLoc().getRawEncoding());
        }
    }

    return nullptr;
}

std::unique_ptr<cpp2_transpiler::ReturnStatement>
ClangToCpp2Visitor::visit_return_statement(const clang::Stmt* stmt) {
    auto ret = clang::dyn_cast<const clang::ReturnStmt>(stmt);
    if (!ret) return nullptr;

    std::unique_ptr<cpp2_transpiler::Expression> value;
    if (ret->getRetValue()) {
        value = visit_expression(ret->getRetValue());
    }

    return std::make_unique<cpp2_transpiler::ReturnStatement>(
        std::move(value), stmt->getBeginLoc().getRawEncoding());
}

std::unique_ptr<cpp2_transpiler::IfStatement>
ClangToCpp2Visitor::visit_if_statement(const clang::Stmt* stmt) {
    auto if_stmt = clang::dyn_cast<const clang::IfStmt>(stmt);
    if (!if_stmt) return nullptr;

    auto condition = visit_expression(if_stmt->getCond());
    auto then_stmt = visit_statement(if_stmt->getThen());

    std::unique_ptr<cpp2_transpiler::Statement> else_stmt;
    if (if_stmt->getElse()) {
        else_stmt = visit_statement(if_stmt->getElse());
    }

    return std::make_unique<cpp2_transpiler::IfStatement>(
        std::move(condition), std::move(then_stmt), std::move(else_stmt),
        stmt->getBeginLoc().getRawEncoding());
}

std::unique_ptr<cpp2_transpiler::WhileStatement>
ClangToCpp2Visitor::visit_while_statement(const clang::Stmt* stmt) {
    auto while_stmt = clang::dyn_cast<const clang::WhileStmt>(stmt);
    if (!while_stmt) return nullptr;

    auto condition = visit_expression(while_stmt->getCond());
    auto body = visit_statement(while_stmt->getBody());

    return std::make_unique<cpp2_transpiler::WhileStatement>(
        std::move(condition), std::move(body), stmt->getBeginLoc().getRawEncoding());
}

std::unique_ptr<cpp2_transpiler::ForStatement>
ClangToCpp2Visitor::visit_for_statement(const clang::Stmt* stmt) {
    auto for_stmt = clang::dyn_cast<const clang::ForStmt>(stmt);
    if (!for_stmt) return nullptr;

    auto init = visit_statement(for_stmt->getInit());
    auto condition = for_stmt->getCond() ? visit_expression(for_stmt->getCond()) : nullptr;
    auto increment = for_stmt->getInc() ? visit_expression(for_stmt->getInc()) : nullptr;
    auto body = visit_statement(for_stmt->getBody());

    return std::make_unique<cpp2_transpiler::ForStatement>(
        std::move(init), std::move(condition), std::move(increment), std::move(body),
        stmt->getBeginLoc().getRawEncoding());
}

std::unique_ptr<cpp2_transpiler::ForRangeStatement>
ClangToCpp2Visitor::visit_for_range_statement(const clang::Stmt* stmt) {
    auto range_stmt = clang::dyn_cast<const clang::CXXForRangeStmt>(stmt);
    if (!range_stmt) return nullptr;

    // Extract loop variable
    std::string var_name;
    if (auto var_decl = range_stmt->getLoopVariable()) {
        var_name = var_decl->getNameAsString();
    }

    auto range = visit_expression(range_stmt->getRangeInit());
    auto var_type = convert_type(range_stmt->getLoopVariable()->getType().getTypePtr());
    auto body = visit_statement(range_stmt->getBody());

    return std::make_unique<cpp2_transpiler::ForRangeStatement>(
        var_name, std::move(var_type), std::move(range), std::move(body),
        stmt->getBeginLoc().getRawEncoding());
}

std::unique_ptr<cpp2_transpiler::BlockStatement>
ClangToCpp2Visitor::visit_compound_statement(const clang::Stmt* stmt) {
    auto compound = clang::dyn_cast<const clang::CompoundStmt>(stmt);
    if (!compound) return nullptr;

    auto block = std::make_unique<cpp2_transpiler::BlockStatement>(
        stmt->getBeginLoc().getRawEncoding());

    for (auto* child : compound->body()) {
        if (auto child_stmt = visit_statement(child)) {
            block->statements.push_back(std::move(child_stmt));
        }
    }

    return block;
}

std::unique_ptr<cpp2_transpiler::Expression>
ClangToCpp2Visitor::visit_expression(const clang::Expr* expr) {
    if (!expr) return nullptr;

    // Handle different expression types
    if (expr->getStmtClass() == clang::Stmt::IntegerLiteralClass ||
        expr->getStmtClass() == clang::Stmt::StringLiteralClass ||
        expr->getStmtClass() == clang::Stmt::FloatingLiteralClass ||
        expr->getStmtClass() == clang::Stmt::CharacterLiteralClass ||
        expr->getStmtClass() == clang::Stmt::CXXBoolLiteralExprClass) {
        return visit_literal(expr);
    } else if (expr->getStmtClass() == clang::Stmt::DeclRefExprClass) {
        return visit_decl_ref(expr);
    } else if (expr->getStmtClass() == clang::Stmt::BinaryOperatorClass ||
               expr->getStmtClass() == clang::Stmt::CXXOperatorCallExprClass) {
        return visit_binary_operator(expr);
    } else if (expr->getStmtClass() == clang::Stmt::CallExprClass) {
        return visit_call(expr, false);  // Non-UFCS by default
    } else if (expr->getStmtClass() == clang::Stmt::CXXMemberCallExprClass) {
        // Could be UFCS or regular member call
        return visit_call(expr, is_potential_ufcs_call(expr));
    } else if (expr->getStmtClass() == clang::Stmt::MemberExprClass) {
        return visit_member_access(expr);
    } else if (expr->getStmtClass() == clang::Stmt::UnaryOperatorClass) {
        return visit_unary_operator(expr);
    }

    return nullptr;
}

std::unique_ptr<cpp2_transpiler::LiteralExpression>
ClangToCpp2Visitor::visit_literal(const clang::Expr* expr) {
    // Handle different literal types
    if (auto int_lit = clang::dyn_cast<const clang::IntegerLiteral>(expr)) {
        return std::make_unique<cpp2_transpiler::LiteralExpression>(
            int_lit->getValue().getSExtValue(), expr->getBeginLoc().getRawEncoding());
    } else if (auto str_lit = clang::dyn_cast<const clang::StringLiteral>(expr)) {
        return std::make_unique<cpp2_transpiler::LiteralExpression>(
            str_lit->getString().str(), expr->getBeginLoc().getRawEncoding());
    } else if (auto float_lit = clang::dyn_cast<const clang::FloatingLiteral>(expr)) {
        return std::make_unique<cpp2_transpiler::LiteralExpression>(
            float_lit->getValueAsApproximateDouble(), expr->getBeginLoc().getRawEncoding());
    } else if (auto bool_lit = clang::dyn_cast<const clang::CXXBoolLiteralExpr>(expr)) {
        return std::make_unique<cpp2_transpiler::LiteralExpression>(
            bool_lit->getValue(), expr->getBeginLoc().getRawEncoding());
    } else if (auto char_lit = clang::dyn_cast<const clang::CharacterLiteral>(expr)) {
        return std::make_unique<cpp2_transpiler::LiteralExpression>(
            static_cast<char>(char_lit->getValue()), expr->getBeginLoc().getRawEncoding());
    }

    return nullptr;
}

std::unique_ptr<cpp2_transpiler::IdentifierExpression>
ClangToCpp2Visitor::visit_decl_ref(const clang::Expr* expr) {
    auto decl_ref = clang::dyn_cast<const clang::DeclRefExpr>(expr);
    if (!decl_ref) return nullptr;

    return std::make_unique<cpp2_transpiler::IdentifierExpression>(
        decl_ref->getNameInfo().getAsString().data(), expr->getBeginLoc().getRawEncoding());
}

std::unique_ptr<cpp2_transpiler::BinaryExpression>
ClangToCpp2Visitor::visit_binary_operator(const clang::Expr* expr) {
    const clang::Expr* left = nullptr;
    const clang::Expr* right = nullptr;

    if (auto bin_op = clang::dyn_cast<const clang::BinaryOperator>(expr)) {
        left = bin_op->getLHS();
        right = bin_op->getRHS();
    } else if (auto op_call = clang::dyn_cast<const clang::CXXOperatorCallExpr>(expr)) {
        // Operator call for overloaded operators
        if (op_call->getNumArgs() >= 2) {
            left = op_call->getArg(0);
            right = op_call->getArg(1);
        }
    }

    if (!left || !right) return nullptr;

    // Default to plus operator for simplicity
    // A full implementation would map all Clang opcodes to cpp2 tokens
    cpp2_transpiler::TokenType token_type = cpp2_transpiler::TokenType::Plus;

    return std::make_unique<cpp2_transpiler::BinaryExpression>(
        visit_expression(left), token_type, visit_expression(right),
        expr->getBeginLoc().getRawEncoding());
}

std::unique_ptr<cpp2_transpiler::CallExpression>
ClangToCpp2Visitor::visit_call(const clang::Expr* expr, bool is_ufcs) {
    const clang::Expr* callee = nullptr;
    std::vector<const clang::Expr*> args;

    if (auto call = clang::dyn_cast<const clang::CallExpr>(expr)) {
        callee = call->getCallee();
        for (unsigned i = 0; i < call->getNumArgs(); ++i) {
            args.push_back(call->getArg(i));
        }
    }

    if (!callee) return nullptr;

    auto call_expr = std::make_unique<cpp2_transpiler::CallExpression>(
        visit_expression(callee), expr->getBeginLoc().getRawEncoding());
    call_expr->is_ufcs = is_ufcs;

    for (auto* arg : args) {
        call_expr->args.push_back(visit_expression(arg));
    }

    return call_expr;
}

std::unique_ptr<cpp2_transpiler::MemberAccessExpression>
ClangToCpp2Visitor::visit_member_access(const clang::Expr* expr) {
    auto member = clang::dyn_cast<const clang::MemberExpr>(expr);
    if (!member) return nullptr;

    auto object = visit_expression(member->getBase());
    std::string member_name = member->getMemberDecl()->getNameAsString();

    return std::make_unique<cpp2_transpiler::MemberAccessExpression>(
        std::move(object), member_name, expr->getBeginLoc().getRawEncoding());
}

std::unique_ptr<cpp2_transpiler::UnaryExpression>
ClangToCpp2Visitor::visit_unary_operator(const clang::Expr* expr) {
    auto unary = clang::dyn_cast<const clang::UnaryOperator>(expr);
    if (!unary) return nullptr;

    // Default to minus for simplicity
    cpp2_transpiler::TokenType token_type = cpp2_transpiler::TokenType::Minus;

    bool is_postfix = unary->isPostfix();
    return std::make_unique<cpp2_transpiler::UnaryExpression>(
        token_type, visit_expression(unary->getSubExpr()),
        expr->getBeginLoc().getRawEncoding(), is_postfix);
}

std::unique_ptr<cpp2_transpiler::Declaration>
ClangToCpp2Visitor::visit_declaration(const clang::Decl* decl) {
    if (!decl) return nullptr;

    if (decl->getKind() == clang::Decl::Function) {
        return visit_function_decl(decl);
    } else if (decl->getKind() == clang::Decl::Var) {
        return visit_variable_decl(decl);
    } else if (decl->getKind() == clang::Decl::CXXRecord ||
               decl->getKind() == clang::Decl::ClassTemplateSpecialization) {
        return visit_type_decl(decl);
    }

    return nullptr;
}

std::unique_ptr<cpp2_transpiler::FunctionDeclaration>
ClangToCpp2Visitor::visit_function_decl(const clang::Decl* decl) {
    auto func = clang::dyn_cast<const clang::FunctionDecl>(decl);
    if (!func) return nullptr;

    auto func_decl = std::make_unique<cpp2_transpiler::FunctionDeclaration>(
        func->getNameAsString(), decl->getBeginLoc().getRawEncoding());

    // Convert parameters
    for (unsigned i = 0; i < func->getNumParams(); ++i) {
        auto* param = func->getParamDecl(i);
        cpp2_transpiler::FunctionDeclaration::Parameter cpp2_param;
        cpp2_param.name = param->getNameAsString();
        cpp2_param.type = convert_qual_type(param->getType());

        // Infer qualifiers from type
        auto qual = ParameterQualifierInference::infer_from_clang_type(
            param->getType(), func->isTemplated());
        if (qual != ParameterQualifierInference::Qualifier::None) {
            cpp2_param.qualifiers.push_back(
                static_cast<cpp2_transpiler::ParameterQualifier>(static_cast<int>(qual)));
        }

        func_decl->parameters.push_back(std::move(cpp2_param));
    }

    // Convert return type
    func_decl->return_type = convert_qual_type(func->getReturnType());

    // Convert body if present
    if (func->hasBody()) {
        func_decl->body = visit_statement(func->getBody());
    }

    return func_decl;
}

std::unique_ptr<cpp2_transpiler::VariableDeclaration>
ClangToCpp2Visitor::visit_variable_decl(const clang::Decl* decl) {
    auto var = clang::dyn_cast<const clang::VarDecl>(decl);
    if (!var) return nullptr;

    auto var_decl = std::make_unique<cpp2_transpiler::VariableDeclaration>(
        var->getNameAsString(), decl->getBeginLoc().getRawEncoding());

    var_decl->type = convert_qual_type(var->getType());

    if (var->hasInit()) {
        var_decl->initializer = visit_expression(var->getInit());
    }

    var_decl->is_const = var->getType().isConstQualified();

    return var_decl;
}

std::unique_ptr<cpp2_transpiler::TypeDeclaration>
ClangToCpp2Visitor::visit_type_decl(const clang::Decl* decl) {
    auto record = clang::dyn_cast<const clang::CXXRecordDecl>(decl);
    if (!record) return nullptr;

    // Simplified: default to struct
    auto type_decl = std::make_unique<cpp2_transpiler::TypeDeclaration>(
        record->getNameAsString(), cpp2_transpiler::TypeDeclaration::TypeKind::Struct,
        decl->getBeginLoc().getRawEncoding());

    // Convert members (simplified)
    for (auto* member : record->decls()) {
        if (auto member_decl = visit_declaration(member)) {
            type_decl->members.push_back(std::move(member_decl));
        }
    }

    return type_decl;
}

std::unique_ptr<cpp2_transpiler::Type>
ClangToCpp2Visitor::convert_type(const clang::Type* clang_type) {
    if (!clang_type) {
        auto type = std::make_unique<cpp2_transpiler::Type>(cpp2_transpiler::Type::Kind::Auto);
        type->name = "auto";
        return type;
    }

    clang::QualType qual_type(clang_type, 0);
    return convert_qual_type(qual_type);
}

std::unique_ptr<cpp2_transpiler::Type>
ClangToCpp2Visitor::convert_qual_type(const clang::QualType& clang_type) {
    if (clang_type.isNull()) {
        auto type = std::make_unique<cpp2_transpiler::Type>(cpp2_transpiler::Type::Kind::Auto);
        type->name = "auto";
        return type;
    }

    clang::PrintingPolicy policy(context_.getLangOpts());
    std::string type_str = clang_type.getAsString(policy);

    // Determine kind
    cpp2_transpiler::Type::Kind kind = cpp2_transpiler::Type::Kind::Builtin;
    const clang::Type* type_ptr = clang_type.getTypePtr();

    if (type_ptr->isPointerType()) {
        kind = cpp2_transpiler::Type::Kind::Pointer;
    } else if (type_ptr->isReferenceType()) {
        kind = cpp2_transpiler::Type::Kind::Reference;
    } else if (type_ptr->isStructureType() || type_ptr->isClassType()) {
        kind = cpp2_transpiler::Type::Kind::UserDefined;
    }

    auto type = std::make_unique<cpp2_transpiler::Type>(kind);
    type->name = type_str;
    type->is_const = clang_type.isConstQualified();

    return type;
}

std::string ClangToCpp2Visitor::generate_cpp2_syntax(const ReverseMappingResult& result) {
    std::ostringstream oss;

    // Generate inferred cpp2 syntax based on result type
    if (result.decl) {
        if (auto func = dynamic_cast<cpp2_transpiler::FunctionDeclaration*>(result.decl.get())) {
            // Function: name: (params) -> return_type = { body }
            oss << func->name << ": (";
            for (size_t i = 0; i < func->parameters.size(); ++i) {
                if (i > 0) oss << ", ";
                auto& param = func->parameters[i];
                // Add qualifier if present
                if (!param.qualifiers.empty()) {
                    oss << static_cast<int>(param.qualifiers[0]) << " ";
                }
                oss << param.name << ": " << (param.type ? param.type->name : "auto");
            }
            oss << ") -> " << (func->return_type ? func->return_type->name : "auto") << " = { /* body */ }";
        } else if (auto var = dynamic_cast<cpp2_transpiler::VariableDeclaration*>(result.decl.get())) {
            // Variable: name: type = value
            oss << var->name << ": " << (var->type ? var->type->name : "auto");
            if (var->initializer) {
                oss << " = /* init */";
            }
        }
    } else if (result.expr) {
        oss << "/* expression */";
    } else if (result.stmt) {
        oss << "/* statement */";
    }

    return oss.str();
}

SHA256Hash ClangToCpp2Visitor::compute_semantic_hash(const void* clang_node) {
    // Build semantic content string from Clang AST
    SemanticContentBuilder builder;

    if (auto stmt = static_cast<const clang::Stmt*>(clang_node)) {
        builder.add("Stmt");
        builder.add(static_cast<int>(stmt->getStmtClass()));
    } else if (auto decl = static_cast<const clang::Decl*>(clang_node)) {
        builder.add("Decl");
        builder.add(static_cast<int>(decl->getKind()));
        if (auto named_decl = clang::dyn_cast<const clang::NamedDecl>(decl)) {
            builder.add(named_decl->getNameAsString());
        }
    }

    return SHA256Hash::compute(builder.build());
}

bool ClangToCpp2Visitor::is_potential_ufcs_call(const clang::Expr* call_expr) {
    // Check if this could be a UFCS call (obj.method() that could be method(obj))
    if (auto member_call = clang::dyn_cast<const clang::CXXMemberCallExpr>(call_expr)) {
        // It's a member call - could be UFCS
        return true;
    }
    return false;
}

std::vector<cpp2_transpiler::FunctionDeclaration::Parameter>
ClangToCpp2Visitor::convert_parameters(const clang::Decl* function_decl) {
    std::vector<cpp2_transpiler::FunctionDeclaration::Parameter> params;

    if (auto func = clang::dyn_cast<const clang::FunctionDecl>(function_decl)) {
        for (unsigned i = 0; i < func->getNumParams(); ++i) {
            auto* param = func->getParamDecl(i);
            cpp2_transpiler::FunctionDeclaration::Parameter cpp2_param;
            cpp2_param.name = param->getNameAsString();
            cpp2_param.type = convert_qual_type(param->getType());
            params.push_back(std::move(cpp2_param));
        }
    }

    return params;
}

std::string ClangToCpp2Visitor::type_to_cpp2_string(const clang::QualType& clang_type) {
    return get_type_string(clang_type);
}

std::string ClangToCpp2Visitor::build_semantic_content(const std::string& kind,
                                                        const std::string& content) {
    SemanticContentBuilder builder;
    builder.add(kind);
    builder.add(content);
    return builder.build();
}

// BidirectionalMappingContext implementation
class BidirectionalMappingContext::Impl {
public:
    std::unique_ptr<ClangToCpp2Visitor> clang_visitor;
    std::unique_ptr<SemanticHashContext> hash_context;
    clang::ASTContext* clang_ctx = nullptr;
};

BidirectionalMappingContext::BidirectionalMappingContext(clang::ASTContext* clang_ctx)
    : impl_(std::make_unique<Impl>()) {
    impl_->hash_context = std::make_unique<SemanticHashContext>(
        create_cpp2_visitor());
    if (clang_ctx) {
        set_clang_context(clang_ctx);
    }
}

BidirectionalMappingContext::~BidirectionalMappingContext() = default;

void BidirectionalMappingContext::set_clang_context(clang::ASTContext* ctx) {
    impl_->clang_ctx = ctx;
    impl_->clang_visitor = std::make_unique<ClangToCpp2Visitor>(*ctx);
}

NodeID BidirectionalMappingContext::register_cpp2_node(const void* cpp2_node,
                                                        const std::string& node_kind) {
    // This would use the Cpp2SemanticHashVisitor to compute hash
    // For now, return placeholder
    return 0;
}

ReverseMappingResult BidirectionalMappingContext::map_clang_to_cpp2(const void* clang_node) {
    if (!impl_->clang_visitor) {
        return ReverseMappingResult{};
    }
    return impl_->clang_visitor->convert(clang_node);
}

CRDTPatch BidirectionalMappingContext::create_equivalence_patch(const void* cpp2_node,
                                                                const void* clang_node) {
    CRDTPatch patch;
    patch.operation = CRDTPatch::Op::UpdateNode;
    patch.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    // Implementation would compute hashes and create patch
    return patch;
}

bool BidirectionalMappingContext::are_semantically_equivalent(NodeID cpp2_id, NodeID clang_id) {
    auto* cpp2_node = impl_->hash_context->get_node(cpp2_id);
    auto* clang_node = impl_->hash_context->get_node(clang_id);

    if (!cpp2_node || !clang_node) return false;

    return cpp2_node->merkle_hash == clang_node->merkle_hash;
}

// Utility: Parse C++ code and generate Clang AST
std::optional<ReverseMappingResult> parse_cpp_to_cpp2(const std::string& cpp_code,
                                                       const std::string& filename) {
    // This would use clang::tooling::runASTAction or similar
    // For now, return empty
    return std::nullopt;
}

} // namespace cppfort::crdt
