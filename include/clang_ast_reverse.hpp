#pragma once

#include "ast.hpp"
#include "semantic_hash.hpp"
#include <memory>
#include <optional>
#include <unordered_map>
#include <string>

// Forward declarations for Clang AST types (avoiding Clang dependency in header)
namespace clang {
class ASTContext;
class Stmt;
class Decl;
class Expr;
class Type;
class QualType;
} // namespace clang

namespace cppfort::crdt {

// Result of reverse mapping Clang AST → cpp2 AST
struct ReverseMappingResult {
    std::unique_ptr<cpp2_transpiler::Expression> expr;
    std::unique_ptr<cpp2_transpiler::Statement> stmt;
    std::unique_ptr<cpp2_transpiler::Declaration> decl;
    std::string inferred_cpp2_syntax;   // The cpp2 syntax that would produce this Clang AST
    SHA256Hash semantic_hash;            // Hash of the semantic content

    bool is_valid() const {
        return expr || stmt || decl;
    }
};

// Parameter qualifier inference from Clang AST analysis
struct ParameterQualifierInference {
    enum class Qualifier {
        InOut,    // T& - mutable reference
        Out,      // T& - definite assignment (requires dataflow analysis)
        Move,     // T&& - rvalue reference
        Forward,  // auto&& - forwarding reference (template context)
        In,       // const T& or T - read-only
        None      // No qualifier
    };

    static Qualifier infer_from_clang_type(const clang::QualType& clang_type,
                                           bool is_template_context);

    static std::string qualifier_to_cpp2(Qualifier q);
};

// Clang AST to cpp2 AST visitor
class ClangToCpp2Visitor {
private:
    clang::ASTContext& context_;

    // Helper: get type string from Clang type
    std::string get_type_string(const clang::QualType& qual_type);
    std::unique_ptr<cpp2_transpiler::Type> make_type(const clang::QualType& qual_type);

public:
    explicit ClangToCpp2Visitor(clang::ASTContext& ctx);
    ~ClangToCpp2Visitor() = default;

    // Main entry point: convert any Clang AST node to cpp2
    ReverseMappingResult convert(const void* clang_node);

    // Statement conversions
    std::unique_ptr<cpp2_transpiler::Statement> visit_statement(const clang::Stmt* stmt);
    std::unique_ptr<cpp2_transpiler::ReturnStatement> visit_return_statement(const clang::Stmt* stmt);
    std::unique_ptr<cpp2_transpiler::IfStatement> visit_if_statement(const clang::Stmt* stmt);
    std::unique_ptr<cpp2_transpiler::WhileStatement> visit_while_statement(const clang::Stmt* stmt);
    std::unique_ptr<cpp2_transpiler::ForStatement> visit_for_statement(const clang::Stmt* stmt);
    std::unique_ptr<cpp2_transpiler::ForRangeStatement> visit_for_range_statement(const clang::Stmt* stmt);
    std::unique_ptr<cpp2_transpiler::BlockStatement> visit_compound_statement(const clang::Stmt* stmt);

    // Expression conversions
    std::unique_ptr<cpp2_transpiler::Expression> visit_expression(const clang::Expr* expr);
    std::unique_ptr<cpp2_transpiler::LiteralExpression> visit_literal(const clang::Expr* expr);
    std::unique_ptr<cpp2_transpiler::IdentifierExpression> visit_decl_ref(const clang::Expr* expr);
    std::unique_ptr<cpp2_transpiler::BinaryExpression> visit_binary_operator(const clang::Expr* expr);
    std::unique_ptr<cpp2_transpiler::CallExpression> visit_call(const clang::Expr* expr, bool is_ufcs);
    std::unique_ptr<cpp2_transpiler::MemberAccessExpression> visit_member_access(const clang::Expr* expr);
    std::unique_ptr<cpp2_transpiler::UnaryExpression> visit_unary_operator(const clang::Expr* expr);

    // Declaration conversions
    std::unique_ptr<cpp2_transpiler::Declaration> visit_declaration(const clang::Decl* decl);
    std::unique_ptr<cpp2_transpiler::FunctionDeclaration> visit_function_decl(const clang::Decl* decl);
    std::unique_ptr<cpp2_transpiler::VariableDeclaration> visit_variable_decl(const clang::Decl* decl);
    std::unique_ptr<cpp2_transpiler::TypeDeclaration> visit_type_decl(const clang::Decl* decl);

    // Type conversion
    std::unique_ptr<cpp2_transpiler::Type> convert_type(const clang::Type* clang_type);
    std::unique_ptr<cpp2_transpiler::Type> convert_qual_type(const clang::QualType& clang_type);

    // Generate inferred cpp2 syntax string
    std::string generate_cpp2_syntax(const ReverseMappingResult& result);

    // Get semantic hash for a Clang AST node
    SHA256Hash compute_semantic_hash(const void* clang_node);

private:
    // Helper: check if a function call could be UFCS
    bool is_potential_ufcs_call(const clang::Expr* call_expr);

    // Helper: infer parameter qualifiers from function signature
    std::vector<cpp2_transpiler::FunctionDeclaration::Parameter> convert_parameters(
        const clang::Decl* function_decl);

    // Helper: convert Clang type to cpp2 type string
    std::string type_to_cpp2_string(const clang::QualType& clang_type);

    // Helper: build semantic content for hashing
    std::string build_semantic_content(const std::string& kind,
                                       const std::string& content);
};

// Context for managing bidirectional mapping
class BidirectionalMappingContext {
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

public:
    explicit BidirectionalMappingContext(clang::ASTContext* clang_ctx = nullptr);
    ~BidirectionalMappingContext();

    // Set Clang AST context (required for reverse mapping)
    void set_clang_context(clang::ASTContext* ctx);

    // Forward: cpp2 AST → semantic hash
    NodeID register_cpp2_node(const void* cpp2_node, const std::string& node_kind);

    // Reverse: Clang AST → cpp2 AST + semantic hash
    ReverseMappingResult map_clang_to_cpp2(const void* clang_node);

    // Generate CRDT patch for semantic equivalence
    CRDTPatch create_equivalence_patch(const void* cpp2_node, const void* clang_node);

    // Check if two nodes are semantically equivalent (by hash)
    bool are_semantically_equivalent(NodeID cpp2_id, NodeID clang_id);
};

// Utility: Parse C++ code and generate Clang AST (for testing)
std::optional<ReverseMappingResult> parse_cpp_to_cpp2(
    const std::string& cpp_code,
    const std::string& filename = "input.cpp");

} // namespace cppfort::crdt
