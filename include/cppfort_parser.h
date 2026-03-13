//===----------------------------------------------------------------------===//
// cppfort_parser.h - Public API Contract
// TrikeShed Math-Based SoN Compiler
// 100% Hand-Written Parser - No LLM-generated internals
//===----------------------------------------------------------------------===//

#ifndef CPPFORT_PARSER_H
#define CPPFORT_PARSER_H

#include <memory>
#include <string_view>
#include <vector>
#include <cstdint>

namespace cppfort {

// ============================================================================
// Canonical AST Node Types
// These map to the tags in src/selfhost/bootstrap_tags.cpp2
// ============================================================================

enum class NodeTag : int {
    // Core semantic types
    join_tag = 1,
    indexed_tag = 2,
    coordinates_tag = 3,
    chart_project_tag = 4,
    chart_embed_tag = 5,
    atlas_locate_tag = 6,
    transition_tag = 7,
    lower_dense_tag = 8,
    normalize_tag = 9,
    canonical_ast_tag = 10,

    // Gradient protocol
    grad_expr_tag = 11,
    grad_backend_tag = 12,

    // Parser API
    parser_parse_tag = 100,
    parser_normalize_tag = 101,
    parser_lower_to_son_tag = 102,

    // Shape descriptors
    shape_tag = 20,
    stride_tag = 21,
    rank_tag = 22,

    // Jacobian/gradient
    jacobian_tag = 30,
    matrix_mul_tag = 31,
    gradient_aggregate_tag = 32,

    // CAS Internment
    intern_string_tag = 40,
    intern_constant_tag = 41
};

// ============================================================================
// Canonical AST Node
// Represents normalized semantic structure
// ============================================================================

struct ASTNode {
    NodeTag tag;
    std::vector<std::unique_ptr<ASTNode>> children;
    std::string_view text;
    int line = 0;
    int column = 0;

    ASTNode(NodeTag t, std::string_view txt = {})
        : tag(t), text(txt) {}
};

// ============================================================================
// Canonical AST
// Root container for parsed and normalized AST
// ============================================================================

class CanonicalAST {
public:
    CanonicalAST() = default;

    void set_root(std::unique_ptr<ASTNode> root) { root_ = std::move(root); }
    ASTNode* root() const { return root_.get(); }

    void add_dependency(std::string_view path) { dependencies_.push_back(std::string(path)); }
    const std::vector<std::string>& dependencies() const { return dependencies_; }

private:
    std::unique_ptr<ASTNode> root_;
    std::vector<std::string> dependencies_;
};

// ============================================================================
// Error Handling
// ============================================================================

struct ParseError {
    int line;
    int column;
    std::string message;

    ParseError(int l, int c, std::string msg)
        : line(l), column(c), message(std::move(msg)) {}
};

struct ParseResult {
    std::unique_ptr<CanonicalAST> ast;
    std::vector<ParseError> errors;
    bool success() const { return errors.empty() && ast != nullptr; }
};

// ============================================================================
// Main Parser API
// Per TrikeShed gospel: "Public API contract in cppfort_parser.h"
// ============================================================================

class Parser {
public:
    Parser();
    ~Parser();

    // -------------------------------------------------------------------------
    // Parsing
    // -------------------------------------------------------------------------

    // Parse Cpp2 source from string
    ParseResult parse(std::string_view source);

    // Parse Cpp2 source from file
    ParseResult parse_file(std::string_view path);

    // -------------------------------------------------------------------------
    // Normalization
    // TrikeShed Notation as Front-End Sugar Only - Normalize early to canonical AST
    // -------------------------------------------------------------------------

    // Normalize parsed AST to canonical form
    // Applies TrikeShed normalization rules:
    // - coords[...] -> coordinates node
    // - chart.project(point) -> chart_project node
    // - chart.embed(coords) -> chart_embed node
    // - atlas.locate(point) -> atlas_locate node (join of chart + coordinates)
    // - coords.lowered() -> lower_dense node
    std::unique_ptr<CanonicalAST> normalize(CanonicalAST* ast);

    // -------------------------------------------------------------------------
    // Lowering to SoN
    // -------------------------------------------------------------------------

    // Lower canonical AST to MLIR SoN operations
    // This bridges to the Cpp2SONDialect in lib/Dialect/
    // Note: Requires MLIR to be available
    // Returns nullptr if MLIR lowering is not available
    // (placeholder - full implementation requires MLIR integration)
    // std::unique_ptr<mlir::ModuleOp> lower_to_son(CanonicalAST* ast);

    // -------------------------------------------------------------------------
    // Error Handling
    // -------------------------------------------------------------------------

    const std::vector<ParseError>& errors() const { return errors_; }
    void clear_errors() { errors_.clear(); }

    // -------------------------------------------------------------------------
    // Source Info
    // -------------------------------------------------------------------------

    std::string_view source() const { return source_; }
    void set_source(std::string_view src) { source_ = src; }

private:
    std::string_view source_;
    std::vector<ParseError> errors_;

    // Internal parsing state
    size_t position_ = 0;
    int current_line_ = 1;
    int current_column_ = 1;

    // Lexer helpers
    void skip_whitespace();
    bool match(char c);
    bool match(std::string_view str);
    char peek();
    char consume();

    // Token types for hand-written lexer
    enum class TokenType {
        EndOfFile,
        Identifier,
        Number,
        String,
        Keyword,
        Operator,
        LBracket,
        RBracket,
        LParen,
        RParen,
        LBrace,
        RBrace,
        Comma,
        Dot,
        Colon,
        Arrow,
    };

    struct Token {
        TokenType type;
        std::string_view text;
        int line;
        int column;
    };

    Token next_token();

    // Parsing functions for expression precedence levels
    std::unique_ptr<ASTNode> parse_assignment_expression();
    std::unique_ptr<ASTNode> parse_pipeline_expression();
    std::unique_ptr<ASTNode> parse_ternary_expression();
    std::unique_ptr<ASTNode> parse_logical_or_expression();
    std::unique_ptr<ASTNode> parse_logical_and_expression();
    std::unique_ptr<ASTNode> parse_bitwise_or_expression();
    std::unique_ptr<ASTNode> parse_bitwise_xor_expression();
    std::unique_ptr<ASTNode> parse_bitwise_and_expression();
    std::unique_ptr<ASTNode> parse_equality_expression();
    std::unique_ptr<ASTNode> parse_comparison_expression();
    std::unique_ptr<ASTNode> parse_range_expression();
    std::unique_ptr<ASTNode> parse_shift_expression();
    std::unique_ptr<ASTNode> parse_additive_expression();
    std::unique_ptr<ASTNode> parse_multiplicative_expression();
    std::unique_ptr<ASTNode> parse_prefix_expression();
    std::unique_ptr<ASTNode> parse_postfix_expression();
    std::unique_ptr<ASTNode> parse_primary_expression();
    std::vector<std::unique_ptr<ASTNode>> parse_argument_list();
    std::unique_ptr<ASTNode> parse_type_specifier();

    // Parsing functions for TrikeShed surface syntax
    std::unique_ptr<ASTNode> parse_expression();
    std::unique_ptr<ASTNode> parse_coordinates();
    std::unique_ptr<ASTNode> parse_join();
    std::unique_ptr<ASTNode> parse_alpha();
    std::unique_ptr<ASTNode> parse_chart();
    std::unique_ptr<ASTNode> parse_atlas();
    std::unique_ptr<ASTNode> parse_manifold();
    std::unique_ptr<ASTNode> parse_transition();
    std::unique_ptr<ASTNode> parse_statement();
    std::unique_ptr<ASTNode> parse_declaration();
    std::unique_ptr<ASTNode> parse_unified_declaration(std::string_view name);

    // Normalization functions
    std::unique_ptr<ASTNode> normalize_node(ASTNode* node);
    std::unique_ptr<ASTNode> normalize_coordinates(ASTNode* node);
    std::unique_ptr<ASTNode> normalize_chart_project(ASTNode* node);
    std::unique_ptr<ASTNode> normalize_chart_embed(ASTNode* node);
    std::unique_ptr<ASTNode> normalize_atlas_locate(ASTNode* node);
    std::unique_ptr<ASTNode> normalize_lower_dense(ASTNode* node);
    std::unique_ptr<ASTNode> normalize_node_from_indexed(ASTNode* node);
};

// ============================================================================
// Convenience Functions
// ============================================================================

// Parse and normalize in one call
inline ParseResult parse_and_normalize(std::string_view source) {
    Parser parser;
    auto result = parser.parse(source);
    if (result.success()) {
        auto normalized = parser.normalize(result.ast.get());
        result.ast = std::move(normalized);
    }
    return result;
}

// Parse from file and normalize
inline ParseResult parse_file_and_normalize(std::string_view path) {
    Parser parser;
    auto result = parser.parse_file(path);
    if (result.success()) {
        auto normalized = parser.normalize(result.ast.get());
        result.ast = std::move(normalized);
    }
    return result;
}

} // namespace cppfort

#endif // CPPFORT_PARSER_H