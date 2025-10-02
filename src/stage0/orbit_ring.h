#pragma once

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "token.h"
#include "ast.h"  // For ParameterKind

namespace cppfort::stage0 {

// Orbit = speculative pattern match with confidence
struct OrbitMatch {
    size_t position;           // Byte offset in source
    size_t length;             // Pattern length
    double confidence;         // Match confidence (0.0-1.0)
    std::string pattern_id;    // Pattern identifier
    std::string grammar;       // C, C++, or CPP2
    SourceLocation location;
};

// Ring = collection of orbit candidates for a scope
struct OrbitRing {
    std::vector<OrbitMatch> candidates;  // All possible interpretations
    size_t start_position;               // Ring start offset
    size_t end_position;                 // Ring end offset
    std::optional<size_t> winner_index;  // Resolved interpretation (if disambiguated)

    // Nested rings for subscopes
    std::vector<std::shared_ptr<OrbitRing>> subscopes;
};

// Parameter with orbit metadata
struct OrbitParameter {
    std::string name;
    std::string type;
    std::vector<OrbitMatch> type_candidates;  // Multiple type interpretations
    ParameterKind kind {ParameterKind::Default};
    SourceLocation location;
};

// Variable declaration with orbit metadata
struct OrbitVariableDecl {
    std::string name;
    std::string type;
    std::vector<OrbitMatch> type_candidates;
    std::optional<std::string> initializer;
    SourceLocation location;
};

// Expression with orbit alternatives
struct OrbitExpressionStmt {
    std::string expression;
    std::vector<OrbitMatch> expr_candidates;  // Multiple interpretations
    SourceLocation location;
};

// Return statement with orbit metadata
struct OrbitReturnStmt {
    std::optional<std::string> expression;
    std::vector<OrbitMatch> expr_candidates;
    SourceLocation location;
};

// Assert with orbit metadata
struct OrbitAssertStmt {
    std::string condition;
    std::optional<std::string> category;
    SourceLocation location;
};

struct OrbitBlock;
struct OrbitForChainStmt;
struct OrbitRawStmt {
    std::string text;
    SourceLocation location;
};

using OrbitStatement = std::variant<
    OrbitVariableDecl,
    OrbitExpressionStmt,
    OrbitReturnStmt,
    OrbitAssertStmt,
    OrbitForChainStmt,
    OrbitRawStmt
>;

struct OrbitBlock {
    std::vector<OrbitStatement> statements;
    OrbitRing ring;  // Scope ring for this block
    SourceLocation location;
};

struct OrbitForChainStmt {
    std::string range_expression;
    std::optional<std::string> next_expression;
    OrbitParameter loop_parameter;
    OrbitBlock body;
    SourceLocation location;
};

struct OrbitExpressionBody {
    std::string expression;
    std::vector<OrbitMatch> expr_candidates;
    SourceLocation location;
};

using OrbitFunctionBody = std::variant<OrbitBlock, OrbitExpressionBody>;

// Function with orbit metadata
struct OrbitFunctionDecl {
    std::string name;
    std::vector<OrbitParameter> parameters;
    std::optional<std::string> return_type;
    std::vector<OrbitMatch> return_type_candidates;
    OrbitFunctionBody body;
    SourceLocation location;
};

struct OrbitTypeDecl {
    std::string name;
    std::string body;
    SourceLocation location;
};

struct OrbitIncludeDecl {
    std::string path;
    bool is_system {false};
    SourceLocation location;
};

struct OrbitRawDecl {
    std::string text;
    SourceLocation location;
};

// Translation unit with orbit rings
struct OrbitTranslationUnit {
    std::vector<OrbitIncludeDecl> includes;
    std::vector<OrbitFunctionDecl> functions;
    std::vector<OrbitTypeDecl> types;
    std::vector<OrbitRawDecl> raw_declarations;

    // Global orbit ring for top-level scope
    OrbitRing global_ring;

    // Resolution metadata
    std::string source_grammar;  // C, C++, or CPP2
    std::vector<std::string> ambiguous_patterns;  // Unresolved ambiguities
};

} // namespace cppfort::stage0
