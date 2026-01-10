#pragma once
// ============================================================================
// Slim AST - Grammar-Derived Parse Tree
// ============================================================================
// AST node kinds derived directly from parser.cpp grammar rules.
// Arena-based storage, no heap allocation per node.
// ============================================================================

#include <cstdint>
#include <span>
#include <vector>
#include <string_view>
#include "core/tokens.hpp"

namespace cpp2::ast {

// ============================================================================
// Node Kinds (1:1 with parser.cpp grammar rules)
// ============================================================================

enum class NodeKind : uint8_t {
    // Terminals
    Identifier,
    Literal,
    
    // Expressions (precedence climbing)
    GroupedExpression,
    PrimaryExpression,
    CallOp,
    MemberOp,
    SubscriptOp,
    PostfixOp,
    PostfixExpression,
    PrefixOp,
    PrefixExpression,
    MultiplicativeExpression,
    AdditiveExpression,
    ShiftExpression,
    ComparisonExpression,
    EqualityExpression,
    BitwiseAndExpression,
    BitwiseXorExpression,
    BitwiseOrExpression,
    LogicalAndExpression,
    LogicalOrExpression,
    TernaryExpression,
    PipelineExpression,
    AssignmentOp,
    AssignmentExpression,
    Expression,
    
    // Types
    BasicType,
    TemplateArgs,
    QualifiedType,
    TypeSpecifier,
    
    // Statements
    BlockStatement,
    ReturnStatement,
    IfStatement,
    WhileStatement,
    ForStatement,
    ExpressionStatement,
    Statement,
    
    // Parameters
    ParamQualifier,
    Parameter,
    ParamList,
    
    // Declarations
    ReturnSpec,
    FunctionBody,
    FunctionSuffix,
    VariableSuffix,
    TypeBody,
    TypeSuffix,
    NamespaceBody,
    NamespaceSuffix,
    DeclarationSuffix,
    UnifiedDeclaration,
    AccessSpecifier,
    Declaration,
    
    // Top-level
    TranslationUnit,
    
    COUNT_  // Sentinel for array sizing
};

// ============================================================================
// EnumMap - O(1) lookup by enum ordinal (like Java EnumMap)
// ============================================================================

template<typename V, std::size_t N = static_cast<std::size_t>(NodeKind::COUNT_)>
struct EnumMap {
    V data[N]{};
    constexpr V& operator[](NodeKind k) { return data[static_cast<std::size_t>(k)]; }
    constexpr const V& operator[](NodeKind k) const { return data[static_cast<std::size_t>(k)]; }
};

// ============================================================================
// NodeKind Metadata (Constexpr Arrays)
// ============================================================================

namespace meta {

// Names for debugging/serialization
inline constexpr const char* names[] = {
    "Identifier", "Literal",
    "GroupedExpression", "PrimaryExpression", "CallOp", "MemberOp", "SubscriptOp",
    "PostfixOp", "PostfixExpression", "PrefixOp", "PrefixExpression",
    "MultiplicativeExpression", "AdditiveExpression", "ShiftExpression",
    "ComparisonExpression", "EqualityExpression", "BitwiseAndExpression",
    "BitwiseXorExpression", "BitwiseOrExpression", "LogicalAndExpression",
    "LogicalOrExpression", "TernaryExpression", "PipelineExpression",
    "AssignmentOp", "AssignmentExpression", "Expression",
    "BasicType", "TemplateArgs", "QualifiedType", "TypeSpecifier",
    "BlockStatement", "ReturnStatement", "IfStatement", "WhileStatement",
    "ForStatement", "ExpressionStatement", "Statement",
    "ParamQualifier", "Parameter", "ParamList",
    "ReturnSpec", "FunctionBody", "FunctionSuffix", "VariableSuffix",
    "TypeBody", "TypeSuffix", "NamespaceBody", "NamespaceSuffix",
    "DeclarationSuffix", "UnifiedDeclaration", "AccessSpecifier", "Declaration",
    "TranslationUnit"
};

constexpr const char* name(NodeKind k) { return names[static_cast<std::size_t>(k)]; }

// Category predicates
constexpr bool is_expression(NodeKind k) {
    auto i = static_cast<uint8_t>(k);
    return i >= static_cast<uint8_t>(NodeKind::GroupedExpression) 
        && i <= static_cast<uint8_t>(NodeKind::Expression);
}

constexpr bool is_statement(NodeKind k) {
    auto i = static_cast<uint8_t>(k);
    return i >= static_cast<uint8_t>(NodeKind::BlockStatement) 
        && i <= static_cast<uint8_t>(NodeKind::Statement);
}

constexpr bool is_declaration(NodeKind k) {
    auto i = static_cast<uint8_t>(k);
    return i >= static_cast<uint8_t>(NodeKind::ReturnSpec) 
        && i <= static_cast<uint8_t>(NodeKind::Declaration);
}

constexpr bool is_type(NodeKind k) {
    auto i = static_cast<uint8_t>(k);
    return i >= static_cast<uint8_t>(NodeKind::BasicType) 
        && i <= static_cast<uint8_t>(NodeKind::TypeSpecifier);
}

} // namespace meta

// ============================================================================
// AST Node (Flyweight)
// ============================================================================
// Each node is a tagged span into the token array.
// Children are indices into the node arena.

struct Node {
    NodeKind kind;
    uint32_t token_start;   // First token index
    uint32_t token_end;     // Past-end token index
    uint32_t child_start;   // First child index in arena
    uint32_t child_count;   // Number of children
    
    constexpr bool has_children() const { return child_count > 0; }
    constexpr std::size_t token_count() const { return token_end - token_start; }
};

// ============================================================================
// Parse Tree (Arena Storage)
// ============================================================================

struct ParseTree {
    std::vector<Node> nodes;                           // Node arena
    std::span<const cpp2_transpiler::Token> tokens;    // Source tokens (external)
    uint32_t root = 0;                                 // Root node index
    
    // Access node by index
    const Node& operator[](uint32_t idx) const { return nodes[idx]; }
    Node& operator[](uint32_t idx) { return nodes[idx]; }
    
    // Get children of a node
    std::span<const Node> children(const Node& n) const {
        return {nodes.data() + n.child_start, n.child_count};
    }
    
    // Get tokens for a node
    std::span<const cpp2_transpiler::Token> node_tokens(const Node& n) const {
        return tokens.subspan(n.token_start, n.token_end - n.token_start);
    }
    
    // Get lexeme for single-token node
    std::string_view lexeme(const Node& n) const {
        if (n.token_count() == 1) return tokens[n.token_start].lexeme;
        return {};
    }
};

// ============================================================================
// Builder (Constructs ParseTree from Features)
// ============================================================================

class TreeBuilder {
    std::vector<Node> nodes_;
    std::vector<uint32_t> stack_;  // Parent node indices
    
public:
    // Start a new node, push onto stack
    void begin(NodeKind kind, uint32_t token_pos) {
        uint32_t idx = static_cast<uint32_t>(nodes_.size());
        nodes_.push_back({kind, token_pos, token_pos, 0, 0});
        stack_.push_back(idx);
    }
    
    // End current node, pop from stack, link to parent
    void end(uint32_t token_pos) {
        if (stack_.empty()) return;
        
        uint32_t idx = stack_.back();
        stack_.pop_back();
        
        nodes_[idx].token_end = token_pos;
        
        // Link as child of parent
        if (!stack_.empty()) {
            uint32_t parent = stack_.back();
            if (nodes_[parent].child_count == 0) {
                nodes_[parent].child_start = idx;
            }
            nodes_[parent].child_count++;
        }
    }
    
    // Finalize and return tree
    ParseTree finish(std::span<const cpp2_transpiler::Token> tokens) {
        ParseTree tree;
        tree.nodes = std::move(nodes_);
        tree.tokens = tokens;
        tree.root = tree.nodes.empty() ? 0 : 0;  // First node is root
        return tree;
    }
};

} // namespace cpp2::ast
