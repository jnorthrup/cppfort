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
#include <string>
#include <iostream>
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
// Children are linked via first_child -> next_sibling indices.

struct Node {
    NodeKind kind;
    uint32_t token_start;   // First token index
    uint32_t token_end;     // Past-end token index
    
    // Topology (Left-Child Right-Sibling)
    uint32_t first_child = UINT32_MAX; 
    uint32_t next_sibling = UINT32_MAX;
    
    // Metadata (optional, but maintained for convenience)
    uint32_t child_count = 0; 
    
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
// Tree Builder (Thread-Local State)
// ============================================================================

class TreeBuilder {
    struct StackFrame {
        uint32_t node_idx;
        uint32_t last_child_idx;
    };

    std::vector<Node> nodes_;
    std::vector<StackFrame> stack_;

public:
    struct Checkpoint { std::size_t nodes_size; std::size_t stack_size; };

    void begin(NodeKind kind, uint32_t token_pos) {
        uint32_t idx = static_cast<uint32_t>(nodes_.size());
        nodes_.push_back({kind, token_pos, token_pos, UINT32_MAX, UINT32_MAX, 0});
        stack_.push_back({idx, UINT32_MAX}); 
    }
    
    void end(uint32_t token_pos) {
        if (stack_.empty()) return;
        
        StackFrame frame = stack_.back();
        stack_.pop_back();
        
        uint32_t current_idx = frame.node_idx;
        if (current_idx < nodes_.size()) {
            nodes_[current_idx].token_end = token_pos;
        }
        
        // Link to parent
        if (!stack_.empty()) {
            StackFrame& parent_frame = stack_.back();
            uint32_t parent_idx = parent_frame.node_idx;
            
            if (parent_frame.last_child_idx == UINT32_MAX) {
                nodes_[parent_idx].first_child = current_idx;
            } else {
                nodes_[parent_frame.last_child_idx].next_sibling = current_idx;
            }
            
            parent_frame.last_child_idx = current_idx;
            nodes_[parent_idx].child_count++;
        }
    }
    
    void start_infix(NodeKind kind, uint32_t token_pos) {
        if (stack_.empty()) {
            begin(kind, token_pos);
            return;
        }

        StackFrame& parent_frame = stack_.back();
        uint32_t parent_idx = parent_frame.node_idx;

        if (nodes_[parent_idx].child_count == 0 || parent_frame.last_child_idx == UINT32_MAX) {
            begin(kind, token_pos);
            return;
        }

        // LHS is the last added child
        uint32_t lhs_idx = parent_frame.last_child_idx;
        
        // Create Infix node
        uint32_t infix_idx = static_cast<uint32_t>(nodes_.size());
        nodes_.push_back({kind, nodes_[lhs_idx].token_start, token_pos, lhs_idx, UINT32_MAX, 1});
        
        // Relink Parent
        if (nodes_[parent_idx].first_child == lhs_idx) {
            nodes_[parent_idx].first_child = infix_idx;
        } else {
            // Traverse to find node pointing to LHS
            uint32_t prev = nodes_[parent_idx].first_child;
            while (prev != UINT32_MAX && nodes_[prev].next_sibling != lhs_idx) {
                prev = nodes_[prev].next_sibling;
            }
            if (prev != UINT32_MAX) {
                nodes_[prev].next_sibling = infix_idx;
            }
        }
        
        // Update Parent state
        parent_frame.last_child_idx = infix_idx;
        // Child count stays same (1 replaced by 1)
        
        // LHS state update
        nodes_[lhs_idx].next_sibling = UINT32_MAX; 
        
        // Push Infix to stack
        stack_.push_back({infix_idx, lhs_idx}); // Infix already has LHS as child
    }

    ParseTree finish(std::span<const cpp2_transpiler::Token> tokens) {
        ParseTree tree;
        tree.nodes = std::move(nodes_);
        tree.tokens = tokens;
        tree.root = 0;
        return tree;
    }

    [[nodiscard]] Checkpoint checkpoint() const { return {nodes_.size(), stack_.size()}; }
    
    void restore(const Checkpoint& cp) {
        nodes_.resize(cp.nodes_size);
        stack_.resize(cp.stack_size);
        // Fixup dangling pointers in the restored stack frame if needed
        if (!stack_.empty()) {
            StackFrame& frame = stack_.back();
            uint32_t last = frame.last_child_idx;
            if (last != UINT32_MAX) {
                if (last >= nodes_.size()) {
                    // Last child was erased. We need to find the new last child.
                    // Or if we can't find it, traverse?
                    // This is complex. For now assume basic restore works for failure atoms.
                    // If we restored, we lose the knowledge of previous sibling?
                    // Ideally check point also saves the `last_child_idx`.
                    // It does (via stack resize restoring old frames).
                    // BUT: The node pointed to by `last_child_idx` MUST exist.
                    // If `nodes_` was resized, and `last_child_idx` < `cp.nodes_size`, we are good.
                    // If `last_child_idx` >= `cp.nodes_size`, that's impossible because the frame was saved WHEN `nodes_size` was smaller (or equal).
                    // Frame is from the past. Nodes it refers to must be from the past (smaller indices).
                    // So `last_child_idx` is strictly < `nodes_.size()`.
                    // So the node exists.
                    // However, `nodes_[last].next_sibling` might point to a node that was just deleted.
                    nodes_[last].next_sibling = UINT32_MAX;
                } else {
                     nodes_[last].next_sibling = UINT32_MAX;
                }
            } else {
                // No children. Reset first child?
                 if (frame.node_idx < nodes_.size()) {
                    nodes_[frame.node_idx].first_child = UINT32_MAX;
                    nodes_[frame.node_idx].child_count = 0;
                 }
            }
        }
    }
};


} // namespace cpp2::ast

// ============================================================================
// Tree Building Primitives (Shared between Parser and Combinators)
// ============================================================================

namespace cpp2::ast {

inline thread_local TreeBuilder g_builder;

inline void begin(NodeKind k, std::size_t pos) { 
    g_builder.begin(k, static_cast<uint32_t>(pos)); 
}

inline void end(std::size_t pos) { 
    g_builder.end(static_cast<uint32_t>(pos)); 
}

inline void start_infix(NodeKind k, std::size_t pos) {
    g_builder.start_infix(k, static_cast<uint32_t>(pos));
}

inline auto tree_checkpoint() { return g_builder.checkpoint(); }
inline void tree_restore(const TreeBuilder::Checkpoint& cp) { g_builder.restore(cp); }

} // namespace cpp2::ast

