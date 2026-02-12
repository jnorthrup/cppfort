#pragma once
// ============================================================================
// Slim AST - Grammar-Derived Parse Tree
// ============================================================================
// AST node kinds derived directly from parser.cpp grammar rules.
// Arena-based storage, no heap allocation per node.
// ============================================================================

#include "core/tokens.hpp"
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

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
  ScopeOp, // Scope resolution ::
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
  BinaryOp,
  IsExpression,
  AsExpression,
  RangeExpression,
  AssignmentOp,
  AssignmentExpression,
  Expression,

  // Types
  BasicType,
  TemplateArgs,
  QualifiedType,
  TypeSpecifier,

  // Contracts
  ContractClause,
  RequiresClause,

  // Pattern Matching
  InspectExpression,
  InspectArm,
  Pattern,
  IsPattern,
  AsPattern,

  // Statements
  BlockStatement,
  UncheckedStatement,
  ScopeStatement,
  LambdaExpression,
  ReturnStatement,
  IfStatement,
  WhileStatement,
  DoWhileStatement,
  ForStatement,
  ExpressionStatement,
  AssertStatement,
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
  TypeAliasSuffix,
  NamespaceBody,
  NamespaceSuffix,
  DeclarationSuffix,
  UnifiedDeclaration,
  AccessSpecifier,
  Declaration,

  // Preprocessor directives (#include, #define, etc.)
  Preprocessor,

  // Metafunctions (@value, @interface, @ordered, etc.)
  Metafunction,

  // Top-level
  TranslationUnit,

  COUNT_ // Sentinel for array sizing
};

// ============================================================================
// EnumMap - O(1) lookup by enum ordinal (like Java EnumMap)
// ============================================================================

template <typename V,
          std::size_t N = static_cast<std::size_t>(NodeKind::COUNT_)>
struct EnumMap {
  V data[N]{};
  constexpr V &operator[](NodeKind k) {
    return data[static_cast<std::size_t>(k)];
  }
  constexpr const V &operator[](NodeKind k) const {
    return data[static_cast<std::size_t>(k)];
  }
};

// ============================================================================
// NodeKind Metadata (Constexpr Arrays)
// ============================================================================

namespace meta {

// Names for debugging/serialization
inline constexpr const char *names[] = {"Identifier",
                                        "Literal",
                                        "GroupedExpression",
                                        "PrimaryExpression",
                                        "CallOp",
                                        "MemberOp",
                                        "ScopeOp",
                                        "SubscriptOp",
                                        "PostfixOp",
                                        "PostfixExpression",
                                        "PrefixOp",
                                        "PrefixExpression",
                                        "MultiplicativeExpression",
                                        "AdditiveExpression",
                                        "ShiftExpression",
                                        "ComparisonExpression",
                                        "EqualityExpression",
                                        "BitwiseAndExpression",
                                        "BitwiseXorExpression",
                                        "BitwiseOrExpression",
                                        "LogicalAndExpression",
                                        "LogicalOrExpression",
                                        "TernaryExpression",
                                        "PipelineExpression",
                                        "BinaryOp",
                                        "IsExpression",
                                        "AsExpression",
                                        "RangeExpression",
                                        "AssignmentOp",
                                        "AssignmentExpression",
                                        "Expression",
                                        "BasicType",
                                        "TemplateArgs",
                                        "QualifiedType",
                                        "TypeSpecifier",
                                        "ContractClause",
                                        "RequiresClause",
                                        "InspectExpression",
                                        "InspectArm",
                                        "Pattern",
                                        "IsPattern",
                                        "AsPattern",
                                        "BlockStatement",
                                        "UncheckedStatement",
                                        "ScopeStatement",
                                        "LambdaExpression",
                                        "ReturnStatement",
                                        "IfStatement",
                                        "WhileStatement",
                                        "DoWhileStatement",
                                        "ForStatement",
                                        "ExpressionStatement",
                                        "AssertStatement",
                                        "Statement",
                                        "ParamQualifier",
                                        "Parameter",
                                        "ParamList",
                                        "ReturnSpec",
                                        "FunctionBody",
                                        "FunctionSuffix",
                                        "VariableSuffix",
                                        "TypeBody",
                                        "TypeSuffix",
                                        "NamespaceBody",
                                        "NamespaceSuffix",
                                        "DeclarationSuffix",
                                        "UnifiedDeclaration",
                                        "AccessSpecifier",
                                        "Declaration",
                                        "Preprocessor",
                                        "Metafunction",
                                        "TranslationUnit"};

constexpr const char *name(NodeKind k) {
  return names[static_cast<std::size_t>(k)];
}

// Category predicates
constexpr bool is_expression(NodeKind k) {
  auto i = static_cast<uint8_t>(k);
  return i <= static_cast<uint8_t>(NodeKind::Expression) &&
         i >= static_cast<uint8_t>(NodeKind::Identifier);
}

constexpr bool is_statement(NodeKind k) {
  auto i = static_cast<uint8_t>(k);
  return i >= static_cast<uint8_t>(NodeKind::BlockStatement) &&
         i <= static_cast<uint8_t>(NodeKind::Statement);
}

constexpr bool is_declaration(NodeKind k) {
  auto i = static_cast<uint8_t>(k);
  return i >= static_cast<uint8_t>(NodeKind::ReturnSpec) &&
         i <= static_cast<uint8_t>(NodeKind::Declaration);
}

constexpr bool is_type(NodeKind k) {
  auto i = static_cast<uint8_t>(k);
  return i >= static_cast<uint8_t>(NodeKind::BasicType) &&
         i <= static_cast<uint8_t>(NodeKind::TypeSpecifier);
}

} // namespace meta

// ============================================================================
// AST Node (Flyweight)
// ============================================================================
// Each node is a tagged span into the token array.
// Children are linked via first_child -> next_sibling indices.

struct Node {
  NodeKind kind;
  uint32_t token_start; // First token index
  uint32_t token_end;   // Past-end token index

  // Topology (Left-Child Right-Sibling)
  uint32_t first_child = UINT32_MAX;
  uint32_t next_sibling = UINT32_MAX;

  // Metadata (optional, but maintained for convenience)
  uint32_t child_count = 0;

  Node() = default;
  Node(NodeKind k, uint32_t ts, uint32_t te, uint32_t fc = UINT32_MAX,
       uint32_t ns = UINT32_MAX, uint32_t cc = 0)
      : kind(k), token_start(ts), token_end(te), first_child(fc),
        next_sibling(ns), child_count(cc) {}

  constexpr bool has_children() const { return child_count > 0; }
  constexpr std::size_t token_count() const { return token_end - token_start; }
};

// ============================================================================
// Parse Tree (Arena Storage)
// ============================================================================

struct ParseTree {
  std::vector<Node> nodes;                        // Node arena
  std::span<const cpp2_transpiler::Token> tokens; // Source tokens (external)
  uint32_t root = 0;                              // Root node index

  // Access node by index
  const Node &operator[](uint32_t idx) const { return nodes[idx]; }
  Node &operator[](uint32_t idx) { return nodes[idx]; }

  // Get tokens for a node
  std::span<const cpp2_transpiler::Token> node_tokens(const Node &n) const {
    return tokens.subspan(n.token_start, n.token_end - n.token_start);
  }

  // Get lexeme for single-token node
  std::string_view lexeme(const Node &n) const {
    if (n.token_count() == 1)
      return tokens[n.token_start].lexeme;
    return {};
  }

  // ============================================================================
  // Children Iterator (LCRS traversal)
  // ============================================================================

  class ChildIterator {
    const ParseTree *tree_;
    uint32_t current_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Node &;
    using difference_type = std::ptrdiff_t;
    using pointer = const Node *;
    using reference = const Node &;

    ChildIterator(const ParseTree *tree, uint32_t idx)
        : tree_(tree), current_(idx) {}

    reference operator*() const { return tree_->nodes[current_]; }
    pointer operator->() const { return &tree_->nodes[current_]; }

    ChildIterator &operator++() {
      current_ = tree_->nodes[current_].next_sibling;
      return *this;
    }

    ChildIterator operator++(int) {
      ChildIterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const ChildIterator &other) const {
      return current_ == other.current_;
    }
    bool operator!=(const ChildIterator &other) const {
      return current_ != other.current_;
    }
  };

  class ChildRange {
    const ParseTree *tree_;
    uint32_t first_;

  public:
    ChildRange(const ParseTree *tree, uint32_t first)
        : tree_(tree), first_(first) {}

    ChildIterator begin() const { return ChildIterator(tree_, first_); }
    ChildIterator end() const { return ChildIterator(tree_, UINT32_MAX); }
    bool empty() const { return first_ == UINT32_MAX; }
  };

  // Get children of a node as iterable range
  ChildRange children(const Node &n) const {
    return ChildRange(this, n.first_child);
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
  struct Checkpoint {
    std::size_t nodes_size;
    std::size_t stack_size;
    uint32_t mod_idx[4];
    Node mod_nodes[4];
    uint32_t mod_count;
    StackFrame top_frame;
  };

  [[nodiscard]] Checkpoint checkpoint() const {
    Checkpoint cp;
    cp.nodes_size = nodes_.size();
    cp.stack_size = stack_.size();
    cp.mod_count = 0;
    if (!stack_.empty()) {
      cp.top_frame = stack_.back();
      uint32_t p_idx = cp.top_frame.node_idx;
      cp.mod_idx[cp.mod_count] = p_idx;
      cp.mod_nodes[cp.mod_count++] = nodes_[p_idx];

      if (cp.top_frame.last_child_idx != UINT32_MAX) {
        // LHS
        uint32_t l_idx = cp.top_frame.last_child_idx;
        cp.mod_idx[cp.mod_count] = l_idx;
        cp.mod_nodes[cp.mod_count++] = nodes_[l_idx];

        // Prev sibling of LHS (if any)
        uint32_t prev_idx = UINT32_MAX;
        uint32_t it = nodes_[p_idx].first_child;
        while (it != UINT32_MAX && it != l_idx) {
          prev_idx = it;
          it = nodes_[it].next_sibling;
        }
        if (prev_idx != UINT32_MAX) {
          cp.mod_idx[cp.mod_count] = prev_idx;
          cp.mod_nodes[cp.mod_count++] = nodes_[prev_idx];
        }
      }
    }
    return cp;
  }

  void restore(const Checkpoint &cp) {
    // 1. Restore modified node contents before truncating
    for (uint32_t i = 0; i < cp.mod_count; ++i) {
      if (cp.mod_idx[i] < nodes_.size()) {
        nodes_[cp.mod_idx[i]] = cp.mod_nodes[i];
      }
    }
    // 2. Truncate
    nodes_.resize(cp.nodes_size);
    stack_.resize(cp.stack_size);
    if (!stack_.empty()) {
      stack_.back() = cp.top_frame;
    }
  }

  void begin(NodeKind kind, uint32_t token_pos) {
    uint32_t idx = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back({kind, token_pos, token_pos, UINT32_MAX, UINT32_MAX, 0});
    stack_.push_back({idx, UINT32_MAX});
  }

  void end(uint32_t token_pos) {
    if (stack_.empty())
      return;

    StackFrame frame = stack_.back();
    stack_.pop_back();

    uint32_t current_idx = frame.node_idx;
    if (current_idx < nodes_.size()) {
      nodes_[current_idx].token_end = token_pos;
    }

    // Link to parent
    if (!stack_.empty()) {
      StackFrame &parent_frame = stack_.back();
      uint32_t parent_idx = parent_frame.node_idx;

      if (parent_frame.last_child_idx == UINT32_MAX) {
        nodes_[parent_idx].first_child = current_idx;
        parent_frame.last_child_idx = current_idx;
        nodes_[parent_idx].child_count++;
      } else if (parent_frame.last_child_idx != current_idx) {
        nodes_[parent_frame.last_child_idx].next_sibling = current_idx;
        parent_frame.last_child_idx = current_idx;
        nodes_[parent_idx].child_count++;
      }
    }
  }

  void start_infix(NodeKind kind, uint32_t token_pos) {
    if (stack_.empty()) {
      begin(kind, token_pos);
      return;
    }

    StackFrame &parent_frame = stack_.back();
    uint32_t parent_idx = parent_frame.node_idx;

    if (nodes_[parent_idx].child_count == 0 ||
        parent_frame.last_child_idx == UINT32_MAX) {
      begin(kind, token_pos);
      return;
    }

    // LHS is the last added child
    uint32_t lhs_idx = parent_frame.last_child_idx;

    // Create Infix node
    uint32_t infix_idx = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(
        {kind, nodes_[lhs_idx].token_start, token_pos, lhs_idx, UINT32_MAX, 1});

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

inline void end(std::size_t pos) { g_builder.end(static_cast<uint32_t>(pos)); }

inline void start_infix(NodeKind k, std::size_t pos) {
  g_builder.start_infix(k, static_cast<uint32_t>(pos));
}

inline auto tree_checkpoint() { return g_builder.checkpoint(); }
inline void tree_restore(const TreeBuilder::Checkpoint &cp) {
  g_builder.restore(cp);
}

} // namespace cpp2::ast
