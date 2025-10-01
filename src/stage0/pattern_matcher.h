#ifndef CPPFORT_PATTERN_MATCHER_H
#define CPPFORT_PATTERN_MATCHER_H

#include "node.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace cppfort::ir {

// TargetLanguage is defined in node.h to avoid circular dependencies

/**
 * Band 5: Pattern structure for declarative lowering.
 *
 * A Pattern encapsulates:
 * - Which nodes it matches (NodeKind)
 * - Target language
 * - Rewrite function (Node* → target code string)
 * - Priority (for disambiguation)
 * - Constraints (type, CFG, scheduling)
 */
struct Pattern {
    NodeKind kind;              // Which node kind this pattern matches
    TargetLanguage target;      // Target language for this lowering
    int priority;               // Higher priority wins (for disambiguation)

    /**
     * Rewrite function: transforms Node* into target language code.
     *
     * Example for AddNode → C:
     *   [](Node* n) {
     *       auto* add = static_cast<AddNode*>(n);
     *       return format("{} + {}", emit(add->lhs()), emit(add->rhs()));
     *   }
     */
    ::std::function<::std::string(Node*)> rewrite;

    /**
     * Optional: Type constraint predicate.
     * Returns true if node's type is compatible with this pattern.
     *
     * Example: Only match integer operations
     *   [](Node* n) { return n->_type->isInteger(); }
     */
    ::std::function<bool(Node*)> typeConstraint;

    /**
     * Optional: CFG legality constraint.
     * Returns true if node's CFG context allows this lowering.
     *
     * Example: Only lower if not in loop
     *   [](Node* n) { return !inLoop(n); }
     */
    ::std::function<bool(Node*)> cfgConstraint;

    Pattern(NodeKind k, TargetLanguage t, ::std::function<::std::string(Node*)> rw, int pri = 0)
        : kind(k), target(t), rewrite(rw), priority(pri),
          typeConstraint(nullptr), cfgConstraint(nullptr) {}
};

/**
 * Band 5: PatternMatcher - Core n-way lowering engine.
 *
 * The PatternMatcher implements cppfort's strategic divergence from
 * Simple compiler: pattern-based multi-target code generation.
 *
 * Architecture:
 *   PatternMatcher registry: (NodeKind, TargetLanguage) → Pattern
 *
 * Usage:
 *   PatternMatcher pm;
 *   pm.registerPattern(NodeKind::ADD, TargetLanguage::C,
 *       [](Node* n) { return "a + b"; });
 *   pm.registerPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH,
 *       [](Node* n) { return "arith.addi"; });
 *
 *   ::std::string c_code = pm.match(addNode, TargetLanguage::C);
 *   ::std::string mlir = pm.match(addNode, TargetLanguage::MLIR_ARITH);
 */
class PatternMatcher {
private:
    /**
     * Pattern registry: maps (NodeKind, TargetLanguage) to list of patterns.
     * Multiple patterns per key supported (sorted by priority).
     */
    using PatternKey = ::std::pair<NodeKind, TargetLanguage>;

    struct PatternKeyHash {
        ::std::size_t operator()(const PatternKey& k) const {
            return ::std::hash<int>()(static_cast<int>(k.first)) ^
                   (::std::hash<int>()(static_cast<int>(k.second)) << 1);
        }
    };

    ::std::unordered_map<PatternKey, ::std::vector<Pattern>, PatternKeyHash> _registry;

public:
    /**
     * Find best matching pattern for (node, target).
     * Returns nullptr if no pattern matches.
     * Made public for InstructionSelection to use.
     */
    const Pattern* findBestMatch(Node* node, TargetLanguage target) const;
    PatternMatcher();
    ~PatternMatcher() = default;

    /**
     * Register a pattern for (NodeKind, TargetLanguage).
     *
     * Patterns are stored sorted by priority (highest first).
     * Multiple patterns for same key are allowed (disambiguation via priority).
     *
     * @param kind       Node kind this pattern matches
     * @param target     Target language for lowering
     * @param rewrite    Function that transforms Node* → target code
     * @param priority   Priority for disambiguation (default 0)
     */
    void registerPattern(
        NodeKind kind,
        TargetLanguage target,
        ::std::function<::std::string(Node*)> rewrite,
        int priority = 0
    );

    /**
     * Register a pattern with a type constraint.
     *
     * @param kind          Node kind this pattern matches
     * @param target        Target language for lowering
     * @param rewrite       Function that transforms Node* → target code
     * @param priority      Priority for disambiguation
     * @param constraint    Type constraint predicate
     */
    void registerPattern(
        NodeKind kind,
        TargetLanguage target,
        ::std::function<::std::string(Node*)> rewrite,
        int priority,
        ::std::function<bool(Node*)> constraint
    );

    /**
     * Register a pattern with constraints.
     *
     * @param pattern    Pattern with type/CFG constraints
     */
    void registerPattern(Pattern pattern);

    /**
     * Match a node and generate target code.
     *
     * Process:
     * 1. Look up patterns for (node->getKind(), target)
     * 2. Filter by type/CFG constraints
     * 3. Select highest priority match
     * 4. Apply rewrite function
     *
     * @param node    Node to lower
     * @param target  Target language
     * @return        Generated target code, or empty string if no match
     */
    ::std::string match(Node* node, TargetLanguage target) const;

    /**
     * Check if a pattern exists for (NodeKind, TargetLanguage).
     *
     * @param kind    Node kind to check
     * @param target  Target language
     * @return        True if at least one pattern registered
     */
    bool hasPattern(NodeKind kind, TargetLanguage target) const;

    /**
     * Get all registered patterns for a NodeKind (any target).
     * Useful for debugging and introspection.
     *
     * @param kind    Node kind to query
     * @return        Vector of all patterns for this kind
     */
    ::std::vector<Pattern> getPatternsForKind(NodeKind kind) const;

    /**
     * Get pattern count for debugging.
     *
     * @return  Total number of registered patterns across all keys
     */
    size_t getPatternCount() const;

    /**
     * Clear all registered patterns.
     * Useful for testing and reinitialization.
     */
    void clear();

    /**
     * Register builtin patterns for common operations.
     * Called by constructor to populate default patterns.
     *
     * Includes:
     * - Arithmetic operations (Add, Sub, Mul, Div) → C, MLIR
     * - Bitwise operations (And, Or, Xor, Shl) → C, MLIR
     * - Comparison operations (Eq, Lt, Gt) → C, MLIR
     * - Control flow (If, Loop, Return) → C, MLIR
     * - Memory operations (Load, Store) → C, MLIR
     */
    void registerBuiltinPatterns();
};

/**
 * Band 5: Helper functions for pattern emission.
 * These utilities simplify writing rewrite functions.
 */
namespace pattern_helpers {
    /**
     * Emit a node recursively (calls PatternMatcher on child nodes).
     *
     * @param node     Node to emit
     * @param matcher  PatternMatcher to use
     * @param target   Target language
     * @return         Emitted code for node
     */
    ::std::string emitNode(Node* node, const PatternMatcher& matcher, TargetLanguage target);

    /**
     * Format binary operation: "lhs op rhs"
     *
     * @param node     BinaryOp node
     * @param op       Operator string (e.g., "+", "&&")
     * @param matcher  PatternMatcher for recursive emission
     * @param target   Target language
     * @return         Formatted binary expression
     */
    ::std::string formatBinaryOp(Node* node, const ::std::string& op,
                               const PatternMatcher& matcher, TargetLanguage target);

    /**
     * Format unary operation: "op expr"
     *
     * @param node     UnaryOp node
     * @param op       Operator string (e.g., "-", "!")
     * @param matcher  PatternMatcher for recursive emission
     * @param target   Target language
     * @return         Formatted unary expression
     */
    ::std::string formatUnaryOp(Node* node, const ::std::string& op,
                              const PatternMatcher& matcher, TargetLanguage target);
}

} // namespace cppfort::ir

#endif // CPPFORT_PATTERN_MATCHER_H
