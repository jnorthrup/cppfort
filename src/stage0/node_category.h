#ifndef CPPFORT_NODE_CATEGORY_H
#define CPPFORT_NODE_CATEGORY_H

#include "node.h"
#include <type_traits>

namespace cppfort::ir {

/**
 * Band 5: Node Category Predicates
 *
 * Helper class providing compile-time and runtime predicates for
 * classifying nodes by their kinds. Used for pattern matching and
 * optimization passes.
 */
class NodeCategory {
public:
    // ============================================================================
    // Control Flow Categories
    // ============================================================================

    static constexpr bool isControlFlow(NodeKind kind) {
        return kind >= NodeKind::CFG_START && kind <= NodeKind::CFG_END;
    }

    static constexpr bool isDataFlow(NodeKind kind) {
        return kind >= NodeKind::DATA_START && kind <= NodeKind::DATA_END;
    }

    static constexpr bool isArithmetic(NodeKind kind) {
        return kind >= NodeKind::ARITH_START && kind <= NodeKind::ARITH_END;
    }

    static constexpr bool isBitwise(NodeKind kind) {
        return kind >= NodeKind::BITWISE_START && kind <= NodeKind::BITWISE_END;
    }

    static constexpr bool isComparison(NodeKind kind) {
        return kind >= NodeKind::COMPARE_START && kind <= NodeKind::COMPARE_END;
    }

    static constexpr bool isBoolean(NodeKind kind) {
        return kind >= NodeKind::BOOL_START && kind <= NodeKind::BOOL_END;
    }

    static constexpr bool isFloat(NodeKind kind) {
        return kind >= NodeKind::FLOAT_START && kind <= NodeKind::FLOAT_END;
    }

    static constexpr bool isMemory(NodeKind kind) {
        return kind >= NodeKind::MEMORY_START && kind <= NodeKind::MEMORY_END;
    }

    static constexpr bool isConstant(NodeKind kind) {
        return kind >= NodeKind::CONSTANT_START && kind <= NodeKind::CONSTANT_END;
    }

    // ============================================================================
    // Runtime Predicates (for Node* arguments)
    // ============================================================================

    static bool isControlFlow(const Node* node) {
        return node && isControlFlow(node->getKind());
    }

    static bool isDataFlow(const Node* node) {
        return node && isDataFlow(node->getKind());
    }

    static bool isArithmetic(const Node* node) {
        return node && isArithmetic(node->getKind());
    }

    static bool isBitwise(const Node* node) {
        return node && isBitwise(node->getKind());
    }

    static bool isComparison(const Node* node) {
        return node && isComparison(node->getKind());
    }

    static bool isBoolean(const Node* node) {
        return node && isBoolean(node->getKind());
    }

    static bool isFloat(const Node* node) {
        return node && isFloat(node->getKind());
    }

    static bool isMemory(const Node* node) {
        return node && isMemory(node->getKind());
    }

    static bool isConstant(const Node* node) {
        return node && isConstant(node->getKind());
    }

    // ============================================================================
    // Specific Node Type Predicates
    // ============================================================================

    static constexpr bool isBinaryOp(NodeKind kind) {
        return isArithmetic(kind) || isBitwise(kind) || isComparison(kind) ||
               isBoolean(kind) || isFloat(kind);
    }

    static constexpr bool isUnaryOp(NodeKind kind) {
        return kind == NodeKind::NEG || kind == NodeKind::NOT ||
               kind == NodeKind::FNEG || kind == NodeKind::FABS;
    }

    static constexpr bool isCommutative(NodeKind kind) {
        return kind == NodeKind::ADD || kind == NodeKind::MUL ||
               kind == NodeKind::AND || kind == NodeKind::OR ||
               kind == NodeKind::XOR || kind == NodeKind::FADD ||
               kind == NodeKind::FMUL || kind == NodeKind::EQ ||
               kind == NodeKind::BOOL_AND || kind == NodeKind::BOOL_OR;
    }

    static constexpr bool isAssociative(NodeKind kind) {
        return kind == NodeKind::ADD || kind == NodeKind::MUL ||
               kind == NodeKind::AND || kind == NodeKind::OR ||
               kind == NodeKind::XOR || kind == NodeKind::FADD ||
               kind == NodeKind::FMUL;
    }

    static constexpr bool hasSideEffects(NodeKind kind) {
        return isMemory(kind) || kind == NodeKind::CALL || kind == NodeKind::THROW;
    }

    // ============================================================================
    // Runtime Specific Predicates
    // ============================================================================

    static bool isBinaryOp(const Node* node) {
        return node && isBinaryOp(node->getKind());
    }

    static bool isUnaryOp(const Node* node) {
        return node && isUnaryOp(node->getKind());
    }

    static bool isCommutative(const Node* node) {
        return node && isCommutative(node->getKind());
    }

    static bool isAssociative(const Node* node) {
        return node && isAssociative(node->getKind());
    }

    static bool hasSideEffects(const Node* node) {
        return node && hasSideEffects(node->getKind());
    }

    // ============================================================================
    // Chapter 16 Bitwise Operation Predicates
    // ============================================================================

    static constexpr bool isShiftOp(NodeKind kind) {
        return kind == NodeKind::SHL || kind == NodeKind::ASHR || kind == NodeKind::LSHR;
    }

    static constexpr bool isLogicalShift(NodeKind kind) {
        return kind == NodeKind::LSHR;
    }

    static constexpr bool isArithmeticShift(NodeKind kind) {
        return kind == NodeKind::ASHR;
    }

    static constexpr bool isBitCountOp(NodeKind kind) {
        return kind == NodeKind::BIT_COUNT;
    }

    static constexpr bool isBitManipOp(NodeKind kind) {
        return kind == NodeKind::BIT_REVERSE || kind == NodeKind::ROTATE_LEFT ||
               kind == NodeKind::ROTATE_RIGHT;
    }

    // ============================================================================
    // Runtime Chapter 16 Predicates
    // ============================================================================

    static bool isShiftOp(const Node* node) {
        return node && isShiftOp(node->getKind());
    }

    static bool isLogicalShift(const Node* node) {
        return node && isLogicalShift(node->getKind());
    }

    static bool isArithmeticShift(const Node* node) {
        return node && isArithmeticShift(node->getKind());
    }

    static bool isBitCountOp(const Node* node) {
        return node && isBitCountOp(node->getKind());
    }

    static bool isBitManipOp(const Node* node) {
        return node && isBitManipOp(node->getKind());
    }
};

} // namespace cppfort::ir

#endif // CPPFORT_NODE_CATEGORY_H