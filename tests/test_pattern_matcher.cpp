#include <gtest/gtest.h>
#include "../src/stage0/pattern_matcher.h"
#include "../src/stage0/node.h"
#include <sstream>

namespace cppfort::ir {

// ============================================================================
// Test Fixtures and Mock Nodes
// ============================================================================

class PatternMatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        matcher = new PatternMatcher();
        matcher->clear();
    }

    void TearDown() override {
        delete matcher;
    }

    PatternMatcher* matcher;
};

// ============================================================================
// Basic Pattern Registration Tests
// ============================================================================

TEST_F(PatternMatcherTest, RegisterSimplePattern) {
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi"; },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
    EXPECT_EQ(matcher->getPatternCount(), 1);
}

TEST_F(PatternMatcherTest, RegisterMultiplePatternsSameKind) {
    // Register two patterns for ADD with different priorities
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi (low priority)"; },
        5
    );

    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi (high priority)"; },
        10
    );

    EXPECT_EQ(matcher->getPatternCount(), 2);
}

TEST_F(PatternMatcherTest, RegisterPatternsMultipleTargets) {
    // Register ADD for different target languages
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi"; },
        10
    );

    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_SCF,
        [](Node* n) { return "scf.add"; },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_SCF));
    EXPECT_EQ(matcher->getPatternCount(), 2);
}

TEST_F(PatternMatcherTest, HasPatternReturnsFalseForUnregistered) {
    EXPECT_FALSE(matcher->hasPattern(NodeKind::MUL, TargetLanguage::MLIR_ARITH));
}

// ============================================================================
// Pattern Priority Tests
// ============================================================================

TEST_F(PatternMatcherTest, PatternPriorityOrdering) {
    // Register patterns in reverse priority order
    matcher->registerPattern(
        NodeKind::SUB,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "low"; },
        1
    );

    matcher->registerPattern(
        NodeKind::SUB,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "medium"; },
        5
    );

    matcher->registerPattern(
        NodeKind::SUB,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "high"; },
        10
    );

    auto patterns = matcher->getPatternsForKind(NodeKind::SUB);
    EXPECT_EQ(patterns.size(), 3);
}

// ============================================================================
// Pattern Retrieval Tests
// ============================================================================

TEST_F(PatternMatcherTest, GetPatternsForKind) {
    matcher->registerPattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.muli"; },
        10
    );

    matcher->registerPattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_SCF,
        [](Node* n) { return "scf.mul"; },
        10
    );

    auto patterns = matcher->getPatternsForKind(NodeKind::MUL);
    EXPECT_EQ(patterns.size(), 2);
}

TEST_F(PatternMatcherTest, GetPatternsForKindEmpty) {
    auto patterns = matcher->getPatternsForKind(NodeKind::DIV);
    EXPECT_TRUE(patterns.empty());
}

// ============================================================================
// Multi-Target Lowering Tests
// ============================================================================

TEST_F(PatternMatcherTest, ArithmeticToMLIRArith) {
    // Register arithmetic operations to MLIR Arithmetic dialect
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi"; },
        10
    );

    matcher->registerPattern(
        NodeKind::SUB,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.subi"; },
        10
    );

    matcher->registerPattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.muli"; },
        10
    );

    matcher->registerPattern(
        NodeKind::DIV,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.divsi"; },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::SUB, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::MUL, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::DIV, TargetLanguage::MLIR_ARITH));
}

TEST_F(PatternMatcherTest, BitwiseToMLIRArith) {
    // Register bitwise operations to MLIR Arithmetic dialect
    matcher->registerPattern(
        NodeKind::AND,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.andi"; },
        10
    );

    matcher->registerPattern(
        NodeKind::OR,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.ori"; },
        10
    );

    matcher->registerPattern(
        NodeKind::XOR,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.xori"; },
        10
    );

    matcher->registerPattern(
        NodeKind::SHL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.shli"; },
        10
    );

    matcher->registerPattern(
        NodeKind::ASHR,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.shrsi"; },
        10
    );

    matcher->registerPattern(
        NodeKind::LSHR,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.shrui"; },
        10
    );

    EXPECT_EQ(matcher->getPatternCount(), 6);
}

TEST_F(PatternMatcherTest, ComparisonToMLIRArith) {
    // Register comparison operations
    matcher->registerPattern(
        NodeKind::EQ,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.cmpi eq"; },
        10
    );

    matcher->registerPattern(
        NodeKind::NE,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.cmpi ne"; },
        10
    );

    matcher->registerPattern(
        NodeKind::LT,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.cmpi slt"; },
        10
    );

    matcher->registerPattern(
        NodeKind::LE,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.cmpi sle"; },
        10
    );

    matcher->registerPattern(
        NodeKind::GT,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.cmpi sgt"; },
        10
    );

    matcher->registerPattern(
        NodeKind::GE,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.cmpi sge"; },
        10
    );

    EXPECT_EQ(matcher->getPatternCount(), 6);
}

// ============================================================================
// Control Flow Lowering Tests
// ============================================================================

TEST_F(PatternMatcherTest, ControlFlowToMLIRCF) {
    // Register control flow operations to MLIR CF dialect
    matcher->registerPattern(
        NodeKind::IF,
        TargetLanguage::MLIR_CF,
        [](Node* n) { return "cf.cond_br"; },
        10
    );

    matcher->registerPattern(
        NodeKind::RETURN,
        TargetLanguage::MLIR_CF,
        [](Node* n) { return "cf.br"; },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::IF, TargetLanguage::MLIR_CF));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::RETURN, TargetLanguage::MLIR_CF));
}

TEST_F(PatternMatcherTest, StructuredControlFlowToMLIRSCF) {
    // Register structured control flow to MLIR SCF dialect
    matcher->registerPattern(
        NodeKind::IF,
        TargetLanguage::MLIR_SCF,
        [](Node* n) { return "scf.if"; },
        10
    );

    matcher->registerPattern(
        NodeKind::LOOP,
        TargetLanguage::MLIR_SCF,
        [](Node* n) { return "scf.while"; },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::IF, TargetLanguage::MLIR_SCF));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::LOOP, TargetLanguage::MLIR_SCF));
}

// ============================================================================
// Memory Operations Lowering Tests
// ============================================================================

TEST_F(PatternMatcherTest, MemoryOpsToMLIRMemRef) {
    // Register memory operations to MLIR MemRef dialect
    matcher->registerPattern(
        NodeKind::LOAD,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) { return "memref.load"; },
        10
    );

    matcher->registerPattern(
        NodeKind::STORE,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) { return "memref.store"; },
        10
    );

    matcher->registerPattern(
        NodeKind::ALLOC,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) { return "memref.alloc"; },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::LOAD, TargetLanguage::MLIR_MEMREF));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::STORE, TargetLanguage::MLIR_MEMREF));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::ALLOC, TargetLanguage::MLIR_MEMREF));
}

// ============================================================================
// Function Operations Lowering Tests
// ============================================================================

TEST_F(PatternMatcherTest, FunctionOpsToMLIRFunc) {
    // Register function operations to MLIR Func dialect
    matcher->registerPattern(
        NodeKind::CALL,
        TargetLanguage::MLIR_FUNC,
        [](Node* n) { return "func.call"; },
        10
    );

    matcher->registerPattern(
        NodeKind::RETURN,
        TargetLanguage::MLIR_FUNC,
        [](Node* n) { return "func.return"; },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::CALL, TargetLanguage::MLIR_FUNC));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::RETURN, TargetLanguage::MLIR_FUNC));
}

// ============================================================================
// Pattern Constraint Tests
// ============================================================================

TEST_F(PatternMatcherTest, PatternWithTypeConstraint) {
    // Register pattern with type constraint
    Pattern pattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi (constrained)"; },
        10
    );

    pattern.typeConstraint = [](Node* n) {
        // Mock constraint: only match if node has type
        return n->_type != nullptr;
    };

    matcher->registerPattern(pattern);
    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
}

TEST_F(PatternMatcherTest, PatternWithCFGConstraint) {
    // Register pattern with CFG constraint
    Pattern pattern(
        NodeKind::LOAD,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) { return "memref.load (constrained)"; },
        10
    );

    pattern.cfgConstraint = [](Node* n) {
        // Mock constraint: only match if node is CFG
        return n->isCFG();
    };

    matcher->registerPattern(pattern);
    EXPECT_TRUE(matcher->hasPattern(NodeKind::LOAD, TargetLanguage::MLIR_MEMREF));
}

TEST_F(PatternMatcherTest, PatternWithBothConstraints) {
    // Register pattern with both type and CFG constraints
    Pattern pattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.muli (double constrained)"; },
        10
    );

    pattern.typeConstraint = [](Node* n) { return n->_type != nullptr; };
    pattern.cfgConstraint = [](Node* n) { return !n->isCFG(); };

    matcher->registerPattern(pattern);
    EXPECT_TRUE(matcher->hasPattern(NodeKind::MUL, TargetLanguage::MLIR_ARITH));
}

// ============================================================================
// Clear and Reset Tests
// ============================================================================

TEST_F(PatternMatcherTest, ClearRemovesAllPatterns) {
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi"; },
        10
    );

    matcher->registerPattern(
        NodeKind::SUB,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.subi"; },
        10
    );

    EXPECT_EQ(matcher->getPatternCount(), 2);

    matcher->clear();

    EXPECT_EQ(matcher->getPatternCount(), 0);
    EXPECT_FALSE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
    EXPECT_FALSE(matcher->hasPattern(NodeKind::SUB, TargetLanguage::MLIR_ARITH));
}

// ============================================================================
// Comprehensive Coverage Tests
// ============================================================================

TEST_F(PatternMatcherTest, RegisterAllArithmeticOperations) {
    // Test coverage for all arithmetic operations
    const std::vector<NodeKind> arithOps = {
        NodeKind::ADD, NodeKind::SUB, NodeKind::MUL, NodeKind::DIV,
        NodeKind::MOD, NodeKind::NEG, NodeKind::ABS
    };

    for (const auto& op : arithOps) {
        matcher->registerPattern(
            op,
            TargetLanguage::MLIR_ARITH,
            [](Node* n) { return "arith.op"; },
            10
        );
    }

    EXPECT_EQ(matcher->getPatternCount(), arithOps.size());

    for (const auto& op : arithOps) {
        EXPECT_TRUE(matcher->hasPattern(op, TargetLanguage::MLIR_ARITH));
    }
}

TEST_F(PatternMatcherTest, RegisterAllBitwiseOperations) {
    // Test coverage for all bitwise operations
    const std::vector<NodeKind> bitwiseOps = {
        NodeKind::AND, NodeKind::OR, NodeKind::XOR,
        NodeKind::SHL, NodeKind::ASHR, NodeKind::LSHR, NodeKind::NOT
    };

    for (const auto& op : bitwiseOps) {
        matcher->registerPattern(
            op,
            TargetLanguage::MLIR_ARITH,
            [](Node* n) { return "arith.bitwise"; },
            10
        );
    }

    EXPECT_EQ(matcher->getPatternCount(), bitwiseOps.size());
}

TEST_F(PatternMatcherTest, RegisterAllComparisonOperations) {
    // Test coverage for all comparison operations
    const std::vector<NodeKind> compareOps = {
        NodeKind::EQ, NodeKind::NE, NodeKind::LT,
        NodeKind::LE, NodeKind::GT, NodeKind::GE
    };

    for (const auto& op : compareOps) {
        matcher->registerPattern(
            op,
            TargetLanguage::MLIR_ARITH,
            [](Node* n) { return "arith.cmpi"; },
            10
        );
    }

    EXPECT_EQ(matcher->getPatternCount(), compareOps.size());
}

TEST_F(PatternMatcherTest, RegisterAllFloatingPointOperations) {
    // Test coverage for floating point operations
    const std::vector<NodeKind> floatOps = {
        NodeKind::FADD, NodeKind::FSUB, NodeKind::FMUL,
        NodeKind::FDIV, NodeKind::FNEG, NodeKind::FABS
    };

    for (const auto& op : floatOps) {
        matcher->registerPattern(
            op,
            TargetLanguage::MLIR_ARITH,
            [](Node* n) { return "arith.float"; },
            10
        );
    }

    EXPECT_EQ(matcher->getPatternCount(), floatOps.size());
}

// ============================================================================
// Cross-Dialect Pattern Tests
// ============================================================================

TEST_F(PatternMatcherTest, SameNodeMultipleDialects) {
    // Register same node kind for multiple dialects
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi"; },
        10
    );

    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_SCF,
        [](Node* n) { return "scf.add"; },
        10
    );

    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_CF,
        [](Node* n) { return "cf.add"; },
        10
    );

    auto patterns = matcher->getPatternsForKind(NodeKind::ADD);
    EXPECT_EQ(patterns.size(), 3);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(PatternMatcherTest, EmptyMatcher) {
    EXPECT_EQ(matcher->getPatternCount(), 0);
    EXPECT_FALSE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
}

TEST_F(PatternMatcherTest, NullNodeMatching) {
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi"; },
        10
    );

    // Match with null node should return empty string
    std::string result = matcher->match(nullptr, TargetLanguage::MLIR_ARITH);
    EXPECT_EQ(result, "");
}

TEST_F(PatternMatcherTest, MatchUnregisteredPattern) {
    // Try to match a pattern that doesn't exist
    // This would require a mock node, which we'll skip for now
    // as it requires more infrastructure
}

// ============================================================================
// Pattern Helper Tests
// ============================================================================

TEST_F(PatternMatcherTest, EmitNodeHelper) {
    // Test pattern_helpers::emitNode with null
    std::string result = pattern_helpers::emitNode(nullptr, *matcher, TargetLanguage::MLIR_ARITH);
    EXPECT_EQ(result, "<null>");
}

TEST_F(PatternMatcherTest, FormatBinaryOpHelper) {
    // This test requires mock nodes - placeholder for integration
    // Would test pattern_helpers::formatBinaryOp
}

TEST_F(PatternMatcherTest, FormatUnaryOpHelper) {
    // This test requires mock nodes - placeholder for integration
    // Would test pattern_helpers::formatUnaryOp
}

} // namespace cppfort::ir
