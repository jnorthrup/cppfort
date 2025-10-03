#include <gtest/gtest.h>
#include "../src/stage0/pattern_matcher.h"
#include "../src/stage0/machine.h"
#include "../src/stage0/node.h"
#include <memory>
#include <sstream>

namespace cppfort::ir {

// ============================================================================
// Integration Test Fixtures
// ============================================================================

class PatternIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        matcher = std::make_unique<PatternMatcher>();
        matcher->clear();
    }

    void TearDown() override {
        matcher.reset();
    }

    std::unique_ptr<PatternMatcher> matcher;
};

// ============================================================================
// Full Pipeline Integration Tests
// ============================================================================

TEST_F(PatternIntegrationTest, ArithmeticPipelineMLIR) {
    // Simulate a complete arithmetic lowering pipeline

    // Register all arithmetic operations for MLIR Arith dialect
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) {
            return "%result = arith.addi %lhs, %rhs : i64";
        },
        10
    );

    matcher->registerPattern(
        NodeKind::SUB,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) {
            return "%result = arith.subi %lhs, %rhs : i64";
        },
        10
    );

    matcher->registerPattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) {
            return "%result = arith.muli %lhs, %rhs : i64";
        },
        10
    );

    matcher->registerPattern(
        NodeKind::DIV,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) {
            return "%result = arith.divsi %lhs, %rhs : i64";
        },
        10
    );

    // Verify all patterns registered
    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::SUB, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::MUL, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::DIV, TargetLanguage::MLIR_ARITH));

    EXPECT_EQ(matcher->getPatternCount(), 4);
}

TEST_F(PatternIntegrationTest, BitwisePipelineMLIR) {
    // Register all bitwise operations for MLIR Arith dialect
    std::vector<std::pair<NodeKind, std::string>> bitwiseOps = {
        {NodeKind::AND, "arith.andi"},
        {NodeKind::OR, "arith.ori"},
        {NodeKind::XOR, "arith.xori"},
        {NodeKind::SHL, "arith.shli"},
        {NodeKind::ASHR, "arith.shrsi"},
        {NodeKind::LSHR, "arith.shrui"}
    };

    for (const auto& [kind, mlirOp] : bitwiseOps) {
        matcher->registerPattern(
            kind,
            TargetLanguage::MLIR_ARITH,
            [mlirOp](Node* n) {
                return "%result = " + mlirOp + " %lhs, %rhs : i64";
            },
            10
        );
    }

    // Verify all patterns registered
    for (const auto& [kind, _] : bitwiseOps) {
        EXPECT_TRUE(matcher->hasPattern(kind, TargetLanguage::MLIR_ARITH));
    }

    EXPECT_EQ(matcher->getPatternCount(), bitwiseOps.size());
}

TEST_F(PatternIntegrationTest, ComparisonPipelineMLIR) {
    // Register comparison operations with proper MLIR syntax
    std::vector<std::pair<NodeKind, std::string>> compareOps = {
        {NodeKind::EQ, "eq"},
        {NodeKind::NE, "ne"},
        {NodeKind::LT, "slt"},
        {NodeKind::LE, "sle"},
        {NodeKind::GT, "sgt"},
        {NodeKind::GE, "sge"}
    };

    for (const auto& [kind, predicate] : compareOps) {
        matcher->registerPattern(
            kind,
            TargetLanguage::MLIR_ARITH,
            [predicate](Node* n) {
                return "%result = arith.cmpi " + predicate + ", %lhs, %rhs : i64";
            },
            10
        );
    }

    EXPECT_EQ(matcher->getPatternCount(), compareOps.size());
}

// ============================================================================
// Multi-Dialect Integration Tests
// ============================================================================

TEST_F(PatternIntegrationTest, MultiDialectLowering) {
    // Register same operation for multiple dialects

    // MLIR Arith dialect
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi %lhs, %rhs : i64"; },
        10
    );

    // MLIR SCF dialect (hypothetical)
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_SCF,
        [](Node* n) { return "scf.add %lhs, %rhs"; },
        10
    );

    // MLIR CF dialect (hypothetical)
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_CF,
        [](Node* n) { return "cf.add %lhs, %rhs"; },
        10
    );

    // Verify patterns registered for all dialects
    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_SCF));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_CF));

    auto patterns = matcher->getPatternsForKind(NodeKind::ADD);
    EXPECT_EQ(patterns.size(), 3);
}

// ============================================================================
// Control Flow Integration Tests
// ============================================================================

TEST_F(PatternIntegrationTest, ControlFlowCFDialect) {
    // Register control flow operations for MLIR CF dialect
    matcher->registerPattern(
        NodeKind::IF,
        TargetLanguage::MLIR_CF,
        [](Node* n) {
            return "cf.cond_br %cond, ^bb_true, ^bb_false";
        },
        10
    );

    matcher->registerPattern(
        NodeKind::RETURN,
        TargetLanguage::MLIR_CF,
        [](Node* n) {
            return "cf.br ^bb_exit";
        },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::IF, TargetLanguage::MLIR_CF));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::RETURN, TargetLanguage::MLIR_CF));
}

TEST_F(PatternIntegrationTest, StructuredControlFlowSCFDialect) {
    // Register structured control flow for MLIR SCF dialect
    matcher->registerPattern(
        NodeKind::IF,
        TargetLanguage::MLIR_SCF,
        [](Node* n) {
            return "scf.if %cond { ... } else { ... }";
        },
        10
    );

    matcher->registerPattern(
        NodeKind::LOOP,
        TargetLanguage::MLIR_SCF,
        [](Node* n) {
            return "scf.while (...) : (...) -> ... { ... }";
        },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::IF, TargetLanguage::MLIR_SCF));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::LOOP, TargetLanguage::MLIR_SCF));
}

// ============================================================================
// Memory Operations Integration Tests
// ============================================================================

TEST_F(PatternIntegrationTest, MemoryOpsMemRefDialect) {
    // Register memory operations for MLIR MemRef dialect
    matcher->registerPattern(
        NodeKind::LOAD,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) {
            return "%value = memref.load %ptr[%idx] : memref<?xi64>";
        },
        10
    );

    matcher->registerPattern(
        NodeKind::STORE,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) {
            return "memref.store %value, %ptr[%idx] : memref<?xi64>";
        },
        10
    );

    matcher->registerPattern(
        NodeKind::ALLOC,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) {
            return "%ptr = memref.alloc() : memref<?xi64>";
        },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::LOAD, TargetLanguage::MLIR_MEMREF));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::STORE, TargetLanguage::MLIR_MEMREF));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::ALLOC, TargetLanguage::MLIR_MEMREF));
}

// ============================================================================
// Function Operations Integration Tests
// ============================================================================

TEST_F(PatternIntegrationTest, FunctionOpsFuncDialect) {
    // Register function operations for MLIR Func dialect
    matcher->registerPattern(
        NodeKind::CALL,
        TargetLanguage::MLIR_FUNC,
        [](Node* n) {
            return "%result = func.call @function(%args) : (i64) -> i64";
        },
        10
    );

    matcher->registerPattern(
        NodeKind::RETURN,
        TargetLanguage::MLIR_FUNC,
        [](Node* n) {
            return "func.return %value : i64";
        },
        10
    );

    EXPECT_TRUE(matcher->hasPattern(NodeKind::CALL, TargetLanguage::MLIR_FUNC));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::RETURN, TargetLanguage::MLIR_FUNC));
}

// ============================================================================
// Priority and Constraint Integration Tests
// ============================================================================

TEST_F(PatternIntegrationTest, PriorityBasedSelection) {
    // Register multiple patterns with different priorities
    matcher->registerPattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.muli (low priority)"; },
        1
    );

    matcher->registerPattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.muli (medium priority)"; },
        5
    );

    matcher->registerPattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.muli (high priority)"; },
        10
    );

    auto patterns = matcher->getPatternsForKind(NodeKind::MUL);
    EXPECT_EQ(patterns.size(), 3);

    // Verify patterns are stored (order doesn't matter for this test)
    EXPECT_TRUE(matcher->hasPattern(NodeKind::MUL, TargetLanguage::MLIR_ARITH));
}

TEST_F(PatternIntegrationTest, ConstraintBasedSelection) {
    // Register patterns with different constraints

    // Pattern 1: No constraints
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi (unconstrained)"; },
        5
    );

    // Pattern 2: Type constraint
    Pattern typeConstrainedPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi (type constrained)"; },
        10
    );
    typeConstrainedPattern.typeConstraint = [](Node* n) {
        return n->_type != nullptr;
    };
    matcher->registerPattern(typeConstrainedPattern);

    // Pattern 3: CFG constraint
    Pattern cfgConstrainedPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi (cfg constrained)"; },
        8
    );
    cfgConstrainedPattern.cfgConstraint = [](Node* n) {
        return !n->isCFG();
    };
    matcher->registerPattern(cfgConstrainedPattern);

    auto patterns = matcher->getPatternsForKind(NodeKind::ADD);
    EXPECT_EQ(patterns.size(), 3);
}

// ============================================================================
// Full Lowering Pipeline Tests
// ============================================================================

TEST_F(PatternIntegrationTest, CompleteArithmeticLoweringPipeline) {
    // Simulate a complete lowering pipeline for arithmetic expression: (a + b) * c

    // Register all needed patterns
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "%add = arith.addi %a, %b : i64"; },
        10
    );

    matcher->registerPattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "%mul = arith.muli %add, %c : i64"; },
        10
    );

    matcher->registerPattern(
        NodeKind::CONSTANT,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "%const = arith.constant 42 : i64"; },
        10
    );

    // Verify complete pipeline is registered
    EXPECT_TRUE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::MUL, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::CONSTANT, TargetLanguage::MLIR_ARITH));

    EXPECT_EQ(matcher->getPatternCount(), 3);
}

TEST_F(PatternIntegrationTest, CompleteMemoryPipeline) {
    // Simulate memory allocation, store, and load pipeline

    matcher->registerPattern(
        NodeKind::ALLOC,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) { return "%ptr = memref.alloc() : memref<10xi64>"; },
        10
    );

    matcher->registerPattern(
        NodeKind::STORE,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) { return "memref.store %value, %ptr[%idx] : memref<10xi64>"; },
        10
    );

    matcher->registerPattern(
        NodeKind::LOAD,
        TargetLanguage::MLIR_MEMREF,
        [](Node* n) { return "%value = memref.load %ptr[%idx] : memref<10xi64>"; },
        10
    );

    EXPECT_EQ(matcher->getPatternCount(), 3);
}

// ============================================================================
// Error Handling and Edge Cases
// ============================================================================

TEST_F(PatternIntegrationTest, NoPatternAvailable) {
    // Try to query pattern that doesn't exist
    EXPECT_FALSE(matcher->hasPattern(NodeKind::SQRT, TargetLanguage::MLIR_ARITH));

    auto patterns = matcher->getPatternsForKind(NodeKind::SQRT);
    EXPECT_TRUE(patterns.empty());
}

TEST_F(PatternIntegrationTest, ClearAndReinitialize) {
    // Register patterns
    matcher->registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.addi"; },
        10
    );

    EXPECT_EQ(matcher->getPatternCount(), 1);

    // Clear
    matcher->clear();
    EXPECT_EQ(matcher->getPatternCount(), 0);

    // Re-register
    matcher->registerPattern(
        NodeKind::SUB,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.subi"; },
        10
    );

    EXPECT_EQ(matcher->getPatternCount(), 1);
    EXPECT_FALSE(matcher->hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher->hasPattern(NodeKind::SUB, TargetLanguage::MLIR_ARITH));
}

// ============================================================================
// Floating Point Integration Tests
// ============================================================================

TEST_F(PatternIntegrationTest, FloatingPointPipeline) {
    // Register floating point operations
    std::vector<std::pair<NodeKind, std::string>> floatOps = {
        {NodeKind::FADD, "arith.addf"},
        {NodeKind::FSUB, "arith.subf"},
        {NodeKind::FMUL, "arith.mulf"},
        {NodeKind::FDIV, "arith.divf"}
    };

    for (const auto& [kind, mlirOp] : floatOps) {
        matcher->registerPattern(
            kind,
            TargetLanguage::MLIR_ARITH,
            [mlirOp](Node* n) {
                return "%result = " + mlirOp + " %lhs, %rhs : f64";
            },
            10
        );
    }

    EXPECT_EQ(matcher->getPatternCount(), floatOps.size());

    for (const auto& [kind, _] : floatOps) {
        EXPECT_TRUE(matcher->hasPattern(kind, TargetLanguage::MLIR_ARITH));
    }
}

// ============================================================================
// Pattern Introspection Tests
// ============================================================================

TEST_F(PatternIntegrationTest, PatternIntrospection) {
    // Register diverse patterns
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
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "arith.muli"; },
        10
    );

    // Test introspection
    auto addPatterns = matcher->getPatternsForKind(NodeKind::ADD);
    EXPECT_EQ(addPatterns.size(), 2);

    auto mulPatterns = matcher->getPatternsForKind(NodeKind::MUL);
    EXPECT_EQ(mulPatterns.size(), 1);

    EXPECT_EQ(matcher->getPatternCount(), 3);
}

} // namespace cppfort::ir
