#include <gtest/gtest.h>
#include "../src/stage0/machine.h"
#include "../src/stage0/pattern_matcher.h"
#include <memory>

namespace cppfort::ir {

// ============================================================================
// Machine Pattern Test Fixtures
// ============================================================================

class MachinePatternTest : public ::testing::Test {
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
// MLIR Arith Machine Tests
// ============================================================================

TEST_F(MachinePatternTest, MLIRArithMachineBasics) {
    MLIRArithMachine arithMachine;

    EXPECT_EQ(arithMachine.name(), "mlir-arith");
    EXPECT_EQ(arithMachine.targetLanguage(), TargetLanguage::MLIR_ARITH);
}

TEST_F(MachinePatternTest, MLIRArithMachineCanHandleArithmetic) {
    MLIRArithMachine arithMachine;

    // Should handle arithmetic operations
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::ADD));
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::SUB));
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::MUL));
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::DIV));
}

TEST_F(MachinePatternTest, MLIRArithMachineCanHandleBitwise) {
    MLIRArithMachine arithMachine;

    // Should handle bitwise operations
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::AND));
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::OR));
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::XOR));
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::SHL));
}

TEST_F(MachinePatternTest, MLIRArithMachineCanHandleComparison) {
    MLIRArithMachine arithMachine;

    // Should handle comparison operations
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::EQ));
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::NE));
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::LT));
    EXPECT_TRUE(arithMachine.canHandle(NodeKind::GT));
}

TEST_F(MachinePatternTest, MLIRArithMachineRegisterPatterns) {
    MLIRArithMachine arithMachine;

    // Register patterns with matcher
    arithMachine.registerPatterns(*matcher);

    // Verify patterns were registered
    EXPECT_GT(matcher->getPatternCount(), 0);
}

// ============================================================================
// MLIR CF Machine Tests
// ============================================================================

TEST_F(MachinePatternTest, MLIRCFMachineBasics) {
    MLIRCFMachine cfMachine;

    EXPECT_EQ(cfMachine.name(), "mlir-cf");
    EXPECT_EQ(cfMachine.targetLanguage(), TargetLanguage::MLIR_CF);
}

TEST_F(MachinePatternTest, MLIRCFMachineCanHandleControlFlow) {
    MLIRCFMachine cfMachine;

    // Should handle control flow operations
    EXPECT_TRUE(cfMachine.canHandle(NodeKind::IF));
    EXPECT_TRUE(cfMachine.canHandle(NodeKind::RETURN));
}

TEST_F(MachinePatternTest, MLIRCFMachineRegisterPatterns) {
    MLIRCFMachine cfMachine;

    cfMachine.registerPatterns(*matcher);

    // Verify patterns were registered
    EXPECT_GT(matcher->getPatternCount(), 0);
}

// ============================================================================
// MLIR SCF Machine Tests
// ============================================================================

TEST_F(MachinePatternTest, MLIRSCFMachineBasics) {
    MLIRSCFMachine scfMachine;

    EXPECT_EQ(scfMachine.name(), "mlir-scf");
    EXPECT_EQ(scfMachine.targetLanguage(), TargetLanguage::MLIR_SCF);
}

TEST_F(MachinePatternTest, MLIRSCFMachineCanHandleStructuredControlFlow) {
    MLIRSCFMachine scfMachine;

    // Should handle structured control flow
    EXPECT_TRUE(scfMachine.canHandle(NodeKind::IF));
    EXPECT_TRUE(scfMachine.canHandle(NodeKind::LOOP));
}

TEST_F(MachinePatternTest, MLIRSCFMachineRegisterPatterns) {
    MLIRSCFMachine scfMachine;

    scfMachine.registerPatterns(*matcher);

    // Verify patterns were registered
    EXPECT_GT(matcher->getPatternCount(), 0);
}

// ============================================================================
// MLIR MemRef Machine Tests
// ============================================================================

TEST_F(MachinePatternTest, MLIRMemRefMachineBasics) {
    MLIRMemRefMachine memrefMachine;

    EXPECT_EQ(memrefMachine.name(), "mlir-memref");
    EXPECT_EQ(memrefMachine.targetLanguage(), TargetLanguage::MLIR_MEMREF);
}

TEST_F(MachinePatternTest, MLIRMemRefMachineCanHandleMemoryOps) {
    MLIRMemRefMachine memrefMachine;

    // Should handle memory operations
    EXPECT_TRUE(memrefMachine.canHandle(NodeKind::LOAD));
    EXPECT_TRUE(memrefMachine.canHandle(NodeKind::STORE));
    EXPECT_TRUE(memrefMachine.canHandle(NodeKind::ALLOC));
}

TEST_F(MachinePatternTest, MLIRMemRefMachineRegisterPatterns) {
    MLIRMemRefMachine memrefMachine;

    memrefMachine.registerPatterns(*matcher);

    // Verify patterns were registered
    EXPECT_GT(matcher->getPatternCount(), 0);
}

// ============================================================================
// MLIR Func Machine Tests
// ============================================================================

TEST_F(MachinePatternTest, MLIRFuncMachineBasics) {
    MLIRFuncMachine funcMachine;

    EXPECT_EQ(funcMachine.name(), "mlir-func");
    EXPECT_EQ(funcMachine.targetLanguage(), TargetLanguage::MLIR_FUNC);
}

TEST_F(MachinePatternTest, MLIRFuncMachineCanHandleFunctionOps) {
    MLIRFuncMachine funcMachine;

    // Should handle function operations
    EXPECT_TRUE(funcMachine.canHandle(NodeKind::CALL));
    EXPECT_TRUE(funcMachine.canHandle(NodeKind::RETURN));
}

TEST_F(MachinePatternTest, MLIRFuncMachineRegisterPatterns) {
    MLIRFuncMachine funcMachine;

    funcMachine.registerPatterns(*matcher);

    // Verify patterns were registered
    EXPECT_GT(matcher->getPatternCount(), 0);
}

// ============================================================================
// Machine Registry Tests
// ============================================================================

class MachineRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry = std::make_unique<MachineRegistry>();
    }

    void TearDown() override {
        registry.reset();
    }

    std::unique_ptr<MachineRegistry> registry;
};

TEST_F(MachineRegistryTest, DefaultConstructorRegistersStandardMachines) {
    // Default constructor should register standard MLIR dialect machines
    EXPECT_TRUE(registry->hasMachine("mlir-arith"));
    EXPECT_TRUE(registry->hasMachine("mlir-cf"));
    EXPECT_TRUE(registry->hasMachine("mlir-scf"));
    EXPECT_TRUE(registry->hasMachine("mlir-memref"));
    EXPECT_TRUE(registry->hasMachine("mlir-func"));
}

TEST_F(MachineRegistryTest, GetMachine) {
    Machine* arithMachine = registry->getMachine("mlir-arith");
    ASSERT_NE(arithMachine, nullptr);
    EXPECT_EQ(arithMachine->name(), "mlir-arith");

    Machine* cfMachine = registry->getMachine("mlir-cf");
    ASSERT_NE(cfMachine, nullptr);
    EXPECT_EQ(cfMachine->name(), "mlir-cf");
}

TEST_F(MachineRegistryTest, GetNonexistentMachine) {
    Machine* machine = registry->getMachine("nonexistent");
    EXPECT_EQ(machine, nullptr);
}

TEST_F(MachineRegistryTest, HasMachine) {
    EXPECT_TRUE(registry->hasMachine("mlir-arith"));
    EXPECT_FALSE(registry->hasMachine("nonexistent"));
}

TEST_F(MachineRegistryTest, AvailableMachines) {
    auto machines = registry->availableMachines();

    // Should have all standard MLIR dialect machines
    EXPECT_GE(machines.size(), 5);

    // Check that standard machines are present
    bool hasArith = false, hasCF = false, hasSCF = false, hasMemRef = false, hasFunc = false;
    for (const auto& name : machines) {
        if (name == "mlir-arith") hasArith = true;
        if (name == "mlir-cf") hasCF = true;
        if (name == "mlir-scf") hasSCF = true;
        if (name == "mlir-memref") hasMemRef = true;
        if (name == "mlir-func") hasFunc = true;
    }

    EXPECT_TRUE(hasArith);
    EXPECT_TRUE(hasCF);
    EXPECT_TRUE(hasSCF);
    EXPECT_TRUE(hasMemRef);
    EXPECT_TRUE(hasFunc);
}

TEST_F(MachineRegistryTest, RegisterCustomMachine) {
    // Create a custom machine
    class CustomMachine : public Machine {
    public:
        std::string name() const override { return "custom"; }
        TargetLanguage targetLanguage() const override { return TargetLanguage::UNKNOWN; }
        void registerPatterns(PatternMatcher& matcher) override {}
        bool canHandle(NodeKind kind) const override { return false; }
    };

    registry->registerMachine(std::make_unique<CustomMachine>());

    EXPECT_TRUE(registry->hasMachine("custom"));

    Machine* customMachine = registry->getMachine("custom");
    ASSERT_NE(customMachine, nullptr);
    EXPECT_EQ(customMachine->name(), "custom");
}

TEST_F(MachineRegistryTest, ReplaceExistingMachine) {
    // Create a custom machine with same name as existing one
    class CustomArithMachine : public Machine {
    public:
        std::string name() const override { return "mlir-arith"; }
        TargetLanguage targetLanguage() const override { return TargetLanguage::MLIR_ARITH; }
        void registerPatterns(PatternMatcher& matcher) override {}
        bool canHandle(NodeKind kind) const override { return true; }
    };

    // Register should replace existing machine
    registry->registerMachine(std::make_unique<CustomArithMachine>());

    EXPECT_TRUE(registry->hasMachine("mlir-arith"));

    Machine* arithMachine = registry->getMachine("mlir-arith");
    ASSERT_NE(arithMachine, nullptr);
    EXPECT_EQ(arithMachine->name(), "mlir-arith");
}

TEST_F(MachineRegistryTest, RegisterStandardMachines) {
    // Create empty registry
    MachineRegistry emptyRegistry;

    // Note: Default constructor already registers standard machines
    // So we can't test truly empty registry without modifying constructor
    // This test verifies the method exists and works

    emptyRegistry.registerStandardMachines();

    EXPECT_TRUE(emptyRegistry.hasMachine("mlir-arith"));
    EXPECT_TRUE(emptyRegistry.hasMachine("mlir-cf"));
    EXPECT_TRUE(emptyRegistry.hasMachine("mlir-scf"));
    EXPECT_TRUE(emptyRegistry.hasMachine("mlir-memref"));
    EXPECT_TRUE(emptyRegistry.hasMachine("mlir-func"));
}

// ============================================================================
// Machine Integration with PatternMatcher Tests
// ============================================================================

TEST_F(MachineRegistryTest, AllMachinesRegisterPatterns) {
    PatternMatcher matcher;
    matcher.clear();

    // Get all machines and register their patterns
    auto machineNames = registry->availableMachines();

    for (const auto& name : machineNames) {
        Machine* machine = registry->getMachine(name);
        ASSERT_NE(machine, nullptr);
        machine->registerPatterns(matcher);
    }

    // Should have patterns registered from all machines
    EXPECT_GT(matcher.getPatternCount(), 0);
}

TEST_F(MachineRegistryTest, MachineSpecificPatternRegistration) {
    PatternMatcher matcher;
    matcher.clear();

    // Register only MLIR Arith machine patterns
    Machine* arithMachine = registry->getMachine("mlir-arith");
    ASSERT_NE(arithMachine, nullptr);

    arithMachine->registerPatterns(matcher);

    // Should have arithmetic patterns
    EXPECT_GT(matcher.getPatternCount(), 0);

    // Verify some arithmetic patterns exist
    EXPECT_TRUE(matcher.hasPattern(NodeKind::ADD, TargetLanguage::MLIR_ARITH));
    EXPECT_TRUE(matcher.hasPattern(NodeKind::SUB, TargetLanguage::MLIR_ARITH));
}

} // namespace cppfort::ir
