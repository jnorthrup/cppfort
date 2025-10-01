#include "orbit_mask.h"
#include <gtest/gtest.h>
#include <string>

namespace cppfort {
namespace ir {

class OrbitContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    context = std::make_unique<OrbitContext>(100);
  }

  std::unique_ptr<OrbitContext> context;
};

TEST_F(OrbitContextTest, Initialization) {
  EXPECT_TRUE(context->isBalanced());
  EXPECT_EQ(context->getDepth(), 0);
  EXPECT_EQ(context->getMaxDepth(), 100);
}

TEST_F(OrbitContextTest, ParenthesesTracking) {
  // Test balanced parentheses
  context->processChar('(');
  EXPECT_EQ(context->getDepth(), 1);
  EXPECT_FALSE(context->isBalanced());

  context->processChar(')');
  EXPECT_EQ(context->getDepth(), 0);
  EXPECT_TRUE(context->isBalanced());
}

TEST_F(OrbitContextTest, NestedStructures) {
  std::string code = "if (x > 0) { return x; }";

  for (char c : code) {
    context->processChar(c);
  }

  EXPECT_TRUE(context->isBalanced());
  EXPECT_EQ(context->getDepth(), 0);
}

TEST_F(OrbitContextTest, UnbalancedStructures) {
  // Missing closing brace
  std::string code = "if (x > 0) { return x;";

  for (char c : code) {
    context->processChar(c);
  }

  EXPECT_FALSE(context->isBalanced());
  EXPECT_GT(context->getDepth(), 0);
}

TEST_F(OrbitContextTest, MultipleStructureTypes) {
  std::string code = "func() { if (true) [1, 2, 3]; }";

  for (char c : code) {
    context->processChar(c);
  }

  EXPECT_TRUE(context->isBalanced());
}

TEST_F(OrbitContextTest, DepthLimit) {
  OrbitContext limitedContext(3);

  // Exceed depth limit
  for (int i = 0; i < 5; ++i) {
    limitedContext.processChar('(');
  }

  EXPECT_GT(limitedContext.getDepth(), 3);
  // Context should still track but may have reduced accuracy
}

TEST_F(OrbitContextTest, ConfidenceCalculation) {
  // Test confidence for balanced code
  std::string balancedCode = "if (x > 0) { return x; }";

  for (char c : balancedCode) {
    context->processChar(c);
  }

  double confidence = context->calculateConfidence();
  EXPECT_GT(confidence, 0.8); // Should be high for balanced code
}

TEST_F(OrbitContextTest, LowConfidenceForUnbalanced) {
  // Test confidence for unbalanced code
  std::string unbalancedCode = "if (x > 0) { return x;";

  for (char c : unbalancedCode) {
    context->processChar(c);
  }

  double confidence = context->calculateConfidence();
  EXPECT_LT(confidence, 0.5); // Should be low for unbalanced code
}

TEST_F(OrbitContextTest, ResetFunctionality) {
  context->processChar('(');
  EXPECT_FALSE(context->isBalanced());

  context->reset();
  EXPECT_TRUE(context->isBalanced());
  EXPECT_EQ(context->getDepth(), 0);
}

TEST_F(OrbitContextTest, ComplexNesting) {
  std::string complexCode = "function() { if (a && (b || c)) { for (i in [1,2,3]) { process(i); } } }";

  for (char c : complexCode) {
    context->processChar(c);
  }

  EXPECT_TRUE(context->isBalanced());
  EXPECT_EQ(context->getDepth(), 0);
}

TEST_F(OrbitContextTest, StringLiteralHandling) {
  // Context should ignore delimiters inside strings
  std::string codeWithStrings = "if (str == \"hello (world)\") { return true; }";

  for (char c : codeWithStrings) {
    context->processChar(c);
  }

  EXPECT_TRUE(context->isBalanced());
}

} // namespace ir
} // namespace cppfort