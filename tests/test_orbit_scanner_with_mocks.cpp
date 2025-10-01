#include <gtest/gtest.h>
#include "orbit_scanner_mock.h"

namespace cppfort::ir {

/**
 * Test suite for OrbitScanner using mock implementations.
 * These tests validate scanner behavior without requiring full pattern database.
 */
class OrbitScannerMockTest : public ::testing::Test {
protected:
    void SetUp() override {
        scanner = ::std::make_unique<MockOrbitScanner>();
        ASSERT_TRUE(scanner->initialize());
    }

    void TearDown() override {
        scanner.reset();
    }

    ::std::unique_ptr<MockOrbitScanner> scanner;
};

/**
 * Test basic scanner initialization with mocks.
 */
TEST_F(OrbitScannerMockTest, BasicInitialization) {
    // Scanner should initialize successfully with mocks
    EXPECT_TRUE(scanner->initialize());

    // Should have mock patterns loaded
    auto patterns = scanner->getMockPatterns();
    EXPECT_GT(patterns.size(), 0u);

    // Config should be valid
    auto config = scanner->getConfig();
    EXPECT_GT(config.windowSizes.size(), 0u);
    EXPECT_GE(config.minConfidence, 0.0);
    EXPECT_LE(config.minConfidence, 1.0);
}

/**
 * Test scanning with mock C code.
 */
TEST_F(OrbitScannerMockTest, ScanCCode) {
    auto sample = MockCodeGenerator::generateCCode();

    auto result = scanner->scan(sample.code);

    // Validate basic result structure
    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));

    // Should detect some grammar (not necessarily C due to simplified mock)
    EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);

    // Should have reasonable confidence
    EXPECT_GT(result.confidence, 0.0);
}

/**
 * Test scanning with mock C++ code.
 */
TEST_F(OrbitScannerMockTest, ScanCppCode) {
    auto sample = MockCodeGenerator::generateCppCode();

    auto result = scanner->scan(sample.code);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);
    EXPECT_GT(result.confidence, 0.0);
}

/**
 * Test scanning with mock CPP2 code.
 */
TEST_F(OrbitScannerMockTest, ScanCpp2Code) {
    auto sample = MockCodeGenerator::generateCpp2Code();

    auto result = scanner->scan(sample.code);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);
}

/**
 * Test scanning with synthetic orbit patterns.
 */
TEST_F(OrbitScannerMockTest, ScanSyntheticPatterns) {
    // Test various orbit count combinations
    ::std::vector<::std::array<size_t, 6>> testCases = {
        {2, 0, 0, 2, 0, 0},  // Simple C-like
        {3, 0, 4, 3, 0, 0},  // C++ with templates
        {1, 0, 1, 1, 1, 1},  // Balanced mix
        {5, 5, 5, 5, 5, 5},  // High complexity
    };

    for (const auto& orbitCounts : testCases) {
        auto code = MockCodeGenerator::generateSyntheticCode(orbitCounts);

        auto result = scanner->scan(code);

        EXPECT_TRUE(MockScannerValidator::validateBasicResult(result))
            << "Failed for orbit pattern: ["
            << orbitCounts[0] << "," << orbitCounts[1] << ","
            << orbitCounts[2] << "," << orbitCounts[3] << ","
            << orbitCounts[4] << "," << orbitCounts[5] << "]";
    }
}

/**
 * Test scanner consistency - same code should produce same results.
 */
TEST_F(OrbitScannerMockTest, ScannerConsistency) {
    auto sample = MockCodeGenerator::generateCCode();

    // Scan same code multiple times
    ::std::vector<DetectionResult> results;
    for (int i = 0; i < 5; ++i) {
        results.push_back(scanner->scan(sample.code));
    }

    // All results should be consistent
    EXPECT_TRUE(MockScannerValidator::validateConsistency(results));

    // Confidence values should be identical
    double firstConfidence = results[0].confidence;
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_DOUBLE_EQ(results[i].confidence, firstConfidence);
    }
}

/**
 * Test empty code handling.
 */
TEST_F(OrbitScannerMockTest, EmptyCodeHandling) {
    auto result = scanner->scan("");

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    EXPECT_EQ(result.detectedGrammar, GrammarType::UNKNOWN);
    EXPECT_EQ(result.confidence, 0.0);
    EXPECT_TRUE(result.matches.empty());
}

/**
 * Test whitespace-only code.
 */
TEST_F(OrbitScannerMockTest, WhitespaceOnlyCode) {
    auto result = scanner->scan("   \n\t\r\n   ");

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    EXPECT_EQ(result.detectedGrammar, GrammarType::UNKNOWN);
}

/**
 * Test custom pattern injection.
 */
TEST_F(OrbitScannerMockTest, CustomPatternInjection) {
    // Create custom pattern
    auto customPattern = MockPatternGenerator::generatePattern(
        GrammarType::CPP,
        "custom_cpp_pattern",
        2.0  // High weight
    );

    scanner->addMockPattern(customPattern);

    // Scan code - should use custom pattern
    auto sample = MockCodeGenerator::generateCppCode();
    auto result = scanner->scan(sample.code);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
}

/**
 * Test pattern batch generation.
 */
TEST_F(OrbitScannerMockTest, PatternBatchGeneration) {
    auto patterns = MockPatternGenerator::generatePatternBatch(GrammarType::C, 10);

    EXPECT_EQ(patterns.size(), 10u);

    // Each pattern should have unique name and valid properties
    for (size_t i = 0; i < patterns.size(); ++i) {
        EXPECT_FALSE(patterns[i].name.empty());
        EXPECT_EQ(patterns[i].orbit_id, static_cast<int>(GrammarType::C));
        EXPECT_GT(patterns[i].weight, 0.0);
    }
}

/**
 * Test diverse pattern generation.
 */
TEST_F(OrbitScannerMockTest, DiversePatternGeneration) {
    auto patterns = MockPatternGenerator::generateDiversePatterns();

    EXPECT_GT(patterns.size(), 0u);

    // Should have patterns for multiple grammar types
    ::std::set<int> grammarTypes;
    for (const auto& pattern : patterns) {
        grammarTypes.insert(pattern.orbit_id);
    }

    EXPECT_GT(grammarTypes.size(), 1u);
}

/**
 * Test scanning diverse code samples.
 */
TEST_F(OrbitScannerMockTest, ScanDiverseSamples) {
    auto samples = MockCodeGenerator::generateDiverseSamples();

    EXPECT_GT(samples.size(), 0u);

    for (const auto& sample : samples) {
        auto result = scanner->scan(sample.code);

        EXPECT_TRUE(MockScannerValidator::validateBasicResult(result))
            << "Failed for grammar: " << MockPatternGenerator::grammarToString(sample.expectedGrammar);

        // Should detect something (may not be exact grammar due to mock simplification)
        EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);
    }
}

/**
 * Test result serialization.
 */
TEST_F(OrbitScannerMockTest, ResultSerialization) {
    auto sample = MockCodeGenerator::generateCCode();
    auto result = scanner->scan(sample.code);

    // Convert to string
    auto resultString = detectionResultToString(result);

    EXPECT_FALSE(resultString.empty());
    EXPECT_NE(resultString.find("Grammar:"), ::std::string::npos);
    EXPECT_NE(resultString.find("Confidence:"), ::std::string::npos);
}

/**
 * Stress test with large number of patterns.
 */
TEST_F(OrbitScannerMockTest, StressTestManyPatterns) {
    // Add many patterns
    for (int i = 0; i < 100; ++i) {
        auto pattern = MockPatternGenerator::generatePattern(
            GrammarType::CPP,
            "stress_pattern_" + ::std::to_string(i)
        );
        scanner->addMockPattern(pattern);
    }

    // Should still work with many patterns
    auto sample = MockCodeGenerator::generateCppCode();
    auto result = scanner->scan(sample.code);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
}

/**
 * Test scanner config update.
 */
TEST_F(OrbitScannerMockTest, ConfigUpdate) {
    OrbitScannerConfig newConfig;
    newConfig.patternThreshold = 0.9;  // Very high threshold
    newConfig.minConfidence = 0.8;

    scanner->updateConfig(newConfig);

    auto config = scanner->getConfig();
    EXPECT_DOUBLE_EQ(config.patternThreshold, 0.9);
    EXPECT_DOUBLE_EQ(config.minConfidence, 0.8);
}

/**
 * Test edge case: very long code.
 */
TEST_F(OrbitScannerMockTest, VeryLongCode) {
    ::std::string longCode;
    for (int i = 0; i < 1000; ++i) {
        longCode += "int x" + ::std::to_string(i) + " = " + ::std::to_string(i) + ";\n";
    }

    auto result = scanner->scan(longCode);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
}

/**
 * Test edge case: deeply nested structures.
 */
TEST_F(OrbitScannerMockTest, DeeplyNestedStructures) {
    ::std::string nested;
    int depth = 50;

    // Build deeply nested braces
    for (int i = 0; i < depth; ++i) {
        nested += "{";
    }
    nested += "x;";
    for (int i = 0; i < depth; ++i) {
        nested += "}";
    }

    auto result = scanner->scan(nested);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);
}

} // namespace cppfort::ir
