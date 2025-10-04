#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "src/stage0/orbit_scanner.h"
#include "src/stage0/cpp2_key_resolver.h"
#include "src/stage0/cpp2_pattern_extractor.h"

namespace fs = std::filesystem;
using namespace cppfort;
using namespace testing;

class OrbitDisambiguationTest : public Test {
protected:
    void SetUp() override {
        // Load pattern databases
        cpp2_extractor.loadPatternsFromYAML("patterns/bnfc_cpp2_complete.yaml");
        key_resolver.build_key_database(cpp2_extractor);
    }

    void TearDown() override {
        // Clean up any test files
    }

    // Helper to scan file and get orbit results
    OrbitRing scan_file(const std::string& content, GrammarMode mode = GrammarMode::CPP) {
        // Create temporary file
        std::ofstream temp_file("temp_test.cpp");
        temp_file << content;
        temp_file.close();

        // Scan file
        OrbitScanner scanner;
        scanner.setGrammarMode(mode);
        scanner.applyCPP2KeyResolution(true);  // Enable CPP2 keying

        auto results = scanner.scanFile("temp_test.cpp");

        // Clean up
        fs::remove("temp_test.cpp");

        return results;
    }

    // Helper to measure scan performance
    std::chrono::microseconds measure_scan_time(const std::string& content,
                                               bool enable_cpp2_keying = true) {
        std::ofstream temp_file("temp_perf_test.cpp");
        temp_file << content;
        temp_file.close();

        OrbitScanner scanner;
        scanner.applyCPP2KeyResolution(enable_cpp2_keying);

        auto start = std::chrono::high_resolution_clock::now();
        scanner.scanFile("temp_perf_test.cpp");
        auto end = std::chrono::high_resolution_clock::now();

        fs::remove("temp_perf_test.cpp");

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }

    CPP2PatternExtractor cpp2_extractor;
    CPP2KeyResolver key_resolver;
};

// Test Cases for Pure C Files (BNFC patterns only)
TEST_F(OrbitDisambiguationTest, PureC_GotoLabels) {
    std::string code = R"(
        void func() {
            goto label;
        label:
            return;
        }
    )";

    auto results = scan_file(code, GrammarMode::C);

    // Should find goto label pattern, no CPP2 interference
    EXPECT_TRUE(results.hasValidInterpretations());
    EXPECT_EQ(results.winner_index(), 0);  // goto label interpretation
}

TEST_F(OrbitDisambiguationTest, PureC_Bitfields) {
    std::string code = R"(
        struct Flags {
            unsigned int enabled : 1;
            unsigned int visible : 1;
        };
    )";

    auto results = scan_file(code, GrammarMode::C);

    // Should correctly identify bitfield declarations
    EXPECT_TRUE(results.hasValidInterpretations());
}

TEST_F(OrbitDisambiguationTest, PureC_TernaryOperator) {
    std::string code = R"(
        int result = (x > 0) ? x : -x;
    )";

    auto results = scan_file(code, GrammarMode::C);

    // Should not confuse colon with other uses
    EXPECT_TRUE(results.hasValidInterpretations());
}

// Test Cases for Pure C++ Files (BNFC + backwards CPP2)
TEST_F(OrbitDisambiguationTest, PureCPP_Inheritance) {
    std::string code = R"(
        class Derived : public Base {
        public:
            void method() override;
        };
    )";

    auto results = scan_file(code, GrammarMode::CPP);

    // Should find inheritance pattern, possibly enhanced by CPP2 keying
    EXPECT_TRUE(results.hasValidInterpretations());
}

TEST_F(OrbitDisambiguationTest, PureCPP_NamespaceAlias) {
    std::string code = R"(
        namespace fs = std::filesystem;
    )";

    auto results = scan_file(code, GrammarMode::CPP);

    // Should identify namespace alias, not confused with CPP2 patterns
    EXPECT_TRUE(results.hasValidInterpretations());
}

TEST_F(OrbitDisambiguationTest, PureCPP_AccessSpecifier) {
    std::string code = R"(
        class MyClass {
        private:
            int data;
        public:
            void setData(int value) { data = value; }
        };
    )";

    auto results = scan_file(code, GrammarMode::CPP);

    // Should correctly parse access specifiers
    EXPECT_TRUE(results.hasValidInterpretations());
}

// Test Cases for Pure CPP2 Files (CPP2 deterministic)
TEST_F(OrbitDisambiguationTest, PureCPP2_TypeAnnotation) {
    std::string code = R"(
        main: () -> int = {
            x: int = 42;
            return x;
        }
    )";

    auto results = scan_file(code, GrammarMode::CPP2);

    // Should deterministically parse CPP2 type annotations
    EXPECT_TRUE(results.hasValidInterpretations());
    EXPECT_EQ(results.winner_index(), 0);  // CPP2 interpretation should win
}

TEST_F(OrbitDisambiguationTest, PureCPP2_FunctionSignature) {
    std::string code = R"(
        add: (x: int, y: int) -> int = x + y;
    )";

    auto results = scan_file(code, GrammarMode::CPP2);

    // Should deterministically parse CPP2 function signatures
    EXPECT_TRUE(results.hasValidInterpretations());
}

TEST_F(OrbitDisambiguationTest, PureCPP2_Namespace) {
    std::string code = R"(
        math: namespace = {
            pi: double = 3.14159;
        }
    )";

    auto results = scan_file(code, GrammarMode::CPP2);

    // Should deterministically parse CPP2 namespaces
    EXPECT_TRUE(results.hasValidInterpretations());
}

TEST_F(OrbitDisambiguationTest, PureCPP2_TypeDefinition) {
    std::string code = R"(
        Point: type = {
            x: int;
            y: int;
        };
    )";

    auto results = scan_file(code, GrammarMode::CPP2);

    // Should deterministically parse CPP2 type definitions
    EXPECT_TRUE(results.hasValidInterpretations());
}

// Test Cases for Mixed C++/CPP2 Files (mode transitions)
TEST_F(OrbitDisambiguationTest, Mixed_CPP2InCPP) {
    std::string code = R"(
        // Traditional C++ class
        class Calculator {
        public:
            // CPP2-style function inside C++ class
            add: (x: int, y: int) -> int = x + y;
        };
    )";

    auto results = scan_file(code, GrammarMode::CPP);

    // Should handle mixed syntax appropriately
    EXPECT_TRUE(results.hasValidInterpretations());
}

TEST_F(OrbitDisambiguationTest, Mixed_CPPInCPP2) {
    std::string code = R"(
        // CPP2 namespace with C++-style content
        math: namespace = {
            class Helper {
            public:
                static double square(double x) { return x * x; }
            };
        };
    )";

    auto results = scan_file(code, GrammarMode::CPP2);

    // Should handle mixed syntax in CPP2 context
    EXPECT_TRUE(results.hasValidInterpretations());
}

// Performance Tests
TEST_F(OrbitDisambiguationTest, Performance_NoCPP2Keying) {
    // Generate large C++ file
    std::string large_code;
    for (int i = 0; i < 1000; ++i) {
        large_code += "class Class" + std::to_string(i) + " {\n";
        large_code += "public:\n";
        large_code += "    int method" + std::to_string(i) + "() { return " + std::to_string(i) + "; }\n";
        large_code += "};\n\n";
    }

    auto time_without = measure_scan_time(large_code, false);
    auto time_with = measure_scan_time(large_code, true);

    // CPP2 keying should not add more than 5% overhead
    double overhead = (time_with.count() - time_without.count()) / (double)time_without.count();
    EXPECT_LT(overhead, 0.05);
}

// Accuracy Tests - All 13 colon contexts
TEST_F(OrbitDisambiguationTest, ColonContexts_Coverage) {
    // Test all 13 colon contexts are represented
    std::vector<std::string> test_cases = {
        // C/C++ contexts (9)
        "label:",           // goto label
        "case 1:",          // switch case
        "public:",          // access specifier
        "private:",         // access specifier
        "protected:",       // access specifier
        "class Derived : public Base",  // inheritance
        "unsigned int flags : 1;",      // bitfield
        "x > 0 ? x : -x;",  // ternary operator
        "namespace alias = std::filesystem;",  // namespace alias

        // CPP2 contexts (4)
        "x: int = 5;",      // type annotation
        "f: () -> int = {}", // function signature
        "ns: namespace = {}", // namespace
        "T: type = {}",     // type definition
    };

    for (const auto& test_case : test_cases) {
        std::string code = "void test() { " + test_case + " }";
        auto results = scan_file(code);

        // Each context should produce valid interpretations
        EXPECT_TRUE(results.hasValidInterpretations())
            << "Failed to disambiguate: " << test_case;
    }
}

// Precision Test - False Positive Reduction
TEST_F(OrbitDisambiguationTest, Precision_FalsePositiveReduction) {
    std::string ambiguous_code = R"(
        // Code that could be ambiguous without CPP2 keying
        struct Data {
            int x : 8;  // bitfield
            int y;      // regular member
        };

        void func() {
            if (x > 0) {
                goto label;  // goto label
            label:
                return;
            }
        }
    )";

    // Scan with CPP2 keying disabled
    OrbitScanner scanner_no_cpp2;
    scanner_no_cpp2.applyCPP2KeyResolution(false);
    std::ofstream temp_file("temp_precision_test.cpp");
    temp_file << ambiguous_code;
    temp_file.close();

    auto results_no_cpp2 = scanner_no_cpp2.scanFile("temp_precision_test.cpp");
    int false_positives_no_cpp2 = results_no_cpp2.totalInterpretations() - results_no_cpp2.validInterpretations();

    // Scan with CPP2 keying enabled
    OrbitScanner scanner_with_cpp2;
    scanner_with_cpp2.applyCPP2KeyResolution(true);
    auto results_with_cpp2 = scanner_with_cpp2.scanFile("temp_precision_test.cpp");

    int false_positives_with_cpp2 = results_with_cpp2.totalInterpretations() - results_with_cpp2.validInterpretations();

    fs::remove("temp_precision_test.cpp");

    // CPP2 keying should reduce false positives
    EXPECT_LE(false_positives_with_cpp2, false_positives_no_cpp2);
}

// Recall Test - No False Negatives
TEST_F(OrbitDisambiguationTest, Recall_NoFalseNegatives) {
    std::string valid_code = R"(
        // All constructs should be correctly identified
        class MyClass : public Base {
        public:
            int data : 8;  // bitfield
            void method() {
                if (condition) {
                    goto label;
                }
            label:
                return;
            }
        };
    )";

    auto results = scan_file(valid_code, GrammarMode::CPP);

    // All valid interpretations should be present
    EXPECT_EQ(results.validInterpretations(), results.totalInterpretations());
    EXPECT_TRUE(results.hasValidInterpretations());
}

// Accuracy Test - Winner Selection
TEST_F(OrbitDisambiguationTest, Accuracy_WinnerSelection) {
    // Test cases where winner selection should be >90% accurate
    std::vector<std::pair<std::string, int>> accuracy_tests = {
        {R"(x: int = 5;)", 0},  // CPP2 type annotation should win
        {R"(goto label; label:)", 0},  // goto label should win
        {R"(class D : public B {})", 0},  // inheritance should win
        {R"(int flags : 1;)", 0},  // bitfield should win
    };

    int correct_selections = 0;
    for (const auto& [code, expected_winner] : accuracy_tests) {
        auto results = scan_file(code);
        if (results.winner_index() == expected_winner) {
            ++correct_selections;
        }
    }

    double accuracy = correct_selections / (double)accuracy_tests.size();
    EXPECT_GE(accuracy, 0.9);  // >90% accuracy target
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}