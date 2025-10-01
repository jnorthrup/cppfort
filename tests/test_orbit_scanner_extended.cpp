#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <string>
#include <random>

#include "orbit_scanner_mock.h"

namespace cppfort::ir {

/**
 * Extended scanner tests focusing on long code samples and performance.
 * Tests realistic file sizes and complex nested structures.
 */
class OrbitScannerExtendedTest : public ::testing::Test {
protected:
    void SetUp() override {
        scanner = ::std::make_unique<MockOrbitScanner>();
        ASSERT_TRUE(scanner->initialize());
    }

    void TearDown() override {
        scanner.reset();
    }

    ::std::unique_ptr<MockOrbitScanner> scanner;

    // Generate large C file with realistic structure
    ::std::string generateLargeCFile(size_t numFunctions = 100) {
        ::std::string code;
        code.reserve(numFunctions * 500);  // Estimate 500 bytes per function

        code += "#include <stdio.h>\n";
        code += "#include <stdlib.h>\n";
        code += "#include <string.h>\n\n";

        // Generate struct definitions
        for (size_t i = 0; i < 10; ++i) {
            code += "struct Data" + ::std::to_string(i) + " {\n";
            code += "    int field1;\n";
            code += "    double field2;\n";
            code += "    char* field3;\n";
            code += "};\n\n";
        }

        // Generate function prototypes
        for (size_t i = 0; i < numFunctions; ++i) {
            code += "int process_" + ::std::to_string(i) + "(int x, int y);\n";
        }
        code += "\n";

        // Generate function implementations
        for (size_t i = 0; i < numFunctions; ++i) {
            code += "int process_" + ::std::to_string(i) + "(int x, int y) {\n";
            code += "    int result = 0;\n";
            code += "    if (x > 0) {\n";
            code += "        result = x + y;\n";
            code += "    } else {\n";
            code += "        result = x - y;\n";
            code += "    }\n";
            code += "    for (int i = 0; i < 10; i++) {\n";
            code += "        result += i;\n";
            code += "    }\n";
            code += "    return result;\n";
            code += "}\n\n";
        }

        // Main function
        code += "int main(int argc, char** argv) {\n";
        code += "    int total = 0;\n";
        for (size_t i = 0; i < 20; ++i) {
            code += "    total += process_" + ::std::to_string(i) + "(" +
                   ::std::to_string(i) + ", " + ::std::to_string(i * 2) + ");\n";
        }
        code += "    printf(\"Total: %d\\n\", total);\n";
        code += "    return 0;\n";
        code += "}\n";

        return code;
    }

    // Generate large C++ file with templates and classes
    ::std::string generateLargeCppFile(size_t numClasses = 50) {
        ::std::string code;
        code.reserve(numClasses * 800);

        code += "#include <iostream>\n";
        code += "#include <vector>\n";
        code += "#include <string>\n";
        code += "#include <memory>\n";
        code += "#include <algorithm>\n\n";

        code += "namespace app {\n\n";

        // Generate template classes
        for (size_t i = 0; i < numClasses; ++i) {
            code += "template<typename T>\n";
            code += "class Container" + ::std::to_string(i) + " {\n";
            code += "private:\n";
            code += "    std::vector<T> data_;\n";
            code += "    size_t capacity_;\n\n";
            code += "public:\n";
            code += "    Container" + ::std::to_string(i) + "() : capacity_(100) {}\n";
            code += "    \n";
            code += "    void add(const T& item) {\n";
            code += "        if (data_.size() < capacity_) {\n";
            code += "            data_.push_back(item);\n";
            code += "        }\n";
            code += "    }\n";
            code += "    \n";
            code += "    T get(size_t index) const {\n";
            code += "        if (index < data_.size()) {\n";
            code += "            return data_[index];\n";
            code += "        }\n";
            code += "        return T{};\n";
            code += "    }\n";
            code += "    \n";
            code += "    size_t size() const { return data_.size(); }\n";
            code += "    \n";
            code += "    template<typename Func>\n";
            code += "    void forEach(Func f) {\n";
            code += "        std::for_each(data_.begin(), data_.end(), f);\n";
            code += "    }\n";
            code += "};\n\n";
        }

        code += "} // namespace app\n\n";

        // Main with usage
        code += "int main() {\n";
        for (size_t i = 0; i < 10; ++i) {
            code += "    app::Container" + ::std::to_string(i) + "<int> c" +
                   ::std::to_string(i) + ";\n";
            code += "    c" + ::std::to_string(i) + ".add(" + ::std::to_string(i * 10) + ");\n";
        }
        code += "    std::cout << \"Done\" << std::endl;\n";
        code += "    return 0;\n";
        code += "}\n";

        return code;
    }

    // Generate large CPP2 file
    ::std::string generateLargeCpp2File(size_t numFunctions = 80) {
        ::std::string code;
        code.reserve(numFunctions * 400);

        // Type definitions
        for (size_t i = 0; i < 10; ++i) {
            code += "Point" + ::std::to_string(i) + ": type = {\n";
            code += "    x: int = 0;\n";
            code += "    y: int = 0;\n";
            code += "}\n\n";
        }

        // Function definitions
        for (size_t i = 0; i < numFunctions; ++i) {
            code += "calculate_" + ::std::to_string(i) + ": (a: int, b: int) -> int = {\n";
            code += "    result: int = 0;\n";
            code += "    if a > b {\n";
            code += "        result = a + b;\n";
            code += "    } else {\n";
            code += "        result = a - b;\n";
            code += "    }\n";
            code += "    return result;\n";
            code += "}\n\n";
        }

        // Main function
        code += "main: () -> int = {\n";
        code += "    total: int = 0;\n";
        for (size_t i = 0; i < 15; ++i) {
            code += "    total += calculate_" + ::std::to_string(i) + "(" +
                   ::std::to_string(i) + ", " + ::std::to_string(i + 1) + ");\n";
        }
        code += "    std::cout << \"Total: \" << total << std::endl;\n";
        code += "    return 0;\n";
        code += "}\n";

        return code;
    }

    // Generate deeply nested code
    ::std::string generateDeeplyNestedCode(size_t depth = 100) {
        ::std::string code;
        code.reserve(depth * 50);

        code += "int deeply_nested() {\n";
        code += "    int result = 0;\n";

        // Build deep nesting
        for (size_t i = 0; i < depth; ++i) {
            code += ::std::string(i + 1, ' ') + "if (result >= 0) {\n";
        }

        code += ::std::string(depth + 1, ' ') + "result = 42;\n";

        // Close all nesting
        for (size_t i = depth; i > 0; --i) {
            code += ::std::string(i, ' ') + "}\n";
        }

        code += "    return result;\n";
        code += "}\n";

        return code;
    }

    // Generate code with many template instantiations
    ::std::string generateHeavyTemplateCode(size_t numInstantiations = 200) {
        ::std::string code;
        code.reserve(numInstantiations * 150);

        code += "#include <vector>\n";
        code += "#include <map>\n";
        code += "#include <set>\n\n";

        code += "template<typename T, typename U, typename V>\n";
        code += "struct Triple {\n";
        code += "    T first;\n";
        code += "    U second;\n";
        code += "    V third;\n";
        code += "};\n\n";

        code += "void instantiate_templates() {\n";
        for (size_t i = 0; i < numInstantiations; ++i) {
            code += "    std::vector<std::map<int, std::vector<double>>> v" +
                   ::std::to_string(i) + ";\n";
            code += "    Triple<int, double, std::string> t" + ::std::to_string(i) + ";\n";
        }
        code += "}\n";

        return code;
    }
};

/**
 * Test scanning large C file (realistic embedded systems code size).
 */
TEST_F(OrbitScannerExtendedTest, ScanLargeCFile) {
    auto code = generateLargeCFile(100);

    EXPECT_GT(code.length(), 20000u) << "Generated code should be substantial";

    auto start = ::std::chrono::high_resolution_clock::now();
    auto result = scanner->scan(code);
    auto end = ::std::chrono::high_resolution_clock::now();

    auto duration = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);

    ::std::cout << "Large C file scan: " << code.length() << " bytes in "
                << duration.count() << "ms" << ::std::endl;
    ::std::cout << "  Detected: " << MockPatternGenerator::grammarToString(result.detectedGrammar)
                << " with confidence " << result.confidence << ::std::endl;
}

/**
 * Test scanning very large C++ file with templates.
 */
TEST_F(OrbitScannerExtendedTest, ScanLargeCppFile) {
    auto code = generateLargeCppFile(50);

    EXPECT_GT(code.length(), 25000u);

    auto start = ::std::chrono::high_resolution_clock::now();
    auto result = scanner->scan(code);
    auto end = ::std::chrono::high_resolution_clock::now();

    auto duration = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);

    ::std::cout << "Large C++ file scan: " << code.length() << " bytes in "
                << duration.count() << "ms" << ::std::endl;
}

/**
 * Test scanning large CPP2 file.
 */
TEST_F(OrbitScannerExtendedTest, ScanLargeCpp2File) {
    auto code = generateLargeCpp2File(80);

    EXPECT_GT(code.length(), 10000u);

    auto start = ::std::chrono::high_resolution_clock::now();
    auto result = scanner->scan(code);
    auto end = ::std::chrono::high_resolution_clock::now();

    auto duration = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));

    ::std::cout << "Large CPP2 file scan: " << code.length() << " bytes in "
                << duration.count() << "ms" << ::std::endl;
}

/**
 * Test extremely deep nesting (stress test for orbit tracking).
 */
TEST_F(OrbitScannerExtendedTest, ScanDeeplyNestedCode) {
    auto code = generateDeeplyNestedCode(150);

    auto start = ::std::chrono::high_resolution_clock::now();
    auto result = scanner->scan(code);
    auto end = ::std::chrono::high_resolution_clock::now();

    auto duration = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);

    ::std::cout << "Deep nesting (150 levels): " << duration.count() << "ms" << ::std::endl;
}

/**
 * Test heavy template usage (stress test for angle bracket tracking).
 */
TEST_F(OrbitScannerExtendedTest, ScanHeavyTemplateCode) {
    auto code = generateHeavyTemplateCode(200);

    EXPECT_GT(code.length(), 15000u);

    auto start = ::std::chrono::high_resolution_clock::now();
    auto result = scanner->scan(code);
    auto end = ::std::chrono::high_resolution_clock::now();

    auto duration = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));

    ::std::cout << "Heavy templates (200 instantiations): " << code.length()
                << " bytes in " << duration.count() << "ms" << ::std::endl;
}

/**
 * Test multiple large files in sequence (memory leak check).
 */
TEST_F(OrbitScannerExtendedTest, ScanMultipleLargeFiles) {
    const size_t NUM_FILES = 20;

    auto start = ::std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUM_FILES; ++i) {
        ::std::string code;

        switch (i % 3) {
            case 0:
                code = generateLargeCFile(50);
                break;
            case 1:
                code = generateLargeCppFile(30);
                break;
            case 2:
                code = generateLargeCpp2File(40);
                break;
        }

        auto result = scanner->scan(code);
        EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    }

    auto end = ::std::chrono::high_resolution_clock::now();
    auto duration = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end - start);

    ::std::cout << "Scanned " << NUM_FILES << " large files in "
                << duration.count() << "ms" << ::std::endl;
}

/**
 * Test scanning massive single file (100K+ lines equivalent).
 */
TEST_F(OrbitScannerExtendedTest, ScanMassiveFile) {
    auto code = generateLargeCppFile(500);  // ~400KB

    EXPECT_GT(code.length(), 100000u) << "Should generate >100KB file";

    auto start = ::std::chrono::high_resolution_clock::now();
    auto result = scanner->scan(code);
    auto end = ::std::chrono::high_resolution_clock::now();

    auto duration = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));

    ::std::cout << "Massive file scan: " << code.length() << " bytes (~"
                << code.length() / 1024 << "KB) in " << duration.count() << "ms" << ::std::endl;
    ::std::cout << "  Throughput: " << (code.length() / 1024.0) / (duration.count() / 1000.0)
                << " KB/sec" << ::std::endl;
}

/**
 * Test mixed language patterns in same file.
 */
TEST_F(OrbitScannerExtendedTest, ScanMixedLanguageFile) {
    ::std::string code;

    // Start with C code
    code += generateLargeCFile(20);

    // Add C++ section
    code += "\n\n// C++ section\n";
    code += generateLargeCppFile(20);

    // Add CPP2 section (hypothetically)
    code += "\n\n// CPP2 section\n";
    code += generateLargeCpp2File(20);

    EXPECT_GT(code.length(), 10000u);

    auto result = scanner->scan(code);

    EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);

    // Should have multiple grammar scores
    EXPECT_GT(result.grammarScores.size(), 0u);

    ::std::cout << "Mixed language file: " << code.length() << " bytes" << ::std::endl;
    ::std::cout << "  Grammar scores: ";
    for (const auto& [grammar, score] : result.grammarScores) {
        ::std::cout << MockPatternGenerator::grammarToString(grammar) << "=" << score << " ";
    }
    ::std::cout << ::std::endl;
}

/**
 * Stress test: random code generation and scanning.
 */
TEST_F(OrbitScannerExtendedTest, StressTestRandomCode) {
    ::std::mt19937 rng(12345);
    ::std::uniform_int_distribution<int> dist(0, 2);

    const size_t NUM_ITERATIONS = 50;
    size_t totalBytes = 0;

    auto start = ::std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        ::std::string code;

        int choice = dist(rng);
        switch (choice) {
            case 0:
                code = generateLargeCFile(30 + (i % 20));
                break;
            case 1:
                code = generateLargeCppFile(20 + (i % 15));
                break;
            case 2:
                code = generateLargeCpp2File(25 + (i % 18));
                break;
        }

        totalBytes += code.length();

        auto result = scanner->scan(code);
        EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));
    }

    auto end = ::std::chrono::high_resolution_clock::now();
    auto duration = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end - start);

    ::std::cout << "Stress test: " << NUM_ITERATIONS << " files, "
                << totalBytes / 1024 << "KB total in " << duration.count() << "ms" << ::std::endl;
    ::std::cout << "  Average throughput: "
                << (totalBytes / 1024.0) / (duration.count() / 1000.0) << " KB/sec" << ::std::endl;
}

/**
 * Performance baseline: scan speed across different file sizes.
 */
TEST_F(OrbitScannerExtendedTest, PerformanceBaseline) {
    const ::std::vector<size_t> fileSizes = {10, 50, 100, 200, 500};

    ::std::cout << "\nPerformance baseline across file sizes:" << ::std::endl;

    for (size_t size : fileSizes) {
        auto code = generateLargeCppFile(size);

        auto start = ::std::chrono::high_resolution_clock::now();
        auto result = scanner->scan(code);
        auto end = ::std::chrono::high_resolution_clock::now();

        auto duration = ::std::chrono::duration_cast<::std::chrono::microseconds>(end - start);

        EXPECT_TRUE(MockScannerValidator::validateBasicResult(result));

        double kbPerSec = (code.length() / 1024.0) / (duration.count() / 1000000.0);

        ::std::cout << "  " << size << " classes (" << code.length() / 1024 << "KB): "
                    << duration.count() / 1000.0 << "ms (" << kbPerSec << " KB/sec)" << ::std::endl;
    }
}

} // namespace cppfort::ir
