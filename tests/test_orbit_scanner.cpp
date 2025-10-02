#include "orbit_scanner.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

namespace cppfort {
namespace ir {

class OrbitScannerTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create test patterns directory
    testDir = std::filesystem::temp_directory_path() / "test_scanner_patterns";
    std::filesystem::create_directory(testDir);

    // Create test pattern files
    createTestPatternFile("c_patterns.yaml", "stdio.h", "C", 0);
    createTestPatternFile("cpp_patterns.yaml", "iostream", "CPP", 1);
    createTestPatternFile("cpp2_patterns.yaml", "inspect", "CPP2", 2);

    // Configure scanner
    config.patternsDir = testDir;
    scanner = std::make_unique<OrbitScanner>(config);
  }

  void TearDown() override {
    // Clean up test directory
    std::filesystem::remove_all(testDir);
  }

  void createTestPatternFile(const std::string& filename, const std::string& signature,
                           const std::string& grammarType, int orbitId) {
    std::string content = "name: " + grammarType + "_pattern\n"
                         "orbit_id: " + std::to_string(orbitId) + "\n"
                         "weight: 0.9\n"
                         "signature_patterns:\n"
                         "  - \"" + signature + "\"\n"
                         "protocol_indicators:\n"
                         "  - \"" + grammarType + "\"\n"
                         "version_patterns:\n"
                         "  - \"1.0\"\n";

    std::ofstream file(testDir / filename);
    file << content;
    file.close();
  }

  OrbitScannerConfig config;
  std::unique_ptr<OrbitScanner> scanner;
  std::filesystem::path testDir;
};

TEST_F(OrbitScannerTest, Initialization) {
  ASSERT_TRUE(scanner->initialize());

  EXPECT_GT(scanner->getPatternCount(), 0);
  EXPECT_FALSE(scanner->getLoadedGrammars().empty());
}

TEST_F(OrbitScannerTest, ScanCCode) {
  scanner->initialize();

  std::string cCode = R"(
#include <stdio.h>
int main() {
    printf("Hello, World!\n");
    return 0;
}
)";

  auto result = scanner->scan(cCode);

  EXPECT_EQ(result.detectedGrammar, GrammarType::C);
  EXPECT_GT(result.confidence, 0.0);
  EXPECT_FALSE(result.matches.empty());
  EXPECT_FALSE(result.reasoning.empty());
}

TEST_F(OrbitScannerTest, ScanCppCode) {
  scanner->initialize();

  std::string cppCode = R"(
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
)";

  auto result = scanner->scan(cppCode);

  EXPECT_EQ(result.detectedGrammar, GrammarType::CPP);
  EXPECT_GT(result.confidence, 0.0);
  EXPECT_FALSE(result.matches.empty());
}

TEST_F(OrbitScannerTest, ScanCpp2Code) {
  scanner->initialize();

  std::string cpp2Code = R"(
#include <iostream>
int main() {
    inspect (x) {
        0 => std::cout << "zero";
        _ => std::cout << "non-zero";
    }
    return 0;
}
)";

  auto result = scanner->scan(cpp2Code);

  EXPECT_EQ(result.detectedGrammar, GrammarType::CPP2);
  EXPECT_GT(result.confidence, 0.0);
  EXPECT_FALSE(result.matches.empty());
}

TEST_F(OrbitScannerTest, ScanUnknownCode) {
  scanner->initialize();

  std::string unknownCode = "some random text without patterns";

  auto result = scanner->scan(unknownCode);

  EXPECT_EQ(result.detectedGrammar, GrammarType::UNKNOWN);
  EXPECT_EQ(result.confidence, 0.0);
  EXPECT_TRUE(result.matches.empty());
}

TEST_F(OrbitScannerTest, ScanMixedCode) {
  scanner->initialize();

  std::string mixedCode = R"(
#include <stdio.h>
#include <iostream>
int main() {
    printf("C style\n");
    std::cout << "C++ style" << std::endl;
    return 0;
}
)";

  auto result = scanner->scan(mixedCode);

  // Should detect the dominant grammar (likely C++ due to more specific patterns)
  EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);
  EXPECT_GT(result.confidence, 0.0);
  EXPECT_FALSE(result.grammarScores.empty());
}

TEST_F(OrbitScannerTest, ConfigurationUpdate) {
  OrbitScannerConfig newConfig = config;
  newConfig.minConfidence = 0.9;
  newConfig.maxMatches = 50;

  scanner->updateConfig(newConfig);

  const auto& updatedConfig = scanner->getConfig();
  EXPECT_DOUBLE_EQ(updatedConfig.minConfidence, 0.9);
  EXPECT_EQ(updatedConfig.maxMatches, 50);
}

TEST_F(OrbitScannerTest, CustomPatterns) {
  // Create custom patterns for testing
  std::vector<OrbitPattern> customPatterns;
  
  OrbitPattern pattern1("custom_c", 1, 0.8);
  pattern1.signature_patterns = {"#include <custom.h>"};
  pattern1.protocol_indicators = {"CUSTOM_C"};
  pattern1.version_patterns = {"1.0"};
  customPatterns.push_back(pattern1);
  
  OrbitPattern pattern2("custom_cpp", 2, 0.9);
  pattern2.signature_patterns = {"std::custom"};
  pattern2.protocol_indicators = {"CUSTOM_CPP"};
  pattern2.version_patterns = {"2.0"};
  customPatterns.push_back(pattern2);

  std::string testCode = "#include <custom.h>\nstd::custom obj;";

  auto result = scanner->scan(testCode, customPatterns);

  EXPECT_NE(result.detectedGrammar, GrammarType::UNKNOWN);
  EXPECT_GT(result.confidence, 0.0);
  EXPECT_EQ(result.matches.size(), 2); // Should match both patterns
}

TEST_F(OrbitScannerTest, ConfidenceScoring) {
  scanner->initialize();

  std::string strongCppCode = R"(
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    vector<int> v = {1, 2, 3, 4, 5};
    for_each(v.begin(), v.end(), [](int x) {
        cout << x << endl;
    });
    return 0;
}
)";

  auto result = scanner->scan(strongCppCode);

  EXPECT_EQ(result.detectedGrammar, GrammarType::CPP);
  EXPECT_GT(result.confidence, 0.5); // Should have high confidence
}

TEST_F(OrbitScannerTest, ReasoningGeneration) {
  scanner->initialize();

  std::string simpleCCode = "#include <stdio.h>\nint main() { return 0; }";

  auto result = scanner->scan(simpleCCode);

  EXPECT_FALSE(result.reasoning.empty());
  EXPECT_NE(result.reasoning.find("Detected C"), std::string::npos);
  EXPECT_NE(result.reasoning.find("confidence"), std::string::npos);
}

TEST_F(OrbitScannerTest, UtilityFunctions) {
  DetectionResult result;
  result.detectedGrammar = GrammarType::CPP;
  result.confidence = 0.85;
  result.matches = {{"test", GrammarType::CPP, 0, 4, 0.85, "test"}};
  result.reasoning = "Test reasoning";
  result.grammarScores[GrammarType::CPP] = 0.85;

  std::string resultString = detectionResultToString(result);
  EXPECT_FALSE(resultString.empty());
  EXPECT_NE(resultString.find("C++"), std::string::npos);
  EXPECT_NE(resultString.find("85%"), std::string::npos);
}

} // namespace ir
} // namespace cppfort