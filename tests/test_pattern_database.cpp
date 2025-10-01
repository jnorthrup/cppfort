#include "tblgen_patterns.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

namespace cppfort {
namespace ir {

class PatternDatabaseTest : public ::testing::Test {
protected:
  void SetUp() override {
    database = std::make_unique<PatternDatabase>();

    // Create a temporary test YAML file
    testYamlContent = R"(
name: test_pattern
orbit_id: 1
weight: 0.8
signature_patterns:
  - "test"
  - "example"
protocol_indicators:
  - "TEST"
version_patterns:
  - "1.0"

---
name: another_pattern
orbit_id: 2
weight: 0.9
signature_patterns:
  - "another"
protocol_indicators:
  - "ANOTHER"
version_patterns:
  - "2.0"
)";

    // Write test file
    testFilePath = std::filesystem::temp_directory_path() / "test_patterns.yaml";
    std::ofstream testFile(testFilePath);
    testFile << testYamlContent;
    testFile.close();
  }

  void TearDown() override {
    // Clean up test file
    std::filesystem::remove(testFilePath);
  }

  std::unique_ptr<PatternDatabase> database;
  std::string testYamlContent;
  std::filesystem::path testFilePath;
};

TEST_F(PatternDatabaseTest, LoadFromYaml) {
  ASSERT_TRUE(database->loadFromYaml(testFilePath));

  const auto& patterns = database->getPatterns();
  ASSERT_EQ(patterns.size(), 2);

  // Check first pattern
  EXPECT_EQ(patterns[0].name, "test_pattern");
  EXPECT_EQ(patterns[0].orbitId, 1);
  EXPECT_DOUBLE_EQ(patterns[0].weight, 0.8);
  EXPECT_EQ(patterns[0].signaturePatterns.size(), 2);
  EXPECT_EQ(patterns[0].protocolIndicators.size(), 1);
  EXPECT_EQ(patterns[0].versionPatterns.size(), 1);

  // Check second pattern
  EXPECT_EQ(patterns[1].name, "another_pattern");
  EXPECT_EQ(patterns[1].orbitId, 2);
  EXPECT_DOUBLE_EQ(patterns[1].weight, 0.9);
}

TEST_F(PatternDatabaseTest, ExportToTableGen) {
  database->loadFromYaml(testFilePath);

  std::string tablegen = database->exportToTableGen();
  EXPECT_FALSE(tablegen.empty());

  // Check for expected TableGen structure
  EXPECT_NE(tablegen.find("class OrbitPattern"), std::string::npos);
  EXPECT_NE(tablegen.find("test_pattern"), std::string::npos);
  EXPECT_NE(tablegen.find("another_pattern"), std::string::npos);
}

TEST_F(PatternDatabaseTest, FindPatternByName) {
  database->loadFromYaml(testFilePath);

  auto pattern = database->findPattern("test_pattern");
  ASSERT_TRUE(pattern.has_value());
  EXPECT_EQ(pattern->name, "test_pattern");
  EXPECT_EQ(pattern->orbitId, 1);

  auto notFound = database->findPattern("nonexistent");
  EXPECT_FALSE(notFound.has_value());
}

TEST_F(PatternDatabaseTest, GetPatternsByOrbitId) {
  database->loadFromYaml(testFilePath);

  auto orbit1Patterns = database->getPatternsByOrbitId(1);
  ASSERT_EQ(orbit1Patterns.size(), 1);
  EXPECT_EQ(orbit1Patterns[0].name, "test_pattern");

  auto orbit2Patterns = database->getPatternsByOrbitId(2);
  ASSERT_EQ(orbit2Patterns.size(), 1);
  EXPECT_EQ(orbit2Patterns[0].name, "another_pattern");

  auto orbit3Patterns = database->getPatternsByOrbitId(3);
  EXPECT_TRUE(orbit3Patterns.empty());
}

TEST_F(PatternDatabaseTest, PatternMatching) {
  database->loadFromYaml(testFilePath);

  // Test matching code against patterns
  std::string testCode = "This is a test example";

  auto matches = database->findMatches(testCode);
  EXPECT_FALSE(matches.empty());

  // Should find matches for both patterns
  bool foundTest = false;
  bool foundAnother = false;

  for (const auto& match : matches) {
    if (match.patternName == "test_pattern") foundTest = true;
    if (match.patternName == "another_pattern") foundAnother = true;
  }

  EXPECT_TRUE(foundTest);
  EXPECT_FALSE(foundAnother); // "another" not in test code
}

TEST_F(PatternDatabaseTest, InvalidYamlFile) {
  std::filesystem::path invalidPath = "nonexistent.yaml";
  EXPECT_FALSE(database->loadFromYaml(invalidPath));
}

TEST_F(PatternDatabaseTest, EmptyDatabase) {
  const auto& patterns = database->getPatterns();
  EXPECT_TRUE(patterns.empty());

  auto matches = database->findMatches("any code");
  EXPECT_TRUE(matches.empty());
}

TEST_F(PatternDatabaseTest, PatternWeights) {
  database->loadFromYaml(testFilePath);

  const auto& patterns = database->getPatterns();

  // Check that weights are preserved
  auto testPattern = std::find_if(patterns.begin(), patterns.end(),
    [](const OrbitPattern& p) { return p.name == "test_pattern"; });
  ASSERT_NE(testPattern, patterns.end());
  EXPECT_DOUBLE_EQ(testPattern->weight, 0.8);

  auto anotherPattern = std::find_if(patterns.begin(), patterns.end(),
    [](const OrbitPattern& p) { return p.name == "another_pattern"; });
  ASSERT_NE(anotherPattern, patterns.end());
  EXPECT_DOUBLE_EQ(anotherPattern->weight, 0.9);
}

TEST_F(PatternDatabaseTest, ClearDatabase) {
  database->loadFromYaml(testFilePath);
  EXPECT_FALSE(database->getPatterns().empty());

  database->clear();
  EXPECT_TRUE(database->getPatterns().empty());
}

} // namespace ir
} // namespace cppfort