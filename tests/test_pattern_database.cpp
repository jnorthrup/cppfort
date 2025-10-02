#include "tblgen_patterns.h"
#include <algorithm>
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
  EXPECT_EQ(patterns[0].orbit_id, 1);
  EXPECT_DOUBLE_EQ(patterns[0].weight, 0.8);
  EXPECT_EQ(patterns[0].signature_patterns.size(), 2);
  EXPECT_EQ(patterns[0].protocol_indicators.size(), 1);
  EXPECT_EQ(patterns[0].version_patterns.size(), 1);

  // Check second pattern
  EXPECT_EQ(patterns[1].name, "another_pattern");
  EXPECT_EQ(patterns[1].orbit_id, 2);
  EXPECT_DOUBLE_EQ(patterns[1].weight, 0.9);
}

TEST_F(PatternDatabaseTest, ExportToTableGen) {
  database->loadFromYaml(testFilePath);

  std::string tablegen = database->exportToTableGen("TestDialect");
  EXPECT_FALSE(tablegen.empty());

  // Check for expected TableGen structure
  EXPECT_NE(tablegen.find("class OrbitPattern"), std::string::npos);
  EXPECT_NE(tablegen.find("test_pattern"), std::string::npos);
  EXPECT_NE(tablegen.find("another_pattern"), std::string::npos);
}

TEST_F(PatternDatabaseTest, GetPatternByName) {
  database->loadFromYaml(testFilePath);

  auto pattern = database->getPattern("test_pattern");
  ASSERT_TRUE(pattern.has_value());
  EXPECT_EQ(pattern->name, "test_pattern");
  EXPECT_EQ(pattern->orbit_id, 1);

  auto notFound = database->getPattern("nonexistent");
  EXPECT_FALSE(notFound.has_value());
}

TEST_F(PatternDatabaseTest, GetPatternsForOrbit) {
  database->loadFromYaml(testFilePath);

  auto orbit1Patterns = database->getPatternsForOrbit(1);
  EXPECT_EQ(orbit1Patterns.size(), 1);
  EXPECT_EQ(orbit1Patterns[0].orbit_id, 1);

  auto orbit2Patterns = database->getPatternsForOrbit(2);
  EXPECT_EQ(orbit2Patterns.size(), 1);
  EXPECT_EQ(orbit2Patterns[0].orbit_id, 2);

  auto orbit3Patterns = database->getPatternsForOrbit(3);
  EXPECT_TRUE(orbit3Patterns.empty());
}

TEST_F(PatternDatabaseTest, InvalidYamlFile) {
  std::filesystem::path invalidPath = "nonexistent.yaml";
  EXPECT_FALSE(database->loadFromYaml(invalidPath));
}

TEST_F(PatternDatabaseTest, EmptyDatabase) {
  const auto& patterns = database->getPatterns();
  EXPECT_TRUE(patterns.empty());
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
