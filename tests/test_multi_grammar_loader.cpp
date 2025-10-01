#include "multi_grammar_loader.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

namespace cppfort {
namespace ir {

class MultiGrammarLoaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    loader = std::make_unique<MultiGrammarLoader>();

    // Create temporary test directory
    testDir = std::filesystem::temp_directory_path() / "test_patterns";
    std::filesystem::create_directory(testDir);

    // Create test pattern files
    createTestPatternFile("c_patterns.yaml", "C");
    createTestPatternFile("cpp_patterns.yaml", "CPP");
    createTestPatternFile("cpp2_patterns.yaml", "CPP2");
  }

  void TearDown() override {
    // Clean up test directory
    std::filesystem::remove_all(testDir);
  }

  void createTestPatternFile(const std::string& filename, const std::string& grammarType) {
    std::string content = "name: " + grammarType + "_test\n"
                         "orbit_id: 1\n"
                         "weight: 0.8\n"
                         "signature_patterns:\n"
                         "  - \"test_" + grammarType + "\"\n"
                         "protocol_indicators:\n"
                         "  - \"" + grammarType + "\"\n"
                         "version_patterns:\n"
                         "  - \"1.0\"\n";

    std::ofstream file(testDir / filename);
    file << content;
    file.close();
  }

  std::unique_ptr<MultiGrammarLoader> loader;
  std::filesystem::path testDir;
};

TEST_F(MultiGrammarLoaderTest, LoadAllGrammars) {
  ASSERT_TRUE(loader->loadAllGrammars(testDir));

  auto loadedGrammars = loader->getLoadedGrammars();
  EXPECT_EQ(loadedGrammars.size(), 3);

  EXPECT_TRUE(loader->isGrammarLoaded(GrammarType::C));
  EXPECT_TRUE(loader->isGrammarLoaded(GrammarType::CPP));
  EXPECT_TRUE(loader->isGrammarLoaded(GrammarType::CPP2));
}

TEST_F(MultiGrammarLoaderTest, LoadSpecificGrammar) {
  std::filesystem::path cPatternPath = testDir / "c_patterns.yaml";
  ASSERT_TRUE(loader->loadGrammar(GrammarType::C, cPatternPath));

  EXPECT_TRUE(loader->isGrammarLoaded(GrammarType::C));
  EXPECT_FALSE(loader->isGrammarLoaded(GrammarType::CPP));

  const auto& cPatterns = loader->getPatterns(GrammarType::C);
  EXPECT_EQ(cPatterns.size(), 1);
  EXPECT_EQ(cPatterns[0].name, "C_test");
}

TEST_F(MultiGrammarLoaderTest, GetAllPatterns) {
  loader->loadAllGrammars(testDir);

  auto allPatterns = loader->getAllPatterns();
  EXPECT_EQ(allPatterns.size(), 3);

  // Check that patterns from different grammars are included
  bool hasCTest = false, hasCppTest = false, hasCpp2Test = false;

  for (const auto& pattern : allPatterns) {
    if (pattern.name == "C_test") hasCTest = true;
    if (pattern.name == "CPP_test") hasCppTest = true;
    if (pattern.name == "CPP2_test") hasCpp2Test = true;
  }

  EXPECT_TRUE(hasCTest);
  EXPECT_TRUE(hasCppTest);
  EXPECT_TRUE(hasCpp2Test);
}

TEST_F(MultiGrammarLoaderTest, LoadStats) {
  loader->loadAllGrammars(testDir);

  auto stats = loader->getLoadStats();
  EXPECT_EQ(stats.totalPatterns, 3);
  EXPECT_EQ(stats.patternsByGrammar[GrammarType::C], 1);
  EXPECT_EQ(stats.patternsByGrammar[GrammarType::CPP], 1);
  EXPECT_EQ(stats.patternsByGrammar[GrammarType::CPP2], 1);
  EXPECT_TRUE(stats.errors.empty());
}

TEST_F(MultiGrammarLoaderTest, InvalidDirectory) {
  std::filesystem::path invalidDir = "nonexistent_directory";
  EXPECT_FALSE(loader->loadAllGrammars(invalidDir));

  auto stats = loader->getLoadStats();
  EXPECT_FALSE(stats.errors.empty());
}

TEST_F(MultiGrammarLoaderTest, InvalidPatternFile) {
  std::filesystem::path invalidFile = testDir / "invalid.txt";
  std::ofstream invalidFileStream(invalidFile);
  invalidFileStream << "not yaml content";
  invalidFileStream.close();

  EXPECT_FALSE(loader->loadGrammar(GrammarType::C, invalidFile));
}

TEST_F(MultiGrammarLoaderTest, ClearLoader) {
  loader->loadAllGrammars(testDir);
  EXPECT_FALSE(loader->getLoadedGrammars().empty());

  loader->clear();
  EXPECT_TRUE(loader->getLoadedGrammars().empty());

  auto stats = loader->getLoadStats();
  EXPECT_EQ(stats.totalPatterns, 0);
}

// TEST_F(MultiGrammarLoaderTest, GrammarTypeDetection) {
//   // Test filename to grammar type detection
//   EXPECT_EQ(loader->detectGrammarType("c_patterns.yaml"), GrammarType::C);
//   EXPECT_EQ(loader->detectGrammarType("cpp_patterns.yaml"), GrammarType::CPP);
//   EXPECT_EQ(loader->detectGrammarType("cpp2_patterns.yaml"), GrammarType::CPP2);
//   EXPECT_EQ(loader->detectGrammarType("unknown.yaml"), GrammarType::UNKNOWN);
// }

TEST_F(MultiGrammarLoaderTest, UtilityFunctions) {
  EXPECT_EQ(grammarTypeToString(GrammarType::C), "C");
  EXPECT_EQ(grammarTypeToString(GrammarType::CPP), "C++");
  EXPECT_EQ(grammarTypeToString(GrammarType::CPP2), "CPP2");
  EXPECT_EQ(grammarTypeToString(GrammarType::UNKNOWN), "UNKNOWN");

  EXPECT_EQ(stringToGrammarType("C"), GrammarType::C);
  EXPECT_EQ(stringToGrammarType("C++"), GrammarType::CPP);
  EXPECT_EQ(stringToGrammarType("CPP2"), GrammarType::CPP2);
  EXPECT_EQ(stringToGrammarType("unknown"), GrammarType::UNKNOWN);
}

TEST_F(MultiGrammarLoaderTest, PartialLoad) {
  // Load only C and CPP, not CPP2
  std::filesystem::remove(testDir / "cpp2_patterns.yaml");

  ASSERT_TRUE(loader->loadAllGrammars(testDir));

  auto loadedGrammars = loader->getLoadedGrammars();
  EXPECT_EQ(loadedGrammars.size(), 2);

  EXPECT_TRUE(loader->isGrammarLoaded(GrammarType::C));
  EXPECT_TRUE(loader->isGrammarLoaded(GrammarType::CPP));
  EXPECT_FALSE(loader->isGrammarLoaded(GrammarType::CPP2));
}

} // namespace ir
} // namespace cppfort