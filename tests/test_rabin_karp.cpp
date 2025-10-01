#include "rabin_karp.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace cppfort {
namespace ir {

class RabinKarpTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Default configuration with word-based windows
    rabinKarp = std::make_unique<RabinKarp>(31, std::vector<size_t>{1, 2, 4, 8, 16, 32, 64});
  }

  std::unique_ptr<RabinKarp> rabinKarp;
};

TEST_F(RabinKarpTest, Initialization) {
  std::string firstWord = "hello";
  rabinKarp->initialize(firstWord);

  // Verify hashes are computed for all levels
  for (size_t level = 0; level < 7; ++level) {
    EXPECT_TRUE(rabinKarp->hashAt(level) != 0);
  }
}

TEST_F(RabinKarpTest, WordBasedRollingHash) {
  std::string firstWord = "hello";
  rabinKarp->initialize(firstWord);

  uint64_t initialHash1 = rabinKarp->hashAt(0); // 1-word window
  uint64_t initialHash2 = rabinKarp->hashAt(1); // 2-word window

  // Update by removing "hello" and adding "world"
  rabinKarp->update("hello", "world");

  uint64_t updatedHash1 = rabinKarp->hashAt(0); // Should now be hash of "world"
  uint64_t updatedHash2 = rabinKarp->hashAt(1); // Should be hash of "hello" + "world"

  // The hashes should be different after rolling update
  EXPECT_NE(initialHash1, updatedHash1);
  EXPECT_NE(initialHash2, updatedHash2);
}

TEST_F(RabinKarpTest, ProcessText) {
  std::string text = "hello world this is a test";
  auto results = rabinKarp->processText(text);

  // Should have results for each word in the text
  EXPECT_EQ(results.size(), 6u); // hello, world, this, is, a, test

  // Each result should have 7 hash levels
  for (const auto& result : results) {
    EXPECT_EQ(result.size(), 7u);
    for (size_t level = 0; level < 7; ++level) {
      EXPECT_TRUE(result[level] != 0);
    }
  }
}

TEST_F(RabinKarpTest, WordBoundaryDetection) {
  // Test various word boundaries
  std::string text1 = "hello, world!";
  std::string text2 = "hello\tworld\n";

  auto results1 = rabinKarp->processText(text1);
  auto results2 = rabinKarp->processText(text2);

  // Both should produce same word sequence: "hello", "world"
  EXPECT_EQ(results1.size(), 2u);
  EXPECT_EQ(results2.size(), 2u);

  // Hashes should be identical for same words regardless of punctuation
  for (size_t level = 0; level < 7; ++level) {
    EXPECT_EQ(results1.back()[level], results2.back()[level]);
  }
}

TEST_F(RabinKarpTest, DifferentWordWindowSizes) {
  std::string text = "word1 word2 word3 word4 word5";
  auto results = rabinKarp->processText(text);

  // Test that different window sizes produce different hashes
  // Level 0 (1 word) vs Level 1 (2 words) should be different
  EXPECT_NE(results.back()[0], results.back()[1]);

  // Verify window sizes
  EXPECT_EQ(rabinKarp->wordWindowSize(0), 1u);
  EXPECT_EQ(rabinKarp->wordWindowSize(1), 2u);
  EXPECT_EQ(rabinKarp->wordWindowSize(2), 4u);
  EXPECT_EQ(rabinKarp->wordWindowSize(6), 64u);
}

TEST_F(RabinKarpTest, RollingWindowBehavior) {
  // Test that rolling windows maintain correct word sequences
  rabinKarp->initialize("first");

  // Add words one by one
  rabinKarp->update("first", "second");
  rabinKarp->update("second", "third");
  rabinKarp->update("third", "fourth");
  rabinKarp->update("fourth", "fifth");

  // Level 0 should have hash of "fifth"
  // Level 1 should have hash of "fourth fifth"
  // Level 2 should have hash of "second third fourth fifth" (4 words)

  uint64_t hashLevel0 = rabinKarp->hashAt(0);
  uint64_t hashLevel1 = rabinKarp->hashAt(1);
  uint64_t hashLevel2 = rabinKarp->hashAt(2);

  EXPECT_TRUE(hashLevel0 != 0);
  EXPECT_TRUE(hashLevel1 != 0);
  EXPECT_TRUE(hashLevel2 != 0);

  // Different levels should have different hashes
  EXPECT_NE(hashLevel0, hashLevel1);
  EXPECT_NE(hashLevel1, hashLevel2);
}

TEST_F(RabinKarpTest, ResetFunctionality) {
  rabinKarp->initialize("test");
  uint64_t hashBeforeReset = rabinKarp->hashAt(0);

  rabinKarp->reset();

  // After reset, hashes should be zero
  for (size_t level = 0; level < 7; ++level) {
    EXPECT_EQ(rabinKarp->hashAt(level), 0u);
  }
}

TEST_F(RabinKarpTest, EmptyText) {
  auto results = rabinKarp->processText("");
  EXPECT_TRUE(results.empty());

  auto resultsWhitespace = rabinKarp->processText("   \t\n  ");
  EXPECT_TRUE(resultsWhitespace.empty());
}

} // namespace ir
} // namespace cppfort