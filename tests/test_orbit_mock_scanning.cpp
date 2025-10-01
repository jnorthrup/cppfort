#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <initializer_list>

#include "orbit_scanner.h"
#include "orbit_mask.h"
#include "rabin_karp.h"

namespace cppfort::ir {

class OrbitMockScanningTest : public ::testing::Test {
protected:
    void SetUp() override {
        OrbitScannerConfig config;
        config.patternThreshold = 0.05;   // Allow low-signal mock matches
        config.minConfidence = 0.0;       // Let the scanner report tentative grammars
        config.maxMatches = 64;           // Keep deterministic while exercising truncation
        config.maxDepth = 256;            // Plenty of headroom for synthetic snippets
        scanner = ::std::make_unique<OrbitScanner>(config);

        mockPatterns = {
            makePattern("mock_c", GrammarType::C, 1.0, {"printf", "%d"}),
            makePattern("mock_cpp", GrammarType::CPP, 1.75, {"std::vector", "template"}),
            makePattern("mock_cpp2", GrammarType::CPP2, 1.75, {"inspect", "=>"})
        };
    }

    static OrbitPattern makePattern(const ::std::string& name,
                                    GrammarType grammar,
                                    double weight,
                                    ::std::initializer_list<::std::string> signatures) {
        OrbitPattern pattern(name, static_cast<uint32_t>(grammar), weight);
        pattern.signature_patterns = signatures;
        pattern.protocol_indicators = {name + "_protocol"};
        pattern.version_patterns = {"1.0"};
        return pattern;
    }

    ::std::unique_ptr<OrbitScanner> scanner;
    ::std::vector<OrbitPattern> mockPatterns;
};

TEST_F(OrbitMockScanningTest, ClassifiesMockGrammarsFromOrbitSignals) {
    struct Scenario {
        ::std::string description;
        GrammarType expected;
        ::std::string code;
    };

    const ::std::vector<Scenario> scenarios = {
        {
            "C constructs",
            GrammarType::C,
            R"(
#include <stdio.h>

int fizzbuzz_c(void) {
    for (int i = 1; i <= 15; ++i) {
        if (i % 15 == 0) {
            printf("FizzBuzz\n");
        } else if (i % 3 == 0) {
            printf("Fizz\n");
        } else if (i % 5 == 0) {
            printf("Buzz\n");
        } else {
            printf("%d\n", i);
        }
    }
    return 0;
}
)"
        },
        {
            "C++ constructs",
            GrammarType::CPP,
            R"(
#include <iostream>
#include <vector>
#include <string>

template <typename Printer>
void fizzbuzz_cpp(int limit, Printer printer) {
    std::vector<std::string> cache;
    cache.reserve(limit);

    for (int i = 1; i <= limit; ++i) {
        if (i % 15 == 0) {
            cache.emplace_back("FizzBuzz");
        } else if (i % 3 == 0) {
            cache.emplace_back("Fizz");
        } else if (i % 5 == 0) {
            cache.emplace_back("Buzz");
        } else {
            cache.emplace_back(std::to_string(i));
        }
    }

    for (const auto& item : cache) {
        printer(item);
    }
}
)"
        },
        {
            "CPP2 constructs",
            GrammarType::CPP2,
            R"(
auto fizzbuzz_cpp2 = [](auto limit) {
    for (auto i = 1; i <= limit; ++i) {
        inspect (i) {
            0 => "zero";
            _ when i % 15 == 0 => "FizzBuzz";
            _ when i % 3 == 0 => "Fizz";
            _ when i % 5 == 0 => "Buzz";
            _ => std::to_string(i);
        };
    }
};
)"
        }
    };

    for (const auto& scenario : scenarios) {
        auto result = scanner->scan(scenario.code, mockPatterns);

        EXPECT_EQ(result.detectedGrammar, scenario.expected) << scenario.description;
        EXPECT_GT(result.confidence, 0.0) << scenario.description;
        EXPECT_FALSE(result.matches.empty()) << scenario.description;
    }
}

TEST_F(OrbitMockScanningTest, AttachesOrbitCountsAndHashesToMatches) {
    const ::std::string code = "void fizz() { printf(\"FizzBuzz\\n\"); }";
    auto result = scanner->scan(code, mockPatterns);

    ASSERT_FALSE(result.matches.empty());
    const auto& match = result.matches.front();
    EXPECT_EQ(match.grammarType, GrammarType::C);
    EXPECT_GT(match.orbitCounts[0], 0u);
    EXPECT_GT(match.orbitHashes[0], 0u);
}

TEST_F(OrbitMockScanningTest, RespectsConfiguredMatchLimit) {
    OrbitScannerConfig limitedConfig = scanner->getConfig();
    limitedConfig.maxMatches = 5;
    scanner->updateConfig(limitedConfig);

    ::std::string noisyCode;
    for (int i = 0; i < 20; ++i) {
        noisyCode += "printf(\"FizzBuzz\\n\");";
    }

    auto result = scanner->scan(noisyCode, mockPatterns);
    EXPECT_LE(result.matches.size(), limitedConfig.maxMatches);
}

TEST_F(OrbitMockScanningTest, HonorsMinimumConfidenceFloor) {
    const ::std::string code = "int main() { printf(\"FizzBuzz\\n\"); }";

    auto baseline = scanner->scan(code, mockPatterns);
    EXPECT_NE(baseline.detectedGrammar, GrammarType::UNKNOWN);
    EXPECT_GT(baseline.confidence, 0.0);

    OrbitScannerConfig strictConfig = scanner->getConfig();
    strictConfig.minConfidence = 0.95;
    scanner->updateConfig(strictConfig);

    auto strictResult = scanner->scan(code, mockPatterns);
    EXPECT_EQ(strictResult.detectedGrammar, GrammarType::UNKNOWN);
    EXPECT_DOUBLE_EQ(strictResult.confidence, 0.0);
    EXPECT_NE(strictResult.reasoning.find("No grammar detected"), ::std::string::npos);
}

TEST_F(OrbitMockScanningTest, OrbitContextBalanceReflectsConfidence) {
    OrbitContext balanced;
    for (char ch : ::std::string("{[()]}")) {
        balanced.update(ch);
    }

    EXPECT_TRUE(balanced.isBalanced());
    EXPECT_DOUBLE_EQ(balanced.calculateConfidence(), 1.0);

    OrbitContext unbalanced;
    for (char ch : ::std::string("{{[")) {
        unbalanced.update(ch);
    }

    EXPECT_FALSE(unbalanced.isBalanced());
    EXPECT_LT(unbalanced.calculateConfidence(), 1.0);
}

TEST_F(OrbitMockScanningTest, OrbitHierarchicalHashing) {
    RabinKarp rabinKarp;

    ::std::vector<::std::array<size_t, 6>> testCounts = {
        {0, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1},
        {3, 2, 1, 4, 2, 3},
        {10, 10, 10, 10, 10, 10},
    };

    for (const auto& counts : testCounts) {
        OrbitContext context;
        for (size_t type = 0; type < counts.size(); ++type) {
            char ch = ' ';
            switch (type) {
                case 0: ch = '{'; break;
                case 1: ch = '['; break;
                case 2: ch = '<'; break;
                case 3: ch = '('; break;
                case 4: ch = '"'; break;
                case 5: ch = '1'; break;
            }

            for (size_t i = 0; i < counts[type]; ++i) {
                context.update(ch);
            }
        }

        auto actualCounts = context.getCounts();
        auto hashes = rabinKarp.processOrbitContext(context);

        bool hasAnyCounts = ::std::any_of(counts.begin(), counts.end(), [](size_t c) { return c > 0; });
        if (hasAnyCounts) {
            bool hasAnyHashes = ::std::any_of(hashes.begin(), hashes.end(), [](uint64_t h) { return h > 0; });
            EXPECT_TRUE(hasAnyHashes) << "Expected non-zero hashes for populated orbit context";
        }

        auto hashes2 = rabinKarp.processOrbitContext(context);
        EXPECT_EQ(hashes, hashes2);

        for (size_t type = 0; type < counts.size(); ++type) {
            if (type == 4) { // quote depth toggles between 0 and 1
                EXPECT_LE(actualCounts[type], 1u);
                continue;
            }
            EXPECT_EQ(actualCounts[type], counts[type]);
        }
    }
}

} // namespace cppfort::ir
