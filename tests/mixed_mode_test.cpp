#include "lexer.hpp"
#include "parser.hpp"
#include <gtest/gtest.h>

static const char* mixed_src = R"(
#pragma mixed-mode
// legacy C++ code
int main() {
    // Cpp2‑only keyword should be ignored in mixed mode
    co_await something();
    auto [a, b] = std::pair<int,int>(1,2);
    return 0;
}
)";

TEST(MixedModeDetection, DetectsPragma) {
    EXPECT_TRUE(cpp2_transpiler::Lexer::isMixedMode(std::string(mixed_src)));
}

TEST(MixedModeLexer, FiltersCpp2OnlyTokens) {
    cpp2_transpiler::Lexer lexer(std::string(mixed_src));
    auto tokens = lexer.tokenize();
    // Ensure no token of type CoAwait (or any Cpp2‑only token) appears.
    for (const auto& t : tokens) {
        EXPECT_NE(t.type, cpp2_transpiler::TokenType::CoAwait);
    }
}

TEST(MixedModeParser, HybridParseSucceeds) {
    cpp2_transpiler::Lexer lexer(std::string(mixed_src));
    cpp2_transpiler::Parser parser;
    // Signal mixed mode to parser (using flag we added earlier)
    // Assuming parser has a setter; if not, it will infer from lexer.
    parser.setMixedMode(true);
    auto result = parser.parseHybrid(std::string(mixed_src));
    EXPECT_TRUE(result.success);
}
