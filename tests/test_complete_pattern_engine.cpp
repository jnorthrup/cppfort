#include "complete_pattern_engine.h"

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

namespace {

std::filesystem::path writeTempPatterns(const std::string& content) {
    std::filesystem::path tmp = std::filesystem::temp_directory_path() / std::filesystem::path("cppfort_patterns_XXXXXX.yaml");
    std::string templ = tmp.string();
    std::vector<char> buffer(templ.begin(), templ.end());
    buffer.push_back('\0');
    int fd = mkstemp(buffer.data());
    if (fd == -1) {
        throw std::runtime_error("mkstemp failed");
    }
    ::close(fd);
    std::ofstream out(buffer.data(), std::ios::trunc);
    out << content;
    return std::filesystem::path(buffer.data());
}

}

TEST(CompletePatternEngineTest, LoadsMultiDocumentYaml) {
    const std::string yaml = R"YAML(
---
name: func_with_return
use_alternating: true
alternating_anchors:
  - "("
  - ")"
grammar_modes: 7
evidence_types:
  - identifier
  - params
  - return_type
  - body
priority: 10
transformation_templates:
  2: "$3 $1($2) { $4 }"
---
name: func_void
use_alternating: true
alternating_anchors:
  - "("
  - ")"
grammar_modes: 7
evidence_types:
  - identifier
  - params
  - body
priority: 5
transformation_templates:
  2: "void $1($2) { $3 }"
)YAML";

    auto path = writeTempPatterns(yaml);
    cppfort::CompletePatternEngine engine;
    ASSERT_TRUE(engine.loadPatterns(path.string()));
}

TEST(CompletePatternEngineTest, TransformationTemplatesProduceValidMain) {
    const std::string yaml = R"YAML(
---
name: main_function
use_alternating: false
grammar_modes: 7
evidence_types:
  - identifier
  - body
priority: 100
transformation_templates:
  2: "int $1() { $2 }"
)YAML";

    auto path = writeTempPatterns(yaml);
    cppfort::CompletePatternEngine engine;
    ASSERT_TRUE(engine.loadPatterns(path.string()));

    const std::string input = "main: () = { return 0; }";
    const std::string output = engine.applyGraphTransformations(input);

    EXPECT_NE(output.find("int main()"), std::string::npos);
}

