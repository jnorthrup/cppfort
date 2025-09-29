#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <memory>

#include "ast.h"
#include "token.h"

namespace cppfort::stage0 {

// Use canonical AST structures from ast.h
// Extended structures for bidirectional transformation

struct TransformOptions {
    bool include_preamble {true};
    bool target_cpp2 {false};  // true for cpp2 output, false for C++ output
    bool preserve_comments {false};
};

// The unified bidirectional transpiler
class BidirectionalTranspiler {
public:
    BidirectionalTranspiler();

    // Parse cpp2 source into AST
    [[nodiscard]] TranslationUnit parse_cpp2(const std::string& source, const std::string& filename);

    // Parse C++ source into AST
    [[nodiscard]] TranslationUnit parse_cpp(const std::string& source, const std::string& filename);

    // Emit AST as cpp2 source
    [[nodiscard]] std::string emit_cpp2(const TranslationUnit& unit, const TransformOptions& options = {});

    // Emit AST as C++ source
    [[nodiscard]] std::string emit_cpp(const TranslationUnit& unit, const TransformOptions& options = {});

    // Roundtrip verification: cpp2 -> AST -> cpp2
    [[nodiscard]] std::string roundtrip_cpp2(const std::string& source, const std::string& filename);

    // Roundtrip verification: C++ -> AST -> C++
    [[nodiscard]] std::string roundtrip_cpp(const std::string& source, const std::string& filename);

private:
    // Internal parsers and emitters
    class Cpp2Parser;
    class CppParser;
    class Cpp2Emitter;
    class CppEmitter;

    std::shared_ptr<Cpp2Parser> m_cpp2_parser;
    std::shared_ptr<CppParser> m_cpp_parser;
    std::shared_ptr<Cpp2Emitter> m_cpp2_emitter;
    std::shared_ptr<CppEmitter> m_cpp_emitter;
};

} // namespace cppfort::stage0