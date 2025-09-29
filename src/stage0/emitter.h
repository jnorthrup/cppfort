#pragma once

#include <string>
#include <string_view>

#include "ast.h"

namespace cppfort::stage0 {

struct EmitOptions {
    bool include_preamble {true};
};

class Emitter {
  public:
    [[nodiscard]] std::string emit(const TranslationUnit& unit, const EmitOptions& options = {}) const;

  private:
    void emit_function(const FunctionDecl& fn, std::string& out, int indent) const;
    void emit_block(const Block& block, std::string& out, int indent, bool add_return_0 = false) const;
    void emit_statement(const Statement& stmt, std::string& out, int indent) const;
    void emit_type(const TypeDecl& type, std::string& out, int indent) const;

    static void append_line(std::string& out, std::string_view text, int indent);
    static std::string indent_string(int indent);
};

} // namespace cppfort::stage0
