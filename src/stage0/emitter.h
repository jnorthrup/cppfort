#pragma once

#include <string>
#include <string_view>

#include "ast.h"

namespace cppfort::stage0 {

enum class EmitBackend {
    Cpp,
    Mlir,
};

struct EmitOptions {
    EmitBackend backend = EmitBackend::Cpp;
    bool include_headers = true;
};

class Emitter {
  public:
    [[nodiscard]] std::string emit(const TranslationUnit& unit, const EmitOptions& options = {}) const;

  private:
    std::string emit_cpp(const TranslationUnit& unit, const EmitOptions& options) const;
    std::string emit_mlir(const TranslationUnit& unit) const;

    void emit_forward_declaration(const FunctionDecl& fn, std::string& out, int indent) const;
    void emit_function(const FunctionDecl& fn, std::string& out, int indent) const;
    void emit_block(const Block& block, std::string& out, int indent, bool add_return_0 = false) const;
    void emit_statement(const Statement& stmt, std::string& out, int indent) const;
    void emit_type(const TypeDecl& type, std::string& out, int indent) const;
    std::string get_param_type(const Parameter& param) const;

    static void append_line(std::string& out, std::string_view text, int indent);
    static std::string indent_string(int indent);
};

} // namespace cppfort::stage0
