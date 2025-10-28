#pragma once

#include <string>
#include <sstream>
#include "ast.h"

namespace cppfort::stage0 {

// Simple text-based IR serialization format
// Format is line-based for easy parsing:
// - Lines starting with @ are directives
// - Nested structures use indentation
class IRSerializer {
public:
    static ::std::string serialize(const TranslationUnit& unit);
    static TranslationUnit deserialize(const ::std::string& ir_text);

private:
    static void serializeParameter(::std::ostringstream& out, const Parameter& param, int indent);
    static void serializeStatement(::std::ostringstream& out, const OrbitStatement& stmt, int indent);
    static void serializeBlock(::std::ostringstream& out, const OrbitBlock& block, int indent);
    static void serializeFunction(::std::ostringstream& out, const FunctionDecl& fn, int indent);
    static void serializeType(::std::ostringstream& out, const TypeDecl& type, int indent);

    static ::std::string escape(const ::std::string& str);
    static ::std::string unescape(const ::std::string& str);
    static ::std::string indent(int level);
    static ::std::string parameterKindToString(ParameterKind kind);
    static ParameterKind stringToParameterKind(const ::std::string& str);
};

} // namespace cppfort::stage0
