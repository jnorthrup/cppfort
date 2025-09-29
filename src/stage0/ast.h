#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "token.h"

namespace cppfort::stage0 {

enum class ParameterKind {
    Default,
    In,
    InOut,
    Out,
    Copy,
    Move,
    Forward,
};

struct Parameter {
    std::string name;
    std::string type;
    ParameterKind kind {ParameterKind::Default};
    SourceLocation location;
};

struct VariableDecl {
    std::string name;
    std::string type;
    std::optional<std::string> initializer;
    SourceLocation location;
};

struct ExpressionStmt {
    std::string expression;
    SourceLocation location;
};

struct ReturnStmt {
    std::optional<std::string> expression;
    SourceLocation location;
};

struct AssertStmt {
    std::string condition;
    SourceLocation location;
};

using Statement = std::variant<VariableDecl, ExpressionStmt, ReturnStmt, AssertStmt>;

struct Block {
    std::vector<Statement> statements;
    SourceLocation location;
};

struct ExpressionBody {
    std::string expression;
    SourceLocation location;
};

using FunctionBody = std::variant<Block, ExpressionBody>;

struct FunctionDecl {
    std::string name;
    std::vector<Parameter> parameters;
    std::optional<std::string> return_type;
    FunctionBody body;
    SourceLocation location;
};

struct TypeDecl {
    std::string name;
    std::string body;
    SourceLocation location;
};

struct IncludeDecl {
    std::string path;
    bool is_system {false};
    SourceLocation location;
};

struct TranslationUnit {
    std::vector<IncludeDecl> includes;
    std::vector<TypeDecl> types;
    std::vector<FunctionDecl> functions;
};

} // namespace cppfort::stage0
