#pragma once

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include "orbit_ring.h"

namespace cppfort::stage0 {

// Forward declarations
struct Statement;
struct Block;
struct EmitOptions;

// Type aliases for compatibility with transpiler expectations
using TranslationUnit = OrbitTranslationUnit;
using Parameter = OrbitParameter;
using FunctionDecl = OrbitFunctionDecl;
using TypeDecl = OrbitTypeDecl;

// Additional structures needed by emitter
struct Statement {
    // Placeholder for now - will be expanded as needed
};

struct Block {
    std::vector<Statement> statements;
};

struct EmitOptions {
    bool include_headers = true;
    bool inline_cpp2_header = false;
    bool bundle_headers = false;
};

} // namespace cppfort::stage0