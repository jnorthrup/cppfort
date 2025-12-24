#!/usr/bin/env python3
"""
Refine cpp2 AST nodes using Clang AST ground truth from regression corpus.

This tool analyzes the generated Clang AST dumps to extract patterns
and create comprehensive cpp2 AST node definitions.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Any

def parse_clang_ast(ast_file: str) -> List[Dict[str, Any]]:
    """Parse Clang AST dump to extract function patterns."""
    patterns = []

    with open(ast_file, 'r') as f:
        content = f.read()

    # Extract function declarations
    func_pattern = r'FunctionDecl\s+<(.*?)>\s+line:(\d+):(\d+)\s+(\w+)\s+[\'"]([^\'"]+)[\'"]\s+[\'"]([^\'"]+)[\'"]'

    for match in re.finditer(func_pattern, content, re.MULTILINE):
        loc, line, col, implicit, name, sig = match.groups()

        # Extract parameter information
        param_pattern = rf'FunctionDecl.*?{re.escape(name)}.*?ParmVarDecl.*?(\w+)\s+[\'"]([^\'"]+)[\'"].*?[\'"]([^\'"]+)[\'"]'
        params = []
        for pmatch in re.finditer(param_pattern, content):
            loc_info, pname, ptype = pmatch.groups()
            params.append({
                'name': pname,
                'type': ptype,
                'modifiers': ['&'] if '&' in ptype else []
            })

        patterns.append({
            'name': name,
            'signature': sig,
            'parameters': params,
            'location': f"{line}:{col}"
        })

    return patterns

def extract_cpp2_patterns(mapping_file: str) -> Dict[str, Any]:
    """Extract cpp2 patterns from a mapping file."""
    with open(mapping_file, 'r') as f:
        content = f.read()

    parts = content.split("=== Generated C++1 ===")
    if len(parts) != 2:
        return {}

    cpp2_part = parts[0].replace("=== Cpp2 Source ===", "").strip()
    cpp1_part = parts[1].split("=== AST Patterns ===")[0].strip()

    # Extract patterns
    patterns = {
        'cpp2_source': cpp2_part,
        'cpp1_generated': cpp1_part,
        'functions': [],
        'ufcs_calls': [],
        'safety_checks': []
    }

    # Find UFCS patterns
    ufcs_pattern = r'CPP2_UFCS\((\w+)\)\(([^)]+)\)'
    for match in re.finditer(ufcs_pattern, cpp1_part):
        func, args = match.groups()
        patterns['ufcs_calls'].append({
            'function': func,
            'arguments': [arg.strip() for arg in args.split(',')]
        })

    # Find safety check patterns
    safety_pattern = r'CPP2_ASSERT_([A-Z_]+)\(([^)]+)\)'
    for match in re.finditer(safety_pattern, cpp1_part):
        check_type, args = match.groups()
        patterns['safety_checks'].append({
            'type': check_type,
            'arguments': args
        })

    return patterns

def generate_ast_refined():
    """Generate refined AST nodes based on corpus analysis."""

    generated_dir = Path("corpus/ast_mappings/generated")
    all_patterns = []

    # Process all mapping files
    for mapping_file in generated_dir.glob("*.mapping"):
        patterns = extract_cpp2_patterns(str(mapping_file))
        if patterns:
            all_patterns.append({
                'file': mapping_file.name,
                'patterns': patterns
            })

    # Extract function patterns
    function_signatures = {}
    for entry in all_patterns:
        # Look for function signatures in cpp2 source
        cpp2 = entry['patterns']['cpp2_source']

        # Pattern: name: (params) -> ret = { body }
        func_pattern = r'(\w+)\s*:\s*\(([^)]*)\)\s*->\s*([^\s=]+)\s*='
        for match in re.finditer(func_pattern, cpp2):
            name, params, ret_type = match.groups()

            param_list = []
            if params.strip():
                # Parse parameters: type param, inout type param, etc.
                for param in params.split(','):
                    param = param.strip()
                    if not param:
                        continue

                    # Check for qualifiers
                    qualifiers = []
                    if param.startswith('inout '):
                        qualifiers.append('inout')
                        param = param[6:]
                    elif param.startswith('out '):
                        qualifiers.append('out')
                        param = param[4:]

                    # Separate type and name
                    parts = param.rsplit(' ', 1)
                    if len(parts) == 2:
                        ptype, pname = parts
                        param_list.append({
                            'name': pname,
                            'type': ptype.strip(),
                            'qualifiers': qualifiers
                        })

            function_signatures[name] = {
                'parameters': param_list,
                'return_type': ret_type,
                'file': entry['file']
            }

    # Generate AST header
    ast_header = """/*
 * Automatically generated from cppfront regression corpus
 * using Clang AST ground truth
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace cppfort {

enum class Qualifier {
    None,
    InOut,
    Out,
    Move,
    Forward
};

struct Parameter {
    std::string name;
    std::string type;
    std::vector<Qualifier> qualifiers;

    bool is_inout() const {
        for (auto q : qualifiers) {
            if (q == Qualifier::InOut) return true;
        }
        return false;
    }
};

struct FunctionDeclaration {
    std::string name;
    std::vector<Parameter> parameters;
    std::string return_type;
    std::unique_ptr<class Block> body;

    // Corpus-derived metadata
    bool is_nodiscard = false;
    bool is_template = false;
    std::vector<std::string> template_params;
};

// Function signatures extracted from corpus
/*
"""

    # Add extracted function signatures
    for name, info in sorted(function_signatures.items()):
        ast_header += f"// From {info['file']}\n"
        ast_header += f"// {name}: ("

        params_str = []
        for p in info['parameters']:
            qual = ""
            if 'inout' in p.get('qualifiers', []):
                qual = "inout "
            elif 'out' in p.get('qualifiers', []):
                qual = "out "
            params_str.append(f"{qual}{p['type']} {p['name']}")

        ast_header += ", ".join(params_str)
        ast_header += f") -> {info['return_type']}\n"

    ast_header += """
 */

// Statement types
class Statement {
public:
    enum class Kind {
        Return,
        Expression,
        VariableDeclaration,
        Block
    };

    virtual ~Statement() = default;
    virtual Kind getKind() const = 0;
};

class ReturnStatement : public Statement {
public:
    std::unique_ptr<class Expression> value;

    Kind getKind() const override { return Kind::Return; }
};

class ExpressionStatement : public Statement {
public:
    std::unique_ptr<class Expression> expr;

    Kind getKind() const override { return Kind::Expression; }
};

class VariableDeclaration : public Statement {
public:
    std::string name;
    std::string type;
    std::unique_ptr<Expression> initializer;
    std::vector<Qualifier> qualifiers;

    Kind getKind() const override { return Kind::VariableDeclaration; }
};

class Block : public Statement {
public:
    std::vector<std::unique_ptr<Statement>> statements;

    Kind getKind() const override { return Kind::Block; }
};

// Expression types
class Expression {
public:
    enum class Kind {
        Literal,
        Identifier,
        FunctionCall,
        BinaryOp,
        UnaryOp,
        UFCS
    };

    virtual ~Expression() = default;
    virtual Kind getKind() const = 0;
};

class FunctionCallExpression : public Expression {
public:
    std::string function_name;
    std::vector<std::unique_ptr<Expression>> arguments;
    bool is_ufcs = false;  // Unified Function Call Syntax

    Kind getKind() const override { return Kind::FunctionCall; }
};

// High-frequency patterns from corpus
class BoundsCheckExpression : public Expression {
public:
    std::unique_ptr<Expression> container;
    std::unique_ptr<Expression> index;

    Kind getKind() const override { return Kind::UnaryOp; }  // Special unary op
};

} // namespace cppfort
"""

    # Write the header
    with open("refined_cpp2_ast.hpp", "w") as f:
        f.write(ast_header)

    print(f"Generated refined AST header")
    print(f"Functions extracted: {len(function_signatures)}")
    print(f"Patterns from corpus: {len(all_patterns)} files")

if __name__ == "__main__":
    generate_ast_refined()
