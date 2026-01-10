#include <iostream>
#include <fstream>
#include <sstream>
#include <span>
#include <string_view>
#include <cstring>

#include "lexer.hpp"
#include "combinator_parser.hpp"
#include "slim_ast.hpp"

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <input.cpp2> <output.cpp>\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --inline    Inline runtime (default, self-contained output)\n";
    std::cerr << "  --header    Use #include <cpp2_runtime.h>\n";
    std::cerr << "  --pch       Use #include <cpp2_pch.h> (precompiled header)\n";
    std::cerr << "  --help      Show this help\n";
}

// Forward declaration - direct ParseTree to C++ code generation
std::string generate_from_tree(const cpp2::ast::ParseTree& tree, 
                                std::span<const cpp2_transpiler::Token> tokens);

int main(int argc, char* argv[]) {
    const char* input_file_arg = nullptr;
    const char* output_file_arg = nullptr;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--inline") == 0 ||
            strcmp(argv[i], "--header") == 0 ||
            strcmp(argv[i], "--pch") == 0) {
            // Options currently ignored - future: pass to emitter
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            if (!input_file_arg) {
                input_file_arg = argv[i];
            } else if (!output_file_arg) {
                output_file_arg = argv[i];
            } else {
                std::cerr << "Error: Too many arguments\n";
                print_usage(argv[0]);
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!input_file_arg || !output_file_arg) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::string input_filename = input_file_arg;
        std::string output_filename = output_file_arg;

        std::ifstream input_file(input_filename);
        if (!input_file) {
            throw std::runtime_error("Cannot open input file: " + input_filename);
        }

        std::string source_code{std::istreambuf_iterator<char>(input_file),
                                std::istreambuf_iterator<char>()};

        cpp2_transpiler::Lexer lexer{std::string_view(source_code)};
        auto tokens = lexer.tokenize();

        // Mixed-mode C++1 passthrough: If no Cpp2-specific syntax is found,
        // emit the original source with minimal changes
        if (!lexer.has_cpp2_syntax()) {
            std::ofstream output_file(output_filename);
            if (!output_file) {
                throw std::runtime_error("Cannot open output file: " + output_filename);
            }

            // For pure C++ files, emit as-is
            output_file << source_code;

            std::cout << "Successfully transpiled " << input_filename << " to " << output_filename << " (C++1 passthrough mode)\n";
            return 0;
        }

        // Parse with slim combinator parser
        auto tree = cpp2::parser::parse(tokens);

        // Check for valid parse
        if (tree.nodes.empty() || tree.nodes[tree.root].child_count == 0) {
            std::cerr << "Error: Parsing failed - no declarations found\n";
            return 1;
        }

        // Generate C++ directly from ParseTree
        std::string cpp1_code = generate_from_tree(tree, tokens);

        std::ofstream output_file(output_filename);
        if (!output_file) {
            throw std::runtime_error("Cannot open output file: " + output_filename);
        }

        output_file << cpp1_code;

        std::cout << "Successfully transpiled " << input_filename << " to " << output_filename << "\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// ============================================================================
// Direct ParseTree to C++ Emitter
// ============================================================================
// No legacy AST - direct tree traversal to emit C++

namespace {

using namespace cpp2::ast;

class TreeEmitter {
    const ParseTree& tree_;
    std::span<const cpp2_transpiler::Token> tokens_;
    std::ostringstream out_;
    int indent_ = 0;

    void emit_indent() {
        for (int i = 0; i < indent_; ++i) out_ << "    ";
    }

    std::string_view token_text(uint32_t idx) const {
        if (idx < tokens_.size()) return tokens_[idx].lexeme;
        return "";
    }

    std::string node_text(const Node& n) const {
        std::string result;
        for (uint32_t i = n.token_start; i < n.token_end && i < tokens_.size(); ++i) {
            if (!result.empty() && !tokens_[i].lexeme.empty()) {
                char prev = result.back();
                char next = tokens_[i].lexeme[0];
                if (std::isalnum(prev) && std::isalnum(next)) result += ' ';
            }
            result += tokens_[i].lexeme;
        }
        return result;
    }

public:
    TreeEmitter(const ParseTree& tree, std::span<const cpp2_transpiler::Token> tokens)
        : tree_(tree), tokens_(tokens) {}

    std::string emit() {
        out_ << "// Generated by cppfort - slim ParseTree emitter\n";
        out_ << "#include <iostream>\n";
        out_ << "#include <string>\n\n";

        if (tree_.nodes.empty()) return out_.str();

        const auto& root = tree_[tree_.root];
        for (const auto& child : tree_.children(root)) {
            emit_declaration(child);
        }

        return out_.str();
    }

private:
    void emit_declaration(const Node& n) {
        if (n.kind != NodeKind::Declaration) return;

        for (const auto& child : tree_.children(n)) {
            if (child.kind == NodeKind::UnifiedDeclaration) {
                emit_unified_decl(child);
            }
        }
    }

    void emit_unified_decl(const Node& n) {
        // Extract name from first token
        std::string name = std::string(token_text(n.token_start));

        // Check for function suffix vs variable suffix
        for (const auto& child : tree_.children(n)) {
            if (child.kind == NodeKind::FunctionSuffix) {
                emit_function(name, child);
                return;
            }
            if (child.kind == NodeKind::VariableSuffix) {
                emit_variable(name, child);
                return;
            }
            if (child.kind == NodeKind::TypeSuffix) {
                emit_type(name, child);
                return;
            }
            if (child.kind == NodeKind::NamespaceSuffix) {
                emit_namespace(name, child);
                return;
            }
        }

        // Fallback: simple variable with := initialization
        emit_indent();
        out_ << "auto " << name << " = /* TODO: emit init */;\n";
    }

    void emit_function(const std::string& name, const Node& suffix) {
        // Extract return type (if any) and parameters
        std::string return_type = "void";
        std::string params;
        const Node* body = nullptr;

        for (const auto& child : tree_.children(suffix)) {
            if (child.kind == NodeKind::ParamList) {
                params = emit_params(child);
            } else if (child.kind == NodeKind::ReturnSpec) {
                return_type = emit_type_spec(child);
            } else if (child.kind == NodeKind::FunctionBody) {
                body = &child;
            }
        }

        // Handle main function specially
        if (name == "main") {
            out_ << "int main(int argc, char* argv[]) {\n";
        } else {
            out_ << "auto " << name << "(" << params << ") -> " << return_type << " {\n";
        }

        ++indent_;
        if (body) emit_function_body(*body);
        --indent_;

        out_ << "}\n\n";
    }

    std::string emit_params(const Node& n) {
        std::string result;
        for (const auto& param : tree_.children(n)) {
            if (param.kind == NodeKind::Parameter) {
                if (!result.empty()) result += ", ";
                result += emit_param(param);
            }
        }
        return result;
    }

    std::string emit_param(const Node& n) {
        // Get param name and optional type
        std::string name = std::string(token_text(n.token_start));
        std::string type = "auto";

        // Check for type specifier child
        for (const auto& child : tree_.children(n)) {
            if (child.kind == NodeKind::TypeSpecifier || child.kind == NodeKind::BasicType) {
                type = node_text(child);
            }
        }

        return type + " " + name;
    }

    std::string emit_type_spec(const Node& n) {
        // Return spec has arrow -> type
        for (const auto& child : tree_.children(n)) {
            if (child.kind == NodeKind::TypeSpecifier || child.kind == NodeKind::BasicType) {
                return node_text(child);
            }
        }
        // Fallback: get text after ->
        std::string text = node_text(n);
        auto pos = text.find("->");
        if (pos != std::string::npos) {
            std::string result = text.substr(pos + 2);
            // Trim whitespace
            while (!result.empty() && std::isspace(result.front())) result.erase(0, 1);
            return result.empty() ? "auto" : result;
        }
        return "auto";
    }

    void emit_function_body(const Node& n) {
        for (const auto& child : tree_.children(n)) {
            if (child.kind == NodeKind::BlockStatement) {
                emit_block(child);
            } else {
                // Expression body: = expr;
                emit_indent();
                out_ << "return " << node_text(child) << ";\n";
            }
        }
    }

    void emit_block(const Node& n) {
        for (const auto& child : tree_.children(n)) {
            emit_statement(child);
        }
    }

    void emit_statement(const Node& n) {
        emit_indent();

        if (n.kind == NodeKind::ReturnStatement) {
            out_ << "return";
            for (const auto& child : tree_.children(n)) {
                out_ << " " << node_text(child);
            }
            out_ << ";\n";
        } else if (n.kind == NodeKind::IfStatement) {
            emit_if(n);
        } else if (n.kind == NodeKind::WhileStatement) {
            emit_while(n);
        } else if (n.kind == NodeKind::ForStatement) {
            emit_for(n);
        } else if (n.kind == NodeKind::BlockStatement) {
            out_ << "{\n";
            ++indent_;
            emit_block(n);
            --indent_;
            emit_indent();
            out_ << "}\n";
        } else if (n.kind == NodeKind::ExpressionStatement) {
            out_ << node_text(n) << "\n";
        } else if (n.kind == NodeKind::Statement) {
            // Generic statement - check children
            for (const auto& child : tree_.children(n)) {
                emit_statement(child);
            }
        } else if (n.kind == NodeKind::UnifiedDeclaration) {
            // Local variable
            emit_local_var(n);
        } else {
            out_ << node_text(n) << ";\n";
        }
    }

    void emit_if(const Node& n) {
        out_ << "if (";
        bool first = true;
        for (const auto& child : tree_.children(n)) {
            if (first && child.kind != NodeKind::BlockStatement && child.kind != NodeKind::Statement) {
                out_ << node_text(child);
                first = false;
            } else if (child.kind == NodeKind::BlockStatement) {
                out_ << ") {\n";
                ++indent_;
                emit_block(child);
                --indent_;
                emit_indent();
                out_ << "}";
            }
        }
        out_ << "\n";
    }

    void emit_while(const Node& n) {
        out_ << "while (";
        bool first = true;
        for (const auto& child : tree_.children(n)) {
            if (first && child.kind != NodeKind::BlockStatement) {
                out_ << node_text(child);
                first = false;
            } else if (child.kind == NodeKind::BlockStatement) {
                out_ << ") {\n";
                ++indent_;
                emit_block(child);
                --indent_;
                emit_indent();
                out_ << "}\n";
            }
        }
    }

    void emit_for(const Node& n) {
        // Cpp2 for: for items do (item) { body }
        // Emit as: for (auto item : items) { body }
        std::string items, var;
        const Node* body = nullptr;

        for (const auto& child : tree_.children(n)) {
            if (child.kind == NodeKind::BlockStatement) {
                body = &child;
            } else if (child.kind == NodeKind::Identifier || 
                       (child.kind == NodeKind::Parameter && var.empty())) {
                var = node_text(child);
            } else if (items.empty() && child.kind != NodeKind::BlockStatement) {
                items = node_text(child);
            }
        }

        out_ << "for (auto " << var << " : " << items << ") {\n";
        ++indent_;
        if (body) emit_block(*body);
        --indent_;
        emit_indent();
        out_ << "}\n";
    }

    void emit_local_var(const Node& n) {
        std::string name = std::string(token_text(n.token_start));
        std::string init;

        for (const auto& child : tree_.children(n)) {
            if (child.kind == NodeKind::VariableSuffix) {
                init = node_text(child);
            }
        }

        out_ << "auto " << name << " " << init << ";\n";
    }

    void emit_variable(const std::string& name, const Node& suffix) {
        emit_indent();
        out_ << "auto " << name << " " << node_text(suffix) << ";\n";
    }

    void emit_type(const std::string& name, const Node& /*suffix*/) {
        out_ << "struct " << name << " {\n";
        out_ << "    // TODO: type body\n";
        out_ << "};\n\n";
    }

    void emit_namespace(const std::string& name, const Node& suffix) {
        out_ << "namespace " << name << " {\n\n";
        for (const auto& child : tree_.children(suffix)) {
            if (child.kind == NodeKind::NamespaceBody) {
                for (const auto& decl : tree_.children(child)) {
                    emit_declaration(decl);
                }
            }
        }
        out_ << "} // namespace " << name << "\n\n";
    }
};

} // anonymous namespace

std::string generate_from_tree(const cpp2::ast::ParseTree& tree,
                                std::span<const cpp2_transpiler::Token> tokens) {
    TreeEmitter emitter(tree, tokens);
    return emitter.emit();
}