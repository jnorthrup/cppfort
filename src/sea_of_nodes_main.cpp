#include "mlir_cpp2_dialect.hpp"
#include "sea_of_nodes_ir.cpp"
#include <iostream>
#include <fstream>
#include <print>
#include "lexer.hpp"

// Replace traditional transpiler with Sea of Nodes implementation
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::println("Usage: {} <input.cpp2> <output.cpp>", argv[0]);
        return 1;
    }

    try {
        std::string input_filename = argv[1];
        std::string output_filename = argv[2];

        std::ifstream input_file(input_filename);
        if (!input_file) {
            throw std::runtime_error("Cannot open input file: " + input_filename);
        }

        std::string source_code(std::istreambuf_iterator<char>(input_file),
                                std::istreambuf_iterator<char>());

        // Create Sea of Nodes builder
        cppfort::mlir_son::SeaOfNodesBuilder son_builder;

        // Build Sea of Nodes graph from tokenized input (no regex/text processing)
        // Use the existing lexer to avoid ad-hoc character scanning
        cpp2_transpiler::Lexer lexer(source_code);
        auto tokens = lexer.tokenize();

        build_graph_from_tokens(son_builder, tokens);

        // Apply global code motion optimization (Chapter 11)
        son_builder.schedule_graph();

        // Generate C++ code from optimized Sea of Nodes
        std::string cpp1_code = generate_cpp_from_sea_of_nodes(son_builder);

        std::ofstream output_file(output_filename);
        if (!output_file) {
            throw std::runtime_error("Cannot open output file: " + output_filename);
        }

        output_file << cpp1_code;

        std::println("Successfully transpiled {} to {} using Sea of Nodes",
                     input_filename, output_filename);
        return 0;
    }
    catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }
}

// Build Sea of Nodes graph using combinator patterns from Simple docs
void build_graph_from_tokens(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                            const std::vector<cpp2_transpiler::Token>& tokens) {
    using namespace cppfort::mlir_son;
    using namespace cppfort::mlir_son::combinators;

    // Token index
    size_t idx = 0;
    auto token_count = tokens.size();

    auto current_token = [&](size_t i) -> const cpp2_transpiler::Token* {
        if (i >= token_count) return nullptr;
        return &tokens[i];
    };

    while (idx < token_count) {
        const auto* tk = current_token(idx);
        if (!tk) break;
        switch (tk->type) {
            case cpp2_transpiler::TokenType::Return:
                idx = parse_return_statement(builder, tokens, idx);
                break;
            case cpp2_transpiler::TokenType::IntegerLiteral:
            case cpp2_transpiler::TokenType::FloatLiteral:
                idx = parse_numeric_constant(builder, tokens, idx);
                break;
            case cpp2_transpiler::TokenType::Identifier: {
                // Check for 'new' lexeme (lexer doesn't include as keyword)
                if (tk->lexeme == "new") {
                    idx = parse_new_expression(builder, tokens, idx);
                } else {
                    idx = parse_identifier_expression(builder, tokens, idx);
                }
                break;
            }
            case cpp2_transpiler::TokenType::LeftBrace:
                idx = parse_block_statement(builder, tokens, idx);
                break;
            case cpp2_transpiler::TokenType::If:
                idx = parse_if_statement(builder, tokens, idx);
                break;
            case cpp2_transpiler::TokenType::Struct:
                idx = parse_struct_definition(builder, tokens, idx);
                break;
            default:
                idx++;
                break;
        }
    }
}

// Forward declarations for token-based parsing helpers
size_t parse_return_statement(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                             const std::vector<cpp2_transpiler::Token>& tokens,
                             size_t idx);
std::pair<NodeID, size_t> parse_expression(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                          const std::vector<cpp2_transpiler::Token>& tokens,
                                          size_t idx);
std::pair<NodeID, size_t> parse_primary(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                       const std::vector<cpp2_transpiler::Token>& tokens,
                                       size_t idx);
std::pair<NodeID, size_t> parse_numeric_constant(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                                const std::vector<cpp2_transpiler::Token>& tokens,
                                                size_t idx);
std::pair<NodeID, size_t> parse_identifier_expression(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                                     const std::vector<cpp2_transpiler::Token>& tokens,
                                                     size_t idx);
size_t parse_block_statement(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                           const std::vector<cpp2_transpiler::Token>& tokens,
                           size_t idx);
size_t parse_if_statement(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                        const std::vector<cpp2_transpiler::Token>& tokens,
                        size_t idx);
size_t parse_struct_definition(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                             const std::vector<cpp2_transpiler::Token>& tokens,
                             size_t idx);
size_t parse_new_expression(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                          const std::vector<cpp2_transpiler::Token>& tokens,
                          size_t idx);

// Parse return statement (Chapter 1 pattern)
size_t parse_return_statement(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                             const std::string& source, size_t pos) {
    // Skip "return"
    pos += 6;

    // Parse return value
    auto [value, new_pos] = parse_expression(builder, source, pos);

    // Create return node
    NodeID return_node = builder.create_node(cppfort::mlir_son::Node::Kind::Return);

    // Connect value to return
    if (value != 0) {
        const_cast<cppfort::mlir_son::Node*>(
            builder.get_graph().get_node(return_node)
        )->inputs = {value};
    }

    return new_pos;
}

// Token-based parsers
size_t parse_return_statement(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                             const std::vector<cpp2_transpiler::Token>& tokens,
                             size_t idx) {
    // tokens[idx] is 'return'
    idx++; // consume 'return'
    auto [value, new_idx] = parse_expression(builder, tokens, idx);

    NodeID return_node = builder.create_node(cppfort::mlir_son::Node::Kind::Return);
    if (value != 0) {
        const_cast<cppfort::mlir_son::Node*>(
            builder.get_graph().get_node(return_node)
        )->inputs = {value};
    }
    return new_idx;
}

std::pair<NodeID, size_t> parse_expression(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                          const std::vector<cpp2_transpiler::Token>& tokens,
                                          size_t idx) {
    auto [left, pos] = parse_primary(builder, tokens, idx);

    while (pos < tokens.size()) {
        auto t = tokens[pos];
        bool is_op = false;
        cppfort::mlir_son::Node::Kind kind = cppfort::mlir_son::Node::Kind::Add;
        switch (t.type) {
            case cpp2_transpiler::TokenType::Plus: kind = cppfort::mlir_son::Node::Kind::Add; is_op = true; break;
            case cpp2_transpiler::TokenType::Minus: kind = cppfort::mlir_son::Node::Kind::Sub; is_op = true; break;
            case cpp2_transpiler::TokenType::Asterisk: kind = cppfort::mlir_son::Node::Kind::Mul; is_op = true; break;
            case cpp2_transpiler::TokenType::Slash: kind = cppfort::mlir_son::Node::Kind::Div; is_op = true; break;
            default: break;
        }

        if (is_op) {
            pos++;
            auto [right, new_pos] = parse_primary(builder, tokens, pos);
            left = builder.create_binary_op(kind, left, right);
            pos = new_pos;
        } else break;
    }

    return {left, pos};
}

std::pair<NodeID, size_t> parse_primary(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                       const std::vector<cpp2_transpiler::Token>& tokens,
                                       size_t idx) {
    if (idx >= tokens.size()) return {0, idx};
    auto t = tokens[idx];
    switch (t.type) {
        case cpp2_transpiler::TokenType::IntegerLiteral:
        case cpp2_transpiler::TokenType::FloatLiteral:
            return parse_numeric_constant(builder, tokens, idx);
        case cpp2_transpiler::TokenType::LeftParen: {
            idx++;
            auto [expr, new_idx] = parse_expression(builder, tokens, idx);
            if (new_idx < tokens.size() && tokens[new_idx].type == cpp2_transpiler::TokenType::RightParen) {
                new_idx++;
            }
            return {expr, new_idx};
        }
        case cpp2_transpiler::TokenType::Identifier:
            return parse_identifier_expression(builder, tokens, idx);
        default:
            return {0, idx};
    }
}

std::pair<NodeID, size_t> parse_numeric_constant(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                                const std::vector<cpp2_transpiler::Token>& tokens,
                                                size_t idx) {
    const auto& tk = tokens[idx];
    if (tk.type == cpp2_transpiler::TokenType::IntegerLiteral) {
        int64_t v = 0;
        try { v = std::stoll(std::string(tk.lexeme)); } catch (...) {}
        NodeID id = builder.create_constant(v);
        return {id, idx + 1};
    }
    if (tk.type == cpp2_transpiler::TokenType::FloatLiteral) {
        double v = 0.0;
        try { v = std::stod(std::string(tk.lexeme)); } catch (...) {}
        NodeID id = builder.create_constant(v);
        return {id, idx + 1};
    }
    return {0, idx};
}

std::pair<NodeID, size_t> parse_identifier_expression(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                                     const std::vector<cpp2_transpiler::Token>& tokens,
                                                     size_t idx) {
    const auto& tk = tokens[idx];
    std::string name(tk.lexeme);
    // Create a variable node (placeholder type int)
    NodeID var = builder.create_variable(name, cppfort::mlir_son::Type::make_int(), false);
    return {var, idx + 1};
}

size_t parse_block_statement(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                           const std::vector<cpp2_transpiler::Token>& tokens,
                           size_t idx) {
    // assume tokens[idx] == LeftBrace
    idx++;
    int depth = 1;
    while (idx < tokens.size() && depth > 0) {
        if (tokens[idx].type == cpp2_transpiler::TokenType::LeftBrace) depth++;
        else if (tokens[idx].type == cpp2_transpiler::TokenType::RightBrace) depth--;
        else {
            // Delegate to top-level dispatcher
            switch (tokens[idx].type) {
                case cpp2_transpiler::TokenType::Return: idx = parse_return_statement(builder, tokens, idx); break;
                default: idx++; break;
            }
        }
    }
    return idx;
}

size_t parse_if_statement(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                        const std::vector<cpp2_transpiler::Token>& tokens,
                        size_t idx) {
    // consume 'if'
    idx++;
    // very simple: parse next primary as condition
    auto [cond, pos] = parse_primary(builder, tokens, idx);
    // create if region
    auto ifregion = builder.create_if(cond);
    return pos;
}

size_t parse_struct_definition(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                             const std::vector<cpp2_transpiler::Token>& tokens,
                             size_t idx) {
    // consume 'struct'
    idx++;
    // expect identifier
    if (idx < tokens.size() && tokens[idx].type == cpp2_transpiler::TokenType::Identifier) {
        std::string name(tokens[idx].lexeme);
        NodeID type_node = builder.create_struct_type(name);
        idx++;
    }
    return idx;
}

size_t parse_new_expression(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                          const std::vector<cpp2_transpiler::Token>& tokens,
                          size_t idx) {
    // consume 'new'
    idx++;
    if (idx < tokens.size() && tokens[idx].type == cpp2_transpiler::TokenType::Identifier) {
        std::string name(tokens[idx].lexeme);
        NodeID struct_type = builder.create_struct_type(name);
        NodeID new_node = builder.create_new_struct(struct_type);
        return idx + 1;
    }
    return idx;
}

// Parse binary arithmetic expression (Chapter 2 pattern)
std::pair<NodeID, size_t> parse_expression(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                          const std::string& source, size_t pos) {
    // Simple precedence climbing parser using combinator patterns

    // Parse primary expression
    auto [left, current_pos] = parse_primary(builder, source, pos);

    // Parse binary operators with precedence
    while (current_pos < source.size()) {
        // Skip whitespace
        while (current_pos < source.size() && isspace(source[current_pos])) {
            current_pos++;
        }

        if (current_pos >= source.size()) break;

        char op = source[current_pos];
        if (op == '+' || op == '-' || op == '*' || op == '/') {
            current_pos++;

            // Parse right operand
            auto [right, new_pos] = parse_primary(builder, source, current_pos);

            // Create binary operation node
            cppfort::mlir_son::Node::Kind kind;
            switch (op) {
                case '+': kind = cppfort::mlir_son::Node::Kind::Add; break;
                case '-': kind = cppfort::mlir_son::Node::Kind::Sub; break;
                case '*': kind = cppfort::mlir_son::Node::Kind::Mul; break;
                case '/': kind = cppfort::mlir_son::Node::Kind::Div; break;
                default: return {left, current_pos};
            }

            left = builder.create_binary_op(kind, left, right);
            current_pos = new_pos;
        } else {
            break;
        }
    }

    return {left, current_pos};
}

// Parse primary expression (constant or identifier)
std::pair<NodeID, size_t> parse_primary(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                       const std::string& source, size_t pos) {
    // Skip whitespace
    while (pos < source.size() && isspace(source[pos])) {
        pos++;
    }

    if (pos >= source.size()) return {0, pos};

    // Numeric literal
    if (isdigit(source[pos]) || source[pos] == '-') {
        return parse_numeric_constant(builder, source, pos);
    }

    // Parenthesized expression
    if (source[pos] == '(') {
        pos++; // Skip '('
        auto [expr, new_pos] = parse_expression(builder, source, pos);
        if (new_pos < source.size() && source[new_pos] == ')') {
            new_pos++; // Skip ')'
        }
        return {expr, new_pos};
    }

    // Identifier or function call
    if (isalpha(source[pos]) || source[pos] == '_') {
        return parse_identifier_expression(builder, source, pos);
    }

    return {0, pos};
}

// Parse numeric constant (Chapter 2 pattern)
std::pair<NodeID, size_t> parse_numeric_constant(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                                const std::string& source, size_t pos) {
    bool negative = false;
    if (pos < source.size() && source[pos] == '-') {
        negative = true;
        pos++;
    }

    size_t start = pos;
    while (pos < source.size() && isdigit(source[pos])) {
        pos++;
    }

    // Handle floating point
    bool is_float = false;
    if (pos < source.size() && source[pos] == '.') {
        is_float = true;
        pos++;
        while (pos < source.size() && isdigit(source[pos])) {
            pos++;
        }
    }

    if (start == pos) {
        return {0, pos}; // No digits found
    }

    std::string num_str = source.substr(start, pos - start);

    NodeID const_node;
    if (is_float) {
        double value = std::stod(num_str);
        if (negative) value = -value;
        const_node = builder.create_constant(value);
    } else {
        int64_t value = std::stoll(num_str);
        if (negative) value = -value;
        const_node = builder.create_constant(value);
    }

    return {const_node, pos};
}

// Parse identifier expression (variable, UFCS call)
std::pair<NodeID, size_t> parse_identifier_expression(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                                     const std::string& source, size_t pos) {
    // Parse identifier name
    size_t start = pos;
    while (pos < source.size() && (isalnum(source[pos]) || source[pos] == '_')) {
        pos++;
    }
    std::string name = source.substr(start, pos - start);

    // Check for function call
    pos = skip_whitespace(source, pos);
    if (pos < source.size() && source[pos] == '(') {
        return parse_function_call(builder, source, pos, name);
    }

    // Check for member access (UFCS)
    if (pos < source.size() && source[pos] == '.') {
        return parse_ufcs_call(builder, source, pos, name);
    }

    // Variable reference - would need symbol table lookup
    return {0, pos};
}

// Parse struct definition (Chapter 10 pattern)
size_t parse_struct_definition(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                              const std::string& source, size_t pos) {
    // Skip "struct"
    pos += 6;
    pos = skip_whitespace(source, pos);

    // Parse struct name
    size_t name_start = pos;
    while (pos < source.size() && (isalnum(source[pos]) || source[pos] == '_')) {
        pos++;
    }
    std::string struct_name = source.substr(name_start, pos - name_start);

    // Create struct type node
    NodeID struct_node = builder.create_struct_type(struct_name);

    // Skip to opening brace
    pos = skip_whitespace(source, pos);
    if (pos < source.size() && source[pos] == '{') {
        pos++;
    }

    // Parse field declarations
    while (pos < source.size() && source[pos] != '}') {
        pos = skip_whitespace_and_comments(source, pos);
        if (pos >= source.size() || source[pos] == '}') break;

        // Parse field: type name
        pos = parse_field_declaration(builder, source, pos, struct_name);
    }

    if (pos < source.size() && source[pos] == '}') {
        pos++;
    }

    return pos;
}

// Parse new expression (Chapter 10 pattern)
std::pair<NodeID, size_t> parse_new_expression(cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                              const std::string& source, size_t pos) {
    // Skip "new"
    pos += 3;
    pos = skip_whitespace(source, pos);

    // Parse type name
    size_t type_start = pos;
    while (pos < source.size() && (isalnum(source[pos]) || source[pos] == '_')) {
        pos++;
    }
    std::string type_name = source.substr(type_start, pos - type_start);

    // Get or create struct type
    NodeID struct_type = builder.create_struct_type(type_name);

    // Create new node
    NodeID new_node = builder.create_new_struct(struct_type);

    return {new_node, pos};
}

// Generate C++ code from Sea of Nodes
std::string generate_cpp_from_sea_of_nodes(const cppfort::mlir_son::SeaOfNodesBuilder& builder) {
    // Walk the optimized Sea of Nodes and generate C++ code
    // This would use the scheduled graph from GCM algorithm

    std::ostringstream output;

    // Header
    output << "#include <memory>\n";
    output << "#include <vector>\n\n";

    // Generate code by walking scheduled blocks
    generate_from_scheduled_blocks(builder, output);

    return output.str();
}

void generate_from_scheduled_blocks(const cppfort::mlir_son::SeaOfNodesBuilder& builder,
                                   std::ostringstream& output) {
    // Generate C++ code from scheduled blocks
    // Each block corresponds to a basic block in the control flow

    const auto& graph = builder.get_graph();

    // Start from entry and follow control edges
    std::unordered_set<NodeID> visited;

    // Find Start node
    NodeID current = 0;
    for (const auto& [id, node] : graph.get_nodes()) {
        if (node.kind == cppfort::mlir_son::Node::Kind::Start) {
            current = id;
            break;
        }
    }

    // Generate code for each scheduled block
    generate_block_code(builder, current, visited, output);
}

void generate_block_code(const cppfort::mlir_son::SeaOfNodesBuilder& builder,
                        NodeID block_id, std::unordered_set<NodeID>& visited,
                        std::ostringstream& output) {
    if (visited.contains(block_id)) return;
    visited.insert(block_id);

    const auto& graph = builder.get_graph();
    const Node* block = graph.get_node(block_id);
    if (!block) return;

    // Generate code for all data nodes in this block
    for (const auto& [id, node] : graph.get_nodes()) {
        // Check if node is scheduled in this block
        if (is_node_in_block(node, block_id) && is_data_node(node)) {
            generate_node_code(builder, id, output);
        }
    }

    // Follow control edges to next blocks
    const auto* outputs = graph.get_outputs(block_id);
    if (outputs) {
        for (NodeID output : *outputs) {
            const Node* out_node = graph.get_node(output);
            if (out_node && is_control_node(*out_node)) {
                output << "\n";
                generate_block_code(builder, output, visited, output);
            }
        }
    }
}

void generate_node_code(const cppfort::mlir_son::SeaOfNodesBuilder& builder,
                       NodeID node_id, std::ostringstream& output) {
    const auto& graph = builder.get_graph();
    const Node* node = graph.get_node(node_id);
    if (!node) return;

    switch (node->kind) {
        case cppfort::mlir_son::Node::Kind::Constant:
            if (std::holds_alternative<int64_t>(node->value)) {
                output << std::get<int64_t>(node->value);
            } else if (std::holds_alternative<double>(node->value)) {
                output << std::get<double>(node->value);
            } else if (std::holds_alternative<bool>(node->value)) {
                output << (std::get<bool>(node->value) ? "true" : "false");
            }
            break;

        case cppfort::mlir_son::Node::Kind::Add:
            output << "(";
            generate_node_code(builder, node->inputs[0], output);
            output << " + ";
            generate_node_code(builder, node->inputs[1], output);
            output << ")";
            break;

        case cppfort::mlir_son::Node::Kind::Sub:
            output << "(";
            generate_node_code(builder, node->inputs[0], output);
            output << " - ";
            generate_node_code(builder, node->inputs[1], output);
            output << ")";
            break;

        case cppfort::mlir_son::Node::Kind::Mul:
            output << "(";
            generate_node_code(builder, node->inputs[0], output);
            output << " * ";
            generate_node_code(builder, node->inputs[1], output);
            output << ")";
            break;

        case cppfort::mlir_son::Node::Kind::Div:
            output << "(";
            generate_node_code(builder, node->inputs[0], output);
            output << " / ";
            generate_node_code(builder, node->inputs[1], output);
            output << ")";
            break;

        case cppfort::mlir_son::Node::Kind::New:
            output << "std::make_unique<";
            // Generate struct name from type node
            output << "StructType";
            output << ">()";
            break;

        case cppfort::mlir_son::Node::Kind::Return:
            output << "return ";
            if (!node->inputs.empty()) {
                generate_node_code(builder, node->inputs[0], output);
            }
            output << ";\n";
            break;

        default:
            // Handle other node types
            break;
    }
}

// Helper functions
size_t skip_whitespace(const std::string& source, size_t pos) {
    while (pos < source.size() && isspace(source[pos])) {
        pos++;
    }
    return pos;
}

size_t skip_whitespace_and_comments(const std::string& source, size_t pos) {
    while (pos < source.size()) {
        // Skip whitespace
        while (pos < source.size() && isspace(source[pos])) {
            pos++;
        }

        // Skip comments
        if (pos < source.size() && source[pos] == '/' && pos + 1 < source.size()) {
            if (source[pos + 1] == '/') {
                // Single line comment
                pos += 2;
                while (pos < source.size() && source[pos] != '\n') {
                    pos++;
                }
            } else if (source[pos + 1] == '*') {
                // Multi-line comment
                pos += 2;
                while (pos + 1 < source.size() &&
                       !(source[pos] == '*' && source[pos + 1] == '/')) {
                    pos++;
                }
                pos += 2; // Skip "*/"
            }
        } else {
            break;
        }
    }
    return pos;
}

bool is_data_node(const cppfort::mlir_son::Node& node) {
    return node.kind != cppfort::mlir_son::Node::Kind::Start &&
           node.kind != cppfort::mlir_son::Node::Kind::Stop &&
           node.kind != cppfort::mlir_son::Node::Kind::If &&
           node.kind != cppfort::mlir_son::Node::Kind::Region &&
           node.kind != cppfort::mlir_son::Node::Kind::Loop &&
           node.kind != cppfort::mlir_son::Node::Kind::Return;
}

bool is_control_node(const cppfort::mlir_son::Node& node) {
    return !is_data_node(node);
}

bool is_node_in_block(const cppfort::mlir_son::Node& node, NodeID block_id) {
    // Check if node is scheduled in the given block
    // This would use scheduling information from GCM
    return true; // Placeholder
}