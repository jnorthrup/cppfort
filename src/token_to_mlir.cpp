#include "mlir_semantic_graph.hpp"
#include "lexer.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include <vector>
#include <span>

namespace cppfort::mlir_son {

// Direct token stream to MLIR semantic graph (no text parsing, no AST)
class TokenToMLIRBuilder {
private:
    mlir::MLIRContext& ctx;
    MLIRSemanticGraph graph;
    std::vector<cpp2_transpiler::Token> tokens;
    size_t pos = 0;

    // Symbol table maps names to OpIDs (semantic references, not strings)
    std::unordered_map<std::string, OpID> symbols;

public:
    explicit TokenToMLIRBuilder(mlir::MLIRContext& context)
        : ctx(context), graph(&context) {}

    MLIRSemanticGraph build(std::span<const cpp2_transpiler::Token> token_stream) {
        tokens = std::vector(token_stream.begin(), token_stream.end());
        pos = 0;

        while (pos < tokens.size()) {
            parse_top_level();
        }

        return std::move(graph);
    }

private:
    const cpp2_transpiler::Token& peek(size_t offset = 0) const {
        size_t index = pos + offset;
        if (index >= tokens.size()) {
            static cpp2_transpiler::Token eof{cpp2_transpiler::TokenType::EndOfFile, "", 0};
            return eof;
        }
        return tokens[index];
    }

    void advance() {
        if (pos < tokens.size()) pos++;
    }

    bool match(cpp2_transpiler::TokenType type) {
        if (peek().type == type) {
            advance();
            return true;
        }
        return false;
    }

    void parse_top_level() {
        using TT = cpp2_transpiler::TokenType;

        switch (peek().type) {
            case TT::Identifier:
                // Could be function, variable, or class
                parse_declaration();
                break;

            case TT::Return:
                parse_return();
                break;

            default:
                advance(); // Skip unknown tokens
                break;
        }
    }

    void parse_declaration() {
        using TT = cpp2_transpiler::TokenType;

        std::string name(peek().lexeme);
        advance();

        if (!match(TT::Colon)) {
            return; // Not a Cpp2 declaration
        }

        // Peek ahead to determine declaration type
        if (peek().type == TT::LeftParen) {
            // Function: name: (params) -> type = { }
            parse_function(name);
        } else if (peek().lexeme == "type") {
            // Class: name: type = { }
            parse_class(name);
        } else if (peek().lexeme == "namespace") {
            // Namespace: name: namespace = { }
            parse_namespace(name);
        } else {
            // Variable: name: type = value
            parse_variable(name);
        }
    }

    void parse_function(const std::string& name) {
        using TT = cpp2_transpiler::TokenType;

        // Parse parameters
        std::vector<OpID> params;
        if (match(TT::LeftParen)) {
            while (!match(TT::RightParen)) {
                if (peek().type == TT::Identifier) {
                    std::string param_name(peek().lexeme);
                    advance();

                    // Create parameter operation
                    auto param_type = mlir::IntegerType::get(&ctx, 64);
                    auto param_attr = mlir::IntegerAttr::get(param_type, 0);
                    OpID param_op = graph.create_constant(param_attr, param_type);
                    params.push_back(param_op);
                    symbols[param_name] = param_op;

                    match(TT::Colon);
                    skip_type();
                }

                if (!match(TT::Comma)) break;
            }
        }

        // Skip return type arrow and type
        if (peek().lexeme == "->") {
            advance();
            skip_type();
        }

        // Parse body
        if (match(TT::Equal)) {
            if (match(TT::LeftBrace)) {
                parse_block();
                match(TT::RightBrace);
            }
        }
    }

    void parse_class(const std::string& name) {
        using TT = cpp2_transpiler::TokenType;

        advance(); // Skip "type"

        if (match(TT::Equal)) {
            if (match(TT::LeftBrace)) {
                // Parse class members
                while (!match(TT::RightBrace)) {
                    if (peek().type == TT::Identifier) {
                        parse_declaration();
                    } else {
                        advance();
                    }
                }
            }
        }
    }

    void parse_namespace(const std::string& name) {
        using TT = cpp2_transpiler::TokenType;

        advance(); // Skip "namespace"

        if (match(TT::Equal)) {
            if (match(TT::LeftBrace)) {
                while (!match(TT::RightBrace)) {
                    parse_top_level();
                }
            }
        }
    }

    void parse_variable(const std::string& name) {
        using TT = cpp2_transpiler::TokenType;

        // Skip type
        skip_type();

        if (match(TT::Equal)) {
            // Parse initializer expression
            OpID init_value = parse_expression();
            if (init_value != 0) {
                symbols[name] = init_value;
            }
        }

        match(TT::Semicolon);
    }

    void parse_block() {
        using TT = cpp2_transpiler::TokenType;

        while (pos < tokens.size() && peek().type != TT::RightBrace) {
            switch (peek().type) {
                case TT::Return:
                    parse_return();
                    break;

                case TT::If:
                    parse_if();
                    break;

                case TT::Identifier:
                    parse_declaration();
                    break;

                default:
                    advance();
                    break;
            }
        }
    }

    void parse_return() {
        using TT = cpp2_transpiler::TokenType;

        advance(); // Skip "return"

        std::vector<OpID> operands;
        OpID value = parse_expression();
        if (value != 0) {
            operands.push_back(value);
        }

        graph.create_return(operands);

        match(TT::Semicolon);
    }

    void parse_if() {
        using TT = cpp2_transpiler::TokenType;

        advance(); // Skip "if"

        OpID condition = parse_expression();
        if (condition != 0) {
            OpID if_op = graph.create_if(condition);

            // Parse then branch
            if (match(TT::LeftBrace)) {
                parse_block();
                match(TT::RightBrace);
            }

            // Parse else branch
            if (peek().lexeme == "else") {
                advance();
                if (match(TT::LeftBrace)) {
                    parse_block();
                    match(TT::RightBrace);
                }
            }
        }
    }

    OpID parse_expression() {
        OpID left = parse_primary();

        while (is_binary_op(peek().type)) {
            auto op_token = peek();
            advance();

            OpID right = parse_primary();
            if (right == 0) break;

            auto result_type = mlir::IntegerType::get(&ctx, 64);

            switch (op_token.type) {
                case cpp2_transpiler::TokenType::Plus:
                    left = graph.create_add(left, right, result_type);
                    break;
                case cpp2_transpiler::TokenType::Minus:
                    left = graph.create_sub(left, right, result_type);
                    break;
                case cpp2_transpiler::TokenType::Star:
                    left = graph.create_mul(left, right, result_type);
                    break;
                case cpp2_transpiler::TokenType::Slash:
                    left = graph.create_div(left, right, result_type);
                    break;
                default:
                    break;
            }
        }

        return left;
    }

    OpID parse_primary() {
        using TT = cpp2_transpiler::TokenType;

        switch (peek().type) {
            case TT::IntegerLiteral: {
                std::string lex(peek().lexeme);
                advance();

                int64_t value = std::stoll(lex);
                auto int_type = mlir::IntegerType::get(&ctx, 64);
                auto attr = mlir::IntegerAttr::get(int_type, value);

                return graph.create_constant(attr, int_type);
            }

            case TT::FloatLiteral: {
                std::string lex(peek().lexeme);
                advance();

                double value = std::stod(lex);
                auto float_type = mlir::Float64Type::get(&ctx);
                auto attr = mlir::FloatAttr::get(float_type, value);

                return graph.create_constant(attr, float_type);
            }

            case TT::True:
            case TT::False: {
                bool value = peek().type == TT::True;
                advance();

                auto bool_type = mlir::IntegerType::get(&ctx, 1);
                auto attr = mlir::IntegerAttr::get(bool_type, value ? 1 : 0);

                return graph.create_constant(attr, bool_type);
            }

            case TT::Identifier: {
                std::string name(peek().lexeme);
                advance();

                // Look up in symbol table
                auto it = symbols.find(name);
                if (it != symbols.end()) {
                    return it->second;
                }

                // Unknown identifier - could be function call or method
                if (match(TT::LeftParen)) {
                    // Parse as call
                    std::vector<OpID> args;
                    while (!match(TT::RightParen)) {
                        OpID arg = parse_expression();
                        if (arg != 0) {
                            args.push_back(arg);
                        }
                        if (!match(TT::Comma)) break;
                    }

                    auto callee = mlir::StringAttr::get(&ctx, name);
                    auto result_type = mlir::IntegerType::get(&ctx, 64);
                    return graph.create_ufcs_call(callee, args, {result_type});
                }

                return 0;
            }

            case TT::LeftParen: {
                advance();
                OpID expr = parse_expression();
                match(TT::RightParen);
                return expr;
            }

            default:
                return 0;
        }
    }

    bool is_binary_op(cpp2_transpiler::TokenType type) const {
        using TT = cpp2_transpiler::TokenType;
        return type == TT::Plus || type == TT::Minus ||
               type == TT::Star || type == TT::Slash;
    }

    void skip_type() {
        using TT = cpp2_transpiler::TokenType;

        // Skip type annotations
        if (peek().type == TT::Identifier) {
            advance();
        }

        // Skip template args
        if (peek().type == TT::Less) {
            int depth = 1;
            advance();
            while (depth > 0 && pos < tokens.size()) {
                if (peek().type == TT::Less) depth++;
                if (peek().type == TT::Greater) depth--;
                advance();
            }
        }
    }
};

} // namespace cppfort::mlir_son