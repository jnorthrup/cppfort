#pragma once

// ============================================================================
// Combinator-Based Parser
// ============================================================================
//
// Replaces the recursive descent parser with combinator composition.
// Matches the AST types defined in ast.hpp exactly.
//
// ============================================================================

#ifndef CPP2_COMBINATOR_PARSER_HPP
#define CPP2_COMBINATOR_PARSER_HPP

#include "../lexer.hpp"
#include "../ast.hpp"
#include <span>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

namespace cpp2::parser::combinators {

using namespace cpp2_transpiler;
using TT = TokenType;

// ============================================================================
// Parse Context
// ============================================================================

struct ParseContext {
    std::span<Token> tokens;
    std::size_t pos = 0;
    bool had_error = false;
    std::string last_error;
    
    ParseContext(std::span<Token> toks) : tokens(toks) {}
    
    [[nodiscard]] bool at_end() const { 
        return pos >= tokens.size() || tokens[pos].type == TT::EndOfFile; 
    }
    
    [[nodiscard]] const Token& current() const {
        return pos < tokens.size() ? tokens[pos] : tokens.back();
    }
    
    [[nodiscard]] const Token& peek(std::size_t ahead = 0) const {
        auto idx = pos + ahead;
        return idx < tokens.size() ? tokens[idx] : tokens.back();
    }
    
    void advance() { if (!at_end()) ++pos; }
    
    bool match(TT type) {
        if (!at_end() && current().type == type) {
            advance();
            return true;
        }
        return false;
    }
    
    void error(std::string_view msg) {
        if (!had_error) {
            had_error = true;
            last_error = msg;
        }
    }
};

// ============================================================================
// Forward Declarations
// ============================================================================

std::unique_ptr<Expression> parse_expression(ParseContext& ctx);
std::unique_ptr<Statement> parse_statement(ParseContext& ctx);
std::unique_ptr<Declaration> parse_declaration(ParseContext& ctx);
std::unique_ptr<Type> parse_type(ParseContext& ctx);

// ============================================================================
// Expression Parsing
// ============================================================================

inline std::unique_ptr<Expression> parse_primary(ParseContext& ctx) {
    auto& tok = ctx.current();
    
    // Literals
    if (tok.type == TT::IntegerLiteral) {
        ctx.advance();
        return std::make_unique<LiteralExpression>(
            LiteralExpression::LiteralKind::Integer, std::string(tok.lexeme), tok.line);
    }
    if (tok.type == TT::FloatLiteral) {
        ctx.advance();
        return std::make_unique<LiteralExpression>(
            LiteralExpression::LiteralKind::Float, std::string(tok.lexeme), tok.line);
    }
    if (tok.type == TT::StringLiteral) {
        ctx.advance();
        return std::make_unique<LiteralExpression>(
            LiteralExpression::LiteralKind::String, std::string(tok.lexeme), tok.line);
    }
    if (tok.type == TT::CharacterLiteral) {
        ctx.advance();
        return std::make_unique<LiteralExpression>(
            LiteralExpression::LiteralKind::Character, std::string(tok.lexeme), tok.line);
    }
    if (tok.type == TT::True) {
        ctx.advance();
        return std::make_unique<LiteralExpression>(
            LiteralExpression::LiteralKind::Boolean, "true", tok.line);
    }
    if (tok.type == TT::False) {
        ctx.advance();
        return std::make_unique<LiteralExpression>(
            LiteralExpression::LiteralKind::Boolean, "false", tok.line);
    }
    
    // this - represented as identifier
    if (tok.type == TT::This) {
        ctx.advance();
        return std::make_unique<IdentifierExpression>("this", tok.line);
    }
    
    // Identifier
    if (tok.type == TT::Identifier || tok.type == TT::Underscore) {
        ctx.advance();
        return std::make_unique<IdentifierExpression>(std::string(tok.lexeme), tok.line);
    }
    
    // Parenthesized expression
    if (tok.type == TT::LeftParen) {
        ctx.advance();
        auto expr = parse_expression(ctx);
        if (!expr) return nullptr;
        if (ctx.current().type != TT::RightParen) {
            ctx.error("Expected ')' after expression");
            return nullptr;
        }
        ctx.advance();
        return expr;
    }
    
    // List literal: [a, b, c]
    if (tok.type == TT::LeftBracket) {
        ctx.advance();
        auto list = std::make_unique<ListExpression>(tok.line);
        
        if (ctx.current().type != TT::RightBracket) {
            do {
                auto elem = parse_expression(ctx);
                if (!elem) return nullptr;
                list->elements.push_back(std::move(elem));
            } while (ctx.match(TT::Comma));
        }
        
        if (!ctx.match(TT::RightBracket)) {
            ctx.error("Expected ']' after list");
            return nullptr;
        }
        return list;
    }
    
    return nullptr;
}

inline std::unique_ptr<Expression> parse_postfix(ParseContext& ctx) {
    auto expr = parse_primary(ctx);
    if (!expr) return nullptr;
    
    while (true) {
        auto& tok = ctx.current();
        
        // Function call: expr(args)
        if (tok.type == TT::LeftParen) {
            ctx.advance();
            auto call = std::make_unique<CallExpression>(std::move(expr), tok.line);
            
            if (ctx.current().type != TT::RightParen) {
                do {
                    auto arg = parse_expression(ctx);
                    if (!arg) return nullptr;
                    call->arguments.push_back(std::move(arg));
                } while (ctx.match(TT::Comma));
            }
            
            if (!ctx.match(TT::RightParen)) {
                ctx.error("Expected ')' after arguments");
                return nullptr;
            }
            expr = std::move(call);
            continue;
        }
        
        // Subscript: expr[index]
        if (tok.type == TT::LeftBracket) {
            ctx.advance();
            auto index = parse_expression(ctx);
            if (!index) return nullptr;
            
            if (!ctx.match(TT::RightBracket)) {
                ctx.error("Expected ']' after subscript");
                return nullptr;
            }
            
            expr = std::make_unique<SubscriptExpression>(
                std::move(expr), std::move(index), tok.line);
            continue;
        }
        
        // Member access: expr.member
        if (tok.type == TT::Dot) {
            ctx.advance();
            if (ctx.current().type != TT::Identifier) {
                ctx.error("Expected identifier after '.'");
                return nullptr;
            }
            auto member = std::string(ctx.current().lexeme);
            ctx.advance();
            
            expr = std::make_unique<MemberAccessExpression>(
                std::move(expr), member, false, tok.line);
            continue;
        }
        
        // Arrow access: expr->member
        if (tok.type == TT::Arrow) {
            ctx.advance();
            if (ctx.current().type != TT::Identifier) {
                ctx.error("Expected identifier after '->'");
                return nullptr;
            }
            auto member = std::string(ctx.current().lexeme);
            ctx.advance();
            
            expr = std::make_unique<MemberAccessExpression>(
                std::move(expr), member, true, tok.line);
            continue;
        }
        
        // Postfix ++/--
        if (tok.type == TT::PlusPlus || tok.type == TT::MinusMinus) {
            ctx.advance();
            expr = std::make_unique<UnaryExpression>(
                tok.type, std::move(expr), false, tok.line);
            continue;
        }
        
        // 'as' cast
        if (tok.type == TT::As) {
            ctx.advance();
            auto type = parse_type(ctx);
            if (!type) return nullptr;
            
            auto cast = std::make_unique<AsExpression>(tok.line);
            cast->expression = std::move(expr);
            cast->type = std::move(type);
            expr = std::move(cast);
            continue;
        }
        
        // 'is' type check
        if (tok.type == TT::Is) {
            ctx.advance();
            auto type = parse_type(ctx);
            if (!type) return nullptr;
            
            auto is_expr = std::make_unique<IsExpression>(tok.line);
            is_expr->expression = std::move(expr);
            is_expr->type = std::move(type);
            expr = std::move(is_expr);
            continue;
        }
        
        break;
    }
    
    return expr;
}

inline std::unique_ptr<Expression> parse_prefix(ParseContext& ctx) {
    auto& tok = ctx.current();
    
    // Prefix unary operators
    switch (tok.type) {
        case TT::Plus: 
        case TT::Minus: 
        case TT::Exclamation: 
        case TT::Tilde: 
        case TT::Ampersand: 
        case TT::Asterisk: 
        case TT::PlusPlus: 
        case TT::MinusMinus: {
            auto op = tok.type;
            ctx.advance();
            auto operand = parse_prefix(ctx);
            if (!operand) return nullptr;
            return std::make_unique<UnaryExpression>(op, std::move(operand), true, tok.line);
        }
        default:
            break;
    }
    
    // move
    if (tok.type == TT::Move) {
        ctx.advance();
        auto operand = parse_prefix(ctx);
        if (!operand) return nullptr;
        return std::make_unique<MoveExpression>(std::move(operand), tok.line);
    }
    
    // await
    if (tok.type == TT::Await) {
        ctx.advance();
        auto operand = parse_prefix(ctx);
        if (!operand) return nullptr;
        return std::make_unique<AwaitExpression>(std::move(operand), tok.line);
    }
    
    return parse_postfix(ctx);
}

inline int get_precedence(TT type) {
    switch (type) {
        case TT::Equal: case TT::PlusEqual: case TT::MinusEqual:
        case TT::AsteriskEqual: case TT::SlashEqual:
            return 1;
        case TT::Question:
            return 2;
        case TT::DoublePipe:
            return 3;
        case TT::DoubleAmpersand:
            return 4;
        case TT::Pipe:
            return 5;
        case TT::Caret:
            return 6;
        case TT::Ampersand:
            return 7;
        case TT::DoubleEqual: case TT::NotEqual:
            return 8;
        case TT::LessThan: case TT::GreaterThan:
        case TT::LessThanOrEqual: case TT::GreaterThanOrEqual:
        case TT::Spaceship:
            return 9;
        case TT::LeftShift: case TT::RightShift:
            return 10;
        case TT::Plus: case TT::Minus:
            return 11;
        case TT::Asterisk: case TT::Slash: case TT::Percent:
            return 12;
        default:
            return 0;
    }
}

inline bool is_right_assoc(TT type) {
    return type == TT::Equal || type == TT::PlusEqual || 
           type == TT::MinusEqual || type == TT::AsteriskEqual ||
           type == TT::SlashEqual || type == TT::Question;
}

inline std::unique_ptr<Expression> parse_binary(ParseContext& ctx, int min_prec) {
    auto left = parse_prefix(ctx);
    if (!left) return nullptr;
    
    while (true) {
        auto& tok = ctx.current();
        int prec = get_precedence(tok.type);
        
        if (prec < min_prec) break;
        
        // Ternary operator
        if (tok.type == TT::Question) {
            ctx.advance();
            auto then_expr = parse_expression(ctx);
            if (!then_expr) return nullptr;
            
            if (!ctx.match(TT::Colon)) {
                ctx.error("Expected ':' in ternary");
                return nullptr;
            }
            
            auto else_expr = parse_binary(ctx, prec);
            if (!else_expr) return nullptr;
            
            auto ternary = std::make_unique<TernaryExpression>(tok.line);
            ternary->condition = std::move(left);
            ternary->then_expr = std::move(then_expr);
            ternary->else_expr = std::move(else_expr);
            left = std::move(ternary);
            continue;
        }
        
        // Binary operator
        auto op = tok.type;
        ctx.advance();
        
        int next_prec = is_right_assoc(op) ? prec : prec + 1;
        auto right = parse_binary(ctx, next_prec);
        if (!right) return nullptr;
        
        left = std::make_unique<BinaryExpression>(
            op, std::move(left), std::move(right), tok.line);
    }
    
    return left;
}

inline std::unique_ptr<Expression> parse_expression(ParseContext& ctx) {
    return parse_binary(ctx, 1);
}

// ============================================================================
// Type Parsing
// ============================================================================

inline std::unique_ptr<Type> parse_type(ParseContext& ctx) {
    auto& tok = ctx.current();
    
    if (tok.type == TT::Auto) {
        ctx.advance();
        auto type = std::make_unique<Type>(Type::Kind::Auto);
        type->name = "auto";
        return type;
    }
    
    if (tok.type == TT::Identifier) {
        auto type = std::make_unique<Type>(Type::Kind::UserDefined);
        type->name = std::string(tok.lexeme);
        ctx.advance();
        
        // Template args
        if (ctx.current().type == TT::LessThan) {
            ctx.advance();
            type->kind = Type::Kind::Template;
            
            while (ctx.current().type != TT::GreaterThan && !ctx.at_end()) {
                auto arg = parse_type(ctx);
                if (!arg) return nullptr;
                type->template_args.push_back(std::move(arg));
                if (!ctx.match(TT::Comma)) break;
            }
            
            if (!ctx.match(TT::GreaterThan)) {
                ctx.error("Expected '>'");
                return nullptr;
            }
        }
        
        // Pointer/reference
        while (true) {
            if (ctx.current().type == TT::Asterisk) {
                ctx.advance();
                auto ptr = std::make_unique<Type>(Type::Kind::Pointer);
                ptr->pointee = std::move(type);
                type = std::move(ptr);
            } else if (ctx.current().type == TT::Ampersand) {
                ctx.advance();
                auto ref = std::make_unique<Type>(Type::Kind::Reference);
                ref->pointee = std::move(type);
                type = std::move(ref);
            } else {
                break;
            }
        }
        
        return type;
    }
    
    ctx.error("Expected type");
    return nullptr;
}

// ============================================================================
// Statement Parsing
// ============================================================================

inline std::unique_ptr<BlockStatement> parse_block(ParseContext& ctx) {
    auto block = std::make_unique<BlockStatement>(ctx.current().line);
    
    while (ctx.current().type != TT::RightBrace && !ctx.at_end()) {
        auto stmt = parse_statement(ctx);
        if (stmt) {
            block->statements.push_back(std::move(stmt));
        }
    }
    
    if (!ctx.match(TT::RightBrace)) {
        ctx.error("Expected '}'");
        return nullptr;
    }
    
    return block;
}

inline std::unique_ptr<Statement> parse_if_statement(ParseContext& ctx) {
    auto line = ctx.current().line;
    
    bool is_constexpr = ctx.match(TT::Const);
    
    auto condition = parse_expression(ctx);
    if (!condition) return nullptr;
    
    if (!ctx.match(TT::LeftBrace)) {
        ctx.error("Expected '{'");
        return nullptr;
    }
    
    auto then_block = parse_block(ctx);
    if (!then_block) return nullptr;
    
    std::unique_ptr<Statement> else_block = nullptr;
    if (ctx.match(TT::Else)) {
        if (ctx.match(TT::If)) {
            else_block = parse_if_statement(ctx);
        } else if (ctx.match(TT::LeftBrace)) {
            else_block = parse_block(ctx);
        } else {
            ctx.error("Expected '{' or 'if'");
            return nullptr;
        }
    }
    
    auto stmt = std::make_unique<IfStatement>(line);
    stmt->condition = std::move(condition);
    stmt->then_branch = std::move(then_block);
    stmt->else_branch = std::move(else_block);
    stmt->is_constexpr = is_constexpr;
    return stmt;
}

inline std::unique_ptr<Statement> parse_while_statement(ParseContext& ctx) {
    auto line = ctx.current().line;
    
    auto condition = parse_expression(ctx);
    if (!condition) return nullptr;
    
    if (!ctx.match(TT::LeftBrace)) {
        ctx.error("Expected '{'");
        return nullptr;
    }
    
    auto body = parse_block(ctx);
    if (!body) return nullptr;
    
    auto stmt = std::make_unique<WhileStatement>(line);
    stmt->condition = std::move(condition);
    stmt->body = std::move(body);
    return stmt;
}

inline std::unique_ptr<Statement> parse_for_statement(ParseContext& ctx) {
    auto line = ctx.current().line;
    
    auto range = parse_expression(ctx);
    if (!range) return nullptr;
    
    if (!ctx.match(TT::Do)) {
        ctx.error("Expected 'do'");
        return nullptr;
    }
    
    if (!ctx.match(TT::LeftParen)) {
        ctx.error("Expected '('");
        return nullptr;
    }
    
    if (ctx.current().type != TT::Identifier) {
        ctx.error("Expected loop variable");
        return nullptr;
    }
    std::string var = std::string(ctx.current().lexeme);
    ctx.advance();
    
    if (!ctx.match(TT::RightParen)) {
        ctx.error("Expected ')'");
        return nullptr;
    }
    
    if (!ctx.match(TT::LeftBrace)) {
        ctx.error("Expected '{'");
        return nullptr;
    }
    
    auto body = parse_block(ctx);
    if (!body) return nullptr;
    
    auto stmt = std::make_unique<ForRangeStatement>(line);
    stmt->variable = var;
    stmt->range = std::move(range);
    stmt->body = std::move(body);
    return stmt;
}

inline std::unique_ptr<Statement> parse_return_statement(ParseContext& ctx) {
    auto line = ctx.current().line;
    
    std::unique_ptr<Expression> value = nullptr;
    if (ctx.current().type != TT::Semicolon) {
        value = parse_expression(ctx);
    }
    
    if (!ctx.match(TT::Semicolon)) {
        ctx.error("Expected ';'");
        return nullptr;
    }
    
    return std::make_unique<ReturnStatement>(std::move(value), line);
}

inline std::unique_ptr<Statement> parse_statement(ParseContext& ctx) {
    auto& tok = ctx.current();
    
    // Block
    if (ctx.match(TT::LeftBrace)) {
        return parse_block(ctx);
    }
    
    // Control flow
    if (ctx.match(TT::If)) {
        return parse_if_statement(ctx);
    }
    if (ctx.match(TT::While)) {
        return parse_while_statement(ctx);
    }
    if (ctx.match(TT::For)) {
        return parse_for_statement(ctx);
    }
    if (ctx.match(TT::Return)) {
        return parse_return_statement(ctx);
    }
    if (ctx.match(TT::Break)) {
        // Skip optional label (not stored in AST)
        if (ctx.current().type == TT::Identifier) {
            ctx.advance();
        }
        if (!ctx.match(TT::Semicolon)) {
            ctx.error("Expected ';'");
            return nullptr;
        }
        return std::make_unique<BreakStatement>(tok.line);
    }
    if (ctx.match(TT::Continue)) {
        // Skip optional label (not stored in AST)
        if (ctx.current().type == TT::Identifier) {
            ctx.advance();
        }
        if (!ctx.match(TT::Semicolon)) {
            ctx.error("Expected ';'");
            return nullptr;
        }
        return std::make_unique<ContinueStatement>(tok.line);
    }
    if (ctx.match(TT::Throw)) {
        std::unique_ptr<Expression> value = nullptr;
        if (ctx.current().type != TT::Semicolon) {
            value = parse_expression(ctx);
        }
        if (!ctx.match(TT::Semicolon)) {
            ctx.error("Expected ';'");
            return nullptr;
        }
        return std::make_unique<ThrowStatement>(std::move(value), tok.line);
    }
    
    // Local variable: name: type = value; or name := value;
    if (tok.type == TT::Identifier) {
        auto lookahead = ctx.peek(1);
        if (lookahead.type == TT::Colon || lookahead.type == TT::ColonEqual) {
            auto name = std::string(tok.lexeme);
            auto line = tok.line;
            ctx.advance();
            
            auto decl = std::make_unique<VariableDeclaration>(name, line);
            
            if (ctx.match(TT::ColonEqual)) {
                decl->type = std::make_unique<Type>(Type::Kind::Auto);
                decl->type->name = "auto";
                decl->initializer = parse_expression(ctx);
            } else {
                ctx.advance(); // consume ':'
                decl->type = parse_type(ctx);
                if (!decl->type) return nullptr;
                
                if (ctx.match(TT::Equal)) {
                    decl->initializer = parse_expression(ctx);
                }
            }
            
            if (!ctx.match(TT::Semicolon)) {
                ctx.error("Expected ';'");
                return nullptr;
            }
            
            return std::make_unique<DeclarationStatement>(std::move(decl), line);
        }
    }
    
    // Expression statement
    auto line = tok.line;
    auto expr = parse_expression(ctx);
    if (!expr) return nullptr;
    
    if (!ctx.match(TT::Semicolon)) {
        ctx.error("Expected ';'");
        return nullptr;
    }
    
    return std::make_unique<ExpressionStatement>(std::move(expr), line);
}

// ============================================================================
// Declaration Parsing
// ============================================================================

inline std::unique_ptr<FunctionDeclaration> parse_function_body(
    ParseContext& ctx,
    const std::string& name,
    std::size_t line)
{
    auto func = std::make_unique<FunctionDeclaration>(name, line);
    
    // Parameters
    if (ctx.current().type != TT::RightParen) {
        do {
            Parameter param;
            
            // Qualifier
            if (ctx.match(TT::In)) param.qualifier = ParameterQualifier::In;
            else if (ctx.match(TT::Out)) param.qualifier = ParameterQualifier::Out;
            else if (ctx.match(TT::Inout)) param.qualifier = ParameterQualifier::Inout;
            else if (ctx.match(TT::Copy)) param.qualifier = ParameterQualifier::Copy;
            else if (ctx.match(TT::Move)) param.qualifier = ParameterQualifier::Move;
            else if (ctx.match(TT::Forward)) param.qualifier = ParameterQualifier::Forward;
            
            // Name
            if (ctx.current().type != TT::Identifier && ctx.current().type != TT::This) {
                ctx.error("Expected parameter name");
                return nullptr;
            }
            param.name = std::string(ctx.current().lexeme);
            ctx.advance();
            
            // Type
            if (ctx.match(TT::Colon)) {
                param.type = parse_type(ctx);
                if (!param.type) return nullptr;
            }
            
            // Default value
            if (ctx.match(TT::Equal)) {
                param.default_value = parse_expression(ctx);
                if (!param.default_value) return nullptr;
            }
            
            func->parameters.push_back(std::move(param));
        } while (ctx.match(TT::Comma));
    }
    
    if (!ctx.match(TT::RightParen)) {
        ctx.error("Expected ')'");
        return nullptr;
    }
    
    // Return type
    if (ctx.match(TT::Arrow)) {
        func->return_type = parse_type(ctx);
        if (!func->return_type) return nullptr;
    }
    
    // Body
    if (ctx.match(TT::Equal)) {
        auto expr = parse_expression(ctx);
        if (!expr) return nullptr;
        
        if (!ctx.match(TT::Semicolon)) {
            ctx.error("Expected ';'");
            return nullptr;
        }
        
        auto ret = std::make_unique<ReturnStatement>(std::move(expr), static_cast<std::uint32_t>(line));
        auto block = std::make_unique<BlockStatement>(static_cast<std::uint32_t>(line));
        block->statements.push_back(std::move(ret));
        func->body = std::move(block);
    }
    else if (ctx.match(TT::LeftBrace)) {
        func->body = parse_block(ctx);
        if (!func->body) return nullptr;
    }
    else if (ctx.match(TT::Semicolon)) {
        // Declaration only
    }
    else {
        ctx.error("Expected '=', '{', or ';'");
        return nullptr;
    }
    
    return func;
}

inline std::unique_ptr<Declaration> parse_declaration(ParseContext& ctx) {
    auto& tok = ctx.current();
    auto line = tok.line;
    
    // Preprocessor
    if (tok.type == TT::Hash) {
        while (!ctx.at_end() && ctx.current().line == line) {
            ctx.advance();
        }
        return std::make_unique<Cpp1PassthroughDeclaration>("", line);
    }
    
    // Import
    if (ctx.match(TT::Import)) {
        if (ctx.current().type != TT::Identifier) {
            ctx.error("Expected module name");
            return nullptr;
        }
        auto name = std::string(ctx.current().lexeme);
        ctx.advance();
        
        if (!ctx.match(TT::Semicolon)) {
            ctx.error("Expected ';'");
            return nullptr;
        }
        
        return std::make_unique<ImportDeclaration>(std::move(name), static_cast<std::uint32_t>(line));
    }
    
    // Access specifier
    if (ctx.match(TT::Public) || ctx.match(TT::Private) || ctx.match(TT::Protected)) {
        // Continue to next token
    }
    
    // func keyword
    if (ctx.match(TT::Func)) {
        if (ctx.current().type != TT::Identifier) {
            ctx.error("Expected function name");
            return nullptr;
        }
        auto name = std::string(ctx.current().lexeme);
        ctx.advance();
        
        ctx.match(TT::Colon); // Optional
        
        if (!ctx.match(TT::LeftParen)) {
            ctx.error("Expected '('");
            return nullptr;
        }
        
        return parse_function_body(ctx, name, line);
    }
    
    // Unified declaration: name: ...
    if (ctx.current().type == TT::Identifier) {
        auto name = std::string(ctx.current().lexeme);
        ctx.advance();
        
        // Type-deduced: name := value
        if (ctx.match(TT::ColonEqual)) {
            auto init = parse_expression(ctx);
            if (!init) return nullptr;
            
            if (!ctx.match(TT::Semicolon)) {
                ctx.error("Expected ';'");
                return nullptr;
            }
            
            auto decl = std::make_unique<VariableDeclaration>(name, line);
            decl->type = std::make_unique<Type>(Type::Kind::Auto);
            decl->type->name = "auto";
            decl->initializer = std::move(init);
            return decl;
        }
        
        // Unified: name: ...
        if (ctx.match(TT::Colon)) {
            // Function: name: (params) -> return { }
            if (ctx.match(TT::LeftParen)) {
                return parse_function_body(ctx, name, line);
            }
            
            // Type: name: type = { }
            if (ctx.match(TT::Type)) {
                if (!ctx.match(TT::Equal)) {
                    ctx.error("Expected '='");
                    return nullptr;
                }
                if (!ctx.match(TT::LeftBrace)) {
                    ctx.error("Expected '{'");
                    return nullptr;
                }
                
                auto decl = std::make_unique<TypeDeclaration>(name, TypeDeclaration::TypeKind::Type, static_cast<std::uint32_t>(line));
                while (ctx.current().type != TT::RightBrace && !ctx.at_end()) {
                    auto member = parse_declaration(ctx);
                    if (member) {
                        decl->members.push_back(std::move(member));
                    }
                }
                
                if (!ctx.match(TT::RightBrace)) {
                    ctx.error("Expected '}'");
                    return nullptr;
                }
                return decl;
            }
            
            // Namespace: name: namespace = { }
            if (ctx.match(TT::Namespace)) {
                ctx.match(TT::Equal); // Optional
                
                if (!ctx.match(TT::LeftBrace)) {
                    ctx.error("Expected '{'");
                    return nullptr;
                }
                
                auto decl = std::make_unique<NamespaceDeclaration>(name, line);
                while (ctx.current().type != TT::RightBrace && !ctx.at_end()) {
                    auto member = parse_declaration(ctx);
                    if (member) {
                        decl->declarations.push_back(std::move(member));
                    }
                }
                
                if (!ctx.match(TT::RightBrace)) {
                    ctx.error("Expected '}'");
                    return nullptr;
                }
                return decl;
            }
            
            // Variable: name: Type = value
            auto var_type = parse_type(ctx);
            if (!var_type) return nullptr;
            
            auto decl = std::make_unique<VariableDeclaration>(name, line);
            decl->type = std::move(var_type);
            
            if (ctx.match(TT::Equal)) {
                decl->initializer = parse_expression(ctx);
                if (!decl->initializer) return nullptr;
            }
            
            if (!ctx.match(TT::Semicolon)) {
                ctx.error("Expected ';'");
                return nullptr;
            }
            
            return decl;
        }
    }
    
    // Skip unknown
    ctx.advance();
    return nullptr;
}

// ============================================================================
// Main Parser Class
// ============================================================================

class CombinatorParser {
public:
    explicit CombinatorParser(std::span<Token> tokens) : ctx_(tokens) {}
    
    std::unique_ptr<AST> parse() {
        auto ast = std::make_unique<AST>();
        
        while (!ctx_.at_end()) {
            auto decl = parse_declaration(ctx_);
            if (decl) {
                ast->declarations.push_back(std::move(decl));
            }
        }
        
        return ast;
    }
    
    [[nodiscard]] bool had_error() const { return ctx_.had_error; }
    [[nodiscard]] const std::string& error_message() const { return ctx_.last_error; }
    
private:
    ParseContext ctx_;
};

} // namespace cpp2::parser::combinators

#endif // CPP2_COMBINATOR_PARSER_HPP
