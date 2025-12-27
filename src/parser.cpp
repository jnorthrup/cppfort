#include "parser.hpp"
#include <stdexcept>
#include <iostream>
#include <format>

namespace cpp2_transpiler {

Parser::Parser(std::span<Token> tokens) : tokens(tokens) {}

std::unique_ptr<AST> Parser::parse() {
    auto ast = std::make_unique<AST>();

    while (!is_at_end()) {
        auto decl = declaration();
        if (decl) {
            ast->declarations.push_back(std::move(decl));
        }
    }

    return ast;
}

// Parsing utilities
const Token& Parser::peek() const {
    if (current >= tokens.size()) {
        return tokens.back(); // Return EOF token
    }
    return tokens[current];
}

const Token& Parser::advance() {
    if (!is_at_end()) current++;
    return previous();
}

const Token& Parser::previous() const {
    if (current == 0) {
        return tokens.front();
    }
    return tokens[current - 1];
}

bool Parser::is_at_end() const {
    return current >= tokens.size() || peek().type == TokenType::EndOfFile;
}

bool Parser::check(TokenType type) const {
    return !is_at_end() && peek().type == type;
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::match(std::initializer_list<TokenType> types) {
    for (TokenType type : types) {
        if (check(type)) {
            advance();
            return true;
        }
    }
    return false;
}

const Token& Parser::consume(TokenType type, const char* message) {
    if (check(type)) return advance();
    error_at(peek(), message);
    return peek(); // Return something to avoid undefined behavior
}

bool Parser::consume_if(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

template<typename F>
auto Parser::synchronize_on_error(F&& func) -> decltype(func()) {
    if (panic_mode) {
        // Skip tokens until we can recover
        while (!is_at_end()) {
            if (previous().type == TokenType::Semicolon) {
                panic_mode = false;
                break;
            }
            if (peek().type == TokenType::Func || peek().type == TokenType::Type ||
                peek().type == TokenType::Namespace || peek().type == TokenType::Let ||
                peek().type == TokenType::Const) {
                panic_mode = false;
                break;
            }
            advance();
        }
        return nullptr;
    }

    try {
        return func();
    } catch (...) {
        panic_mode = true;
        return nullptr;
    }
}

// Entry point
std::unique_ptr<Declaration> Parser::declaration() {
    return synchronize_on_error([this]() -> std::unique_ptr<Declaration> {
        // Skip preprocessor directives (#include, #define, etc.)
        if (match(TokenType::Hash)) {
            // Preprocessor directive - skip it and move to next declaration
            return nullptr;
        }

        // Cpp2 unified syntax: identifier: type = initializer
        if (check(TokenType::Identifier)) {
            std::size_t saved = current;
            advance(); // consume identifier
            if (check(TokenType::Colon)) {
                current = saved; // backtrack
                // Could be function or variable
                advance(); // consume identifier (makes it previous())
                consume(TokenType::Colon, "Expected ':' after identifier");

                // Check if it's a function (has parameter list)
                if (check(TokenType::LeftParen)) {
                    return function_declaration(); // Will parse params, return type, body
                } else {
                    return variable_declaration(); // Will parse type and initializer
                }
            }
            current = saved; // backtrack if no colon
        }

        if (match({TokenType::Let, TokenType::Const})) {
            return variable_declaration();
        }
        if (match(TokenType::Func)) {
            return function_declaration();
        }
        if (match(TokenType::Type)) {
            return type_declaration();
        }
        if (match(TokenType::Namespace)) {
            return namespace_declaration();
        }
        if (match(TokenType::Operator)) {
            return operator_declaration();
        }
        if (match(TokenType::Import)) {
            return import_declaration();
        }
        if (match(TokenType::Using)) {
            return using_declaration();
        }
        if (is_template_start()) {
            return template_declaration();
        }

        // If we don't recognize the declaration, treat as statement
        auto stmt = statement();
        if (stmt) {
            // Wrap statement in a dummy declaration or handle appropriately
            error_at_current("Expected declaration");
        }
        return nullptr;
    });
}

std::unique_ptr<Statement> Parser::statement() {
    return synchronize_on_error([this]() -> std::unique_ptr<Statement> {
        // Local variable declarations in statement position
        if (match({TokenType::Let, TokenType::Const})) {
            auto decl = variable_declaration();
            if (decl) {
                return std::make_unique<DeclarationStatement>(std::move(decl), decl->line);
            }
            return nullptr;
        }

        if (match(TokenType::LeftBrace)) {
            return block_statement();
        }
        if (match(TokenType::If)) {
            return if_statement();
        }
        if (match(TokenType::While)) {
            return while_statement();
        }
        if (match(TokenType::For)) {
            return for_statement();
        }
        if (match(TokenType::Switch)) {
            return switch_statement();
        }
        if (match(TokenType::Inspect)) {
            return inspect_statement();
        }
        if (match(TokenType::Return)) {
            return return_statement();
        }
        if (match(TokenType::Break)) {
            return break_statement();
        }
        if (match(TokenType::Continue)) {
            return continue_statement();
        }
        if (match(TokenType::Try)) {
            return try_statement();
        }
        if (match(TokenType::Throw)) {
            return throw_statement();
        }
        if (match({TokenType::ContractPre, TokenType::ContractPost, TokenType::ContractAssert})) {
            return contract_statement();
        }
        if (match(TokenType::Static_assert)) {
            return static_assert_statement();
        }

        // Concurrency statements
        if (match(TokenType::CoroutineScope)) {
            return coroutine_scope_statement();
        }
        if (match(TokenType::Channel)) {
            return channel_declaration_statement();
        }
        if (match(TokenType::ParallelFor)) {
            return parallel_for_statement();
        }

        // Variable declaration with unified syntax: name: type = value;
        // Check for identifier followed by colon
        if (check(TokenType::Identifier)) {
            std::size_t saved = current;
            advance(); // consume identifier
            if (check(TokenType::Colon)) {
                // Found unified syntax variable declaration
                Token name = previous(); // the identifier we just consumed
                advance(); // consume the colon

                // Parse type and initializer
                std::unique_ptr<Type> var_type = nullptr;
                if (!check(TokenType::Equal) && !check(TokenType::DoubleEqual)) {
                    var_type = type();
                }

                std::unique_ptr<Expression> initializer = nullptr;
                bool is_compile_time = false;
                if (match(TokenType::Equal)) {
                    initializer = expression();
                } else if (match(TokenType::DoubleEqual)) {
                    is_compile_time = true;
                    initializer = expression();
                }

                consume(TokenType::Semicolon, "Expected ';' after variable declaration");

                auto decl = std::make_unique<VariableDeclaration>(std::string(name.lexeme), name.line);
                decl->type = std::move(var_type);
                decl->initializer = std::move(initializer);
                decl->is_const = false;
                decl->is_mut = false;
                decl->is_compile_time = is_compile_time;

                return std::make_unique<DeclarationStatement>(std::move(decl), previous().line);
            }
            current = saved; // backtrack if no colon
        }

        // Expression statement
        auto expr = expression();
        if (expr) {
            consume(TokenType::Semicolon, "Expected ';' after expression");
            return std::make_unique<ExpressionStatement>(std::move(expr), previous().line);
        }
        return nullptr;
    });
}

std::unique_ptr<Expression> Parser::expression() {
    return assignment_expression();
}

// Declarations
std::unique_ptr<Declaration> Parser::variable_declaration() {
    bool is_const = false;
    bool is_compile_time = false;
    bool is_mut = false;

    // Check if called from keyword syntax (let/const) or unified syntax (name:)
    Token start = previous();
    bool from_keyword = (start.type == TokenType::Let || start.type == TokenType::Const);
    Token name = start; // Will be identifier or keyword

    if (from_keyword) {
        // Keyword syntax: let/const name[: type] (=|==) init;
        is_const = start.type == TokenType::Const;
        name = consume(TokenType::Identifier, "Expected variable name");
    } else if (start.type == TokenType::Colon) {
        // Unified syntax from declaration(): colon is previous(), need to get identifier
        if (current >= 2) {
            name = tokens[current - 2]; // identifier before colon
        }
    }
    // else: name is already set from some other path (e.g., function parameter)

    std::unique_ptr<Type> var_type = nullptr;
    if (from_keyword) {
        // Optional type annotation.
        if (match(TokenType::Colon)) {
            var_type = type();
        }
    } else {
        // Unified syntax (name: ...) or other contexts: type is expected unless an initializer follows.
        if (!check(TokenType::Equal) && !check(TokenType::DoubleEqual)) {
            var_type = type();
        }
    }

    std::unique_ptr<Expression> initializer = nullptr;
    if (match(TokenType::Equal)) {
        initializer = expression();
    } else if (match(TokenType::DoubleEqual)) {
        is_compile_time = true;
        initializer = expression();
    }

    if (!var_type && !initializer) {
        error_at_current("Expected ':' type annotation or initializer for variable");
    }

    consume(TokenType::Semicolon, "Expected ';' after variable declaration");

    auto decl = std::make_unique<VariableDeclaration>(std::string(name.lexeme), name.line);
    decl->type = std::move(var_type);
    decl->initializer = std::move(initializer);
    decl->is_const = is_const;
    decl->is_mut = is_mut;
    decl->is_compile_time = is_compile_time;

    return decl;
}

std::unique_ptr<Declaration> Parser::function_declaration() {
    Token prev = previous();
    Token name = prev; // Initialize with prev, will be overridden if needed
    std::vector<std::string> template_params;

    // Check if called from keyword syntax (func) or unified syntax (name:)
    if (prev.type == TokenType::Func) {
        // Keyword syntax: func name(...) -> type = body
        // Historically this parser also accepted/expected a ':' after the function name.
        // Keep ':' optional for compatibility with existing tests and cppfront-like syntax.
        name = consume(TokenType::Identifier, "Expected function name");
        // Template parameters
        if (match(TokenType::LessThan)) {
            template_params = template_parameters();
            consume(TokenType::GreaterThan, "Expected '>' after template parameters");
        }
        match(TokenType::Colon); // Optional
    } else {
        // Unified syntax: name: (...) -> type = body
        // identifier and colon were already consumed in declaration()
        // After consuming identifier and colon, tokens[current - 2] is the identifier
        if (current >= 2 && tokens[current - 2].type == TokenType::Identifier) {
            name = tokens[current - 2]; // identifier before colon
        }
        // else: keep prev as fallback (though this indicates a parsing bug)
    }

    consume(TokenType::LeftParen, "Expected '(' after function name");

    std::vector<FunctionDeclaration::Parameter> parameters;
    if (!check(TokenType::RightParen)) {
        do {
            // Parse qualifiers before parameter name
            std::vector<ParameterQualifier> qualifiers = parse_parameter_qualifiers();

            Token param_name = consume(TokenType::Identifier, "Expected parameter name");
            consume(TokenType::Colon, "Expected ':' after parameter name");

            std::unique_ptr<Type> param_type = type();
            if (!param_type) {
                error_at_current("Expected parameter type");
            }

            std::unique_ptr<Expression> default_value = nullptr;
            if (match(TokenType::Equal)) {
                default_value = expression();
            }

            parameters.push_back({
                std::string(param_name.lexeme),
                std::move(param_type),
                std::move(default_value)
            });
            // Add qualifiers to the parameter
            parameters.back().qualifiers = std::move(qualifiers);
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RightParen, "Expected ')' after parameters");

    // Return type
    std::unique_ptr<Type> return_type = nullptr;
    if (match(TokenType::Arrow)) {
        return_type = type();
    }

    // Contracts
    auto contracts = parse_contracts();

    // Function body - Cpp2 supports both block bodies and expression bodies.
    // - name: (params) -> type = { body }
    // - name: (params) -> type = expr;
    std::unique_ptr<Statement> body = nullptr;
    bool has_equals = match(TokenType::Equal); // '=' is optional for historical compatibility
    if (match(TokenType::LeftBrace)) {
        body = block_statement();
    } else if (has_equals) {
        // Expression-bodied function
        auto expr = expression();
        if (expr) {
            if (return_type && return_type->name != "void") {
                body = std::make_unique<ReturnStatement>(std::move(expr), previous().line);
            } else {
                body = std::make_unique<ExpressionStatement>(std::move(expr), previous().line);
            }
        } else {
            error_at_current("Expected expression");
        }
        consume(TokenType::Semicolon, "Expected ';' after function body expression");
    } else {
        consume(TokenType::Semicolon, "Expected ';' or function body");
    }

    // Lower contracts into the function body as statements so later phases (and
    // tests that only run CodeGenerator) can see them.
    if (!contracts.empty()) {
        if (auto* block = dynamic_cast<BlockStatement*>(body.get())) {
            std::vector<std::unique_ptr<Statement>> new_stmts;
            new_stmts.reserve(contracts.size() + block->statements.size());
            for (auto& c : contracts) {
                std::size_t line = c ? c->line : name.line;
                new_stmts.push_back(std::make_unique<ContractStatement>(std::move(c), line));
            }
            for (auto& s : block->statements) {
                new_stmts.push_back(std::move(s));
            }
            block->statements = std::move(new_stmts);
        } else {
            auto new_block = std::make_unique<BlockStatement>(name.line);
            new_block->statements.reserve(contracts.size() + (body ? 1 : 0));
            for (auto& c : contracts) {
                std::size_t line = c ? c->line : name.line;
                new_block->statements.push_back(std::make_unique<ContractStatement>(std::move(c), line));
            }
            if (body) {
                new_block->statements.push_back(std::move(body));
            }
            body = std::move(new_block);
        }
    }

    auto func = std::make_unique<FunctionDeclaration>(std::string(name.lexeme), name.line);
    func->parameters = std::move(parameters);
    func->return_type = std::move(return_type);
    func->body = std::move(body);

    return func;
}

std::unique_ptr<Declaration> Parser::type_declaration() {
    Token name = consume(TokenType::Identifier, "Expected type name");
    consume(TokenType::Colon, "Expected ':' after type name");

    TypeDeclaration::TypeKind kind = TypeDeclaration::TypeKind::Struct;
    if (check(TokenType::Identifier)) {
        if (peek().lexeme == "struct") {
            advance();
            kind = TypeDeclaration::TypeKind::Struct;
        } else if (peek().lexeme == "class") {
            advance();
            kind = TypeDeclaration::TypeKind::Class;
        } else if (peek().lexeme == "interface") {
            advance();
            kind = TypeDeclaration::TypeKind::Interface;
        } else if (peek().lexeme == "enum") {
            advance();
            kind = TypeDeclaration::TypeKind::Enum;
        } else if (peek().lexeme == "union") {
            advance();
            kind = TypeDeclaration::TypeKind::Union;
        } else if (peek().lexeme == "type") {
            advance();
            kind = TypeDeclaration::TypeKind::Alias;
        }
    }

    // Type alias handling
    if (kind == TypeDeclaration::TypeKind::Alias) {
        consume(TokenType::Equal, "Expected '=' for type alias");
        auto underlying_type = type();
        consume(TokenType::Semicolon, "Expected ';' after type alias");

        auto type_decl = std::make_unique<TypeDeclaration>(std::string(name.lexeme), kind, name.line);
        type_decl->underlying_type = std::move(underlying_type);
        return type_decl;
    }

    consume(TokenType::LeftBrace, "Expected '{' for type definition");

    auto type_decl = std::make_unique<TypeDeclaration>(std::string(name.lexeme), kind, name.line);

    while (!check(TokenType::RightBrace) && !is_at_end()) {
        auto member = declaration();
        if (member) {
            type_decl->members.push_back(std::move(member));
        }
    }

    consume(TokenType::RightBrace, "Expected '}' after type definition");

    return type_decl;
}

std::unique_ptr<Declaration> Parser::namespace_declaration() {
    Token name = consume(TokenType::Identifier, "Expected namespace name");
    consume(TokenType::Equal, "Expected '=' after namespace name");

    auto ns = std::make_unique<NamespaceDeclaration>(std::string(name.lexeme), name.line);

    if (match(TokenType::LeftBrace)) {
        while (!check(TokenType::RightBrace) && !is_at_end()) {
            auto member = declaration();
            if (member) {
                ns->members.push_back(std::move(member));
            }
        }
        consume(TokenType::RightBrace, "Expected '}' after namespace");
    } else {
        // Single declaration
        auto member = declaration();
        if (member) {
            ns->members.push_back(std::move(member));
        }
        consume(TokenType::Semicolon, "Expected ';' after namespace declaration");
    }

    return ns;
}

std::unique_ptr<Declaration> Parser::operator_declaration() {
    // Handle operator overloading
    Token op = advance(); // This should be the operator token
    consume(TokenType::Colon, "Expected ':' after operator");
    consume(TokenType::LeftParen, "Expected '(' after operator ':'");

    auto op_decl = std::make_unique<OperatorDeclaration>(std::string(op.lexeme), op.line);

    if (!check(TokenType::RightParen)) {
        do {
            // Parse qualifiers before parameter name
            std::vector<ParameterQualifier> qualifiers = parse_parameter_qualifiers();

            Token param_name = consume(TokenType::Identifier, "Expected parameter name");
            consume(TokenType::Colon, "Expected ':' after parameter name");
            auto param_type = type();

            auto param = std::make_unique<FunctionDeclaration::Parameter>();
            param->name = std::string(param_name.lexeme);
            param->type = std::move(param_type);
            param->qualifiers = std::move(qualifiers);

            op_decl->parameters.push_back(std::move(param));
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RightParen, "Expected ')' after parameters");

    if (match(TokenType::Arrow)) {
        op_decl->return_type = type();
    }

    consume(TokenType::LeftBrace, "Expected '{' for operator body");
    op_decl->body = block_statement();

    return op_decl;
}

std::unique_ptr<Declaration> Parser::using_declaration() {
    Token name = consume(TokenType::Identifier, "Expected alias name");
    consume(TokenType::Equal, "Expected '=' after using alias");
    Token target = consume(TokenType::Identifier, "Expected target name for using alias");
    consume(TokenType::Semicolon, "Expected ';' after using declaration");

    return std::make_unique<UsingDeclaration>(
        std::string(name.lexeme),
        std::string(target.lexeme),
        name.line
    );
}

std::unique_ptr<Declaration> Parser::import_declaration() {
    Token module = consume(TokenType::Identifier, "Expected module name");
    consume(TokenType::Semicolon, "Expected ';' after import");

    return std::make_unique<ImportDeclaration>(std::string(module.lexeme), module.line);
}

std::unique_ptr<Declaration> Parser::template_declaration() {
    consume(TokenType::LessThan, "Expected '<' after template");
    auto params = template_parameters();
    consume(TokenType::GreaterThan, "Expected '>' after template parameters");

    auto decl = declaration();
    if (!decl) {
        error_at_current("Expected declaration after template parameters");
    }

    // TODO: Attach template parameters to declaration
    // For now, just return the inner declaration
    return decl;
}

// Types
std::unique_ptr<Type> Parser::type() {
    auto t = qualified_type();

    // Handle pointers, references
    while (match({TokenType::Asterisk, TokenType::Ampersand})) {
        auto ptr = std::make_unique<Type>(Type::Kind::Pointer);
        ptr->pointee = std::move(t);
        if (previous().type == TokenType::Asterisk) {
            ptr->kind = Type::Kind::Pointer;
        } else {
            ptr->kind = Type::Kind::Reference;
        }
        t = std::move(ptr);
    }

    return t;
}

std::unique_ptr<Type> Parser::qualified_type() {
    auto t = basic_type();

    while (match(TokenType::DoubleColon)) {
        Token name = consume(TokenType::Identifier, "Expected identifier after '::'");
        t->name += "::" + std::string(name.lexeme);
    }

    return t;
}

std::unique_ptr<Type> Parser::basic_type() {
    if (match(TokenType::Identifier)) {
        auto t = std::make_unique<Type>(Type::Kind::UserDefined);
        t->name = std::string(previous().lexeme);

        // Check for template arguments
        if (match(TokenType::LessThan)) {
            do {
                auto arg = type();
                if (arg) {
                    t->template_args.push_back(std::move(arg));
                }
            } while (match(TokenType::Comma));
            consume(TokenType::GreaterThan, "Expected '>' after template arguments");
            t->kind = Type::Kind::Template;
        }

        return t;
    }

    if (match(TokenType::Auto)) {
        auto t = std::make_unique<Type>(Type::Kind::Auto);
        t->name = "auto";
        return t;
    }

    if (match(TokenType::Underscore)) {
        auto t = std::make_unique<Type>(Type::Kind::Deduced);
        t->name = "_";
        return t;
    }

    error_at_current("Expected type");
    return nullptr;
}

std::unique_ptr<Type> Parser::function_type() {
    consume(TokenType::LeftParen, "Expected '(' for function type");

    auto func_type = std::make_unique<Type>(Type::Kind::Function);

    if (!check(TokenType::RightParen)) {
        do {
            auto param_type = type();
            if (param_type) {
                func_type->template_args.push_back(std::move(param_type));
            }
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RightParen, "Expected ')' after function type parameters");

    if (match(TokenType::Arrow)) {
        func_type->pointee = type(); // Return type
    }

    return func_type;
}

// Statements
std::unique_ptr<Statement> Parser::block_statement() {
    auto block = std::make_unique<BlockStatement>(previous().line);
    while (!check(TokenType::RightBrace) && !is_at_end()) {
        std::size_t before = current;
        auto stmt = statement();
        if (stmt) {
            block->statements.push_back(std::move(stmt));
        } else if (current == before) {
            // Avoid infinite loop when a statement cannot be parsed and no tokens
            // are consumed; advance to allow error recovery to proceed.
            advance();
        }
    }

    consume(TokenType::RightBrace, "Expected '}' after block");
    return block;
}

std::unique_ptr<Statement> Parser::if_statement() {
    consume(TokenType::LeftParen, "Expected '(' after 'if'");
    auto condition = expression();
    consume(TokenType::RightParen, "Expected ')' after if condition");

    auto then_stmt = statement();

    std::unique_ptr<Statement> else_stmt = nullptr;
    if (match(TokenType::Else)) {
        else_stmt = statement();
    }

    return std::make_unique<IfStatement>(
        std::move(condition),
        std::move(then_stmt),
        std::move(else_stmt),
        peek().line
    );
}

std::unique_ptr<Statement> Parser::while_statement() {
    consume(TokenType::LeftParen, "Expected '(' after 'while'");
    auto condition = expression();
    consume(TokenType::RightParen, "Expected ')' after while condition");

    auto body = statement();

    return std::make_unique<WhileStatement>(
        std::move(condition),
        std::move(body),
        peek().line
    );
}

std::unique_ptr<Statement> Parser::for_statement() {
    consume(TokenType::LeftParen, "Expected '(' after 'for'");

    std::unique_ptr<Statement> init = nullptr;
    if (!check(TokenType::Semicolon)) {
        init = statement();
        if (!init && !check(TokenType::Semicolon)) {
            init = std::make_unique<ExpressionStatement>(expression(), peek().line);
            consume(TokenType::Semicolon, "Expected ';' after for initializer");
        }
    } else {
        advance(); // Consume semicolon
    }

    std::unique_ptr<Expression> condition = nullptr;
    if (!check(TokenType::Semicolon)) {
        condition = expression();
    }
    consume(TokenType::Semicolon, "Expected ';' after for condition");

    std::unique_ptr<Expression> increment = nullptr;
    if (!check(TokenType::RightParen)) {
        increment = expression();
    }
    consume(TokenType::RightParen, "Expected ')' after for increment");

    auto body = statement();

    return std::make_unique<ForStatement>(
        std::move(init),
        std::move(condition),
        std::move(increment),
        std::move(body),
        peek().line
    );
}

std::unique_ptr<Statement> Parser::for_range_statement() {
    Token var = consume(TokenType::Identifier, "Expected variable name");
    consume(TokenType::Colon, "Expected ':' after variable name");

    std::unique_ptr<Type> var_type = nullptr;
    if (!check(TokenType::In)) {
        var_type = type();
    }

    consume(TokenType::In, "Expected 'in' in for-range statement");
    auto range = expression();

    auto body = statement();

    return std::make_unique<ForRangeStatement>(
        std::string(var.lexeme),
        std::move(var_type),
        std::move(range),
        std::move(body),
        var.line
    );
}

std::unique_ptr<Statement> Parser::switch_statement() {
    consume(TokenType::LeftParen, "Expected '(' after 'switch'");
    auto value = expression();
    consume(TokenType::RightParen, "Expected ')' after switch value");

    consume(TokenType::LeftBrace, "Expected '{' after switch");

    auto switch_stmt = std::make_unique<SwitchStatement>(std::move(value), value->line);

    while (!check(TokenType::RightBrace) && !is_at_end()) {
        if (match(TokenType::Case)) {
            auto case_expr = expression();
            consume(TokenType::Colon, "Expected ':' after case");
            auto case_stmt = statement();
            switch_stmt->cases.emplace_back(std::move(case_expr), std::move(case_stmt));
        } else if (match(TokenType::Default)) {
            consume(TokenType::Colon, "Expected ':' after default");
            switch_stmt->default_case = statement();
        } else {
            error_at_current("Expected 'case' or 'default' in switch");
        }
    }

    consume(TokenType::RightBrace, "Expected '}' after switch");

    return switch_stmt;
}

std::unique_ptr<Statement> Parser::inspect_statement() {
    consume(TokenType::LeftParen, "Expected '(' after 'inspect'");
    auto value = expression();
    consume(TokenType::RightParen, "Expected ')' after inspect value");

    auto inspect = std::make_unique<InspectStatement>(std::move(value), value->line);

    while (!is_at_end() && !check(TokenType::Else)) {
        auto arm = inspect_arm();
        inspect->arms.push_back(std::move(arm));
    }

    if (match(TokenType::Else)) {
        inspect->else_arm = statement();
    }

    return inspect;
}

std::unique_ptr<Statement> Parser::return_statement() {
    std::unique_ptr<Expression> value = nullptr;
    if (!check(TokenType::Semicolon)) {
        value = expression();
    }
    consume(TokenType::Semicolon, "Expected ';' after return");
    return std::make_unique<ReturnStatement>(std::move(value), previous().line);
}

std::unique_ptr<Statement> Parser::break_statement() {
    consume(TokenType::Semicolon, "Expected ';' after break");
    return std::make_unique<BreakStatement>(previous().line);
}

std::unique_ptr<Statement> Parser::continue_statement() {
    consume(TokenType::Semicolon, "Expected ';' after continue");
    return std::make_unique<ContinueStatement>(previous().line);
}

std::unique_ptr<Statement> Parser::try_statement() {
    auto try_block = block_statement();

    auto try_stmt = std::make_unique<TryStatement>(std::move(try_block), try_block->line);

    while (match(TokenType::Catch)) {
        consume(TokenType::LeftParen, "Expected '(' after catch");
        Token exception_type = consume(TokenType::Identifier, "Expected exception type");
        Token exception_name = consume(TokenType::Identifier, "Expected exception name");
        consume(TokenType::RightParen, "Expected ')' after catch exception");

        auto catch_block = block_statement();
        try_stmt->catch_blocks.emplace_back(
            std::string(exception_type.lexeme),
            std::move(catch_block)
        );
    }

    return try_stmt;
}

std::unique_ptr<Statement> Parser::throw_statement() {
    auto value = expression();
    consume(TokenType::Semicolon, "Expected ';' after throw");
    return std::make_unique<ThrowStatement>(std::move(value), previous().line);
}

std::unique_ptr<Statement> Parser::contract_statement() {
    Token contract_type = previous();
    ContractExpression::ContractKind kind;

    if (contract_type.type == TokenType::ContractPre) {
        kind = ContractExpression::ContractKind::Pre;
    } else if (contract_type.type == TokenType::ContractPost) {
        kind = ContractExpression::ContractKind::Post;
    } else {
        kind = ContractExpression::ContractKind::Assert;
    }

    consume(TokenType::Colon, "Expected ':' after contract keyword");
    auto condition = expression();

    std::optional<std::string> message;
    if (match(TokenType::Colon)) {
        Token msg_token = consume(TokenType::StringLiteral, "Expected string message");
        message = std::string(msg_token.lexeme);
    }

    consume(TokenType::Semicolon, "Expected ';' after contract");

    auto contract_expr = std::make_unique<ContractExpression>(kind, std::move(condition), contract_type.line);
    if (message) {
        contract_expr->message = *message;
    }

    return std::make_unique<ContractStatement>(std::move(contract_expr), contract_type.line);
}

std::unique_ptr<Statement> Parser::static_assert_statement() {
    consume(TokenType::LeftParen, "Expected '(' after static_assert");
    auto condition = expression();

    std::optional<std::string> message;
    if (match(TokenType::Comma)) {
        Token msg_token = consume(TokenType::StringLiteral, "Expected string message");
        message = std::string(msg_token.lexeme);
    }

    consume(TokenType::RightParen, "Expected ')' after static_assert");
    consume(TokenType::Semicolon, "Expected ';' after static_assert");

    auto assert_stmt = std::make_unique<StaticAssertStatement>(std::move(condition), condition->line);
    if (message) {
        assert_stmt->message = *message;
    }

    return assert_stmt;
}

// ============================================================================
// Concurrency Statements (Kotlin-style)
// ============================================================================

std::unique_ptr<Statement> Parser::coroutine_scope_statement() {
    // coroutineScope { ... }
    consume(TokenType::LeftBrace, "Expected '{' after 'coroutineScope'");
    auto body = block_statement();
    return std::make_unique<CoroutineScopeStatement>(std::move(body), previous().line);
}

std::unique_ptr<Statement> Parser::channel_declaration_statement() {
    // channel name: Type (capacity N)?;
    Token name = consume(TokenType::Identifier, "Expected channel name");
    consume(TokenType::Colon, "Expected ':' after channel name");
    auto elem_type = type();

    auto channel = std::make_unique<ChannelDeclarationStatement>(
        std::string(name.lexeme), std::move(elem_type), name.line);

    // Optional buffer size
    if (match(TokenType::LeftParen)) {
        Token capacity = consume(TokenType::IntegerLiteral, "Expected capacity");
        channel->buffer_size = std::stoull(std::string(capacity.lexeme));
        consume(TokenType::RightParen, "Expected ')' after capacity");
    }

    consume(TokenType::Semicolon, "Expected ';' after channel declaration");
    return channel;
}

std::unique_ptr<Statement> Parser::parallel_for_statement() {
    // parallel_for (var: lower..upper [step S] [mapping M]) { ... }
    consume(TokenType::LeftParen, "Expected '(' after 'parallel_for'");

    Token var = consume(TokenType::Identifier, "Expected loop variable");
    consume(TokenType::Colon, "Expected ':' after loop variable");

    auto lower = expression();
    consume(TokenType::DoubleDot, "Expected '..' for range");
    auto upper = expression();

    std::unique_ptr<Expression> step = nullptr;
    if (check(TokenType::Identifier) && peek().lexeme == "step") {
        advance();
        step = expression();
    }

    std::string mapping = "thread_id";
    if (check(TokenType::Identifier) && peek().lexeme == "mapping") {
        advance();
        Token map_token = consume(TokenType::StringLiteral, "Expected mapping string");
        std::string_view lexeme = map_token.lexeme;
        if (lexeme.size() >= 2 && lexeme.front() == '"' && lexeme.back() == '"') {
            lexeme = lexeme.substr(1, lexeme.size() - 2);
        }
        mapping = std::string(lexeme);
    }

    consume(TokenType::RightParen, "Expected ')' after parallel_for header");
    consume(TokenType::LeftBrace, "Expected '{' for parallel_for body");
    auto body = block_statement();

    return std::make_unique<ParallelForStatement>(
        std::string(var.lexeme), std::move(lower), std::move(upper),
        std::move(step), std::move(mapping), std::move(body), var.line);
}

// Expressions (precedence climbing)
std::unique_ptr<Expression> Parser::assignment_expression() {
    auto expr = ternary_expression();

    if (match({TokenType::Equal, TokenType::PlusEqual, TokenType::MinusEqual,
               TokenType::AsteriskEqual, TokenType::SlashEqual, TokenType::PercentEqual,
               TokenType::LeftShiftEqual, TokenType::RightShiftEqual,
               TokenType::AmpersandEqual, TokenType::PipeEqual, TokenType::CaretEqual})) {
        Token op = previous();
        auto value = assignment_expression();

        expr = std::make_unique<BinaryExpression>(
            std::move(expr),
            op.type,
            std::move(value),
            expr->line
        );
    }

    return expr;
}

std::unique_ptr<Expression> Parser::ternary_expression() {
    auto expr = logical_or_expression();

    if (match(TokenType::Question)) {
        auto then_expr = expression();
        consume(TokenType::Colon, "Expected ':' in ternary expression");
        auto else_expr = ternary_expression();

        expr = std::make_unique<TernaryExpression>(
            std::move(expr),
            std::move(then_expr),
            std::move(else_expr),
            expr->line
        );
    }

    return expr;
}

std::unique_ptr<Expression> Parser::logical_or_expression() {
    auto expr = logical_and_expression();

    while (match(TokenType::DoublePipe)) {
        Token op = previous();
        auto right = logical_and_expression();
        expr = std::make_unique<BinaryExpression>(
            std::move(expr),
            op.type,
            std::move(right),
            expr->line
        );
    }

    return expr;
}

std::unique_ptr<Expression> Parser::logical_and_expression() {
    auto expr = bitwise_or_expression();

    while (match(TokenType::DoubleAmpersand)) {
        Token op = previous();
        auto right = bitwise_or_expression();
        expr = std::make_unique<BinaryExpression>(
            std::move(expr),
            op.type,
            std::move(right),
            expr->line
        );
    }

    return expr;
}

std::unique_ptr<Expression> Parser::bitwise_and_expression() {
    auto expr = equality_expression();
    while (match(TokenType::Ampersand)) {
        Token op = previous();
        auto right = equality_expression();
        expr = std::make_unique<BinaryExpression>(std::move(expr), op.type, std::move(right), expr->line);
    }
    return expr;
}

std::unique_ptr<Expression> Parser::bitwise_xor_expression() {
    auto expr = bitwise_and_expression();
    while (match(TokenType::Caret)) {
        Token op = previous();
        auto right = bitwise_and_expression();
        expr = std::make_unique<BinaryExpression>(std::move(expr), op.type, std::move(right), expr->line);
    }
    return expr;
}

std::unique_ptr<Expression> Parser::bitwise_or_expression() {
    auto expr = bitwise_xor_expression();
    while (match(TokenType::Pipe)) {
        Token op = previous();
        auto right = bitwise_xor_expression();
        expr = std::make_unique<BinaryExpression>(std::move(expr), op.type, std::move(right), expr->line);
    }
    return expr;
}

std::unique_ptr<Expression> Parser::equality_expression() {
    auto expr = comparison_expression();

    while (match({TokenType::DoubleEqual, TokenType::NotEqual})) {
        Token op = previous();
        auto right = comparison_expression();
        expr = std::make_unique<BinaryExpression>(
            std::move(expr),
            op.type,
            std::move(right),
            expr->line
        );
    }

    return expr;
}

std::unique_ptr<Expression> Parser::comparison_expression() {
    auto expr = range_expression();

    while (match({TokenType::LessThan, TokenType::GreaterThan,
                  TokenType::LessThanOrEqual, TokenType::GreaterThanOrEqual})) {
        Token op = previous();
        auto right = range_expression();
        expr = std::make_unique<BinaryExpression>(
            std::move(expr),
            op.type,
            std::move(right),
            expr->line
        );
    }

    return expr;
}

std::unique_ptr<Expression> Parser::range_expression() {
    auto expr = shift_expression();

    while (match({TokenType::RangeInclusive, TokenType::RangeExclusive})) {
        Token op = previous();
        auto right = addition_expression();

        bool inclusive = (op.type == TokenType::RangeInclusive);
        expr = std::make_unique<RangeExpression>(
            std::move(expr),
            std::move(right),
            inclusive,
            expr->line
        );
    }

    return expr;
}

std::unique_ptr<Expression> Parser::shift_expression() {
    auto expr = addition_expression();

    while (match({TokenType::LeftShift, TokenType::RightShift})) {
        Token op = previous();
        auto right = addition_expression();
        expr = std::make_unique<BinaryExpression>(
            std::move(expr), op.type, std::move(right), expr->line
        );
    }
    return expr;
}

std::unique_ptr<Expression> Parser::addition_expression() {
    auto expr = multiplication_expression();

    while (match({TokenType::Plus, TokenType::Minus})) {
        Token op = previous();
        auto right = multiplication_expression();
        expr = std::make_unique<BinaryExpression>(
            std::move(expr),
            op.type,
            std::move(right),
            expr->line
        );
    }

    return expr;
}

std::unique_ptr<Expression> Parser::multiplication_expression() {
    auto expr = prefix_expression();

    while (match({TokenType::Asterisk, TokenType::Slash, TokenType::Percent})) {
        Token op = previous();
        auto right = prefix_expression();
        expr = std::make_unique<BinaryExpression>(
            std::move(expr),
            op.type,
            std::move(right),
            expr->line
        );
    }

    return expr;
}

std::unique_ptr<Expression> Parser::prefix_expression() {
    // Concurrency prefix expressions
    if (match(TokenType::Await)) {
        return await_expression();
    }
    if (match(TokenType::Launch)) {
        return spawn_expression();
    }
    if (match(TokenType::Select)) {
        return select_expression();
    }

    if (match({TokenType::Minus, TokenType::Exclamation, TokenType::Tilde, TokenType::PlusPlus,
               TokenType::MinusMinus, TokenType::Ampersand, TokenType::Asterisk})) {
        Token op = previous();
        auto operand = prefix_expression();
        return std::make_unique<UnaryExpression>(
            op.type,
            std::move(operand),
            op.line,
            false // prefix
        );
    }

    return postfix_expression();
}

std::unique_ptr<Expression> Parser::postfix_expression() {
    auto expr = primary_expression();

    while (true) {
        if (match(TokenType::LeftParen)) {
            expr = call_expression(std::move(expr));
        } else if (match(TokenType::Dot)) {
            expr = member_access_expression(std::move(expr));
        } else if (match(TokenType::LeftBracket)) {
            expr = subscript_expression(std::move(expr));
        } else if (check(TokenType::Asterisk) || check(TokenType::Ampersand)) {
            // Cpp2 allows postfix deref/address-of: `p*`, `x&`.
            // Disambiguate from binary '*' by only treating it as postfix when
            // the following token cannot start a new expression.
            TokenType next = TokenType::EndOfFile;
            if (current + 1 < tokens.size()) {
                next = tokens[current + 1].type;
            }
            bool looks_like_postfix =
                next == TokenType::Semicolon || next == TokenType::Comma ||
                next == TokenType::RightParen || next == TokenType::RightBracket ||
                next == TokenType::RightBrace || next == TokenType::Dot ||
                next == TokenType::PlusPlus || next == TokenType::MinusMinus;

            if (!looks_like_postfix) {
                break;
            }

            advance();
            Token op = previous();
            expr = std::make_unique<UnaryExpression>(
                op.type,
                std::move(expr),
                op.line,
                true // postfix
            );
        } else if (match({TokenType::PlusPlus, TokenType::MinusMinus})) {
            Token op = previous();
            expr = std::make_unique<UnaryExpression>(
                op.type,
                std::move(expr),
                op.line,
                true // postfix
            );
        } else {
            break;
        }
    }

    return expr;
}

std::unique_ptr<Expression> Parser::primary_expression() {
    if (match(TokenType::True)) {
        return std::make_unique<LiteralExpression>(true, previous().line);
    }
    if (match(TokenType::False)) {
        return std::make_unique<LiteralExpression>(false, previous().line);
    }
    if (match(TokenType::IntegerLiteral)) {
        return std::make_unique<LiteralExpression>(
            std::stoll(std::string(previous().lexeme)),
            previous().line
        );
    }
    if (match(TokenType::FloatLiteral)) {
        return std::make_unique<LiteralExpression>(
            std::stod(std::string(previous().lexeme)),
            previous().line
        );
    }
    if (match(TokenType::StringLiteral)) {
        // Strip the surrounding quotes from the string literal lexeme
        std::string_view lexeme = previous().lexeme;
        if (lexeme.size() >= 2 && lexeme.front() == '"' && lexeme.back() == '"') {
            lexeme = lexeme.substr(1, lexeme.size() - 2);
        }
        return std::make_unique<LiteralExpression>(
            std::string(lexeme),
            previous().line
        );
    }
    if (match(TokenType::CharacterLiteral)) {
        return std::make_unique<LiteralExpression>(
            previous().lexeme[0],
            previous().line
        );
    }
    if (match(TokenType::Identifier)) {
        std::string qname(previous().lexeme);
        std::size_t line = previous().line;
        // Support scope resolution :: chains
        while (match(TokenType::DoubleColon)) {
            Token next = consume(TokenType::Identifier, "Expected identifier after '::'");
            qname += "::";
            qname += std::string(next.lexeme);
            line = next.line;
        }
        return std::make_unique<IdentifierExpression>(
            std::move(qname),
            line
        );
    }
    if (match(TokenType::LeftParen)) {
        auto expr = expression();
        consume(TokenType::RightParen, "Expected ')' after expression");
        return expr;
    }
    if (match(TokenType::LeftBracket)) {
        return list_literal();
    }
    if (match(TokenType::LeftBrace)) {
        return struct_initializer();
    }
    if (match(TokenType::Is)) {
        return is_expression();
    }
    if (match(TokenType::As)) {
        return as_expression();
    }
    if (match(TokenType::At)) {
        return metafunction_call();
    }
    if (match(TokenType::Underscore)) {
        return std::make_unique<IdentifierExpression>("_", previous().line);
    }

    error_at_current("Expected expression");
    return nullptr;
}

// Expression helpers
std::unique_ptr<Expression> Parser::call_expression(std::unique_ptr<Expression> callee) {
    auto call = std::make_unique<CallExpression>(std::move(callee), callee->line);

    if (!check(TokenType::RightParen)) {
        do {
            call->args.push_back(expression());
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RightParen, "Expected ')' after arguments");
    return call;
}

std::unique_ptr<Expression> Parser::member_access_expression(std::unique_ptr<Expression> object) {
    Token member = consume(TokenType::Identifier, "Expected member name after '.'");
    return std::make_unique<MemberAccessExpression>(
        std::move(object),
        std::string(member.lexeme),
        member.line
    );
}

std::unique_ptr<Expression> Parser::subscript_expression(std::unique_ptr<Expression> array) {
    auto index = expression();
    consume(TokenType::RightBracket, "Expected ']' after subscript");
    return std::make_unique<SubscriptExpression>(
        std::move(array),
        std::move(index),
        array->line
    );
}

std::unique_ptr<Expression> Parser::list_literal() {
    auto list = std::make_unique<ListExpression>(previous().line);

    if (!check(TokenType::RightBracket)) {
        do {
            list->elements.push_back(expression());
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RightBracket, "Expected ']' after list");
    return list;
}

std::unique_ptr<Expression> Parser::struct_initializer() {
    Token type_name = consume(TokenType::Identifier, "Expected type name for struct initializer");
    auto type_expr = std::make_unique<IdentifierExpression>(std::string(type_name.lexeme), type_name.line);

    auto init = std::make_unique<StructInitializerExpression>(std::move(type_expr), type_name.line);

    if (!check(TokenType::RightBrace)) {
        do {
            Token field_name = consume(TokenType::Identifier, "Expected field name");
            consume(TokenType::Colon, "Expected ':' after field name");
            auto value = expression();
            init->fields.emplace_back(std::string(field_name.lexeme), std::move(value));
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RightBrace, "Expected '}' after struct initializer");
    return init;
}

std::unique_ptr<Expression> Parser::lambda_expression() {
    auto lambda = std::make_unique<LambdaExpression>(peek().line);

    consume(TokenType::LeftParen, "Expected '(' for lambda");

    if (!check(TokenType::RightParen)) {
        do {
            LambdaExpression::Parameter param;

            // Parse qualifiers before parameter name
            param.qualifiers = parse_parameter_qualifiers();

            Token name = consume(TokenType::Identifier, "Expected parameter name");
            param.name = std::string(name.lexeme);

            if (match(TokenType::Colon)) {
                param.type = type();
            }

            if (match(TokenType::Equal)) {
                param.default_value = expression();
            }

            lambda->parameters.push_back(std::move(param));
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RightParen, "Expected ')' after lambda parameters");

    if (match(TokenType::Arrow)) {
        lambda->return_type = type();
    }

    consume(TokenType::LeftBrace, "Expected '{' for lambda body");

    while (!check(TokenType::RightBrace) && !is_at_end()) {
        auto stmt = statement();
        if (stmt) {
            lambda->body.push_back(std::move(stmt));
        }
    }

    consume(TokenType::RightBrace, "Expected '}' after lambda body");
    return lambda;
}

std::unique_ptr<Expression> Parser::metafunction_call() {
    Token name = consume(TokenType::Identifier, "Expected metafunction name after '@'");

    auto call = std::make_unique<MetafunctionCallExpression>(std::string(name.lexeme), name.line);

    if (match(TokenType::LeftParen)) {
        if (!check(TokenType::RightParen)) {
            do {
                call->args.push_back(expression());
            } while (match(TokenType::Comma));
        }
        consume(TokenType::RightParen, "Expected ')' after metafunction arguments");
    }

    return call;
}

std::unique_ptr<Expression> Parser::is_expression() {
    auto expr = expression();
    consume(TokenType::Is, "Expected 'is' in type test expression");
    auto type_expr = type();

    return std::make_unique<IsExpression>(
        std::move(expr),
        std::move(type_expr),
        expr->line
    );
}

std::unique_ptr<Expression> Parser::as_expression() {
    auto expr = expression();
    consume(TokenType::As, "Expected 'as' in cast expression");
    auto type_expr = type();

    return std::make_unique<AsExpression>(
        std::move(expr),
        std::move(type_expr),
        expr->line
    );
}

// ============================================================================
// Concurrency Expressions (Kotlin-style)
// ============================================================================

std::unique_ptr<Expression> Parser::await_expression() {
    // await expr
    auto value = prefix_expression();
    return std::make_unique<AwaitExpression>(std::move(value), previous().line);
}

std::unique_ptr<Expression> Parser::spawn_expression() {
    // launch expr  (fire-and-forget coroutine)
    auto task = prefix_expression();
    return std::make_unique<SpawnExpression>(std::move(task), previous().line);
}

std::unique_ptr<Expression> Parser::channel_send_expression() {
    // channel <- value
    // Note: This is called from member access when we see a channel send operator
    // The channel identifier should be previous()
    Token channel_name = previous();
    auto value = expression();
    return std::make_unique<ChannelSendExpression>(
        std::string(channel_name.lexeme), std::move(value), channel_name.line);
}

std::unique_ptr<Expression> Parser::channel_recv_expression() {
    // <- channel
    Token channel_name = consume(TokenType::Identifier, "Expected channel name after '<-'");
    return std::make_unique<ChannelRecvExpression>(
        std::string(channel_name.lexeme), channel_name.line);
}

std::unique_ptr<Expression> Parser::select_expression() {
    // select { onSend(ch, v) { ... } onRecv(ch) { v -> ... } }
    consume(TokenType::LeftBrace, "Expected '{' after 'select'");

    std::vector<ChannelSelectExpression::SelectCase> cases;
    std::unique_ptr<Expression> default_case = nullptr;

    while (!check(TokenType::RightBrace) && !is_at_end()) {
        if (check(TokenType::Identifier) && peek().lexeme == "onSend") {
            advance();
            consume(TokenType::LeftParen, "Expected '(' after 'onSend'");
            Token ch = consume(TokenType::Identifier, "Expected channel name");
            consume(TokenType::Comma, "Expected ',' after channel name");
            auto value = expression();
            consume(TokenType::RightParen, "Expected ')' after onSend arguments");
            consume(TokenType::LeftBrace, "Expected '{' for onSend body");
            auto action = expression();
            consume(TokenType::RightBrace, "Expected '}' after onSend body");

            ChannelSelectExpression::SelectCase case_item;
            case_item.channel = std::string(ch.lexeme);
            case_item.kind = ChannelSelectExpression::SelectCase::Kind::Send;
            case_item.value = std::move(value);
            case_item.action = std::move(action);
            cases.push_back(std::move(case_item));
        } else if (check(TokenType::Identifier) && peek().lexeme == "onRecv") {
            advance();
            consume(TokenType::LeftParen, "Expected '(' after 'onRecv'");
            Token ch = consume(TokenType::Identifier, "Expected channel name");
            consume(TokenType::RightParen, "Expected ')' after onRecv arguments");
            consume(TokenType::LeftBrace, "Expected '{' for onRecv body");
            auto action = expression();
            consume(TokenType::RightBrace, "Expected '}' after onRecv body");

            ChannelSelectExpression::SelectCase case_item;
            case_item.channel = std::string(ch.lexeme);
            case_item.kind = ChannelSelectExpression::SelectCase::Kind::Recv;
            case_item.action = std::move(action);
            cases.push_back(std::move(case_item));
        } else if (check(TokenType::Default)) {
            advance();
            consume(TokenType::Arrow, "Expected '=>' after 'default'");
            default_case = expression();
        } else {
            error_at_current("Expected 'onSend', 'onRecv', or 'default' in select");
            break;
        }
    }

    consume(TokenType::RightBrace, "Expected '}' after select");

    auto select = std::make_unique<ChannelSelectExpression>(std::move(cases), previous().line);
    select->default_case = std::move(default_case);
    return select;
}

// Pattern matching for inspect
std::pair<InspectStatement::Pattern, std::unique_ptr<Statement>> Parser::inspect_arm() {
    auto pat = pattern();
    consume(TokenType::Arrow, "Expected '=>' after inspect pattern");
    auto stmt = statement();
    return {std::move(pat), std::move(stmt)};
}

InspectStatement::Pattern Parser::pattern() {
    if (match(TokenType::Underscore)) {
        InspectStatement::Pattern pat;
        pat.kind = InspectStatement::Pattern::Kind::Wildcard;
        return pat;
    }

    if (match(TokenType::Identifier)) {
        Token name = previous();
        if (match(TokenType::Colon)) {
            // Type pattern
            InspectStatement::Pattern pat;
            pat.kind = InspectStatement::Pattern::Kind::Type;
            pat.type = type();
            return pat;
        } else {
            // Binding pattern
            InspectStatement::Pattern pat;
            pat.kind = InspectStatement::Pattern::Kind::Binding;
            pat.binding_name = std::string(name.lexeme);
            return pat;
        }
    }

    // Value pattern
    auto value = expression();
    InspectStatement::Pattern pat;
    pat.kind = InspectStatement::Pattern::Kind::Value;
    pat.value = std::move(value);
    return pat;
}

// Contract handling
std::vector<std::unique_ptr<ContractExpression>> Parser::parse_contracts() {
    std::vector<std::unique_ptr<ContractExpression>> contracts;

    while (match({TokenType::ContractPre, TokenType::ContractPost, TokenType::ContractAssert})) {
        Token contract_type = previous();
        ContractExpression::ContractKind kind;

        if (contract_type.type == TokenType::ContractPre) {
            kind = ContractExpression::ContractKind::Pre;
        } else if (contract_type.type == TokenType::ContractPost) {
            kind = ContractExpression::ContractKind::Post;
        } else {
            kind = ContractExpression::ContractKind::Assert;
        }

        consume(TokenType::Colon, "Expected ':' after contract keyword");
        auto condition = expression();

        auto contract = std::make_unique<ContractExpression>(kind, std::move(condition), contract_type.line);

        if (match(TokenType::Colon)) {
            Token msg = consume(TokenType::StringLiteral, "Expected string message");
            contract->message = std::string(msg.lexeme);
        }

        contracts.push_back(std::move(contract));
    }

    return contracts;
}

// Parameter qualifier parsing (Cpp2-specific)
// Parses: inout, out, move, forward, virtual, override
std::vector<ParameterQualifier> Parser::parse_parameter_qualifiers() {
    std::vector<ParameterQualifier> qualifiers;

    while (true) {
        if (match(TokenType::Inout)) {
            qualifiers.push_back(ParameterQualifier::InOut);
        } else if (match(TokenType::Out)) {
            qualifiers.push_back(ParameterQualifier::Out);
        } else if (match(TokenType::Move)) {
            qualifiers.push_back(ParameterQualifier::Move);
        } else if (match(TokenType::Forward)) {
            qualifiers.push_back(ParameterQualifier::Forward);
        } else if (match(TokenType::Virtual)) {
            qualifiers.push_back(ParameterQualifier::Virtual);
        } else if (match(TokenType::Override)) {
            qualifiers.push_back(ParameterQualifier::Override);
        } else {
            break;
        }
    }

    return qualifiers;
}

// Template handling
std::vector<std::string> Parser::template_parameters() {
    std::vector<std::string> params;

    do {
        Token param = consume(TokenType::Identifier, "Expected template parameter");
        params.push_back(std::string(param.lexeme));
    } while (match(TokenType::Comma));

    return params;
}

bool Parser::is_template_start() {
    return check(TokenType::Template) ||
           (check(TokenType::Identifier) && peek().lexeme == "template");
}

// Error handling
void Parser::error(const Token& token, const char* message) {
    if (token.type == TokenType::EndOfFile) {
        error_at(token, " at end");
    } else {
        error_at(token, std::format(" at '{}'", token.lexeme).c_str());
    }
}

void Parser::error_at(const Token& token, const char* message) {
    std::string msg = message ? message : std::string();
    // Suppress duplicate error printing for the same token/position
    if (token.position == last_error_position && msg == last_error_text) {
        panic_mode = true;
        return;
    }

    last_error_position = token.position;
    last_error_text = msg;

    std::cerr << std::format("[line {}] Error", token.line);
    if (message) {
        std::cerr << ": " << message;
    }
    std::cerr << std::endl;
    panic_mode = true;
}

void Parser::error_at_current(const char* message) {
    error_at(peek(), message);
}

} // namespace cpp2_transpiler