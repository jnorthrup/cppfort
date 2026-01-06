#include "parser.hpp"
#include "markdown_hash.hpp"
#include <stdexcept>
#include <iostream>
#include <format>

namespace cpp2_transpiler {

// Helper function to convert a Type* to a string representation
static std::string type_to_string(const Type* t) {
    if (!t) return "";
    
    std::string result;
    if (t->is_const) result += "const ";
    
    switch (t->kind) {
        case Type::Kind::Builtin:
        case Type::Kind::UserDefined:
            result += t->name;
            break;
        case Type::Kind::Pointer:
            result += type_to_string(t->pointee.get()) + "*";
            break;
        case Type::Kind::Reference:
            result += type_to_string(t->pointee.get()) + "&";
            break;
        case Type::Kind::Auto:
            result += "auto";
            break;
        case Type::Kind::Template:
            result += t->name + "<";
            for (size_t i = 0; i < t->template_args.size(); ++i) {
                if (i > 0) result += ", ";
                result += type_to_string(t->template_args[i].get());
            }
            result += ">";
            break;
        default:
            result += t->name;
            break;
    }
    
    return result;
}

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

// Check if the current token can be used as an identifier (for contextual keywords)
bool Parser::is_identifier_like() const {
    if (check(TokenType::Identifier)) return true;
    // Underscore can be used as discard/placeholder in declarations
    if (check(TokenType::Underscore)) return true;
    // Keywords that can also be used as identifiers in C++1 contexts
    if (check(TokenType::In) || check(TokenType::Copy) || check(TokenType::Move) ||
        check(TokenType::Forward) || check(TokenType::Out) || check(TokenType::Inout) ||
        check(TokenType::InRef) || check(TokenType::ForwardRef)) {
        // These are Cpp2 parameter keywords but valid C++1 identifiers
        return true;
    }
    // 'func', 'type', 'namespace' can be used as identifiers when followed by ':'
    // in unified declaration syntax (e.g., func: () = { }) or := in type-deduced syntax
    if (check(TokenType::Func) || check(TokenType::Type) || check(TokenType::Namespace)) {
        std::size_t lookahead = current + 1;
        if (lookahead < tokens.size()) {
            TokenType next_tok = tokens[lookahead].type;
            if (next_tok == TokenType::Colon || next_tok == TokenType::ColonEqual) {
                return true;
            }
        }
    }
    // 'base' can be used as identifier (type names, variable names) even though
    // it's a keyword for base class access
    if (check(TokenType::Base)) {
        std::size_t lookahead = current + 1;
        if (lookahead < tokens.size()) {
            TokenType next_tok = tokens[lookahead].type;
            if (next_tok == TokenType::Colon || next_tok == TokenType::ColonEqual ||
                next_tok == TokenType::Asterisk || next_tok == TokenType::Ampersand ||
                next_tok == TokenType::Dot || next_tok == TokenType::Semicolon ||
                next_tok == TokenType::Comma || next_tok == TokenType::RightParen ||
                next_tok == TokenType::RightBracket) {
                return true;
            }
        }
    }
    // 'next' can be used as identifier in many contexts (e.g., variable names)
    if (check(TokenType::Next)) {
        // Look ahead to see if it's being used as a variable name
        std::size_t lookahead = current + 1;
        if (lookahead < tokens.size()) {
            TokenType next_tok = tokens[lookahead].type;
            if (next_tok == TokenType::Colon || next_tok == TokenType::ColonEqual ||
                next_tok == TokenType::Asterisk || next_tok == TokenType::Ampersand ||
                next_tok == TokenType::Dot || next_tok == TokenType::PlusPlus ||
                next_tok == TokenType::MinusMinus || next_tok == TokenType::LeftBracket ||
                next_tok == TokenType::LeftParen || next_tok == TokenType::Equal ||
                next_tok == TokenType::NotEqual || next_tok == TokenType::DoubleEqual ||
                next_tok == TokenType::Semicolon || next_tok == TokenType::Comma ||
                next_tok == TokenType::RightParen || next_tok == TokenType::RightBracket) {
                return true;
            }
        }
    }
    return false;
}

// Get the lexeme for identifier-like tokens
std::string_view Parser::get_identifier_lexeme() const {
    return peek().lexeme;
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

// Special helper to consume '>' that also handles '>>' (for C++11 nested template syntax)
// Uses pending_gt to track when we've conceptually consumed the first '>' of a '>>'
bool Parser::consume_template_close() {
    if (pending_gt) {
        // We previously saw '>>' and consumed the first '>'
        // Now we consume the second '>' by actually advancing past '>>'
        pending_gt = false;
        advance();  // consume the actual '>>' token
        return true;
    }
    if (check(TokenType::GreaterThan)) {
        advance();
        return true;
    }
    if (check(TokenType::RightShift)) {
        // '>>' is being used as two '>' tokens for nested templates
        // Don't advance - just mark that we've used the first '>'
        pending_gt = true;
        return true;
    }
    return false;
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
        std::size_t iterations = 0;
        constexpr std::size_t MAX_ITERATIONS = 1000; // Lower limit for faster debugging
        std::size_t stall_count = 0;
        std::size_t last_current = current;
        while (!is_at_end() && iterations < MAX_ITERATIONS) {
            iterations++;
            if (previous().type == TokenType::Semicolon) {
                panic_mode = false;
                break;
            }
            if (peek().type == TokenType::Func || peek().type == TokenType::Type ||
                peek().type == TokenType::Namespace || peek().type == TokenType::Let ||
                peek().type == TokenType::Const) {
                // Advance past the recovery keyword to avoid re-parsing the same error
                advance();
                panic_mode = false;
                break;
            }
            advance();
            // Check if we're stuck (not advancing)
            if (current == last_current) {
                stall_count++;
                if (stall_count > 10) {
                    std::cerr << "Error: Parser stuck at token " << current << " of " << tokens.size() << ", breaking out\n";
                    break;
                }
            } else {
                stall_count = 0;
                last_current = current;
            }
        }
        // If we hit the iteration limit, we're likely stuck - force exit
        if (iterations >= MAX_ITERATIONS) {
            std::cerr << "Error recovery exceeded iteration limit (" << iterations << ") - forcing exit\n";
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
        static int decl_count = 0;
        decl_count++;
        if (decl_count > 1000) {
            std::cerr << "Error: declaration() called " << decl_count << " times - forcing advance to break loop\n";
            // Force advance to break the infinite loop
            if (!is_at_end()) {
                advance();
                decl_count = 0; // Reset counter after recovery
            }
            return nullptr;
        }
        // Collect any markdown blocks before this declaration
        collect_markdown_blocks();

        // Pass through preprocessor directives (#include, #define, etc.)
        if (check(TokenType::Hash)) {
            const Token& hash_tok = advance();
            // Create a passthrough declaration with the full directive
            auto decl = std::make_unique<Cpp1PassthroughDeclaration>(
                std::string(hash_tok.lexeme), hash_tok.line);
            return decl;
        }

        // Handle access specifiers in type bodies: public/private/protected
        // Cpp2 syntax: public x: int = 0; or public f: () = {...}
        bool has_access_specifier = false;
        if (match(TokenType::Public) || match(TokenType::Private) || match(TokenType::Protected)) {
            // Access specifier consumed, continue to parse the declaration
            has_access_specifier = true;
            // TODO: Store the access specifier in the declaration
        }

        // Handle decorators starting a declaration (e.g. @value type Name ...)
        if (check(TokenType::At)) {
            std::size_t saved = current;
            std::vector<std::string> decorators;
            while (match(TokenType::At)) {
                if (check(TokenType::Identifier)) {
                    decorators.push_back(std::string(advance().lexeme));
                } else {
                    // Handle keywords used as decorators
                    decorators.push_back(std::string(advance().lexeme));
                }
            }
            
            if (match(TokenType::Type)) {
                auto decl = type_declaration(std::move(decorators));
                attach_markdown_blocks(decl.get());
                return decl;
            }
            
            // If not a type declaration, backtrack (or handle other decorated declarations)
            current = saved;
        }

        // Cpp2 unified syntax: identifier: type = initializer
        // Cpp2 type-deduced syntax: identifier := initializer
        if (is_identifier_like()) {
            std::size_t saved = current;
            advance(); // consume identifier
            if (check(TokenType::ColonEqual)) {
                // Type-deduced variable: name := initializer
                current = saved; // backtrack
                advance(); // consume identifier (makes it previous())
                consume(TokenType::ColonEqual, "Expected ':=' after identifier");
                auto decl = variable_declaration();
                attach_markdown_blocks(decl.get());
                return decl;
            } else if (check(TokenType::Colon)) {
                current = saved; // backtrack
                // Could be function or variable
                advance(); // consume identifier (makes it previous())
                consume(TokenType::Colon, "Expected ':' after identifier");

                // Check what kind of declaration follows
                if (check(TokenType::LeftParen) || check(TokenType::LessThan)) {
                    // Function: name: (params) or name: <T> (params)
                    // Or type with template params: name: <T> type = {...}
                    // Or variable with template params: name: <T> type = value
                    auto decl = function_declaration();
                    if (decl) {
                        attach_markdown_blocks(decl.get());
                        return decl;
                    }
                    // If function_declaration returned nullptr, check what kind of declaration this is
                    // After attempting to parse as function, check the current position
                    if (check(TokenType::At) || check(TokenType::Type) || check(TokenType::Concept)) {
                        // Type declaration with template params: name: <T> type = {...}
                        // or name: <T> @decorator type = {...}
                        // or concept: name: <T> concept = expr
                        // The template params were already consumed by function_declaration
                        // But we need to re-parse from the start because type_declaration
                        // needs to parse the template params itself
                        current = saved; // backtrack to start
                        advance(); // consume identifier
                        consume(TokenType::Colon, "Expected ':' after identifier");
                        auto type_decl = type_declaration();
                        attach_markdown_blocks(type_decl.get());
                        return type_decl;
                    }

                    // If function_declaration returned nullptr, it might be a variable
                    // with template parameters. Fall through to variable parsing.
                    current = saved; // backtrack to try variable declaration
                    advance(); // consume identifier
                    consume(TokenType::Colon, "Expected ':' after identifier");
                    auto var_decl = variable_declaration();
                    attach_markdown_blocks(var_decl.get());
                    return var_decl;
                } else if (check(TokenType::At) || check(TokenType::Type) ||
                           check(TokenType::Concept) || is_type_qualifier()) {
                    // Check for type alias: name: type == underlying_type;
                    if (check(TokenType::Type)) {
                        std::size_t type_saved = current;
                        advance();  // consume 'type'
                        if (check(TokenType::DoubleEqual)) {
                            // This is a type alias declaration
                            advance();  // consume '=='
                            Token name = tokens[saved];
                            auto underlying = type();  // Parse the underlying type
                            consume(TokenType::Semicolon, "Expected ';' after type alias");

                            auto type_decl = std::make_unique<TypeDeclaration>(
                                std::string(name.lexeme),
                                TypeDeclaration::TypeKind::Alias,
                                name.line
                            );
                            type_decl->underlying_type = std::move(underlying);
                            attach_markdown_blocks(type_decl.get());
                            return type_decl;
                        }
                        // Not a type alias, backtrack
                        current = type_saved;
                    }

                    // Type with decorators: name: @value @ordered type = {...}
                    // Or type keyword: name: type = {...}
                    // Or type with qualifiers: name: final type = {...}
                    // Or concept: name: concept = expr
                    auto decl = type_declaration();
                    attach_markdown_blocks(decl.get());
                    return decl;
                } else if (check(TokenType::Namespace)) {
                    // Namespace: name: namespace = {...}
                    auto decl = namespace_declaration();
                    attach_markdown_blocks(decl.get());
                    return decl;
                } else {
                    // Variable: name: type = value
                    auto decl = variable_declaration();
                    attach_markdown_blocks(decl.get());
                    return decl;
                }
            }
            current = saved; // backtrack if no colon or colon-equal
        }

        if (match({TokenType::Let, TokenType::Const})) {
            auto decl = variable_declaration();
            attach_markdown_blocks(decl.get());
            return decl;
        }
        if (match(TokenType::Func)) {
            auto decl = function_declaration();
            attach_markdown_blocks(decl.get());
            return decl;
        }
        if (match(TokenType::Type)) {
            auto decl = type_declaration();
            attach_markdown_blocks(decl.get());
            return decl;
        }
        if (match(TokenType::Namespace)) {
            auto decl = namespace_declaration();
            attach_markdown_blocks(decl.get());
            return decl;
        }
        if (match(TokenType::Operator)) {
            auto decl = operator_declaration();
            attach_markdown_blocks(decl.get());
            return decl;
        }
        if (match(TokenType::Import)) {
            auto decl = import_declaration();
            attach_markdown_blocks(decl.get());
            return decl;
        }
        if (match(TokenType::Using)) {
            auto decl = using_declaration();
            attach_markdown_blocks(decl.get());
            return decl;
        }

        // C++1 template passthrough: template<...> struct/class/union/namespace
        // Must check BEFORE Cpp2 template handling
        if (is_cpp1_template_start()) {
            auto decl = cpp1_passthrough_declaration();
            attach_markdown_blocks(decl.get());
            return decl;
        }

        if (is_template_start()) {
            auto decl = template_declaration();
            attach_markdown_blocks(decl.get());
            return decl;
        }

        // C++1 passthrough detection: detect C++1 function syntax
        // Patterns: "auto name(...) -> type {" or "type name(...)"
        // Only trigger if NOT preceded by Cpp2 ':' marker
        if (check_cpp1_function_syntax()) {
            auto decl = cpp1_passthrough_declaration();
            if (decl) {
                attach_markdown_blocks(decl.get());
                return decl;
            }
        }
        
        // C++1 passthrough detection: detect constexpr/inline variable declarations
        // Pattern: "constexpr auto name = ...;" or "inline constexpr auto name = ...;"
        if (check_cpp1_constexpr_syntax()) {
            auto decl = cpp1_passthrough_declaration();
            if (decl) {
                attach_markdown_blocks(decl.get());
                return decl;
            }
        }

        // C++1 passthrough detection: detect C++1 struct/class/union/enum syntax
        // Patterns: "struct Name {" or "class Name :" (vs Cpp2 "Name: struct = {")
        if (check_cpp1_struct_syntax()) {
            auto decl = cpp1_passthrough_declaration(true);  // Pass true for struct types
            if (decl) {
                attach_markdown_blocks(decl.get());
                return decl;
            }
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
        if (match(TokenType::Do)) {
            return do_while_statement();
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
        if (match(TokenType::Using)) {
            // Allow 'using' declarations inside function bodies (block-scope using)
            auto decl = using_declaration();
            if (decl) {
                return std::make_unique<DeclarationStatement>(std::move(decl), decl->line);
            }
            return nullptr;
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

        // Check for labeled loops: label: while/for/do ...
        // AND variable declarations: name: type = value or name := value
        // Both start with identifier:, so we need to disambiguate
        if (is_identifier_like()) {
            std::size_t saved = current;
            advance(); // consume identifier
            if (check(TokenType::Colon)) {
                // Peek ahead to see if this is followed by a loop keyword
                advance(); // consume colon
                if (check(TokenType::While) || check(TokenType::For) || check(TokenType::Do)) {
                    // This is a labeled loop
                    std::string label = std::string(tokens[saved].lexeme);
                    return labeled_loop_statement(std::move(label));
                }
                
                // Check for local type alias: name: type == underlying_type;
                if (check(TokenType::Type)) {
                    std::size_t type_saved = current;
                    advance();  // consume 'type'
                    if (check(TokenType::DoubleEqual)) {
                        // This is a type alias declaration
                        advance();  // consume '=='
                        Token name = tokens[saved];
                        auto underlying = type();  // Parse the underlying type
                        consume(TokenType::Semicolon, "Expected ';' after type alias");
                        
                        auto type_decl = std::make_unique<TypeDeclaration>(
                            std::string(name.lexeme),
                            TypeDeclaration::TypeKind::Alias,
                            name.line
                        );
                        type_decl->underlying_type = std::move(underlying);
                        return std::make_unique<DeclarationStatement>(std::move(type_decl), name.line);
                    }
                    // Not a type alias, backtrack
                    current = type_saved;
                }
                
                // Check for local namespace alias: name: namespace == target_namespace;
                if (check(TokenType::Namespace)) {
                    std::size_t ns_saved = current;
                    advance();  // consume 'namespace'
                    if (check(TokenType::DoubleEqual)) {
                        // This is a namespace alias declaration
                        advance();  // consume '=='
                        Token name = tokens[saved];
                        std::string target;
                        // Handle qualified names like ::std or std::literals
                        if (match(TokenType::DoubleColon)) {
                            target = "::";
                        }
                        do {
                            Token id = consume(TokenType::Identifier, "Expected namespace name");
                            target += std::string(id.lexeme);
                            if (match(TokenType::DoubleColon)) {
                                target += "::";
                            } else {
                                break;
                            }
                        } while (true);
                        consume(TokenType::Semicolon, "Expected ';' after namespace alias");
                        
                        auto ns_decl = std::make_unique<NamespaceDeclaration>(
                            std::string(name.lexeme), name.line);
                        ns_decl->alias_target = target;
                        return std::make_unique<DeclarationStatement>(std::move(ns_decl), name.line);
                    }
                    // Not a namespace alias, backtrack
                    current = ns_saved;
                }
                
                // Not a labeled loop, so it's a variable declaration - continue parsing
                // Don't backtrack, continue with variable declaration parsing
                Token name = tokens[saved]; // the identifier
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
            } else if (check(TokenType::ColonEqual)) {
                // Found type-deduced variable declaration: name := value
                Token name = previous(); // the identifier we just consumed
                advance(); // consume the :=

                // Type is auto
                auto var_type = std::make_unique<Type>(Type::Kind::Auto);
                var_type->name = "auto";

                // Parse initializer
                std::unique_ptr<Expression> initializer = expression();
                if (!initializer) {
                    error_at_current("Expected initializer after ':='");
                }

                consume(TokenType::Semicolon, "Expected ';' after variable declaration");

                auto decl = std::make_unique<VariableDeclaration>(std::string(name.lexeme), name.line);
                decl->type = std::move(var_type);
                decl->initializer = std::move(initializer);
                decl->is_const = false;
                decl->is_mut = false;
                decl->is_compile_time = false;

                return std::make_unique<DeclarationStatement>(std::move(decl), previous().line);
            }
            current = saved; // backtrack if no colon or colon-equal
        }

        // Check for Cpp2 loop initializer: (copy/move name := expr) while/for/do
        if (check(TokenType::LeftParen)) {
            auto loop_stmt = try_loop_with_initializer();
            if (loop_stmt) {
                return loop_stmt;
            }
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

    // Check if called from keyword syntax (let/const), unified syntax (name:), or type-deduced syntax (name:=)
    Token start = previous();
    bool from_keyword = (start.type == TokenType::Let || start.type == TokenType::Const);
    bool from_colon_equal = (start.type == TokenType::ColonEqual);
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
    } else if (from_colon_equal) {
        // Type-deduced syntax: := was consumed, get identifier before it
        if (current >= 2) {
            name = tokens[current - 2]; // identifier before :=
        }
    }
    // else: name is already set from some other path (e.g., function parameter)

    // Check for template parameters before the type (Cpp2 syntax: name: <T> type = value)
    std::vector<std::string> template_params;
    if (match(TokenType::LessThan)) {
        template_params = template_parameters();
        // Handle > or >> using consume_template_close for proper pending_gt handling
        if (!consume_template_close()) {
            consume(TokenType::GreaterThan, "Expected '>' after template parameters");
        }
    }

    std::unique_ptr<Type> var_type = nullptr;
    std::unique_ptr<Expression> initializer = nullptr;
    
    if (from_colon_equal) {
        // Type-deduced declaration: name := initializer
        // Type is auto, initializer follows immediately after :=
        var_type = std::make_unique<Type>(Type::Kind::Auto);
        var_type->name = "auto";
        initializer = expression();
        if (!initializer) {
            error_at_current("Expected initializer after ':='");
        }
    } else if (from_keyword) {
        // Optional type annotation.
        if (match(TokenType::Colon)) {
            var_type = type();
        }
        // Then parse initializer if present
        if (match(TokenType::Equal)) {
            initializer = expression();
        } else if (match(TokenType::DoubleEqual)) {
            is_compile_time = true;
            initializer = expression();
        }
    } else {
        // Unified syntax (name: ...) or other contexts: type is expected unless an initializer follows.
        if (!check(TokenType::Equal) && !check(TokenType::DoubleEqual)) {
            var_type = type();
        }

        // Optional requires clause for constrained template variables
        if (match(TokenType::Requires)) {
            // Parse and discard the constraint for now
            auto requires_constraint = requires_constraint_expression();
            // TODO: Store constraint in AST if needed
        }

        // Then parse initializer if present
        if (match(TokenType::Equal)) {
            initializer = expression();
        } else if (match(TokenType::DoubleEqual)) {
            is_compile_time = true;
            initializer = expression();
        }
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
            // Handle > or >> using consume_template_close for proper pending_gt handling
            if (!consume_template_close()) {
                consume(TokenType::GreaterThan, "Expected '>' after template parameters");
            }
        }
        match(TokenType::Colon); // Optional
    } else {
        // Unified syntax: name: <T> (...) -> type = body
        // identifier and colon were already consumed in declaration()
        // After consuming identifier and colon, tokens[current - 2] is the identifier
        if (current >= 2) {
            TokenType name_type = tokens[current - 2].type;
            // Accept identifier or contextual keywords as function names
            if (name_type == TokenType::Identifier || 
                name_type == TokenType::Func || 
                name_type == TokenType::Type ||
                name_type == TokenType::Namespace ||
                name_type == TokenType::Operator) {
                name = tokens[current - 2]; // identifier/keyword before colon
            }
        }
        // else: keep prev as fallback (though this indicates a parsing bug)

        // Template parameters in unified syntax: name: <T> (params)
        if (match(TokenType::LessThan)) {
            template_params = template_parameters();
            // Handle > or >> using consume_template_close for proper pending_gt handling
            if (!consume_template_close()) {
                consume(TokenType::GreaterThan, "Expected '>' after template parameters");
            }
        }
    }

    // Check if this is actually a function (next token should be '(')
    // If not, this might be:
    // - A type declaration: name: <T> type = {...}
    // - A variable with template parameters: name: <T> type = value
    if (!check(TokenType::LeftParen)) {
        // Check if this is a type declaration (next token is 'type' or starts with '@')
        if (check(TokenType::Type) || check(TokenType::At)) {
            // This is a type declaration with template parameters
            // The type_declaration() function will be called by the caller
            return nullptr;
        }
        // Not a function or type - might be a variable with template parameters
        return nullptr;
    }

    consume(TokenType::LeftParen, "Expected '(' after function name");

    std::vector<FunctionDeclaration::Parameter> parameters;
    if (!check(TokenType::RightParen)) {
        do {
            // Check for type-first parameter syntax: in_ref name, forward_ref name
            std::unique_ptr<Type> param_type = nullptr;
            if (check(TokenType::InRef) || check(TokenType::ForwardRef)) {
                // Parse the type first (in_ref or forward_ref)
                param_type = basic_type();

                // Then parse the parameter name
                Token param_name = consume(TokenType::Identifier, "Expected parameter name after type");
                std::string param_name_str(param_name.lexeme);

                // Check for variadic
                if (match(TokenType::TripleDot) || match(TokenType::Ellipsis)) {
                    param_name_str += "...";
                }

                std::unique_ptr<Expression> default_value = nullptr;
                if (match(TokenType::Equal)) {
                    default_value = expression();
                }

                parameters.push_back({
                    param_name_str,
                    std::move(param_type),
                    std::move(default_value)
                });

                continue; // Skip to next parameter
            }

            // Parse qualifiers before parameter name
            std::vector<ParameterQualifier> qualifiers = parse_parameter_qualifiers();

            // Parameter name can be an identifier, 'this' keyword, '_' placeholder, or contextual keywords
            Token param_name = [this]() -> Token {
                if (check(TokenType::Identifier)) {
                    return advance();
                } else if (check(TokenType::This) || check(TokenType::That) || check(TokenType::Underscore)) {
                    return advance();
                } else if (check(TokenType::Func) || check(TokenType::Type) || check(TokenType::Namespace) ||
                           check(TokenType::Out) || check(TokenType::In) || check(TokenType::Inout) ||
                           check(TokenType::Copy) || check(TokenType::Move) || check(TokenType::Forward)) {
                    // Allow contextual keywords as parameter names
                    return advance();
                } else {
                    return consume(TokenType::Identifier, "Expected parameter name");
                }
            }();
            
            // Check for variadic pack parameter: name... or _...
            std::string param_name_str(param_name.lexeme);
            bool is_variadic = false;
            if (match(TokenType::TripleDot) || match(TokenType::Ellipsis)) {
                param_name_str += "...";
                is_variadic = true;
            }

            // Type annotation is optional in Cpp2 (type can be deduced)
            // param_type already declared at top of loop
            if (match(TokenType::Colon)) {
                param_type = type();
            }

            std::unique_ptr<Expression> default_value = nullptr;
            if (match(TokenType::Equal)) {
                default_value = expression();
            }

            parameters.push_back({
                param_name_str,
                std::move(param_type),
                std::move(default_value)
            });
            // Add qualifiers to the parameter
            parameters.back().qualifiers = std::move(qualifiers);
        } while (match(TokenType::Comma) && !check(TokenType::RightParen));
    }

    consume(TokenType::RightParen, "Expected ')' after parameters");

    // Return type - Cpp2 supports several syntaxes:
    // -> type             - simple return type
    // -> forward type     - forward return (pass-through)
    // -> (name: type)     - named return (single)
    // -> (name: type = val) - named return with default value
    // -> (n1: t1, n2: t2) - multiple named returns (tuple)
    // =: return_type      - alternative syntax
    std::unique_ptr<Type> return_type = nullptr;
    struct NamedReturnLocal {
        std::string name;
        std::unique_ptr<Type> type;
        std::unique_ptr<Expression> default_value;
    };
    std::vector<NamedReturnLocal> named_returns;
    bool is_forward_return = false;
    
    if (match(TokenType::Arrow)) {
        // Check for forward_ref as a return type: -> forward_ref _
        if (match(TokenType::ForwardRef)) {
            // forward_ref can be followed by _ or a type
            if (match(TokenType::Underscore)) {
                // forward_ref _ → auto&&
                return_type = std::make_unique<Type>(Type::Kind::UserDefined);
                return_type->name = "auto&&";
            } else {
                // forward_ref SomeType → SomeType&&
                auto base = type();
                return_type = std::make_unique<Type>(Type::Kind::Reference);
                return_type->pointee = std::move(base);
                return_type->name = return_type->pointee->name + "&&";
            }
        }
        // Check for forward return modifier: -> forward type
        else if (match(TokenType::Forward)) {
            is_forward_return = true;
            return_type = type(); // Parse the actual type after forward
        }
        else if (match(TokenType::LeftParen)) {
            // Return type can be:
            // - Named returns: -> (name: type) or -> (n1: t1, n2: t2)
            // - Unnamed tuple: -> (int, int) or -> (int)
            // Check if first element looks like named or unnamed
            std::size_t saved_pos = current;
            bool is_named = false;
            if (check(TokenType::Identifier)) {
                std::size_t peek_pos = current;
                advance();  // Skip the identifier
                if (check(TokenType::Colon)) {
                    is_named = true;
                }
                current = peek_pos;  // Restore position
            }
            
            if (is_named) {
                // Named returns: -> (name: type) or -> (name: type = default)
                // Support trailing commas: -> (i: int,)
                do {
                    // Check for trailing comma before )
                    if (check(TokenType::RightParen)) break;
                    Token ret_name = consume(TokenType::Identifier, "Expected return value name");
                    consume(TokenType::Colon, "Expected ':' after return name");
                    auto ret_type = type();
                    std::unique_ptr<Expression> default_val = nullptr;
                    if (match(TokenType::Equal)) {
                        default_val = expression();
                    }
                    named_returns.push_back({std::string(ret_name.lexeme), std::move(ret_type), std::move(default_val)});
                } while (match(TokenType::Comma));
                consume(TokenType::RightParen, "Expected ')' after named return types");
                
                // Create the return type(s) for code generation
                if (named_returns.size() == 1) {
                    // Clone the type for return_type since we need to keep named_returns
                    auto cloned_type = std::make_unique<Type>(named_returns[0].type->kind);
                    cloned_type->name = named_returns[0].type->name;
                    cloned_type->is_const = named_returns[0].type->is_const;
                    return_type = std::move(cloned_type);
                } else {
                    // Multiple returns - create a std::tuple type
                    auto tuple_type = std::make_unique<Type>(Type::Kind::Template);
                    tuple_type->name = "std::tuple";
                    for (auto& nr : named_returns) {
                        auto cloned = std::make_unique<Type>(nr.type->kind);
                        cloned->name = nr.type->name;
                        cloned->is_const = nr.type->is_const;
                        tuple_type->template_args.push_back(std::move(cloned));
                    }
                    return_type = std::move(tuple_type);
                }
            } else {
                // Unnamed tuple return: -> (int, int) or -> (int)
                // Also handles placeholders: -> (a, b) where a,b are deduced (auto)
                std::vector<std::unique_ptr<Type>> tuple_types;
                do {
                    // Check if this is a placeholder (identifier followed by , or ))
                    if (check(TokenType::Identifier)) {
                        std::size_t peek_pos = current + 1;
                        if (peek_pos < tokens.size() &&
                            (tokens[peek_pos].type == TokenType::Comma ||
                             tokens[peek_pos].type == TokenType::RightParen)) {
                            // Placeholder - use auto (deduced type)
                            auto auto_type = std::make_unique<Type>(Type::Kind::Builtin);
                            auto_type->name = "auto";
                            tuple_types.push_back(std::move(auto_type));
                            advance(); // Skip the identifier
                        } else {
                            tuple_types.push_back(type());
                        }
                    } else {
                        tuple_types.push_back(type());
                    }
                } while (match(TokenType::Comma));
                consume(TokenType::RightParen, "Expected ')' after tuple types");
                
                if (tuple_types.size() == 1) {
                    // Single type in parens - just use it directly
                    return_type = std::move(tuple_types[0]);
                } else {
                    // Multiple types - create a std::tuple
                    auto tuple_type = std::make_unique<Type>(Type::Kind::Template);
                    tuple_type->name = "std::tuple";
                    for (auto& t : tuple_types) {
                        tuple_type->template_args.push_back(std::move(t));
                    }
                    return_type = std::move(tuple_type);
                }
            }
        } else if (!return_type) {
            // Normal return type (not forward_ref or forward modifier)
            return_type = type();
        }
    } else if (match(TokenType::EqualColon)) {
        // Cpp2 named return type: (params) =: return_type
        return_type = type();
    }

    // Optional requires clause: requires expression
    // Be careful not to consume '=' or '==' which are function body separators
    std::unique_ptr<Expression> requires_clause = nullptr;
    if (match(TokenType::Requires)) {
        // Parse requires clause, stopping at '=' or '=='
        requires_clause = requires_constraint_expression();
    }

    // Exception specification (Cpp2-style): throws or noexcept
    [[maybe_unused]] bool can_throw = false;
    [[maybe_unused]] bool is_noexcept = false;
    if (match(TokenType::Throws)) {
        can_throw = true;
    } else if (match(TokenType::Noexcept)) {
        is_noexcept = true;
    }

    // Contracts
    auto contracts = parse_contracts();

    // Function body - Cpp2 supports both block bodies and expression bodies.
    // - name: (params) -> type = { body }
    // - name: (params) -> type = expr;
    // - name: (params) -> type == { body }  (compile-time function)
    // - name: (params) -> type == expr;     (compile-time function)
    std::unique_ptr<Statement> body = nullptr;
    bool is_compile_time = match(TokenType::DoubleEqual); // '==' for compile-time functions
    bool has_equals = is_compile_time || match(TokenType::Equal); // '=' or '==' before body
    std::cerr << "[DEBUG] function_declaration: is_compile_time=" << is_compile_time 
              << " has_equals=" << has_equals 
              << " current_token=" << static_cast<int>(peek().type) 
              << " lexeme=" << peek().lexeme << std::endl;
    if (match(TokenType::LeftBrace)) {
        std::cerr << "[DEBUG] Matched LeftBrace, parsing block" << std::endl;
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
    func->template_parameters = std::move(template_params);
    func->is_constexpr = is_compile_time;  // '==' means compile-time function
    func->is_forward_return = is_forward_return;  // forward return type
    
    // Store named return parameters
    for (auto& nr : named_returns) {
        FunctionDeclaration::NamedReturn named_ret;
        named_ret.name = std::move(nr.name);
        named_ret.type = std::move(nr.type);
        named_ret.default_value = std::move(nr.default_value);
        func->named_returns.push_back(std::move(named_ret));
    }

    return func;
}

std::unique_ptr<Declaration> Parser::type_declaration(std::vector<std::string> decorators) {
    Token prev = previous();
    std::vector<std::string> template_params;  // Store template parameters

    // Track if we saw a kind-defining decorator
    TypeDeclaration::TypeKind decorator_kind = TypeDeclaration::TypeKind::Struct;
    bool has_kind_decorator = false;

    // Process passed-in decorators for kind
    for (const auto& d : decorators) {
        if (!has_kind_decorator) {
            if (d == "enum" || d == "flag_enum") {
                decorator_kind = TypeDeclaration::TypeKind::Enum;
                has_kind_decorator = true;
            } else if (d == "struct") {
                decorator_kind = TypeDeclaration::TypeKind::Struct;
                has_kind_decorator = true;
            } else if (d == "class") {
                decorator_kind = TypeDeclaration::TypeKind::Class;
                has_kind_decorator = true;
            } else if (d == "union") {
                decorator_kind = TypeDeclaration::TypeKind::Union;
                has_kind_decorator = true;
            } else if (d == "interface") {
                decorator_kind = TypeDeclaration::TypeKind::Interface;
                has_kind_decorator = true;
            }
        }
    }

    // Get the name token based on syntax
    Token name = prev; // Initialize with prev as default

    // Check if called from keyword syntax (type) or unified syntax (name: @decorators type)
    if (prev.type == TokenType::Type) {
        // Keyword syntax: type Name = { ... }
        name = consume(TokenType::Identifier, "Expected type name");
    } else {
        // Unified syntax: Name: <T> @decorators type = { ... } or Name: @decorators type = { ... }
        // identifier and colon were already consumed in declaration()
        if (current >= 2 && tokens[current - 2].type == TokenType::Identifier) {
            name = tokens[current - 2]; // identifier before colon
        }

        // Parse template parameters before decorators: <T, U>
        if (match(TokenType::LessThan)) {
            template_params = template_parameters();
            // Handle > or >> (via pending_gt mechanism from nested templates)
            if (!consume_template_close()) {
                consume(TokenType::GreaterThan, "Expected '>' after template parameters");
            }
        }

        // Parse decorators: @value @ordered ... @struct <T> ...
        while (match(TokenType::At)) {
            // Decorator name can be an identifier or a keyword (struct, class, etc.)
            std::string decorator_name;
            if (check(TokenType::Identifier)) {
                decorator_name = std::string(advance().lexeme);
                // Check for enum-like metafunctions: @enum, @flag_enum
                if (!has_kind_decorator) {
                    if (decorator_name == "enum" || decorator_name == "flag_enum") {
                        decorator_kind = TypeDeclaration::TypeKind::Enum;
                        has_kind_decorator = true;
                    }
                }
            } else if (match(TokenType::Struct) || match(TokenType::Class) || match(TokenType::Union) ||
                       match(TokenType::Enum) || match(TokenType::Interface) || match(TokenType::Type) ||
                       match(TokenType::Public) || match(TokenType::Private) || match(TokenType::Virtual) ||
                       match(TokenType::Override) || match(TokenType::Final) || match(TokenType::Explicit) ||
                       match(TokenType::Const)) {
                decorator_name = std::string(previous().lexeme);

                // Set kind based on decorator keyword (first one wins)
                if (!has_kind_decorator) {
                    if (decorator_name == "enum") {
                        decorator_kind = TypeDeclaration::TypeKind::Enum;
                        has_kind_decorator = true;
                    } else if (decorator_name == "struct") {
                        decorator_kind = TypeDeclaration::TypeKind::Struct;
                        has_kind_decorator = true;
                    } else if (decorator_name == "class") {
                        decorator_kind = TypeDeclaration::TypeKind::Class;
                        has_kind_decorator = true;
                    } else if (decorator_name == "union") {
                        decorator_kind = TypeDeclaration::TypeKind::Union;
                        has_kind_decorator = true;
                    } else if (decorator_name == "interface") {
                        decorator_kind = TypeDeclaration::TypeKind::Interface;
                        has_kind_decorator = true;
                    }
                }
            } else {
                error_at_current("Expected decorator name after '@'");
                decorator_name = "_unknown"; // fallback
            }

            // Check for template arguments after decorator name
            if (match(TokenType::LessThan)) {
                // Parse template parameters for the decorator
                // These can be:
                // - Types (e.g., <T>)
                // - Parameter declarations (e.g., T:type)
                // - Expressions/values (e.g., <"order=6">, <42>)
                // - Variadic (e.g., <_...>, <T...:_>)
                do {
                    // Check for string literals, numbers, or other expression arguments
                    if (check(TokenType::StringLiteral) || check(TokenType::IntegerLiteral) ||
                        check(TokenType::FloatLiteral) || check(TokenType::CharacterLiteral) ||
                        check(TokenType::True) || check(TokenType::False)) {
                        // Simple value argument - just consume it
                        advance();
                    } else if (check(TokenType::Underscore)) {
                        // _ as a wildcard type
                        advance();
                        // Check for variadic: _...
                        if (check(TokenType::TripleDot) || check(TokenType::Ellipsis)) {
                            advance(); // consume ...
                        }
                        // Check for constraint: _: type
                        if (match(TokenType::Colon)) {
                            if (check(TokenType::Type) || check(TokenType::Auto) ||
                                check(TokenType::Identifier) || check(TokenType::Underscore)) {
                                advance();
                            }
                        }
                    } else if (check(TokenType::Identifier)) {
                        // Check if this is a parameter declaration (name:type or name : type)
                        std::size_t saved = current;
                        std::string param_name = std::string(advance().lexeme);
                        // Check for variadic: T...
                        if (check(TokenType::TripleDot) || check(TokenType::Ellipsis)) {
                            advance(); // consume ...
                        }
                        if (match(TokenType::Colon)) {
                            // This is a parameter declaration - consume the type constraint
                            // The type constraint can be:
                            // - A keyword like "type" (meaning any type)
                            // - A regular type
                            // - _ (wildcard)
                            if (check(TokenType::Type) || check(TokenType::Auto) ||
                                check(TokenType::Identifier) || check(TokenType::Underscore)) {
                                advance(); // consume the type constraint keyword/type
                                // For now, just ignore the parameter - we could store it later
                            } else {
                                auto param_type = type();
                            }
                        } else if (current != saved + 1 || (current == saved + 1 && !check(TokenType::Comma) && !check(TokenType::GreaterThan))) {
                            // Not a parameter declaration but we consumed something other than just the name
                            // or there's more to parse - backtrack
                            current = saved;
                            // Parse as a regular type
                            auto arg = type();
                            // Check for pack expansion
                            if (check(TokenType::TripleDot) || check(TokenType::Ellipsis)) {
                                advance();
                            }
                        }
                    } else if (!check(TokenType::GreaterThan)) {
                        // Parse as a regular type (only if not at closing >)
                        auto arg = type();
                        // Check for pack expansion
                        if (check(TokenType::TripleDot) || check(TokenType::Ellipsis)) {
                            advance();
                        }
                    }
                } while (match(TokenType::Comma));
                consume(TokenType::GreaterThan, "Expected '>' after decorator template arguments");
            }

            decorators.push_back(decorator_name);
        }

        // Handle type qualifiers without @ decorator syntax (e.g., final, virtual)
        // These appear before the 'type' keyword: mytype: final type = {...}
        while (is_type_qualifier() && !check(TokenType::At)) {
            // Skip the qualifier - for now, just consume it
            // TODO: Store qualifiers in the type declaration
            if (check(TokenType::Identifier)) {
                // Check if this identifier followed by 'type' keyword
                if (current + 1 < tokens.size() && tokens[current + 1].type == TokenType::Type) {
                    // This is likely a type qualifier (final, abstract, etc.)
                    std::string qual = std::string(advance().lexeme);
                    decorators.push_back(qual);  // Store as pseudo-decorator for now
                } else {
                    // Not followed by 'type', might be something else
                    break;
                }
            } else {
                // For keyword qualifiers (Final, Virtual, etc.)
                std::string qual = std::string(advance().lexeme);
                decorators.push_back(qual);  // Store as pseudo-decorator for now
            }
        }

        // Expect 'type' or 'concept' keyword in unified syntax
        if (match(TokenType::Concept)) {
            // Concept definition: name: <T> concept = constraint_expr;
            decorator_kind = TypeDeclaration::TypeKind::Alias;  // Treat as alias
            has_kind_decorator = true;
            decorators.push_back("concept");  // Mark as concept
        } else if (match(TokenType::Type)) {
            // We saw 'type' keyword, this might be a type alias or type declaration
            // Check for the pattern "type ==" which is a type alias with expression
            if (check(TokenType::DoubleEqual)) {
                // "type ==" is a type alias with expression on the right
                decorator_kind = TypeDeclaration::TypeKind::Alias;
                has_kind_decorator = true;
                // Don't do anything here - the DoubleEqual will be consumed in type alias handling
            }
            // Only set kind from the 'type' keyword if not already set by decorator
            else if (!has_kind_decorator) {
                if (check(TokenType::Identifier)) {
                    if (peek().lexeme == "struct") {
                        advance();
                        decorator_kind = TypeDeclaration::TypeKind::Struct;
                        has_kind_decorator = true;
                    } else if (peek().lexeme == "class") {
                        advance();
                        decorator_kind = TypeDeclaration::TypeKind::Class;
                        has_kind_decorator = true;
                    } else if (peek().lexeme == "enum") {
                        advance();
                        decorator_kind = TypeDeclaration::TypeKind::Enum;
                        has_kind_decorator = true;
                    } else if (peek().lexeme == "union") {
                        advance();
                        decorator_kind = TypeDeclaration::TypeKind::Union;
                        has_kind_decorator = true;
                    } else if (peek().lexeme == "interface") {
                        advance();
                        decorator_kind = TypeDeclaration::TypeKind::Interface;
                        has_kind_decorator = true;
                    } else if (peek().lexeme == "type") {
                        advance();
                        decorator_kind = TypeDeclaration::TypeKind::Alias;
                        has_kind_decorator = true;
                    }
                } else {
                    // Just "type" without any identifier - check what comes next
                    if (check(TokenType::DoubleEqual)) {
                        // "type ==" is a type alias with expression
                        decorator_kind = TypeDeclaration::TypeKind::Alias;
                        has_kind_decorator = true;
                    } else if (check(TokenType::Equal)) {
                        // "type =" is a type alias ONLY if not followed by "{"
                        // If followed by "{", it's a type definition with body
                        if (current + 1 < tokens.size() && tokens[current + 1].type != TokenType::LeftBrace) {
                            decorator_kind = TypeDeclaration::TypeKind::Alias;
                            has_kind_decorator = true;
                        }
                    }
                    // Otherwise "type = {" is a type definition with body (default Struct kind)
                }
            }
        } else {
            // If no 'type' or 'concept' keyword, might be using implicit struct syntax
            // For now, require explicit 'type' keyword
            error_at_current("Expected 'type' or 'concept' keyword after decorators");
        }
    }

    TypeDeclaration::TypeKind kind = has_kind_decorator ? decorator_kind : TypeDeclaration::TypeKind::Struct;
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
        // Accept either '=' or '==' for type aliases
        if (!match(TokenType::Equal)) {
            match(TokenType::DoubleEqual);  // also accept '=='
        }
        
        // Check if this is a concept definition
        bool is_concept = std::find(decorators.begin(), decorators.end(), "concept") != decorators.end();
        
        if (is_concept) {
            // Concept definition: parse constraint expression as a type (user-defined type with the expression text)
            // We need to collect the constraint text until the semicolon
            // But we need to track paren/brace/bracket depth to handle nested expressions like :() = {}
            // Also handle function expressions with immediate invocation: :() = {}();
            std::string constraint_text;
            TokenType prev_type = TokenType::EndOfFile;  // sentinel
            int paren_depth = 0;
            int brace_depth = 0;
            int bracket_depth = 0;
            bool seen_function_expr = false;  // Track if we've seen a function expression start

            while (!is_at_end()) {
                // Check for semicolon at depth 0 (potential end of constraint)
                if (check(TokenType::Semicolon) && paren_depth == 0 && brace_depth == 0 && bracket_depth == 0) {
                    // If we've seen a function expression and the next token is '(', this is an immediate invocation
                    // Continue collecting to include the invocation
                    if (seen_function_expr && current + 1 < tokens.size() && tokens[current + 1].type == TokenType::LeftParen) {
                        // Consume the semicolon and continue to capture the invocation
                        Token tok = advance();  // consume ';'
                        constraint_text += std::string(tok.lexeme);
                        prev_type = tok.type;
                        continue;
                    } else {
                        break;  // This is the real end of the constraint
                    }
                }
                Token tok = advance();

                // Track if we've seen a function expression start ':'
                if (tok.type == TokenType::Colon) {
                    seen_function_expr = true;
                }

                // Track depth
                if (tok.type == TokenType::LeftParen) paren_depth++;
                else if (tok.type == TokenType::RightParen) paren_depth--;
                else if (tok.type == TokenType::LeftBrace) brace_depth++;
                else if (tok.type == TokenType::RightBrace) brace_depth--;
                else if (tok.type == TokenType::LeftBracket) bracket_depth++;
                else if (tok.type == TokenType::RightBracket) bracket_depth--;

                // Add spacing between tokens intelligently
                if (!constraint_text.empty()) {
                    // No space after: <, (, ::
                    // No space before: >, ), ::, <
                    // Always space around: ||, &&
                    bool space_around = (tok.type == TokenType::DoublePipe ||
                                        tok.type == TokenType::DoubleAmpersand ||
                                        prev_type == TokenType::DoublePipe ||
                                        prev_type == TokenType::DoubleAmpersand);
                    bool no_space_before = !space_around && (tok.type == TokenType::GreaterThan ||
                                           tok.type == TokenType::RightParen ||
                                           tok.type == TokenType::DoubleColon ||
                                           tok.type == TokenType::LessThan);
                    bool no_space_after = !space_around && (prev_type == TokenType::LessThan ||
                                          prev_type == TokenType::LeftParen ||
                                          prev_type == TokenType::DoubleColon ||
                                          prev_type == TokenType::GreaterThan);
                    if (space_around || (!no_space_before && !no_space_after)) {
                        constraint_text += " ";
                    }
                }
                constraint_text += std::string(tok.lexeme);
                prev_type = tok.type;
            }
            consume(TokenType::Semicolon, "Expected ';' after concept constraint");

            // Store the constraint as a user-defined type
            auto constraint_type = std::make_unique<Type>(Type::Kind::UserDefined);
            constraint_type->name = constraint_text;
            
            auto type_decl = std::make_unique<TypeDeclaration>(std::string(name.lexeme), kind, name.line);
            type_decl->underlying_type = std::move(constraint_type);
            type_decl->metafunctions = std::move(decorators);
            type_decl->template_parameters = std::move(template_params);
            return type_decl;
        }
        
        auto underlying_type = type();
        consume(TokenType::Semicolon, "Expected ';' after type alias");

        auto type_decl = std::make_unique<TypeDeclaration>(std::string(name.lexeme), kind, name.line);
        type_decl->underlying_type = std::move(underlying_type);
        type_decl->metafunctions = std::move(decorators);
        type_decl->template_parameters = std::move(template_params);
        return type_decl;
    }

    // Expect '=' before type body in unified syntax, optional in keyword syntax
    // But first, check for optional requires clause
    std::unique_ptr<Expression> requires_constraint = nullptr;
    if (match(TokenType::Requires)) {
        // Parse the constraint expression, stopping at '=' or '=='
        requires_constraint = requires_constraint_expression();
    }

    match(TokenType::Equal);

    consume(TokenType::LeftBrace, "Expected '{' for type definition");

    auto type_decl = std::make_unique<TypeDeclaration>(std::string(name.lexeme), kind, name.line);
    type_decl->metafunctions = std::move(decorators);
    type_decl->template_parameters = std::move(template_params);
    type_decl->requires_clause = std::move(requires_constraint);

    while (!check(TokenType::RightBrace) && !is_at_end()) {
        std::size_t before = current;

        // Skip standalone semicolons in type body
        if (match(TokenType::Semicolon)) {
            continue;
        }

        // Check for base class declaration: "this: BaseType" or "this: BaseType = ();"
        if (check(TokenType::This)) {
            std::size_t saved = current;
            advance();  // consume 'this'
            
            if (match(TokenType::Colon)) {
                // This is a base class declaration
                auto base_type = type();
                std::unique_ptr<Expression> initializer = nullptr;
                
                if (match(TokenType::Equal)) {
                    initializer = expression();
                }
                
                consume(TokenType::Semicolon, "Expected ';' after base class declaration");
                
                TypeDeclaration::BaseClass base;
                base.type = std::move(base_type);
                base.initializer = std::move(initializer);
                type_decl->base_classes.push_back(std::move(base));
                continue;
            } else {
                // Not a base class, backtrack
                current = saved;
            }
        }

        // Special case for enum members: just an identifier followed by ; or :=
        // e.g., "member;" or "member := value;"
        if (kind == TypeDeclaration::TypeKind::Enum && check(TokenType::Identifier)) {
            std::size_t saved = current;
            std::string member_name = std::string(advance().lexeme); // consume identifier

            if (check(TokenType::ColonEqual) || check(TokenType::Semicolon)) {
                // This is an enum member declaration
                std::unique_ptr<Expression> init_value = nullptr;
                if (match(TokenType::ColonEqual)) {
                    init_value = expression();
                }
                consume(TokenType::Semicolon, "Expected ';' after enum member");

                // Create a variable declaration for the enum member
                auto enum_member = std::make_unique<VariableDeclaration>(
                    member_name,
                    previous().line
                );
                enum_member->initializer = std::move(init_value);
                type_decl->members.push_back(std::move(enum_member));
                continue;
            } else {
                // Not an enum member, backtrack
                current = saved;
            }
        }

        auto member = declaration();
        if (member) {
            type_decl->members.push_back(std::move(member));
        } else if (current == before) {
            // Avoid infinite loop when a declaration cannot be parsed and no tokens
            // are consumed; advance to allow error recovery to proceed.
            advance();
        }
    }

    consume(TokenType::RightBrace, "Expected '}' after type definition");
    // Optional semicolon after type definition
    match(TokenType::Semicolon);

    return type_decl;
}

std::unique_ptr<Declaration> Parser::namespace_declaration() {
    Token prev = previous();
    std::string name;
    bool unified_syntax = false;

    // Check if called from keyword syntax (namespace) or unified syntax (name: namespace)
    if (prev.type == TokenType::Namespace) {
        // Keyword syntax: namespace name { ... } or namespace name = { ... }
        Token name_token = consume(TokenType::Identifier, "Expected namespace name");
        name = std::string(name_token.lexeme);
    } else {
        // Unified syntax: name: namespace = { ... }
        // The name and colon were already consumed in declaration()
        // Previous token should be the identifier name
        if (current >= 2 && tokens[current - 2].type == TokenType::Identifier) {
            name = std::string(tokens[current - 2].lexeme);
        }
        unified_syntax = true;
        // Consume the namespace keyword
        consume(TokenType::Namespace, "Expected 'namespace' keyword");
    }

    // Check for namespace alias: N1: namespace == N;
    if (match(TokenType::DoubleEqual)) {
        // This is a namespace alias
        std::string target;
        // Handle qualified names like ::std or std::literals
        if (match(TokenType::DoubleColon)) {
            target = "::";
        }
        do {
            Token id = consume(TokenType::Identifier, "Expected namespace name");
            target += std::string(id.lexeme);
            if (match(TokenType::DoubleColon)) {
                target += "::";
            } else {
                break;
            }
        } while (true);
        auto ns = std::make_unique<NamespaceDeclaration>(name, prev.line);
        ns->alias_target = target;
        consume(TokenType::Semicolon, "Expected ';' after namespace alias");
        return ns;
    }

    // Support both C++1 style: namespace name { ... }
    // and Cpp2 style: namespace name = { ... }
    match(TokenType::Equal);

    auto ns = std::make_unique<NamespaceDeclaration>(name, prev.line);

    if (match(TokenType::LeftBrace)) {
        while (!check(TokenType::RightBrace) && !is_at_end()) {
            std::size_t before = current;
            auto member = declaration();
            if (member) {
                ns->members.push_back(std::move(member));
            } else if (current == before) {
                // Avoid infinite loop when a declaration cannot be parsed and no tokens
                // are consumed; advance to allow error recovery to proceed.
                advance();
            }
        }
        consume(TokenType::RightBrace, "Expected '}' after namespace");
    } else {
        // Single declaration
        auto member = declaration();
        if (member) {
            ns->members.push_back(std::move(member));
        }
        // The declaration itself should have consumed the necessary terminator (semicolon or brace)
        // consume(TokenType::Semicolon, "Expected ';' after namespace declaration");
    }

    return ns;
}

std::unique_ptr<Declaration> Parser::operator_declaration() {
    // Handle operator overloading
    // Cpp2 syntax: operator=: (...) or operator[]: (...) or operator++: (...) etc.
    
    // Collect operator tokens - operators like [] are multi-token
    std::string op_name;
    std::size_t op_line = peek().line;
    
    // Check for special multi-character operators
    if (check(TokenType::LeftBracket)) {
        advance();  // consume [
        if (match(TokenType::RightBracket)) {
            op_name = "[]";
        } else {
            error_at_current("Expected ']' after '[' in operator[]");
            return nullptr;
        }
    } else if (check(TokenType::LeftParen)) {
        advance();  // consume (
        if (match(TokenType::RightParen)) {
            op_name = "()";
        } else {
            error_at_current("Expected ')' after '(' in operator()");
            return nullptr;
        }
    } else if (check(TokenType::PlusPlus)) {
        op_name = "++";
        op_line = advance().line;
    } else if (check(TokenType::MinusMinus)) {
        op_name = "--";
        op_line = advance().line;
    } else if (check(TokenType::Arrow)) {
        op_name = "->";
        op_line = advance().line;
    } else if (check(TokenType::Spaceship)) {
        op_name = "<=>";
        op_line = advance().line;
    } else {
        // Single-token operator like = + - etc.
        Token op = advance();
        op_name = std::string(op.lexeme);
        op_line = op.line;
    }

    // In Cpp2, after operator name comes ':'
    // operator=: (...) where = is operator, : is separator
    // For operators like ++ we have operator++: (...)
    // For operator= we might have operator=: which parses as EqualColon
    // Template operators: operator++: <T> (...) 
    bool is_cpp2_assign_op = (op_name == "=:");

    if (!is_cpp2_assign_op) {
        consume(TokenType::Colon, "Expected ':' after operator");
    }

    // Parse optional template parameters: operator++: <T> (...)
    std::vector<std::string> template_params;
    if (match(TokenType::LessThan)) {
        template_params = template_parameters();
        // Handle > or >> (via pending_gt mechanism from nested templates)
        if (!consume_template_close()) {
            consume(TokenType::GreaterThan, "Expected '>' after template parameters");
        }
    }

    consume(TokenType::LeftParen, "Expected '(' after operator");

    auto op_decl = std::make_unique<OperatorDeclaration>(op_name, op_line);
    op_decl->template_parameters = std::move(template_params);

    if (!check(TokenType::RightParen)) {
        do {
            // Parse qualifiers before parameter name
            std::vector<ParameterQualifier> qualifiers = parse_parameter_qualifiers();

            // Parameter name can be identifier or 'this' keyword (for constructors/assignment)
            Token param_name = [this]() -> Token {
                if (check(TokenType::Identifier)) {
                    return advance();
                } else if (check(TokenType::This) || check(TokenType::That) ||
                           check(TokenType::Underscore) || check(TokenType::Implicit)) {
                    return advance();
                } else {
                    return consume(TokenType::Identifier, "Expected parameter name");
                }
            }();

            // Type annotation is optional for operator parameters
            // (e.g., operator=: (out this) - type is implied from context)
            std::unique_ptr<Type> param_type = nullptr;
            if (match(TokenType::Colon)) {
                param_type = type();
            }

            auto param = std::make_unique<FunctionDeclaration::Parameter>();
            param->name = std::string(param_name.lexeme);
            param->type = std::move(param_type);
            param->qualifiers = std::move(qualifiers);

            op_decl->parameters.push_back(std::move(param));
        } while (match(TokenType::Comma) && !check(TokenType::RightParen));
    }

    consume(TokenType::RightParen, "Expected ')' after parameters");

    // Return type - Cpp2 supports both -> and =: syntax
    // Also supports -> forward T or -> forward_ref T
    // Also supports named returns: -> (x: int, y: int)
    bool is_forward_return = false;
    if (match(TokenType::Arrow)) {
        // Check for forward modifier before type
        if (match(TokenType::Forward) || match(TokenType::ForwardRef)) {
            is_forward_return = true;
        }
        
        // Check for named return types: -> (name: type, ...)
        if (check(TokenType::LeftParen)) {
            advance();  // consume '('
            
            // Check if first element looks like named or unnamed
            bool is_named = false;
            if (check(TokenType::Identifier)) {
                std::size_t peek_pos = current;
                advance();  // Skip the identifier
                if (check(TokenType::Colon)) {
                    is_named = true;
                }
                current = peek_pos;  // Restore position
            }
            
            if (is_named) {
                // Named returns: -> (name: type) or -> (n1: t1, n2: t2)
                std::vector<std::pair<std::string, std::unique_ptr<Type>>> named_returns;
                do {
                    if (check(TokenType::RightParen)) break;
                    Token ret_name = consume(TokenType::Identifier, "Expected return value name");
                    consume(TokenType::Colon, "Expected ':' after return name");
                    auto ret_type = type();
                    named_returns.push_back({std::string(ret_name.lexeme), std::move(ret_type)});
                } while (match(TokenType::Comma));
                consume(TokenType::RightParen, "Expected ')' after named return types");
                
                // Create the return type
                if (named_returns.size() == 1) {
                    auto cloned_type = std::make_unique<Type>(named_returns[0].second->kind);
                    cloned_type->name = named_returns[0].second->name;
                    cloned_type->is_const = named_returns[0].second->is_const;
                    op_decl->return_type = std::move(cloned_type);
                } else {
                    auto tuple_type = std::make_unique<Type>(Type::Kind::Template);
                    tuple_type->name = "std::tuple";
                    for (auto& nr : named_returns) {
                        auto cloned = std::make_unique<Type>(nr.second->kind);
                        cloned->name = nr.second->name;
                        cloned->is_const = nr.second->is_const;
                        tuple_type->template_args.push_back(std::move(cloned));
                    }
                    op_decl->return_type = std::move(tuple_type);
                }
                op_decl->named_returns = std::move(named_returns);
            } else {
                // Unnamed tuple return - parse as type
                current--;  // go back before '('
                op_decl->return_type = type();
            }
        } else {
            op_decl->return_type = type();
        }
        op_decl->is_forward_return = is_forward_return;
    } else if (match(TokenType::EqualColon)) {
        // Cpp2 named return type: (params) =: return_type
        op_decl->return_type = type();
    }

    // Optional requires clause: requires expression
    if (match(TokenType::Requires)) {
        // Parse the constraint expression, stopping at '=' or '=='
        auto constraint = requires_constraint_expression();
        // TODO: Store constraint in AST if needed
    }

    // Cpp2 supports both block bodies and expression bodies
    // Cpp2 supports both block bodies and expression bodies
    // - operator=: (params) -> type = { body }
    // - operator=: (params) -> type = expr;
    // - operator=: (params) -> type == { body }  (compile-time operator)
    // - operator=: (params) -> type == expr;     (compile-time operator)
    // - operator=: (params) -> type;  (forward declaration)
    std::unique_ptr<Statement> body = nullptr;
    
    // Check for forward declaration (no body, just semicolon)
    if (match(TokenType::Semicolon)) {
        // Forward declaration - no body
        op_decl->body = nullptr;
        return op_decl;
    }
    
    bool is_compile_time = match(TokenType::DoubleEqual); // '==' for compile-time functions
    bool has_equals = is_compile_time || match(TokenType::Equal); // '=' or '==' before body
    if (match(TokenType::LeftBrace)) {
        body = block_statement();
    } else if (has_equals) {
        // Expression-bodied operator
        auto expr = expression();
        if (expr) {
            if (op_decl->return_type && op_decl->return_type->name != "void") {
                body = std::make_unique<ReturnStatement>(std::move(expr), previous().line);
            } else {
                body = std::make_unique<ExpressionStatement>(std::move(expr), previous().line);
            }
        } else {
            error_at_current("Expected expression");
        }
        consume(TokenType::Semicolon, "Expected ';' after operator body expression");
    } else {
        consume(TokenType::LeftBrace, "Expected '{' for operator body");
        body = block_statement();
    }

    op_decl->body = std::move(body);
    op_decl->is_constexpr = is_compile_time;

    return op_decl;
}

std::unique_ptr<Declaration> Parser::using_declaration() {
    // Supports forms:
    //  - using name = target;
    //  - using qualified::path::name;
    //  - using ::qualified::path::name;  (global scope)
    //  - using qualified::path::_;   (wildcard shorthand -> using namespace qualified::path;)

    // Check for global scope: using ::...
    if (check(TokenType::DoubleColon)) {
        advance();  // consume ::
        std::string path = "::";
        
        // First identifier after ::
        if (!check(TokenType::Identifier) && !check(TokenType::Underscore)) {
            error_at_current("Expected identifier after '::' in using declaration");
            return nullptr;
        }
        path += std::string(advance().lexeme);
        
        // Continue with rest of path
        while (match(TokenType::DoubleColon)) {
            if (check(TokenType::Identifier) || check(TokenType::Underscore)) {
                path += "::";
                path += std::string(advance().lexeme);
            } else {
                error_at_current("Expected identifier or '_' after '::' in using declaration");
                break;
            }
        }
        
        consume(TokenType::Semicolon, "Expected ';' after using declaration");
        return std::make_unique<UsingDeclaration>("", path, previous().line);
    }

    // If it's an alias form: identifier '=' target ';'
    if (check(TokenType::Identifier)) {
        // Peek ahead to see if this is an alias
        Token first = advance(); // consume first identifier
        if (check(TokenType::Equal)) {
            // Alias: using name = target;
            advance(); // consume '='
            Token target = consume(TokenType::Identifier, "Expected target name for using alias");
            consume(TokenType::Semicolon, "Expected ';' after using declaration");
            return std::make_unique<UsingDeclaration>(
                std::string(first.lexeme),
                std::string(target.lexeme),
                first.line
            );
        }

        // Not an alias - parse a qualified path (Identifier(:: Identifier|_)* )
        std::string path = std::string(first.lexeme);
        while (true) {
            if (match(TokenType::DoubleColon)) {
                if (check(TokenType::Identifier) || check(TokenType::Underscore)) {
                    path += "::";
                    path += std::string(advance().lexeme);
                    continue;
                } else {
                    error_at_current("Expected identifier or '_' after '::' in using declaration");
                    break;
                }
            }
            break;
        }

        consume(TokenType::Semicolon, "Expected ';' after using declaration");
        // For non-alias using declarations we store name as empty and target as the path
        return std::make_unique<UsingDeclaration>("", path, first.line);
    }

    // 'using namespace X;' form
    if (match(TokenType::Namespace)) {
        Token ns = consume(TokenType::Identifier, "Expected namespace name after 'namespace'");
        consume(TokenType::Semicolon, "Expected ';' after using namespace declaration");
        return std::make_unique<UsingDeclaration>("", std::string(ns.lexeme), ns.line);
    }

    // If we get here, it's a syntax error
    error_at_current("Expected identifier or 'namespace' in using declaration");
    return nullptr;
}

std::unique_ptr<Declaration> Parser::import_declaration() {
    Token module = consume(TokenType::Identifier, "Expected module name");
    consume(TokenType::Semicolon, "Expected ';' after import");

    return std::make_unique<ImportDeclaration>(std::string(module.lexeme), module.line);
}

std::unique_ptr<Declaration> Parser::template_declaration() {
    // Consume 'template' keyword if present (it may already be consumed in some call paths)
    match(TokenType::Template);
    
    consume(TokenType::LessThan, "Expected '<' after template");
    auto params = template_parameters();
    // Handle > or >> (via pending_gt mechanism from nested templates)
    if (!consume_template_close()) {
        consume(TokenType::GreaterThan, "Expected '>' after template parameters");
    }

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
    // Check for function type: (params) -> return_type
    // This syntax is used for function pointers and std::function template args
    if (check(TokenType::LeftParen)) {
        // Look ahead to see if this is a function type: (params) ->
        std::size_t saved = current;
        advance();  // consume (
        
        // Skip past matching parens to find ->
        int depth = 1;
        bool found_arrow = false;
        while (!is_at_end() && depth > 0) {
            if (check(TokenType::LeftParen)) {
                depth++;
            } else if (check(TokenType::RightParen)) {
                depth--;
            }
            advance();
        }
        
        // Check if followed by ->
        if (check(TokenType::Arrow)) {
            found_arrow = true;
        }
        
        current = saved;  // restore position
        
        if (found_arrow) {
            // Parse as function type by collecting the entire signature
            std::string func_type;
            advance();  // consume (
            
            // Collect parameter types
            depth = 1;
            while (!is_at_end() && depth > 0) {
                if (check(TokenType::LeftParen)) {
                    depth++;
                    func_type += "(";
                    advance();
                } else if (check(TokenType::RightParen)) {
                    depth--;
                    if (depth == 0) break;
                    func_type += ")";
                    advance();
                } else if (check(TokenType::Identifier) && peek().lexeme == "in") {
                    // Skip 'in' parameter kind
                    advance();
                    func_type += " ";
                } else if (check(TokenType::Identifier) && peek().lexeme == "inout") {
                    // Skip 'inout' parameter kind
                    advance();
                    func_type += " ";
                } else if (check(TokenType::Identifier) && peek().lexeme == "out") {
                    // Skip 'out' parameter kind
                    advance();
                    func_type += " ";
                } else if (check(TokenType::Identifier) && peek().lexeme == "copy") {
                    // Skip 'copy' parameter kind
                    advance();
                    func_type += " ";
                } else if (check(TokenType::Identifier) && peek().lexeme == "move") {
                    // Skip 'move' parameter kind
                    advance();
                    func_type += " ";
                } else if (check(TokenType::Identifier) && peek().lexeme == "forward") {
                    // Skip 'forward' parameter kind
                    advance();
                    func_type += " ";
                } else {
                    func_type += peek().lexeme;
                    advance();
                }
            }
            
            consume(TokenType::RightParen, "Expected ')' after function type parameters");
            
            // Parse -> return_type
            consume(TokenType::Arrow, "Expected '->' in function type");
            
            // Check for return kind: forward, move, etc.
            std::string return_kind;
            if (check(TokenType::Forward)) {
                return_kind = "forward ";
                advance();
            } else if (check(TokenType::Move)) {
                return_kind = "move ";
                advance();
            } else if (check(TokenType::Identifier)) {
                std::string_view kw = peek().lexeme;
                if (kw == "forward" || kw == "move") {
                    return_kind = std::string(kw) + " ";
                    advance();
                }
            }
            
            auto return_type = type();
            
            // Construct function type as C++ syntax: return_type(*)(params)
            // But for std::function, we need different output
            // Store as a special function type
            auto t = std::make_unique<Type>(Type::Kind::FunctionType);
            t->name = "(" + func_type + ") -> " + return_kind + 
                      (return_type ? return_type->name : "void");
            return t;
        }
    }
    
    // Check for C++1 style decltype(type) syntax
    if (match(TokenType::Decltype)) {
        // decltype( type ) - parse the parenthesized type/expression
        consume(TokenType::LeftParen, "Expected '(' after decltype");
        // Collect everything until matching ')'
        std::string decltype_content;
        int depth = 1;
        while (!is_at_end() && depth > 0) {
            if (check(TokenType::LeftParen)) {
                depth++;
            } else if (check(TokenType::RightParen)) {
                depth--;
                if (depth == 0) break;
            }
            decltype_content += std::string(advance().lexeme);
        }
        consume(TokenType::RightParen, "Expected ')' after decltype content");

        auto t = std::make_unique<Type>(Type::Kind::UserDefined);
        t->name = "decltype(" + decltype_content + ")";
        return t;
    }

    // Check for C++1 style type syntax: *const char, const char*, etc.
    // This starts with '*' or 'const' keyword
    // Also handle Cpp2 prefix pointer syntax: *int, *void, **int, * const int, etc.
    // Also handle Cpp2 const prefix: const _, const int, etc.
    if (check(TokenType::Asterisk) || check(TokenType::Const)) {
        // Check for Cpp2 prefix pointer syntax: *Type or **Type... or * const Type
        // This should be parsed as a proper Pointer type, not C++1 passthrough
        if (check(TokenType::Asterisk)) {
            std::size_t peek_pos = current + 1;
            // Check if next is another *, const, a simple type, 'type', 'auto', '_', or '(' (for function pointer)
            if (peek_pos < tokens.size() &&
                (tokens[peek_pos].type == TokenType::Asterisk ||
                 tokens[peek_pos].type == TokenType::Const ||
                 tokens[peek_pos].type == TokenType::Identifier ||
                 tokens[peek_pos].type == TokenType::Type ||
                 tokens[peek_pos].type == TokenType::Auto ||
                 tokens[peek_pos].type == TokenType::Underscore ||
                 tokens[peek_pos].type == TokenType::LeftParen)) {
                // This is Cpp2 prefix pointer syntax: *int, **int, *void, *auto, * const _, *(params)->ret etc.
                match(TokenType::Asterisk);  // Consume the *
                
                // Check for const after *: * const Type
                bool pointee_is_const = match(TokenType::Const);
                
                // For function pointer: * (params) -> ret
                // We need to recursively call type() which handles function types
                auto pointee_base = type();
                if (!pointee_base) {
                    error_at_current("Expected type after '*'");
                    return nullptr;
                }
                if (pointee_is_const) {
                    pointee_base->is_const = true;
                }
                auto ptr = std::make_unique<Type>(Type::Kind::Pointer);
                ptr->pointee = std::move(pointee_base);
                return ptr;
            }
        }

        // Check for Cpp2 const prefix: const Type
        if (check(TokenType::Const)) {
            std::size_t peek_pos = current + 1;
            // Check if next is a simple type or '_'
            if (peek_pos < tokens.size() &&
                (tokens[peek_pos].type == TokenType::Identifier ||
                 tokens[peek_pos].type == TokenType::Type ||
                 tokens[peek_pos].type == TokenType::Auto ||
                 tokens[peek_pos].type == TokenType::Underscore)) {
                // This is Cpp2 const prefix: const int, const _, etc.
                match(TokenType::Const);  // Consume const
                auto base_type = qualified_type();  // Parse the base type
                base_type->is_const = true;
                return base_type;
            }
        }

        // Otherwise, treat as C++1 passthrough
        // C++1 style pointer type: *const char, *char, const char*, etc.
        // For now, collect tokens until we hit a delimiter and store as raw C++1 type
        std::string cpp1_type;

        // Handle leading * (pointer)
        if (match(TokenType::Asterisk)) {
            cpp1_type += "*";
        }

        // Handle const qualifier
        if (match(TokenType::Const)) {
            cpp1_type += "const ";
        }

        // Now expect the base type
        if (check(TokenType::Identifier)) {
            cpp1_type += std::string(advance().lexeme);
        } else if (check(TokenType::Type)) {
            cpp1_type += "type";  // 'type' keyword
            advance();
        }

        // Handle trailing const qualifier
        while (match(TokenType::Const)) {
            cpp1_type += " const";
        }

        // Handle trailing * (pointer)
        while (match(TokenType::Asterisk)) {
            cpp1_type += "*";
        }

        auto t = std::make_unique<Type>(Type::Kind::UserDefined);
        t->name = cpp1_type;
        return t;
    }

    auto t = qualified_type();

    // Handle pointers, references (Cpp2 style - suffix)
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
    
    // If basic_type failed, return null
    if (!t) {
        return nullptr;
    }

    while (match(TokenType::DoubleColon)) {
        // For each :: component, we need to parse the next name and its template args
        Token name = consume(TokenType::Identifier, "Expected identifier after '::'");
        t->name += "::" + std::string(name.lexeme);

        // Check for template arguments on this component
        if (match(TokenType::LessThan)) {
            do {
                // Check for empty template argument list (e.g., std::plus<>)
                if (check(TokenType::GreaterThan) || check(TokenType::RightShift)) {
                    break;
                }
                // Template arguments can be types or expressions (including function calls like f(), o.f())
                // Use a token-collecting approach to handle complex expressions
                std::string arg_text;
                int angle_depth = 0;
                int paren_depth = 0;
                TokenType prev_type = TokenType::EndOfFile;
                [[maybe_unused]] bool split_right_shift = false;  // Set when we split >> into > and pending >
                
                while (!is_at_end()) {
                    // Check termination conditions at depth 0
                    if (angle_depth == 0 && paren_depth == 0) {
                        if (check(TokenType::GreaterThan) || check(TokenType::RightShift) ||
                            check(TokenType::Comma)) {
                            break;
                        }
                    }
                    
                    // Track nesting
                    if (check(TokenType::LessThan)) {
                        angle_depth++;
                    } else if (check(TokenType::GreaterThan)) {
                        if (angle_depth > 0) angle_depth--;
                        else break;  // This > closes our template
                    } else if (check(TokenType::RightShift)) {
                        // >> could close two levels or one level + our template
                        if (angle_depth >= 2) {
                            // Add >> to close two nested templates
                            arg_text += ">>";
                            advance();
                            angle_depth -= 2;
                            continue;  // Skip the normal token add below
                        } else if (angle_depth == 1) {
                            // Split >>: add one > to close the nested template,
                            // set pending_gt for the outer template
                            arg_text += ">";
                            split_right_shift = true;
                            pending_gt = true;
                            angle_depth--;
                            break;  // Stop - the outer close will handle the second >
                        } else {
                            break;  // Both > close outer templates
                        }
                    } else if (check(TokenType::LeftParen)) {
                        paren_depth++;
                    } else if (check(TokenType::RightParen)) {
                        if (paren_depth > 0) paren_depth--;
                    }
                    
                    // Add spacing intelligently
                    if (!arg_text.empty()) {
                        TokenType curr = peek().type;
                        bool no_space_before = (curr == TokenType::RightParen ||
                                               curr == TokenType::GreaterThan ||
                                               curr == TokenType::DoubleColon ||
                                               curr == TokenType::Dot ||
                                               curr == TokenType::Comma);
                        bool no_space_after = (prev_type == TokenType::LeftParen ||
                                              prev_type == TokenType::LessThan ||
                                              prev_type == TokenType::DoubleColon ||
                                              prev_type == TokenType::Dot);
                        if (!no_space_before && !no_space_after) {
                            arg_text += " ";
                        }
                    }
                    
                    arg_text += std::string(peek().lexeme);
                    prev_type = peek().type;
                    advance();
                }
                
                if (!arg_text.empty()) {
                    auto arg_type = std::make_unique<Type>(Type::Kind::UserDefined);
                    arg_type->name = arg_text;
                    if (match(TokenType::TripleDot) || match(TokenType::Ellipsis)) {
                        arg_type->name += "...";
                    }
                    t->template_args.push_back(std::move(arg_type));
                }
            } while (match(TokenType::Comma));
            // Handle both > and >> (for nested templates)
            if (!consume_template_close()) {
                consume(TokenType::GreaterThan, "Expected '>' after template arguments");
            }
            t->kind = Type::Kind::Template;
        }
    }

    return t;
}

std::unique_ptr<Type> Parser::basic_type() {
    // Handle multi-word C types like: unsigned char, long long, unsigned int, etc.
    // Also: signed/unsigned/short/long modifiers before int/char/float/double
    std::string type_name;
    
    // Check for type modifiers: unsigned, signed, short, long
    while (check(TokenType::Identifier)) {
        std::string_view lexeme = peek().lexeme;
        if (lexeme == "unsigned" || lexeme == "signed" || 
            lexeme == "short" || lexeme == "long") {
            if (!type_name.empty()) type_name += " ";
            type_name += lexeme;
            advance();
        } else {
            break;
        }
    }
    
    // Check for base type: int, char, double, float, void
    if (check(TokenType::Identifier)) {
        std::string_view lexeme = peek().lexeme;
        if (lexeme == "int" || lexeme == "char" || lexeme == "double" || 
            lexeme == "float" || lexeme == "void" || lexeme == "bool" ||
            type_name.empty()) {  // If no modifiers, allow any identifier
            if (!type_name.empty()) type_name += " ";
            type_name += lexeme;
            advance();
        }
    }
    
    if (!type_name.empty()) {
        auto t = std::make_unique<Type>(Type::Kind::UserDefined);
        t->name = type_name;

        // Check for template arguments
        if (match(TokenType::LessThan)) {
            // Template arguments can be types or constant expressions
            // Track nesting depth for proper > matching
            do {
                // Check for empty template argument list (e.g., std::plus<>)
                if (check(TokenType::GreaterThan) || check(TokenType::RightShift)) {
                    break;
                }
                // Template arguments can be types or expressions (including function calls like f(), o.f())
                // Use a token-collecting approach to handle complex expressions
                std::string arg_text;
                int angle_depth = 0;
                int paren_depth = 0;
                TokenType prev_type = TokenType::EndOfFile;
                [[maybe_unused]] bool split_right_shift = false;  // Set when we split >> into > and pending >
                
                while (!is_at_end()) {
                    // Check termination conditions at depth 0
                    if (angle_depth == 0 && paren_depth == 0) {
                        if (check(TokenType::GreaterThan) || check(TokenType::RightShift) ||
                            check(TokenType::Comma)) {
                            break;
                        }
                    }
                    
                    // Track nesting
                    if (check(TokenType::LessThan)) {
                        angle_depth++;
                    } else if (check(TokenType::GreaterThan)) {
                        if (angle_depth > 0) angle_depth--;
                        else break;  // This > closes our template
                    } else if (check(TokenType::RightShift)) {
                        // >> could close two levels or one level + our template
                        if (angle_depth >= 2) {
                            // Add >> to close two nested templates
                            arg_text += ">>";
                            advance();
                            angle_depth -= 2;
                            continue;  // Skip the normal token add below
                        } else if (angle_depth == 1) {
                            // Split >>: add one > to close the nested template,
                            // set pending_gt for the outer template
                            arg_text += ">";
                            split_right_shift = true;
                            pending_gt = true;
                            angle_depth--;
                            break;  // Stop - the outer close will handle the second >
                        } else {
                            break;  // Both > close outer templates
                        }
                    } else if (check(TokenType::LeftParen)) {
                        paren_depth++;
                    } else if (check(TokenType::RightParen)) {
                        if (paren_depth > 0) paren_depth--;
                    }
                    
                    // Add spacing intelligently
                    if (!arg_text.empty()) {
                        TokenType curr = peek().type;
                        bool no_space_before = (curr == TokenType::RightParen ||
                                               curr == TokenType::GreaterThan ||
                                               curr == TokenType::DoubleColon ||
                                               curr == TokenType::Dot ||
                                               curr == TokenType::Comma);
                        bool no_space_after = (prev_type == TokenType::LeftParen ||
                                              prev_type == TokenType::LessThan ||
                                              prev_type == TokenType::DoubleColon ||
                                              prev_type == TokenType::Dot);
                        if (!no_space_before && !no_space_after) {
                            arg_text += " ";
                        }
                    }
                    
                    arg_text += std::string(peek().lexeme);
                    prev_type = peek().type;
                    advance();
                }
                
                if (!arg_text.empty()) {
                    // Store as a pseudo-type with the expression text as its name
                    auto arg_type = std::make_unique<Type>(Type::Kind::UserDefined);
                    arg_type->name = arg_text;
                    // Check for pack expansion: ...
                    if (match(TokenType::TripleDot) || match(TokenType::Ellipsis)) {
                        arg_type->name += "...";
                    }
                    t->template_args.push_back(std::move(arg_type));
                }
            } while (match(TokenType::Comma));
            // Handle both > and >> (for nested templates like X<Y<Z>>)
            if (!consume_template_close()) {
                consume(TokenType::GreaterThan, "Expected '>' after template arguments");
            }
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
        // Check for type constraint: _ is constraint
        if (match(TokenType::Is)) {
            // Parse the constraint expression as a type name
            auto constraint = qualified_type();
            if (constraint) {
                t->name = "_ is " + constraint->name;
                // Copy template args from constraint if present
                t->template_args = std::move(constraint->template_args);
            }
        }
        return t;
    }

    // 'type' keyword can be used as a type name in some contexts
    if (match(TokenType::Type)) {
        auto t = std::make_unique<Type>(Type::Kind::UserDefined);
        t->name = "type";
        return t;
    }
    
    // 'base' keyword can be used as a type name (user-defined type called "base")
    if (match(TokenType::Base)) {
        auto t = std::make_unique<Type>(Type::Kind::UserDefined);
        t->name = "base";
        return t;
    }

    // in_ref type alias: in_ref → auto const&
    if (match(TokenType::InRef)) {
        auto t = std::make_unique<Type>(Type::Kind::UserDefined);
        t->name = "auto const&";
        return t;
    }

    // forward_ref type alias: forward_ref → auto&&
    if (match(TokenType::ForwardRef)) {
        auto t = std::make_unique<Type>(Type::Kind::UserDefined);
        t->name = "auto&&";
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
    // Cpp2 syntax: if <condition> { <then_body> } else { <else_body> }
    // C syntax: if (<condition>) { <then_body> } else { <else_body> }
    // Also: if constexpr <condition> { ... } for compile-time branching

    bool is_constexpr = false;
    if (check(TokenType::Identifier) && peek().lexeme == "constexpr") {
        advance();  // consume 'constexpr'
        is_constexpr = true;
    }
    
    std::unique_ptr<Expression> condition = nullptr;

    // Check if condition is in parentheses (traditional syntax)
    if (match(TokenType::LeftParen)) {
        condition = expression();
        consume(TokenType::RightParen, "Expected ')' after if condition");
    } else {
        // Cpp2 syntax without parentheses
        condition = expression();
    }

    auto then_stmt = statement();

    std::unique_ptr<Statement> else_stmt = nullptr;
    if (match(TokenType::Else)) {
        else_stmt = statement();
    }

    return std::make_unique<IfStatement>(
        std::move(condition),
        std::move(then_stmt),
        std::move(else_stmt),
        peek().line,
        is_constexpr
    );
}

std::unique_ptr<Statement> Parser::while_statement() {
    // Cpp2 syntax: while <cond> next <increment> { <body> }
    // Traditional C syntax: while (<cond>) { <body> }

    std::unique_ptr<Expression> condition = nullptr;
    std::unique_ptr<Expression> increment = nullptr;

    // Check if condition is in parentheses (traditional syntax)
    if (match(TokenType::LeftParen)) {
        condition = expression();
        consume(TokenType::RightParen, "Expected ')' after while condition");

        // Check for Cpp2 'next' clause after closing paren
        if (match(TokenType::Next)) {
            increment = expression();
        }
    } else {
        // Cpp2 syntax without parentheses: while cond next increment { body }
        condition = expression();

        // Check for Cpp2 'next' clause
        if (match(TokenType::Next)) {
            increment = expression();
        }
    }

    auto body = statement();

    if (increment) {
        return std::make_unique<WhileStatement>(
            std::move(condition),
            std::move(increment),
            std::move(body),
            peek().line
        );
    }

    return std::make_unique<WhileStatement>(
        std::move(condition),
        std::move(body),
        peek().line
    );
}

std::unique_ptr<Statement> Parser::do_while_statement() {
    // Cpp2 do-while syntax: do { body } next increment while condition
    // Or: label: do { ... } next inc while cond

    consume(TokenType::LeftBrace, "Expected '{' after 'do'");
    auto body = block_statement();

    std::unique_ptr<Expression> increment = nullptr;
    if (match(TokenType::Next)) {
        increment = expression();
    }

    consume(TokenType::While, "Expected 'while' after do-while body (or next clause)");
    auto condition = expression();

    consume(TokenType::Semicolon, "Expected ';' after do-while statement");

    if (increment) {
        return std::make_unique<DoWhileStatement>(
            std::move(body),
            std::move(increment),
            std::move(condition),
            previous().line
        );
    }

    return std::make_unique<DoWhileStatement>(
        std::move(body),
        std::move(condition),
        previous().line
    );
}

std::unique_ptr<Statement> Parser::for_statement() {
    // Try to parse as Cpp2 "for collection do(var)" syntax first
    // This handles both:
    //   for collection do(var) { body }
    //   for (expr).method() do(var) { body }  - where expr starts with paren

    std::size_t saved_pos = current;
    bool saved_panic = panic_mode;
    suppress_errors = true;  // Speculative parse

    auto collection = expression();

    // Check if this could be Cpp2 syntax (followed by 'next' or 'do')
    bool is_cpp2_for = collection && (check(TokenType::Next) || check(TokenType::Do));

    suppress_errors = false;

    if (is_cpp2_for) {
        // cpp2: for <collection> do(<var>) { body }
        // cpp2: for <collection> next <expr> do(<var>) { body }
        // cpp2: for <collection> do(inout var) { body }
        // cpp2: for <collection> do(copy var) { body }

        // Optional: next clause comes before do
        std::unique_ptr<Expression> next_clause = nullptr;
        if (match(TokenType::Next)) {
            next_clause = expression();
        }

        consume(TokenType::Do, "Expected 'do' in for-do loop");
        consume(TokenType::LeftParen, "Expected '(' after 'do'");

        // Handle optional parameter modifiers: in, inout, out, copy, move, forward, in_ref, forward_ref
        std::string var_kind;
        if (match(TokenType::In) || match(TokenType::Inout) || match(TokenType::Out) || 
            match(TokenType::Copy) || match(TokenType::Move) || match(TokenType::Forward) ||
            match(TokenType::InRef) || match(TokenType::ForwardRef)) {
            var_kind = std::string(previous().lexeme);
        }

        // Variable name can be identifier or underscore (discard)
        if (!check(TokenType::Identifier) && !check(TokenType::Underscore)) {
            error_at_current("Expected variable name in do clause");
            return nullptr;
        }
        Token var = advance();

        consume(TokenType::RightParen, "Expected ')' after variable in do clause");

        auto body = statement();

        // Create a ForRangeStatement to represent this
        auto stmt = std::make_unique<ForRangeStatement>(
            std::string(var.lexeme),
            nullptr,  // type will be deduced
            std::move(collection),
            std::move(body),
            var.line,
            std::move(var_kind)
        );
        stmt->next_clause = std::move(next_clause);
        return stmt;
    }

    // Not Cpp2 syntax, backtrack and parse as traditional C-style for loop
    current = saved_pos;
    panic_mode = saved_panic;

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

std::unique_ptr<Statement> Parser::labeled_loop_statement(std::string label) {
    // Called when we've already seen 'label:' and confirmed it's followed by while/for/do
    // The next token should be the loop keyword
    if (match(TokenType::While)) {
        auto stmt = while_statement();
        // Add label to the statement
        if (auto* while_stmt = dynamic_cast<WhileStatement*>(stmt.get())) {
            while_stmt->label = std::move(label);
        }
        return stmt;
    }
    if (match(TokenType::For)) {
        // Need to check if this is for-range (for collection do(var)) or traditional for
        if (!check(TokenType::LeftParen)) {
            // for collection do(var) - for-each loop
            auto collection = expression();
            consume(TokenType::Do, "Expected 'do' in for-do loop");
            consume(TokenType::LeftParen, "Expected '(' after 'do'");
            
            // Handle optional parameter modifiers: in, inout, out, copy, move, forward, in_ref, forward_ref
            std::string var_kind;
            if (match(TokenType::In) || match(TokenType::Inout) || match(TokenType::Out) || 
                match(TokenType::Copy) || match(TokenType::Move) || match(TokenType::Forward) ||
                match(TokenType::InRef) || match(TokenType::ForwardRef)) {
                var_kind = std::string(previous().lexeme);
            }
            
            // Variable name can be identifier or underscore (discard)
            if (!check(TokenType::Identifier) && !check(TokenType::Underscore)) {
                error_at_current("Expected variable name in do clause");
                return nullptr;
            }
            Token var = advance();
            consume(TokenType::RightParen, "Expected ')' after variable in do clause");
            auto body = statement();

            auto for_range = std::make_unique<ForRangeStatement>(
                std::string(var.lexeme),
                nullptr,
                std::move(collection),
                std::move(body),
                var.line,
                std::move(var_kind)
            );
            for_range->label = std::move(label);
            return for_range;
        } else {
            // Traditional for loop - need to implement since for_statement() assumes no label
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

            auto for_stmt = std::make_unique<ForStatement>(
                std::move(init),
                std::move(condition),
                std::move(increment),
                std::move(body),
                peek().line
            );
            for_stmt->label = std::move(label);
            return for_stmt;
        }
    }
    if (match(TokenType::Do)) {
        // do-while with next clause: do { body } next increment while condition
        // Or: label: do { ... } next inc while cond
        auto stmt = do_while_statement();
        if (auto* do_stmt = dynamic_cast<DoWhileStatement*>(stmt.get())) {
            do_stmt->label = std::move(label);
        }
        return stmt;
    }

    error_at_current("Expected loop statement after label");
    return nullptr;
}

std::unique_ptr<Statement> Parser::try_loop_with_initializer() {
    // Try to parse: (copy/move name := expr, copy/move name2 := expr2, ...) while/for/do
    // Returns nullptr if this doesn't match the pattern
    
    std::size_t saved = current;
    
    if (!match(TokenType::LeftParen)) {
        return nullptr;
    }
    
    std::vector<std::unique_ptr<LoopInitializer>> initializers;
    
    // Parse one or more initializers separated by commas
    do {
        // Check for parameter kind keyword (copy, move, inout for scope blocks)
        std::string param_kind;
        if (match(TokenType::Copy) || match(TokenType::Move) || match(TokenType::Inout)) {
            param_kind = std::string(previous().lexeme);
        } else if (is_identifier_like()) {
            // No qualifier - check if this still looks like an initializer pattern
            // Must have identifier := value or identifier : type = value
            std::size_t peek_pos = current;
            advance();  // skip identifier
            if (!check(TokenType::ColonEqual) && !check(TokenType::Colon)) {
                current = saved;
                return nullptr;
            }
            current = peek_pos;  // restore to identifier position
        } else if (check(TokenType::RightParen)) {
            // Handle trailing comma: (copy a := 42,)
            break;
        } else {
            // Not a loop initializer pattern
            current = saved;
            return nullptr;
        }
        
        // Need identifier
        if (!is_identifier_like()) {
            current = saved;
            return nullptr;
        }
        Token name = advance();
        
        // Need := or : type =
        std::unique_ptr<Type> var_type = nullptr;
        std::unique_ptr<Expression> init_value = nullptr;
        
        if (match(TokenType::ColonEqual)) {
            // Type-deduced: copy i := 0
            var_type = std::make_unique<Type>(Type::Kind::Auto);
            var_type->name = "auto";
            init_value = expression();
        } else if (match(TokenType::Colon)) {
            // Typed: copy i : int = 0
            if (!check(TokenType::Equal)) {
                var_type = type();
            }
            if (!match(TokenType::Equal)) {
                current = saved;
                return nullptr;
            }
            init_value = expression();
        } else {
            current = saved;
            return nullptr;
        }
        
        // Add this initializer to the list
        initializers.push_back(std::make_unique<LoopInitializer>(
            std::string(name.lexeme),
            std::move(param_kind),
            std::move(var_type),
            std::move(init_value)
        ));
        
    } while (match(TokenType::Comma) && !check(TokenType::RightParen));
    
    if (!match(TokenType::RightParen)) {
        current = saved;
        return nullptr;
    }
    
    // Statement scope: (decl) stmt
    // Handle loops specially to attach initializers to loop statements
    // For any other statement, wrap in a ScopeBlockStatement
    
    // Check for scope block: (copy x := value) { body }
    if (check(TokenType::LeftBrace)) {
        advance();  // consume '{'
        auto body = block_statement();
        // Wrap in a scope statement with the initializers
        auto scope_stmt = std::make_unique<ScopeBlockStatement>(std::move(initializers), std::move(body), previous().line);
        return scope_stmt;
    }
    
    // Parse the loop statement
    if (match(TokenType::While)) {
        auto stmt = while_statement();
        if (auto* while_stmt = dynamic_cast<WhileStatement*>(stmt.get())) {
            while_stmt->loop_inits = std::move(initializers);
        }
        return stmt;
    }
    if (match(TokenType::For)) {
        // Could be for-range or traditional for
        if (!check(TokenType::LeftParen)) {
            // for-range: for collection do(var)
            auto collection = expression();
            
            std::unique_ptr<Expression> next_clause = nullptr;
            if (match(TokenType::Next)) {
                next_clause = expression();
            }
            
            consume(TokenType::Do, "Expected 'do' in for-do loop");
            consume(TokenType::LeftParen, "Expected '(' after 'do'");
            
            std::string var_kind;
            if (match(TokenType::In) || match(TokenType::Inout) || match(TokenType::Out) || 
                match(TokenType::Copy) || match(TokenType::Move) || match(TokenType::Forward) ||
                match(TokenType::InRef) || match(TokenType::ForwardRef)) {
                var_kind = std::string(previous().lexeme);
            }
            
            // Variable name can be identifier or underscore (discard)
            if (!check(TokenType::Identifier) && !check(TokenType::Underscore)) {
                error_at_current("Expected variable name in do clause");
                return nullptr;
            }
            Token var = advance();
            consume(TokenType::RightParen, "Expected ')' after variable in do clause");
            
            auto body = statement();
            
            auto for_range = std::make_unique<ForRangeStatement>(
                std::string(var.lexeme),
                nullptr,
                std::move(collection),
                std::move(body),
                var.line,
                std::move(var_kind)
            );
            for_range->next_clause = std::move(next_clause);
            for_range->loop_inits = std::move(initializers);
            return for_range;
        }
        // Traditional for loop with initializer - less common but possible
        auto stmt = for_statement();
        // Traditional for loops already have an init, so this is unusual
        return stmt;
    }
    if (match(TokenType::Do)) {
        auto stmt = do_while_statement();
        if (auto* do_stmt = dynamic_cast<DoWhileStatement*>(stmt.get())) {
            do_stmt->loop_inits = std::move(initializers);
        }
        return stmt;
    }
    
    // For any other statement (assert, if, expression, etc.), wrap in a ScopeBlockStatement
    auto body = statement();
    if (!body) {
        current = saved;
        return nullptr;
    }
    auto scope_stmt = std::make_unique<ScopeBlockStatement>(std::move(initializers), std::move(body), previous().line);
    return scope_stmt;
}

std::unique_ptr<Statement> Parser::switch_statement() {
    consume(TokenType::LeftParen, "Expected '(' after 'switch'");
    auto value = expression();
    consume(TokenType::RightParen, "Expected ')' after switch value");

    consume(TokenType::LeftBrace, "Expected '{' after switch");

    auto switch_stmt = std::make_unique<SwitchStatement>(std::move(value), value->line);

    while (!check(TokenType::RightBrace) && !is_at_end()) {
        std::size_t before = current;
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
            // If we haven't advanced after the error, break to avoid infinite loop
            if (current == before) {
                break;
            }
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

std::unique_ptr<Expression> Parser::inspect_expression() {
    // cpp2 syntax: inspect <value> -> <type> { is <pattern> = <result>; }
    auto value = expression();
    std::size_t line = value->line;

    std::unique_ptr<Type> result_type = nullptr;
    if (match(TokenType::Arrow)) {
        result_type = type();
    }

    consume(TokenType::LeftBrace, "Expected '{' after inspect value");

    auto inspect = std::make_unique<InspectExpression>(std::move(value), line);
    inspect->result_type = std::move(result_type);

    // Parse arms: is <pattern> = <result>;
    while (!check(TokenType::RightBrace) && !is_at_end()) {
        std::size_t before = current;
        if (!consume_if(TokenType::Is)) {
            // Not an 'is' keyword - check for other constructs or break
            error_at_current("Expected 'is' in inspect arm");
            // If we haven't advanced after the error, break to avoid infinite loop
            if (current == before) {
                break;
            }
            continue;
        }

        InspectExpression::Arm arm;

        // Parse pattern
        if (match(TokenType::Underscore)) {
            arm.pattern_kind = InspectExpression::Arm::PatternKind::Wildcard;
        } else {
            // Try to determine if this is a type pattern or value pattern
            // Type patterns: is std::string, is int, is std::variant<int, string>
            // Value patterns: is 42, is "hello", is (x > 0)
            
            // Heuristic: if it starts with an identifier (possibly qualified) followed
            // by '=' or starts with a template like identifier<...>, it's likely a type pattern
            
            // Save position to potentially backtrack
            std::size_t pattern_start = current;
            bool is_type_pattern = false;
            
            // Check if this looks like a type pattern
            // Types typically start with: identifier, qualified names (std::string),
            // or template types (variant<...>)
            if (check(TokenType::Identifier)) {
                // Try to parse as a type
                auto maybe_type = type();
                
                // If we successfully parsed a type and next is '=', it's a type pattern
                if (maybe_type && check(TokenType::Equal)) {
                    is_type_pattern = true;
                    arm.pattern_kind = InspectExpression::Arm::PatternKind::Type;
                    arm.pattern_type = std::move(maybe_type);
                } else {
                    // Backtrack and parse as value
                    current = pattern_start;
                }
            }
            
            if (!is_type_pattern) {
                // Parse as value pattern
                // Use ternary_expression to avoid parsing '=' as assignment
                arm.pattern_kind = InspectExpression::Arm::PatternKind::Value;
                arm.pattern_value = ternary_expression();
            }
        }

        consume(TokenType::Equal, "Expected '=' after pattern");

        // Parse result value
        arm.result_value = expression();

        consume(TokenType::Semicolon, "Expected ';' after inspect arm");

        inspect->arms.push_back(std::move(arm));

        // If we haven't advanced at all after parsing an arm, break to avoid infinite loop
        if (current == before) {
            break;
        }
    }

    consume(TokenType::RightBrace, "Expected '}' after inspect arms");

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
    // Cpp2 supports: break; or break label;
    std::string label;
    if (check(TokenType::Identifier)) {
        label = std::string(advance().lexeme);
    }
    consume(TokenType::Semicolon, "Expected ';' after break");
    if (label.empty()) {
        return std::make_unique<BreakStatement>(previous().line);
    }
    return std::make_unique<BreakStatement>(std::move(label), previous().line);
}

std::unique_ptr<Statement> Parser::continue_statement() {
    // Cpp2 supports: continue; or continue label;
    std::string label;
    if (check(TokenType::Identifier)) {
        label = std::string(advance().lexeme);
    }
    consume(TokenType::Semicolon, "Expected ';' after continue");
    if (label.empty()) {
        return std::make_unique<ContinueStatement>(previous().line);
    }
    return std::make_unique<ContinueStatement>(std::move(label), previous().line);
}

std::unique_ptr<Statement> Parser::try_statement() {
    auto try_block = block_statement();

    auto try_stmt = std::make_unique<TryStatement>(std::move(try_block), try_block->line);

    while (match(TokenType::Catch)) {
        consume(TokenType::LeftParen, "Expected '(' after catch");
        Token exception_type = consume(TokenType::Identifier, "Expected exception type");
        [[maybe_unused]] Token exception_name = consume(TokenType::Identifier, "Expected exception name");
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

    // Check for annotation level syntax: assert<name>(condition)
    // vs function contract syntax: assert : condition
    [[maybe_unused]] bool has_annotation = false;
    std::optional<std::string> annotation;

    if (match(TokenType::LessThan)) {
        // Parse annotation level: <bounds_safety>, <bounds_safety, audit>, etc.
        // Multiple predicates separated by commas are allowed
        std::vector<std::string> annotations;
        do {
            if (check(TokenType::Identifier)) {
                Token ann = advance();
                annotations.push_back(std::string(ann.lexeme));
            }
        } while (match(TokenType::Comma));
        
        consume(TokenType::GreaterThan, "Expected '>' after contract annotation");
        has_annotation = true;
        
        // Use first annotation as primary (additional predicates are control flags)
        if (!annotations.empty()) {
            annotation = annotations[0];
        }
    }

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
    if (annotation) {
        contract_expr->annotation = *annotation;
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
    auto expr = pipeline_expression();

    if (match({TokenType::Equal, TokenType::PlusEqual, TokenType::MinusEqual,
               TokenType::AsteriskEqual, TokenType::SlashEqual, TokenType::PercentEqual,
               TokenType::LeftShiftEqual, TokenType::RightShiftEqual,
               TokenType::AmpersandEqual, TokenType::PipeEqual, TokenType::CaretEqual})) {
        Token op = previous();
        auto value = assignment_expression();

        if (!expr) {
             // Error recovery: we have an operator but no LHS.
             // This can happen if the LHS expression failed to parse.
             // We can't create a BinaryExpression with null LHS.
             return nullptr;
        }

        expr = std::make_unique<BinaryExpression>(
            std::move(expr),
            op.type,
            std::move(value),
            expr->line
        );
    }

    return expr;
}

// Pipeline expression - handles |> operator for left-to-right function composition
// a |> b |> c is parsed as c(b(a)) - left associative
// Precedence: lower than assignment, so x = a |> b means x = (a |> b)
std::unique_ptr<Expression> Parser::pipeline_expression() {
    auto expr = ternary_expression();

    while (match(TokenType::Pipeline)) {
        Token op = previous();
        auto right = ternary_expression();  // Right side is the function to apply

        if (!expr) {
            // Error recovery
            return nullptr;
        }

        // Create pipeline expression: left |> right
        expr = std::make_unique<PipelineExpression>(
            std::move(expr),
            std::move(right),
            op.line
        );
    }

    return expr;
}

// Special requires constraint expression parser that stops at '=' and '=='
// These are function body separators in Cpp2, not comparison operators in requires clauses
std::unique_ptr<Expression> Parser::requires_constraint_expression() {
    // Parse like a comparison expression but exclude == as an operator
    auto expr = requires_comparison_expression();
    
    // Handle && and || for compound constraints
    while (match({TokenType::DoubleAmpersand, TokenType::DoublePipe})) {
        Token op = previous();
        auto right = requires_comparison_expression();
        if (!expr) return nullptr;
        expr = std::make_unique<BinaryExpression>(
            std::move(expr),
            op.type,
            std::move(right),
            expr->line
        );
    }
    
    return expr;
}

// Comparison for requires clauses - excludes == and !=
std::unique_ptr<Expression> Parser::requires_comparison_expression() {
    auto expr = prefix_expression();
    
    // Parse template instantiations and postfix operations
    while (true) {
        if (check(TokenType::LessThan) && is_template_start()) {
            std::size_t saved = current;
            bool saved_panic = panic_mode;
            suppress_errors = true;
            
            advance();  // consume <
            std::vector<std::unique_ptr<Type>> template_args;
            bool success = true;
            
            if (!check(TokenType::GreaterThan)) {
                do {
                    if (check(TokenType::IntegerLiteral) || check(TokenType::FloatLiteral) ||
                        check(TokenType::True) || check(TokenType::False)) {
                        auto const_type = std::make_unique<Type>(Type::Kind::Builtin);
                        const_type->name = std::string(advance().lexeme);
                        template_args.push_back(std::move(const_type));
                    } else {
                        auto arg = type();
                        if (!arg || panic_mode) {
                            success = false;
                            break;
                        }
                        template_args.push_back(std::move(arg));
                    }
                } while (match(TokenType::Comma));
            }
            
            suppress_errors = false;
            
            if (success && !panic_mode && match(TokenType::GreaterThan)) {
                panic_mode = saved_panic;
                auto inst = std::make_unique<CallExpression>(std::move(expr), expr->line);
                inst->template_args = std::move(template_args);
                inst->is_template_instantiation = true;
                expr = std::move(inst);
            } else {
                current = saved;
                panic_mode = saved_panic;
                break;
            }
        } else if (match(TokenType::DoubleColon)) {
            Token member = consume(TokenType::Identifier, "Expected identifier after '::'");
            expr = std::make_unique<MemberAccessExpression>(
                std::move(expr),
                "::" + std::string(member.lexeme),
                member.line
            );
        } else {
            break;
        }
    }
    
    return expr;
}

std::unique_ptr<Expression> Parser::ternary_expression() {
    auto expr = logical_or_expression();

    if (match(TokenType::Question)) {
        auto then_expr = expression();
        consume(TokenType::Colon, "Expected ':' in ternary expression");
        auto else_expr = ternary_expression();

        if (!expr) return nullptr;

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
        if (!expr) return nullptr;
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
        if (!expr) return nullptr;
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
        if (!expr) return nullptr;
        expr = std::make_unique<BinaryExpression>(std::move(expr), op.type, std::move(right), expr->line);
    }
    return expr;
}

std::unique_ptr<Expression> Parser::bitwise_xor_expression() {
    auto expr = bitwise_and_expression();
    while (match(TokenType::Caret)) {
        Token op = previous();
        auto right = bitwise_and_expression();
        if (!expr) return nullptr;
        expr = std::make_unique<BinaryExpression>(std::move(expr), op.type, std::move(right), expr->line);
    }
    return expr;
}

std::unique_ptr<Expression> Parser::bitwise_or_expression() {
    auto expr = bitwise_xor_expression();
    while (match(TokenType::Pipe)) {
        Token op = previous();
        auto right = bitwise_xor_expression();
        if (!expr) return nullptr;
        expr = std::make_unique<BinaryExpression>(std::move(expr), op.type, std::move(right), expr->line);
    }
    return expr;
}

std::unique_ptr<Expression> Parser::equality_expression() {
    auto expr = comparison_expression();

    while (match({TokenType::DoubleEqual, TokenType::NotEqual})) {
        Token op = previous();
        auto right = comparison_expression();
        if (!expr) return nullptr;
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
        if (!expr) return nullptr;
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
        if (!expr) return nullptr;
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
        if (!expr) return nullptr;
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
        if (!expr) return nullptr;
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
        if (!expr) return nullptr;
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

    // Cpp2 move/forward expressions: move x, forward x, copy x
    if (match({TokenType::Move, TokenType::Forward, TokenType::Copy})) {
        Token op = previous();
        auto operand = prefix_expression();
        // Create a call expression like std::move(operand) or std::forward<decltype(operand)>(operand)
        return std::make_unique<MoveExpression>(
            op.type,
            std::move(operand),
            op.line
        );
    }

    if (match({TokenType::Plus, TokenType::Minus, TokenType::Exclamation, TokenType::Tilde, TokenType::PlusPlus,
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
    
    // If primary_expression failed (returned nullptr), bail out early
    if (!expr) {
        return nullptr;
    }

    while (true) {
        if (match(TokenType::LeftParen)) {
            expr = call_expression(std::move(expr));
        } else if (check(TokenType::LessThan)) {
            // Try to parse as template arguments: func<T>() or identifier<T>
            // Valid if followed by type arguments and >
            // May optionally be followed by ( for a function call
            std::size_t saved = current;
            bool saved_panic = panic_mode;
            panic_mode = false;
            suppress_errors = true;  // Speculative parse - don't emit errors
            advance(); // consume <
            
            std::vector<std::unique_ptr<Type>> template_args;
            bool success = true;
            
            // Try to parse template argument list
            if (!check(TokenType::GreaterThan)) {
                do {
                    // Check for NTTP (non-type template parameters): numeric literals
                    if (check(TokenType::IntegerLiteral) || check(TokenType::FloatLiteral) ||
                        check(TokenType::True) || check(TokenType::False)) {
                        // Create a pseudo-type for the constant value
                        auto const_type = std::make_unique<Type>(Type::Kind::Builtin);
                        const_type->name = std::string(advance().lexeme);
                        template_args.push_back(std::move(const_type));
                    } else if (check(TokenType::Minus)) {
                        // Negative literal: -42
                        std::string const_val = "-";
                        advance();  // consume -
                        if (check(TokenType::IntegerLiteral) || check(TokenType::FloatLiteral)) {
                            const_val += advance().lexeme;
                        }
                        auto const_type = std::make_unique<Type>(Type::Kind::Builtin);
                        const_type->name = const_val;
                        template_args.push_back(std::move(const_type));
                    } else if (check(TokenType::Identifier) && current + 1 < tokens.size() && tokens[current + 1].type == TokenType::LeftParen) {
                        // Non-type template parameter: identifier(args) like CPP2_TYPEOF(x) or func(args)
                        std::string nttp_text;
                        nttp_text += advance().lexeme;  // identifier
                        nttp_text += advance().lexeme;  // (

                        // Parse the argument list, tracking parens
                        int paren_depth = 1;
                        while (!is_at_end() && paren_depth > 0) {
                            if (check(TokenType::LeftParen)) paren_depth++;
                            else if (check(TokenType::RightParen)) paren_depth--;
                            nttp_text += advance().lexeme;
                        }

                        auto const_type = std::make_unique<Type>(Type::Kind::Builtin);
                        const_type->name = nttp_text;
                        template_args.push_back(std::move(const_type));
                    } else {
                        auto arg = type();
                        if (!arg || panic_mode) {
                            success = false;
                            break;
                        }
                        template_args.push_back(std::move(arg));
                    }
                } while (match(TokenType::Comma));
            }
            
            suppress_errors = false;  // Re-enable errors
            
            if (success && !panic_mode && match(TokenType::GreaterThan)) {
                // Valid template instantiation: identifier<T, U>
                panic_mode = saved_panic;  // Restore panic state
                
                if (check(TokenType::LeftParen)) {
                    // Template call: func<T, U>(args)
                    advance(); // consume (
                    auto call = std::make_unique<CallExpression>(std::move(expr), expr->line);
                    call->template_args = std::move(template_args);
                    
                    if (!check(TokenType::RightParen)) {
                        do {
                            call->args.push_back(expression());
                        } while (match(TokenType::Comma));
                    }
                    consume(TokenType::RightParen, "Expected ')' after arguments");
                    expr = std::move(call);
                } else if (check(TokenType::DoubleColon)) {
                    // Nested qualified access after template: Type<T>::member
                    auto inst = std::make_unique<CallExpression>(std::move(expr), expr->line);
                    inst->template_args = std::move(template_args);
                    inst->is_template_instantiation = true;
                    expr = std::move(inst);
                    
                    // Continue to process :: in the next iteration via scope resolution handling
                    // Actually we handle :: here directly:
                    while (match(TokenType::DoubleColon)) {
                        Token next_member = consume(TokenType::Identifier, "Expected identifier after '::'");
                        // Create a scope resolution expression or member access
                        auto scope_expr = std::make_unique<MemberAccessExpression>(
                            std::move(expr),
                            "::" + std::string(next_member.lexeme),
                            next_member.line
                        );
                        expr = std::move(scope_expr);
                    }
                } else {
                    // Template instantiation without call: std::integral<T>
                    // Wrap in a TemplateInstantiationExpression or just use the identifier
                    // with template args attached
                    auto inst = std::make_unique<CallExpression>(std::move(expr), expr->line);
                    inst->template_args = std::move(template_args);
                    inst->is_template_instantiation = true;  // Mark as instantiation, not call
                    expr = std::move(inst);
                }
            } else {
                // Not a template instantiation, restore and let binary < be handled elsewhere
                current = saved;
                panic_mode = saved_panic;  // Restore panic state
                break;
            }
        } else if (match(TokenType::Dot)) {
            expr = member_access_expression(std::move(expr));
        } else if (match(TokenType::DoubleDot)) {
            // Cpp2 explicit non-UFCS syntax: obj..method(args)
            // The .. explicitly calls the method on the object without UFCS rewrite
            Token member = consume(TokenType::Identifier, "Expected member name after '..'");
            auto member_expr = std::make_unique<MemberAccessExpression>(
                std::move(expr),
                std::string(member.lexeme),
                member.line,
                true  // explicit_non_ufcs
            );
            // If followed by call, parse it
            if (match(TokenType::LeftParen)) {
                expr = call_expression(std::move(member_expr));
            } else {
                expr = std::move(member_expr);
            }
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
                next == TokenType::RightBrace || next == TokenType::LeftBrace ||  // { starts new block (if p* {)
                next == TokenType::Dot || next == TokenType::Arrow ||  // -> for inspect return type
                next == TokenType::PlusPlus || next == TokenType::MinusMinus ||
                next == TokenType::Dollar ||  // Cpp2 capture operator
                next == TokenType::Asterisk || next == TokenType::Ampersand ||  // chained postfix
                // Also treat as postfix when followed by binary operators
                // (e.g., p* + q* means (*p) + (*q))
                next == TokenType::Plus || next == TokenType::Minus ||
                next == TokenType::Slash || next == TokenType::Percent ||
                next == TokenType::Caret || next == TokenType::Pipe ||  // bitwise XOR and OR
                next == TokenType::DoubleAmpersand || next == TokenType::DoublePipe ||  // logical AND/OR
                next == TokenType::Equal || next == TokenType::DoubleEqual ||
                next == TokenType::NotEqual || next == TokenType::LessThanOrEqual || next == TokenType::GreaterThanOrEqual ||
                next == TokenType::LessThan || next == TokenType::GreaterThan ||
                next == TokenType::LeftShift || next == TokenType::RightShift ||
                // Compound assignment operators (p* += 1 means (*p) += 1)
                next == TokenType::PlusEqual || next == TokenType::MinusEqual ||
                next == TokenType::AsteriskEqual || next == TokenType::SlashEqual ||
                next == TokenType::PercentEqual || next == TokenType::AmpersandEqual ||
                next == TokenType::PipeEqual || next == TokenType::CaretEqual ||
                next == TokenType::LeftShiftEqual || next == TokenType::RightShiftEqual ||
                // Cpp2 postfix type operators
                next == TokenType::Is || next == TokenType::As;

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
        } else if (match(TokenType::Dollar)) {
            // Cpp2 postfix $ capture operator: captures expression in lambda
            Token op = previous();
            expr = std::make_unique<UnaryExpression>(
                op.type,
                std::move(expr),
                op.line,
                true // postfix
            );
        } else if (match(TokenType::TripleDot) || match(TokenType::Ellipsis)) {
            // C++ pack expansion: expr... expands a parameter pack
            // Represent as a special expression or add "..." to the identifier
            if (auto* ident = dynamic_cast<IdentifierExpression*>(expr.get())) {
                ident->name += "...";
            } else {
                // For non-identifier expressions, wrap it
                Token op = previous();
                expr = std::make_unique<UnaryExpression>(
                    TokenType::Ellipsis,
                    std::move(expr),
                    op.line,
                    true // postfix
                );
            }
        } else if (match(TokenType::As)) {
            // Cpp2 postfix 'as' cast: expression as type
            auto cast_type = type();
            if (!cast_type) {
                error_at_current("Expected type after 'as'");
            }
            expr = std::make_unique<AsExpression>(
                std::move(expr),
                std::move(cast_type),
                previous().line
            );
        } else if (match(TokenType::Is)) {
            // Cpp2 postfix 'is' type test: expression is type
            // Can also be: expression is (value) or expression is literal for value pattern matching
            std::unique_ptr<Type> is_type = nullptr;
            std::unique_ptr<Expression> is_value = nullptr;
            
            if (match(TokenType::LeftParen)) {
                // Value pattern: x is (value)
                // The predicate can be a lambda: x is (:(x) = x > 3;)
                // Note: Cpp2 allows a semicolon to terminate lambda expression bodies
                is_value = expression();
                // Consume optional semicolon before closing paren (lambda body terminator)
                match(TokenType::Semicolon);
                consume(TokenType::RightParen, "Expected ')' after is value");
            } else if (check(TokenType::IntegerLiteral) || check(TokenType::FloatLiteral) ||
                       check(TokenType::StringLiteral) || check(TokenType::CharacterLiteral) ||
                       check(TokenType::True) || check(TokenType::False)) {
                // Direct literal value pattern: x is 123, x is true, x is "str"
                is_value = primary_expression();
            } else {
                // Type pattern: x is type
                is_type = type();
                if (!is_type) {
                    error_at_current("Expected type or (value) after 'is'");
                }
            }
            
            if (is_type) {
                expr = std::make_unique<IsExpression>(
                    std::move(expr),
                    std::move(is_type),
                    previous().line
                );
            } else {
                // For value patterns, create an equality comparison
                // x is (v) means x == v
                expr = std::make_unique<BinaryExpression>(
                    std::move(expr),
                    TokenType::DoubleEqual,
                    std::move(is_value),
                    previous().line
                );
            }
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
    if (match(TokenType::This)) {
        // 'this' is a reference to the current object
        return std::make_unique<IdentifierExpression>("this", previous().line);
    }
    if (match(TokenType::Underscore)) {
        // '_' is a discard/wildcard pattern or placeholder
        return std::make_unique<IdentifierExpression>("_", previous().line);
    }
    // Handle 'next' keyword when used as an identifier (common in Cpp2 code)
    // Context: 'next' is a keyword for while/for loops, but can also be a variable name
    if (check(TokenType::Next)) {
        // Check if 'next' is followed by something that makes it look like a variable
        // (e.g., postfix operators, member access, assignment, etc.)
        std::size_t lookahead = current + 1;
        if (lookahead < tokens.size()) {
            TokenType next_tok = tokens[lookahead].type;
            if (next_tok == TokenType::Asterisk || next_tok == TokenType::Ampersand ||
                next_tok == TokenType::Dot || next_tok == TokenType::PlusPlus ||
                next_tok == TokenType::MinusMinus || next_tok == TokenType::LeftBracket ||
                next_tok == TokenType::LeftParen || next_tok == TokenType::Equal ||
                next_tok == TokenType::ColonEqual || next_tok == TokenType::NotEqual ||
                next_tok == TokenType::DoubleEqual || next_tok == TokenType::Colon ||
                next_tok == TokenType::Semicolon || next_tok == TokenType::Comma ||
                next_tok == TokenType::RightParen || next_tok == TokenType::RightBracket) {
                advance();
                return std::make_unique<IdentifierExpression>("next", previous().line);
            }
        }
    }
    if (check(TokenType::IntegerLiteral)) {
        Token int_tok = advance(); // consume integer
        // Support user-defined literal suffixes like `10s` where lexer yields
        // IntegerLiteral `10` followed by Identifier `s`. Treat them as a single
        // identifier-like literal so codegen can emit `10s` directly.
        if (check(TokenType::Identifier) || check(TokenType::Underscore)) {
            Token suffix = advance();
            std::string combined = std::string(int_tok.lexeme) + std::string(suffix.lexeme);
            return std::make_unique<IdentifierExpression>(combined, int_tok.line);
        }
        return std::make_unique<LiteralExpression>(
            std::stoll(std::string(int_tok.lexeme)),
            int_tok.line
        );
    }
    if (check(TokenType::FloatLiteral)) {
        Token float_tok = advance();
        // Support UDL for floating point values e.g., `1.0s`
        if (check(TokenType::Identifier) || check(TokenType::Underscore)) {
            Token suffix = advance();
            std::string combined = std::string(float_tok.lexeme) + std::string(suffix.lexeme);
            return std::make_unique<IdentifierExpression>(combined, float_tok.line);
        }
        return std::make_unique<LiteralExpression>(
            std::stod(std::string(float_tok.lexeme)),
            float_tok.line
        );
    }
    if (match(TokenType::StringLiteral)) {
        // Build full string, supporting adjacent string literal concatenation
        std::string full_lexeme;
        std::size_t start_line = previous().line;

        // Handle first string - keep the full raw lexeme for code generation
        full_lexeme = std::string(previous().lexeme);

        // Handle adjacent string literals: "a" "b" "c" becomes "a" "b" "c" (passed through)
        while (check(TokenType::StringLiteral)) {
            Token next_str = advance();
            full_lexeme += " ";  // Space between concatenated literals
            full_lexeme += std::string(next_str.lexeme);
        }

        // Handle Cpp2 string interpolation: (expr)$ - anywhere in the string
        // Check if the string contains the (expr)$ pattern
        bool has_interpolation = false;
        if (full_lexeme.find(")$") != std::string::npos) {
            has_interpolation = true;
        }

        // Create a literal expression with the raw string (including quotes)
        auto str_expr = std::make_unique<LiteralExpression>(
            full_lexeme,
            start_line
        );

        if (has_interpolation) {
            auto interp = std::make_unique<StringInterpolationExpression>(start_line);
            interp->parts.push_back(full_lexeme);
            return interp;
        }

        return str_expr;
    }
    if (match(TokenType::InterpolatedRawStringLiteral)) {
        // Cpp2 interpolated raw string: $R"delimiter(...)delimiter"
        // These always have interpolation enabled
        std::string full_lexeme = std::string(previous().lexeme);
        std::size_t start_line = previous().line;

        // Handle adjacent interpolated raw strings: $R"a(...)" $R"b(...)"
        while (check(TokenType::InterpolatedRawStringLiteral) || check(TokenType::StringLiteral)) {
            Token next_str = advance();
            full_lexeme += " ";
            full_lexeme += std::string(next_str.lexeme);
        }

        auto interp = std::make_unique<StringInterpolationExpression>(start_line);
        interp->parts.push_back(full_lexeme);
        return interp;
    }
    if (match(TokenType::CharacterLiteral)) {
        return std::make_unique<LiteralExpression>(
            previous().lexeme[0],
            previous().line
        );
    }
    // Handle global scope resolution starting with ::
    // E.g., ::print(...), ::std::cout, etc.
    if (match(TokenType::DoubleColon)) {
        std::size_t line = previous().line;
        // Must have an identifier following ::
        Token next = [this]() -> Token {
            if (check(TokenType::Identifier) || check(TokenType::Func) || 
                check(TokenType::Type) || check(TokenType::Namespace)) {
                return advance();
            }
            return consume(TokenType::Identifier, "Expected identifier after '::'");
        }();
        std::string qname = "::";
        qname += std::string(next.lexeme);
        // Continue with any further :: chains
        while (match(TokenType::DoubleColon)) {
            Token next_part = [this]() -> Token {
                if (check(TokenType::Identifier) || check(TokenType::Func) || 
                    check(TokenType::Type) || check(TokenType::Namespace)) {
                    return advance();
                }
                return consume(TokenType::Identifier, "Expected identifier after '::'");
            }();
            qname += "::";
            qname += std::string(next_part.lexeme);
            line = next_part.line;
        }
        return std::make_unique<IdentifierExpression>(std::move(qname), line);
    }
    // Allow contextual keywords (func, type, namespace, in, out, etc.) as identifiers in expressions
    // These are used both as parameter passing modes and as regular identifiers
    if (match(TokenType::Identifier) || match(TokenType::Func) || match(TokenType::Type) ||
        match(TokenType::Namespace) || match(TokenType::In) || match(TokenType::Out) ||
        match(TokenType::Inout) || match(TokenType::Copy) || match(TokenType::Move) ||
        match(TokenType::Forward) || match(TokenType::That) || match(TokenType::Base)) {
        std::string qname(previous().lexeme);
        std::size_t line = previous().line;
        // Support scope resolution :: chains
        while (match(TokenType::DoubleColon)) {
            Token next = [this]() -> Token {
                if (check(TokenType::Identifier) || check(TokenType::Func) || 
                    check(TokenType::Type) || check(TokenType::Namespace)) {
                    return advance();
                }
                return consume(TokenType::Identifier, "Expected identifier after '::'");
            }();
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
        // Check if this is empty parens () - default construction
        if (check(TokenType::RightParen)) {
            advance(); // consume ')'
            // Return an empty list expression to represent () default construction
            return std::make_unique<ListExpression>(previous().line);
        }
        
        // Check if this starts with a parameter qualifier (out, inout, etc.)
        // This indicates a constructor call with qualified arguments: (out y)
        // BUT only if the qualifier is followed by an identifier (not by an operator or colon)
        // NOTE: We specifically exclude 'move' and 'forward' here because they can also be
        // prefix operators: (move x) is a grouped move expression, not a call with move qualifier
        auto is_qualifier_context = [this]() -> bool {
            // Only out/inout/in_ref/forward_ref unambiguously indicate qualifier context
            // move/forward can be prefix operators so don't trigger qualifier context
            if (!check(TokenType::Out) && !check(TokenType::Inout) &&
                !check(TokenType::InRef) && !check(TokenType::ForwardRef)) {
                return false;
            }
            // Check if next token after qualifier is an identifier (qualifier is actually a qualifier)
            // or if it's an operator/colon (qualifier is being used as a variable name)
            if (current + 1 >= tokens.size()) return false;
            TokenType next = tokens[current + 1].type;
            // If next is an identifier, this looks like a qualifier context
            return next == TokenType::Identifier || next == TokenType::This ||
                   next == TokenType::Underscore;
        };
        
        bool has_qualifier = is_qualifier_context();
        
        if (has_qualifier) {
            // Parse as a constructor-style call with qualified arguments
            auto call = std::make_unique<CallExpression>(nullptr, previous().line);
            
            do {
                // Handle trailing comma
                if (check(TokenType::RightParen)) break;
                
                // Check for argument qualifiers
                ParameterQualifier qualifier = ParameterQualifier::None;
                if (match(TokenType::Out)) {
                    qualifier = ParameterQualifier::Out;
                } else if (match(TokenType::Inout)) {
                    qualifier = ParameterQualifier::InOut;
                } else if (match(TokenType::Move)) {
                    qualifier = ParameterQualifier::Move;
                } else if (match(TokenType::Forward)) {
                    qualifier = ParameterQualifier::Forward;
                } else if (match(TokenType::InRef)) {
                    qualifier = ParameterQualifier::In;
                } else if (match(TokenType::ForwardRef)) {
                    qualifier = ParameterQualifier::Forward;
                }
                
                auto arg_expr = expression();
                
                CallExpression::Argument arg;
                arg.expr = std::move(arg_expr);
                arg.qualifier = qualifier;
                call->arguments.push_back(std::move(arg));
            } while (match(TokenType::Comma));
            
            consume(TokenType::RightParen, "Expected ')' after arguments");
            return call;
        }
        
        // Check if this is a fold expression: ( init op ... op pack ) or ( ... op pack )
        // We need to detect this before regular expression parsing
        std::size_t saved = current;
        
        // Check for left fold: ( ... op pack )
        if (check(TokenType::TripleDot) || check(TokenType::Ellipsis)) {
            advance(); // consume ...
            // Next should be an operator
            if (is_fold_operator()) {
                Token op_tok = advance();
                auto pack_expr = expression();
                consume(TokenType::RightParen, "Expected ')' after fold expression");
                // Generate as passthrough: ( ... op pack )
                auto fold = std::make_unique<IdentifierExpression>(
                    "( ... " + std::string(op_tok.lexeme) + " " + 
                    generate_fold_expr(pack_expr.get()) + " )",
                    saved);
                return fold;
            }
            // Not a fold expression, backtrack
            current = saved;
        }

        // Look ahead to detect binary fold pattern: init op ... op pack
        // We need to scan for "... op" pattern after initial "expr op"
        {
            std::size_t lookahead = current;
            int paren_depth = 1;
            bool found_fold = false;
            [[maybe_unused]] std::size_t fold_pos = 0;
            
            while (lookahead < tokens.size() && paren_depth > 0) {
                if (tokens[lookahead].type == TokenType::LeftParen) paren_depth++;
                else if (tokens[lookahead].type == TokenType::RightParen) paren_depth--;
                else if (paren_depth == 1 && 
                         (tokens[lookahead].type == TokenType::TripleDot || 
                          tokens[lookahead].type == TokenType::Ellipsis)) {
                    found_fold = true;
                    fold_pos = lookahead;
                    break;
                }
                lookahead++;
            }
            
            if (found_fold) {
                // This is a fold expression - collect tokens as raw passthrough
                std::string fold_text = "( ";
                while (current < tokens.size() && !check(TokenType::RightParen)) {
                    fold_text += std::string(advance().lexeme);
                    fold_text += " ";
                }
                fold_text += ")";
                consume(TokenType::RightParen, "Expected ')' after fold expression");
                return std::make_unique<IdentifierExpression>(fold_text, saved);
            }
        }

        auto first_expr = expression();

        if (match(TokenType::Comma)) {
            // This is a tuple expression: (expr, expr, ...)
            auto tuple = std::make_unique<ListExpression>(previous().line);
            tuple->elements.push_back(std::move(first_expr));

            // Parse remaining elements
            do {
                // Handle trailing comma
                if (check(TokenType::RightParen)) break;
                
                auto elem = expression();
                if (elem) {
                    tuple->elements.push_back(std::move(elem));
                } else {
                    error_at_current("Expected expression in tuple");
                }
            } while (match(TokenType::Comma));

            consume(TokenType::RightParen, "Expected ')' after tuple");
            return tuple;
        } else {
            // Single parenthesized expression
            consume(TokenType::RightParen, "Expected ')' after expression");
            return first_expr;
        }
    }
    if (match(TokenType::LeftBracket)) {
        // Check if this is a C++1 lambda expression: [capture](params) {...}
        // Lambda captures are followed by ]( pattern
        // List literals are just [expr, expr, ...]
        std::size_t save = current;
        
        // Try to parse as C++1 lambda capture
        // Skip to closing bracket
        int bracket_depth = 1;
        while (!is_at_end() && bracket_depth > 0) {
            if (check(TokenType::LeftBracket)) bracket_depth++;
            else if (check(TokenType::RightBracket)) bracket_depth--;
            if (bracket_depth > 0) advance();
        }
        
        if (match(TokenType::RightBracket) && check(TokenType::LeftParen)) {
            // This is a C++1 lambda expression - restore and parse it
            current = save - 1;  // Go back to include the [
            return cpp1_lambda_expression();
        } else {
            // This is a list literal
            current = save;
            return list_literal();
        }
    }
    if (match(TokenType::LeftBrace)) {
        return struct_initializer();
    }
    if (match(TokenType::Is)) {
        return is_expression();
    }
    // Note: 'as' is a postfix operator only, handled in call() function (line ~3925)
    if (match(TokenType::Inspect)) {
        return inspect_expression();
    }
    if (match(TokenType::At)) {
        return metafunction_call();
    }
    if (match(TokenType::Underscore)) {
        return std::make_unique<IdentifierExpression>("_", previous().line);
    }
    // Cpp2 function expression: :(params) -> type = { body }
    // or :(params) = { body } or :(params) { body } or :(params) = expr
    // or :<T> (params) = { body } (templated function expression)
    // Also: : type = value (typed construction expression)
    if (match(TokenType::Colon)) {
        if (check(TokenType::LeftParen) || check(TokenType::LessThan)) {
            return function_expression();
        }
        // : type = value - typed construction
        // Parse type and value
        auto t = type();
        if (match(TokenType::Equal)) {
            auto value = expression();
            // Create a typed construction expression - use AsExpression
            auto typed_expr = std::make_unique<AsExpression>(
                std::move(value),
                std::move(t),
                previous().line
            );
            return typed_expr;
        }
        // Just : type - create a placeholder
        return std::make_unique<IdentifierExpression>("_type_", previous().line);
    }

    // Handle 'decltype' as an identifier so it can be used in expressions like decltype(expr)
    if (match(TokenType::Decltype)) {
        return std::make_unique<IdentifierExpression>("decltype", previous().line);
    }

    error_at_current("Expected expression");
    return nullptr;
}

// Expression helpers
std::unique_ptr<Expression> Parser::call_expression(std::unique_ptr<Expression> callee) {
    int line = callee ? callee->line : previous().line;
    auto call = std::make_unique<CallExpression>(std::move(callee), line);

    if (!check(TokenType::RightParen)) {
        do {
            // Check for trailing comma
            if (check(TokenType::RightParen)) break;
            
            // Check for argument qualifiers: out, inout, move, forward, in_ref, forward_ref
            ParameterQualifier qualifier = ParameterQualifier::None;
            if (match(TokenType::Out)) {
                qualifier = ParameterQualifier::Out;
            } else if (match(TokenType::Inout)) {
                qualifier = ParameterQualifier::InOut;
            } else if (match(TokenType::Move)) {
                qualifier = ParameterQualifier::Move;
            } else if (match(TokenType::Forward)) {
                qualifier = ParameterQualifier::Forward;
            } else if (match(TokenType::InRef)) {
                qualifier = ParameterQualifier::In;  // in_ref maps to reference In
            } else if (match(TokenType::ForwardRef)) {
                qualifier = ParameterQualifier::Forward;  // forward_ref maps to Forward
            }
            
            auto arg_expr = expression();
            
            // Store in new structure
            CallExpression::Argument arg;
            arg.expr = std::move(arg_expr);
            arg.qualifier = qualifier;
            call->arguments.push_back(std::move(arg));
            
            // Also store in legacy args for backward compat (with cloned expression... can't do that)
            // For now, we'll update code generator to use arguments instead
        } while (match(TokenType::Comma));
    }

    consume(TokenType::RightParen, "Expected ')' after arguments");
    return call;
}

std::unique_ptr<Expression> Parser::member_access_expression(std::unique_ptr<Expression> object) {
    // Handle qualified member access: obj.ns::func or obj.Outer::Inner::method
    // Also allow contextual keywords as member names (func, type, etc.)
    Token member = [this]() -> Token {
        if (check(TokenType::Identifier)) {
            return advance();
        } else if (check(TokenType::Func) || check(TokenType::Type) || check(TokenType::Namespace) ||
                   check(TokenType::In) || check(TokenType::Out) || check(TokenType::Inout) ||
                   check(TokenType::Copy) || check(TokenType::Move) || check(TokenType::Forward)) {
            return advance();
        }
        return consume(TokenType::Identifier, "Expected member name after '.'");
    }();
    std::string member_name(member.lexeme);
    
    // Check for qualified name: ns::func, Outer::Inner::method
    while (match(TokenType::DoubleColon)) {
        Token next = consume(TokenType::Identifier, "Expected identifier after '::'");
        member_name += "::";
        member_name += next.lexeme;
    }
    
    return std::make_unique<MemberAccessExpression>(
        std::move(object),
        member_name,
        member.line
    );
}

std::unique_ptr<Expression> Parser::subscript_expression(std::unique_ptr<Expression> array) {
    auto index = expression();
    int line = array ? array->line : previous().line;
    consume(TokenType::RightBracket, "Expected ']' after subscript");
    return std::make_unique<SubscriptExpression>(
        std::move(array),
        std::move(index),
        line
    );
}

std::unique_ptr<Expression> Parser::list_literal() {
    auto list = std::make_unique<ListExpression>(previous().line);

    if (!check(TokenType::RightBracket)) {
        do {
            // Handle trailing comma
            if (check(TokenType::RightBracket)) break;
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
            // Handle trailing comma
            if (check(TokenType::RightBrace)) break;
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

            Token name = [this]() -> Token {
                if (check(TokenType::Identifier)) {
                    return advance();
                } else if (check(TokenType::This) || check(TokenType::That) || check(TokenType::Underscore)) {
                    return advance();
                } else {
                    return consume(TokenType::Identifier, "Expected parameter name");
                }
            }();
            param.name = std::string(name.lexeme);

            if (match(TokenType::Colon)) {
                param.type = type();
            }

            if (match(TokenType::Equal)) {
                param.default_value = expression();
            }

            lambda->parameters.push_back(std::move(param));
        } while (match(TokenType::Comma) && !check(TokenType::RightParen));
    }

    consume(TokenType::RightParen, "Expected ')' after lambda parameters");

    if (match(TokenType::Arrow)) {
        lambda->return_type = type();
    }

    consume(TokenType::LeftBrace, "Expected '{' for lambda body");

    while (!check(TokenType::RightBrace) && !is_at_end()) {
        std::size_t before = current;
        auto stmt = statement();
        if (stmt) {
            lambda->body.push_back(std::move(stmt));
        } else if (current == before) {
            // Avoid infinite loop when a statement cannot be parsed and no tokens
            // are consumed; advance to allow error recovery to proceed.
            advance();
        }
    }

    consume(TokenType::RightBrace, "Expected '}' after lambda body");
    return lambda;
}

std::unique_ptr<Expression> Parser::cpp1_lambda_expression() {
    // C++1 lambda syntax: [capture](params) -> type { body }
    // Called when we see [ at the start of an expression
    std::size_t line = peek().line;
    
    consume(TokenType::LeftBracket, "Expected '[' for lambda capture");
    
    auto lambda = std::make_unique<Cpp1LambdaExpression>(line);
    
    // Parse capture clause: =, &, identifier, =identifier, &identifier
    if (!check(TokenType::RightBracket)) {
        do {
            Cpp1LambdaExpression::Capture cap;
            if (match(TokenType::Equal)) {
                // Default capture by copy [=] or [=, ...]
                cap.mode = Cpp1LambdaExpression::Capture::Mode::DefaultCopy;
            } else if (match(TokenType::Ampersand)) {
                // Check if this is default capture [&] or a specific capture [&x]
                if (check(TokenType::RightBracket) || check(TokenType::Comma)) {
                    cap.mode = Cpp1LambdaExpression::Capture::Mode::DefaultRef;
                } else {
                    // &name capture
                    cap.mode = Cpp1LambdaExpression::Capture::Mode::ByRef;
                    Token name = consume(TokenType::Identifier, "Expected captured variable name");
                    cap.name = std::string(name.lexeme);
                }
            } else if (match(TokenType::This)) {
                cap.mode = Cpp1LambdaExpression::Capture::Mode::This;
                cap.name = "this";
            } else if (match(TokenType::Identifier)) {
                cap.mode = Cpp1LambdaExpression::Capture::Mode::ByCopy;
                cap.name = std::string(previous().lexeme);
            } else {
                break;
            }
            lambda->captures.push_back(std::move(cap));
        } while (match(TokenType::Comma));
    }
    
    consume(TokenType::RightBracket, "Expected ']' after lambda capture");
    
    // Parse parameter list
    consume(TokenType::LeftParen, "Expected '(' for lambda parameters");
    
    if (!check(TokenType::RightParen)) {
        do {
            Cpp1LambdaExpression::Parameter param;
            
            // Check for auto type
            if (match(TokenType::Auto)) {
                param.type_str = "auto";
            } else {
                // Parse type (may be qualified like std::string)
                std::string type_str;
                
                // Handle const
                if (match(TokenType::Const)) {
                    type_str = "const ";
                }
                
                // Parse the type itself
                auto t = type();
                if (t) {
                    // Convert type to string representation
                    // For now, just use the lexeme of what we parsed
                    // This is a simplification; proper implementation would render the Type
                    type_str += type_to_string(t.get());
                }
                
                // Handle references/pointers
                if (match(TokenType::Ampersand)) {
                    type_str += "&";
                    if (match(TokenType::Ampersand)) {
                        type_str += "&";  // rvalue reference
                    }
                }
                
                param.type_str = type_str;
            }
            
            // Parameter name
            Token name = consume(TokenType::Identifier, "Expected parameter name");
            param.name = std::string(name.lexeme);
            
            lambda->parameters.push_back(std::move(param));
        } while (match(TokenType::Comma));
    }
    
    consume(TokenType::RightParen, "Expected ')' after lambda parameters");
    
    // Optional trailing return type
    if (match(TokenType::Arrow)) {
        lambda->return_type = type();
    }
    
    // Lambda body
    consume(TokenType::LeftBrace, "Expected '{' for lambda body");
    
    while (!check(TokenType::RightBrace) && !is_at_end()) {
        std::size_t before = current;
        auto stmt = statement();
        if (stmt) {
            lambda->body.push_back(std::move(stmt));
        } else if (current == before) {
            advance();
        }
    }
    
    consume(TokenType::RightBrace, "Expected '}' after lambda body");
    
    return lambda;
}

std::unique_ptr<Expression> Parser::function_expression() {
    // Cpp2 function expression: :(params) -> type = { body }
    // or :(params) = { body } or :(params) { body } or :(params) = expr
    // or :<T> (params) = { body } (templated function expression)
    // Called after ':' has been consumed

    auto lambda = std::make_unique<LambdaExpression>(previous().line);

    // Check for template parameters: :<T, U> (params)
    if (match(TokenType::LessThan)) {
        lambda->template_params = template_parameters();
        // Handle > or >> (via pending_gt mechanism from nested templates)
        if (!consume_template_close()) {
            consume(TokenType::GreaterThan, "Expected '>' after template parameters");
        }
    }

    consume(TokenType::LeftParen, "Expected '(' after ':' for function expression");

    if (!check(TokenType::RightParen)) {
        do {
            LambdaExpression::Parameter param;

            // Parse qualifiers before parameter name
            param.qualifiers = parse_parameter_qualifiers();

            Token name = [this]() -> Token {
                if (check(TokenType::Identifier)) {
                    return advance();
                } else if (check(TokenType::This) || check(TokenType::That) || check(TokenType::Underscore)) {
                    return advance();
                } else {
                    return consume(TokenType::Identifier, "Expected parameter name");
                }
            }();
            param.name = std::string(name.lexeme);

            if (match(TokenType::Colon)) {
                param.type = type();
            }

            if (match(TokenType::Equal)) {
                param.default_value = expression();
            }

            lambda->parameters.push_back(std::move(param));
        } while (match(TokenType::Comma) && !check(TokenType::RightParen));
    }

    consume(TokenType::RightParen, "Expected ')' after function expression parameters");

    if (match(TokenType::Arrow)) {
        lambda->return_type = type();
    }

    // Function expressions can have = { body } or == { body } or = expr or just { body }
    // == indicates compile-time function (constexpr)
    bool is_compile_time = match(TokenType::DoubleEqual);
    if (is_compile_time || match(TokenType::Equal)) {
        lambda->is_constexpr = is_compile_time;
        if (check(TokenType::LeftBrace)) {
            advance(); // consume '{'
            while (!check(TokenType::RightBrace) && !is_at_end()) {
                std::size_t before = current;
                auto stmt = statement();
                if (stmt) {
                    lambda->body.push_back(std::move(stmt));
                } else if (current == before) {
                    advance();
                }
            }
            consume(TokenType::RightBrace, "Expected '}' after function expression body");
        } else {
            // Single expression body: :(x) = x + 1
            // Or immediately-invoked: :(x) = x;(args) - call with args
            auto expr = expression();
            auto ret_stmt = std::make_unique<ReturnStatement>(std::move(expr), previous().line);
            lambda->body.push_back(std::move(ret_stmt));

            // Check for immediately-invoked function expression: ;(args)
            // Only consume semicolon if it's followed by ( to avoid breaking regular function expressions
            if (check(TokenType::Semicolon) && current + 1 < tokens.size() && tokens[current + 1].type == TokenType::LeftParen) {
                match(TokenType::Semicolon);  // Consume the semicolon
                // Parse the call arguments
                auto call = std::make_unique<CallExpression>(std::move(lambda), previous().line);
                consume(TokenType::LeftParen, "Expected '(' after ';' for IIFE");
                if (!check(TokenType::RightParen)) {
                    do {
                        CallExpression::Argument arg;
                        arg.expr = expression();
                        call->arguments.push_back(std::move(arg));
                    } while (match(TokenType::Comma));
                }
                consume(TokenType::RightParen, "Expected ')' after IIFE arguments");
                return call;
            }
        }
    } else if (check(TokenType::LeftBrace)) {
        advance(); // consume '{'
        while (!check(TokenType::RightBrace) && !is_at_end()) {
            std::size_t before = current;
            auto stmt = statement();
            if (stmt) {
                lambda->body.push_back(std::move(stmt));
            } else if (current == before) {
                advance();
            }
        }
        consume(TokenType::RightBrace, "Expected '}' after function expression body");
    }

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

        // Contract syntax: pre: expr or pre( expr ) or pre<bounds_safety>( expr )
        // Handle optional template argument like <bounds_safety>
        if (match(TokenType::LessThan)) {
            // Skip template arguments for now (e.g., <bounds_safety>)
            int depth = 1;
            while (!is_at_end() && depth > 0) {
                if (match(TokenType::LessThan)) depth++;
                else if (match(TokenType::GreaterThan)) depth--;
                else advance();
            }
        }

        std::unique_ptr<Expression> condition;
        std::string message;
        if (match(TokenType::Colon)) {
            condition = expression();
        } else if (match(TokenType::LeftParen)) {
            condition = expression();
            // Check for optional message: pre( condition, "message" )
            if (match(TokenType::Comma)) {
                Token msg = consume(TokenType::StringLiteral, "Expected string message in contract");
                message = std::string(msg.lexeme);
            }
            consume(TokenType::RightParen, "Expected ')' after contract condition");
        } else {
            consume(TokenType::Colon, "Expected ':' or '(' after contract keyword");
            condition = expression();
        }

        auto contract = std::make_unique<ContractExpression>(kind, std::move(condition), contract_type.line);
        if (!message.empty()) {
            contract->message = std::move(message);
        }

        if (match(TokenType::Colon)) {
            Token msg = consume(TokenType::StringLiteral, "Expected string message");
            contract->message = std::string(msg.lexeme);
        }

        contracts.push_back(std::move(contract));
    }

    return contracts;
}

// Parameter qualifier parsing (Cpp2-specific)
// Parses: in, inout, out, move, forward, copy, in_ref, forward_ref, virtual, override, implicit
std::vector<ParameterQualifier> Parser::parse_parameter_qualifiers() {
    std::vector<ParameterQualifier> qualifiers;

    while (true) {
        // Look ahead: if next token after qualifier is ':', '::', or '...' then 
        // the current token is actually a parameter NAME, not a qualifier
        auto peek_for_name = [this]() -> bool {
            if (current + 1 >= tokens.size()) return false;
            TokenType next = tokens[current + 1].type;
            return next == TokenType::Colon || next == TokenType::DoubleColon || 
                   next == TokenType::TripleDot || next == TokenType::Ellipsis ||
                   next == TokenType::RightParen || next == TokenType::Comma ||
                   next == TokenType::Equal;
        };
        
        if (check(TokenType::In) && !peek_for_name()) {
            advance();
            qualifiers.push_back(ParameterQualifier::In);
        } else if (check(TokenType::Inout) && !peek_for_name()) {
            advance();
            qualifiers.push_back(ParameterQualifier::InOut);
        } else if (check(TokenType::Out) && !peek_for_name()) {
            advance();
            qualifiers.push_back(ParameterQualifier::Out);
        } else if (check(TokenType::Copy) && !peek_for_name()) {
            advance();
            qualifiers.push_back(ParameterQualifier::None);  // 'copy' means pass-by-value
        } else if (check(TokenType::Move) && !peek_for_name()) {
            advance();
            qualifiers.push_back(ParameterQualifier::Move);
        } else if (check(TokenType::Forward) && !peek_for_name()) {
            advance();
            qualifiers.push_back(ParameterQualifier::Forward);
        } else if (check(TokenType::Virtual) && !peek_for_name()) {
            advance();
            qualifiers.push_back(ParameterQualifier::Virtual);
        } else if (check(TokenType::Override) && !peek_for_name()) {
            advance();
            qualifiers.push_back(ParameterQualifier::Override);
        } else if (match(TokenType::Implicit)) {
            qualifiers.push_back(ParameterQualifier::Implicit);
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
        // Handle trailing comma
        if (check(TokenType::GreaterThan) || check(TokenType::RightShift)) break;
        
        // Template parameter can be an identifier or underscore (_)
        const Token& param = check(TokenType::Underscore) ? advance() : consume(TokenType::Identifier, "Expected template parameter");
        std::string param_name = std::string(param.lexeme);

        // Check for variadic parameter: Ts... or T...
        if (match(TokenType::TripleDot) || match(TokenType::Ellipsis)) {
            param_name += "...";
        }

        params.push_back(param_name);

        // Check for type constraint: T:type, T:_, T: i8, T:t<o.f()>, etc.
        if (match(TokenType::Colon)) {
            // Parse the type constraint - always use type() to handle complex types
            // including templates like t<o.f()>
            auto constraint = type();
            // Note: constraint result is discarded for now, but type() advances the parser
            // to the correct position after the constraint

            // Check for default value: T:type = int, T:type = std::plus<>
            if (match(TokenType::Equal)) {
                // Parse the default value as a type expression
                // This could be: int, std::plus<>, some_type<args>, function_call(args), etc.

                // Skip the default value, tracking nesting for template argument lists and parens
                int angle_depth = 0;  // depth of nested <...>
                int paren_depth = 0;  // depth of nested (...)
                while (!is_at_end()) {
                    if (check(TokenType::Comma) && angle_depth == 0 && paren_depth == 0) {
                        break;  // , at top level ends the default value
                    }
                    if (check(TokenType::GreaterThan) && angle_depth == 0 && paren_depth == 0) {
                        break;  // > at top level ends the default value (closes template params)
                    }
                    if (check(TokenType::RightShift) && angle_depth == 0 && paren_depth == 0) {
                        break;  // >> at top level ends the default value (closes template params)
                    }

                    // Track nesting for <...>
                    if (match(TokenType::LessThan)) {
                        angle_depth++;
                    } else if (match(TokenType::GreaterThan)) {
                        if (angle_depth > 0) angle_depth--;
                    } else if (check(TokenType::RightShift)) {
                        // >> is two > characters for template closing
                        if (angle_depth >= 2) {
                            angle_depth -= 2;
                            advance();  // consume >>
                        } else if (angle_depth == 1) {
                            angle_depth = 0;
                            // Don't consume >> - let outer parser handle remaining >
                            break;
                        } else {
                            advance();
                        }
                    } else if (match(TokenType::LeftParen)) {
                        paren_depth++;
                    } else if (match(TokenType::RightParen)) {
                        if (paren_depth > 0) {
                            paren_depth--;
                        } else {
                            // Unbalanced ) - should not consume
                            break;
                        }
                    } else {
                        advance();
                    }
                }
            }
        }
    } while (match(TokenType::Comma) && !check(TokenType::GreaterThan) && !check(TokenType::RightShift));

    return params;
}

bool Parser::is_fold_operator() {
    // Valid fold operators in C++17/20
    TokenType t = peek().type;
    return t == TokenType::Plus || t == TokenType::Minus ||
           t == TokenType::Asterisk || t == TokenType::Slash ||
           t == TokenType::Percent || t == TokenType::Caret ||
           t == TokenType::Ampersand || t == TokenType::Pipe ||
           t == TokenType::DoubleAmpersand || t == TokenType::DoublePipe ||
           t == TokenType::LessThan || t == TokenType::LessThanOrEqual ||
           t == TokenType::GreaterThan || t == TokenType::GreaterThanOrEqual ||
           t == TokenType::DoubleEqual || t == TokenType::NotEqual ||
           t == TokenType::LeftShift || t == TokenType::RightShift ||
           t == TokenType::Comma;
}

std::string Parser::generate_fold_expr(Expression* expr) {
    if (!expr) return "";
    // Generate a simple string representation of the expression
    if (auto* ident = dynamic_cast<IdentifierExpression*>(expr)) {
        return ident->name;
    }
    if (auto* lit = dynamic_cast<LiteralExpression*>(expr)) {
        if (std::holds_alternative<std::string>(lit->value)) {
            return std::get<std::string>(lit->value);
        }
        if (std::holds_alternative<long long>(lit->value)) {
            return std::to_string(std::get<long long>(lit->value));
        }
        if (std::holds_alternative<double>(lit->value)) {
            return std::to_string(std::get<double>(lit->value));
        }
        if (std::holds_alternative<bool>(lit->value)) {
            return std::get<bool>(lit->value) ? "true" : "false";
        }
    }
    if (auto* binary = dynamic_cast<BinaryExpression*>(expr)) {
        // Convert TokenType to operator string
        std::string op_str;
        switch (binary->op) {
            case TokenType::Plus: op_str = "+"; break;
            case TokenType::Minus: op_str = "-"; break;
            case TokenType::Asterisk: op_str = "*"; break;
            case TokenType::Slash: op_str = "/"; break;
            case TokenType::Percent: op_str = "%"; break;
            case TokenType::Ampersand: op_str = "&"; break;
            case TokenType::Pipe: op_str = "|"; break;
            case TokenType::Caret: op_str = "^"; break;
            case TokenType::DoubleAmpersand: op_str = "&&"; break;
            case TokenType::DoublePipe: op_str = "||"; break;
            case TokenType::LessThan: op_str = "<"; break;
            case TokenType::GreaterThan: op_str = ">"; break;
            case TokenType::LessThanOrEqual: op_str = "<="; break;
            case TokenType::GreaterThanOrEqual: op_str = ">="; break;
            case TokenType::DoubleEqual: op_str = "=="; break;
            case TokenType::NotEqual: op_str = "!="; break;
            default: op_str = "?op?"; break;
        }
        return generate_fold_expr(binary->left.get()) + " " + 
               op_str + " " + 
               generate_fold_expr(binary->right.get());
    }
    if (auto* unary = dynamic_cast<UnaryExpression*>(expr)) {
        std::string op_str;
        switch (unary->op) {
            case TokenType::Minus: op_str = "-"; break;
            case TokenType::Plus: op_str = "+"; break;
            case TokenType::Exclamation: op_str = "!"; break;
            case TokenType::Tilde: op_str = "~"; break;
            case TokenType::Asterisk: op_str = "*"; break;
            case TokenType::Ampersand: op_str = "&"; break;
            default: op_str = "?"; break;
        }
        return op_str + generate_fold_expr(unary->operand.get());
    }
    // Fallback: return a placeholder
    return "/*expr*/";
}

bool Parser::is_template_start() {
    return check(TokenType::Template) ||
           (check(TokenType::Identifier) && peek().lexeme == "template");
}

bool Parser::is_type_qualifier() {
    // Check if current token is a type qualifier that can appear before 'type'
    // e.g., final, abstract, sealed, etc.
    // NOTE: const by itself is ambiguous - it could be a type qualifier (const type = ...)
    // or a type annotation (const int = 42). We need to look ahead.
    
    // For const, check if followed by 'type' keyword eventually
    if (check(TokenType::Const)) {
        std::size_t peek_pos = current + 1;
        // Look ahead for 'type' keyword after skipping other qualifiers
        while (peek_pos < tokens.size()) {
            auto next_type = tokens[peek_pos].type;
            if (next_type == TokenType::Type ||
                (next_type == TokenType::Identifier && 
                 (tokens[peek_pos].lexeme == "type" || 
                  tokens[peek_pos].lexeme == "struct" ||
                  tokens[peek_pos].lexeme == "class"))) {
                return true;  // const is a type qualifier
            }
            // Skip other type qualifiers
            if (next_type == TokenType::Final || 
                next_type == TokenType::Virtual ||
                next_type == TokenType::Override ||
                next_type == TokenType::Explicit ||
                next_type == TokenType::Const) {
                peek_pos++;
                continue;
            }
            // Not followed by type keyword - const is a type annotation
            return false;
        }
        return false;
    }
    
    return check(TokenType::Final) ||
           check(TokenType::Virtual) ||
           check(TokenType::Override) ||
           check(TokenType::Explicit) ||
           (check(TokenType::Identifier) &&
            (peek().lexeme == "type" || peek().lexeme == "struct" ||
             peek().lexeme == "class" || peek().lexeme == "enum" ||
             peek().lexeme == "union" || peek().lexeme == "interface"));
}

bool Parser::is_cpp1_template_start() {
    // Check for C++1 template patterns:
    // template<...> struct/class/union/namespace name {...}
    // template<...> auto/type name(...) {...}  (function template)
    //
    // Key: After template<...>, we expect C++1 keywords, not Cpp2 syntax

    std::size_t saved = current;

    // Check for 'template' keyword (TokenType::Template)
    if (!check(TokenType::Template)) {
        return false;
    }

    advance(); // consume 'template'

    // Check for '<' after template
    if (!check(TokenType::LessThan)) {
        current = saved;
        return false;
    }

    advance(); // consume '<', now current points to first template param

    // Skip template parameters: template<...>
    // Start at depth=1 for the '<' we already consumed
    int depth = 1;
    while (depth > 0 && !is_at_end()) {
        if (peek().type == TokenType::LessThan) {
            depth++;
        } else if (peek().type == TokenType::GreaterThan) {
            depth--;
        }
        advance(); // move to next token (or past '>')
    }

    // After loop, current is past the final '>'
    // Now check what follows - C++1 keywords (struct/class/union/namespace/enum/using)
    // or function templates (auto/identifier followed by identifier and '(')
    bool is_cpp1 = false;
    if (!is_at_end()) {
        TokenType next_type = peek().type;
        if (next_type == TokenType::Struct || next_type == TokenType::Class ||
            next_type == TokenType::Union || next_type == TokenType::Namespace ||
            next_type == TokenType::Enum || next_type == TokenType::Using) {
            is_cpp1 = true;
        }
        // Check for C++1 function template: template<...> auto/type name(...)
        // The return type can be: auto, const, or identifier (like void, int, static, inline)
        else if (next_type == TokenType::Auto || next_type == TokenType::Const ||
                 next_type == TokenType::Identifier) {
            // Could be a C++1 function template - check if followed by identifier then '('
            std::size_t lookahead_saved = current;
            // Skip qualifiers like const, static, inline, constexpr (all identifiers in lexer)
            while (peek().type == TokenType::Const || peek().type == TokenType::Identifier) {
                std::string_view lexeme = peek().lexeme;
                // Only skip known C++1 qualifiers/specifiers
                if (peek().type == TokenType::Const ||
                    lexeme == "static" || lexeme == "inline" || lexeme == "constexpr") {
                    advance();
                } else {
                    break;
                }
            }
            // Skip return type tokens (auto or identifier like void, int, etc.)
            if (peek().type == TokenType::Auto || peek().type == TokenType::Identifier) {
                advance();
                // Check for pointer/reference
                while (peek().type == TokenType::Asterisk || peek().type == TokenType::Ampersand ||
                       peek().type == TokenType::Const) {
                    advance();
                }
                // After return type, expect function name (identifier)
                if (peek().type == TokenType::Identifier) {
                    advance();
                    // After function name, expect '(' for C++1 function template
                    if (peek().type == TokenType::LeftParen) {
                        is_cpp1 = true;
                    }
                }
            }
            current = lookahead_saved;
        }
    }

    current = saved;
    return is_cpp1;
}

bool Parser::check_cpp1_struct_syntax() {
    // Check for C++1 struct/class patterns:
    // - "struct Name {" or "struct Name :" (with inheritance)
    // - "class Name {" or "class Name :"
    // - "union Name {"
    // - "enum Name {" or "enum class Name {"
    // 
    // Key distinction from Cpp2: Cpp2 uses "Name: struct = {" syntax
    // C++1 uses "struct Name {" syntax

    std::size_t saved = current;

    // Check for struct/class/union/enum keyword at current position
    if (check(TokenType::Struct) || check(TokenType::Class) || 
        check(TokenType::Union) || check(TokenType::Enum)) {
        advance(); // consume struct/class/union/enum
        
        // enum class Name is also valid
        if (previous().type == TokenType::Enum && check(TokenType::Class)) {
            advance(); // consume 'class' after 'enum'
        }
        
        // Expect identifier (name of the type)
        if (check(TokenType::Identifier)) {
            advance(); // consume name
            
            // Could be followed by:
            // - { for body start
            // - : for inheritance list
            // - ; for forward declaration
            // - final/override before : or {
            while (check(TokenType::Final) || check(TokenType::Override) ||
                   (check(TokenType::Identifier) && (peek().lexeme == "final" || peek().lexeme == "override"))) {
                advance();
            }
            
            if (check(TokenType::LeftBrace) || check(TokenType::Colon) || check(TokenType::Semicolon)) {
                current = saved;
                return true;
            }
        }
    }
    
    current = saved;
    return false;
}

bool Parser::check_cpp1_function_syntax() {
    // Check for C++1 function patterns (per docs/cppfront/mixed.md):
    // - "auto name(...) -> type {" - trailing return type
    // - "type name(...) {" - standard function
    // - "[qualifiers] type name(...) {" - with C++1 qualifiers (constexpr, inline, static, virtual)
    //
    // Key distinction from Cpp2: Cpp2 uses ':' after name, C++1 does NOT
    // We only trigger this if we're NOT at a Cpp2 declaration

    std::size_t saved = current;

    // Skip C++1 function qualifiers: constexpr, inline, static, virtual, extern, friend
    while (check(TokenType::Identifier) &&
           (peek().lexeme == "constexpr" || peek().lexeme == "inline" ||
            peek().lexeme == "static" || peek().lexeme == "extern" ||
            peek().lexeme == "friend")) {
        advance();
    }
    if (check(TokenType::Virtual)) {
        advance();
    }

    // Check for "auto name(..." pattern - C++1 trailing return type
    if (check(TokenType::Auto) ||
        (check(TokenType::Identifier) && peek().lexeme == "auto")) {
        advance(); // consume 'auto'
        // In C++1, function name can be: identifier, keyword, or 'operator' for operator overloads
        if (check(TokenType::Identifier) || is_identifier_like() || check(TokenType::Operator)) {
            advance(); // consume function name
            
            // For operator overloads like operator<<, skip the operator symbol(s)
            if (previous().type == TokenType::Operator) {
                // Skip operator symbol(s): <<, ==, [], etc.
                while (!is_at_end() && !check(TokenType::LeftParen)) {
                    advance();
                }
            }
            
            if (check(TokenType::LeftParen)) {
                current = saved;
                return true; // "[qualifiers] auto name(..." - C++1
            }
        }
    }
    // Don't reset here, continue checking with qualifiers already consumed

    // Check for "[qualifiers] type name(..." pattern - C++1 standard function
    // Need to skip: Cpp2 keywords (func, type, namespace, let, const, etc.)
    if (check(TokenType::Identifier)) {
        std::string_view first_lexeme = peek().lexeme;
        // Skip Cpp2 keywords - but not if we already consumed qualifiers
        if (current == saved) {
            if (first_lexeme == "func" || first_lexeme == "type" ||
                first_lexeme == "namespace" || first_lexeme == "struct" ||
                first_lexeme == "class" || first_lexeme == "union" ||
                first_lexeme == "interface" || first_lexeme == "enum" ||
                first_lexeme == "let" ||
                first_lexeme == "const" || first_lexeme == "import" ||
                first_lexeme == "using" || first_lexeme == "template") {
                // Note: 'operator' is NOT excluded - "auto operator<<(...)" is valid C++1
                return false;
            }
        }

        advance(); // consume type name

        // Skip template arguments: type<...>
        if (check(TokenType::LessThan)) {
            int depth = 1;
            advance();
            while (depth > 0 && !is_at_end()) {
                if (check(TokenType::LessThan)) depth++;
                else if (check(TokenType::GreaterThan)) depth--;
                advance();
            }
        }

        // Skip pointers/references: type*, type&, type&&
        while (check(TokenType::Asterisk) || check(TokenType::Ampersand)) {
            advance();
        }

        // Now we expect function name - can be identifier or 'operator' for operator overloads
        if (check(TokenType::Identifier) || is_identifier_like() || check(TokenType::Operator)) {
            advance(); // consume function name
            
            // For operator overloads like operator<<, skip the operator symbol(s)
            if (previous().type == TokenType::Operator) {
                // Skip operator symbol(s): <<, ==, [], etc.
                while (!is_at_end() && !check(TokenType::LeftParen)) {
                    advance();
                }
            }
            
            if (check(TokenType::LeftParen)) {
                current = saved;
                return true; // "[qualifiers] type name(..." - C++1
            }
        }
    }
    current = saved;

    return false;
}

bool Parser::check_cpp1_constexpr_syntax() {
    // Detect C++1 constexpr/inline variable declarations
    // Patterns:
    //   constexpr auto name = ...;
    //   inline constexpr auto name = ...;
    //   static constexpr auto name = ...;
    //   constexpr type name = ...;
    //   constexpr type<T> name{};   // brace initialization
    //   constexpr type<T> name(...);  // parenthesized initialization
    
    std::size_t saved = current;
    
    // Skip optional inline/static
    while (check(TokenType::Identifier) && 
           (peek().lexeme == "inline" || peek().lexeme == "static" || peek().lexeme == "constexpr")) {
        advance();
    }
    
    // Now check if we have "constexpr" or if we already passed it
    // We should have at least "constexpr" in the sequence
    bool has_constexpr = false;
    for (std::size_t i = saved; i < current; i++) {
        if (tokens[i].lexeme == "constexpr") {
            has_constexpr = true;
            break;
        }
    }
    
    if (!has_constexpr) {
        current = saved;
        return false;
    }
    
    // Now expect auto or a type
    if (check(TokenType::Auto) || 
        (check(TokenType::Identifier) && peek().lexeme == "auto")) {
        advance();
    } else if (check(TokenType::Identifier)) {
        advance();
        // Skip template args
        if (check(TokenType::LessThan)) {
            int depth = 1;
            advance();
            while (depth > 0 && !is_at_end()) {
                if (check(TokenType::LessThan)) depth++;
                else if (check(TokenType::GreaterThan)) depth--;
                advance();
            }
        }
    } else {
        current = saved;
        return false;
    }
    
    // Now expect identifier (variable name) - may be a keyword used as identifier in C++1
    // In C++1, 'in' can be a variable name
    if (check(TokenType::Identifier) || check(TokenType::In)) {
        advance();
        // Check for = (assignment), { (brace init), or ( (paren init) 
        // which distinguishes C++1 variable from Cpp2
        if (check(TokenType::Equal) || check(TokenType::LeftBrace) || check(TokenType::LeftParen)) {
            current = saved;
            return true;
        }
        // Also handle direct initialization without initializer (constexpr T x;)
        if (check(TokenType::Semicolon)) {
            current = saved;
            return true;
        }
    }
    
    current = saved;
    return false;
}

std::unique_ptr<Declaration> Parser::cpp1_passthrough_declaration(bool is_struct_type) {
    // Capture all tokens until matching closing brace
    std::size_t start_pos = current;
    std::string raw_code;

    // Build raw source from token lexemes
    while (!is_at_end()) {
        const Token& tok = peek();
        raw_code += tok.lexeme;

        // Add whitespace based on token positions for readability
        if (!is_at_end() && current + 1 < tokens.size()) {
            const Token& next_tok = tokens[current + 1];
            if (next_tok.line == tok.line && next_tok.column > tok.column + tok.lexeme.length()) {
                // Same line, add space
                raw_code += " ";
            } else if (next_tok.line > tok.line) {
                // Different line, add newline
                raw_code += "\n";
            }
        }

        advance();

        // Stop at semicolon or matching closing brace
        if (tok.type == TokenType::Semicolon) {
            break;
        }
        if (tok.type == TokenType::LeftBrace) {
            // Find matching closing brace
            int depth = 1;
            while (!is_at_end() && depth > 0) {
                const Token& inner = peek();
                if (inner.type == TokenType::LeftBrace) depth++;
                else if (inner.type == TokenType::RightBrace) depth--;
                raw_code += inner.lexeme;

                advance();

                // Add whitespace
                if (!is_at_end() && current < tokens.size()) {
                    const Token& next_tok = peek();
                    if (next_tok.line == inner.line && next_tok.column > inner.column + inner.lexeme.length()) {
                        raw_code += " ";
                    } else if (next_tok.line > inner.line) {
                        raw_code += "\n";
                    }
                }
            }
            // After closing brace, handle differently for structs vs functions
            if (is_struct_type && !is_at_end()) {
                // For structs/classes, check for variable declarations or semicolon
                // e.g., "struct X { } x;" or "struct X { } x, y;" or just "struct X { };"
                // But don't consume Cpp2 declarations like "name :" after the brace
                if (check(TokenType::Identifier)) {
                    std::size_t lookahead = current + 1;
                    if (lookahead < tokens.size() && tokens[lookahead].type == TokenType::Colon) {
                        // This is a Cpp2 declaration, stop here
                        break;
                    }
                }
                // Capture any identifiers (variable names) and commas before semicolon
                while (check(TokenType::Identifier) || check(TokenType::Comma)) {
                    raw_code += " ";
                    raw_code += peek().lexeme;
                    advance();
                    // Check if next identifier is followed by colon (Cpp2 declaration)
                    if (check(TokenType::Identifier)) {
                        std::size_t lookahead = current + 1;
                        if (lookahead < tokens.size() && tokens[lookahead].type == TokenType::Colon) {
                            break;
                        }
                    }
                }
                // Now consume the semicolon
                if (check(TokenType::Semicolon)) {
                    raw_code += peek().lexeme;
                    advance();
                }
            } else {
                // For functions, check for semicolon OR function-try-catch handlers
                // After function body closing brace, we might have:
                // - Semicolon: end of declaration
                // - Catch handler: function-try-catch block
                while (!is_at_end()) {
                    if (check(TokenType::Semicolon)) {
                        raw_code += " ";
                        raw_code += peek().lexeme;
                        advance();
                        break;
                    }
                    if (check(TokenType::Catch)) {
                        // Found catch handler - capture it completely
                        raw_code += " ";
                        raw_code += peek().lexeme;
                        advance();

                        // Expect '(' after catch
                        if (check(TokenType::LeftParen)) {
                            raw_code += peek().lexeme;
                            advance();

                            // Find matching closing paren for exception declaration
                            int paren_depth = 1;
                            while (!is_at_end() && paren_depth > 0) {
                                const Token& inner = peek();
                                if (inner.type == TokenType::LeftParen) paren_depth++;
                                else if (inner.type == TokenType::RightParen) paren_depth--;
                                raw_code += inner.lexeme;
                                advance();

                                // Add whitespace
                                if (!is_at_end() && current < tokens.size()) {
                                    const Token& next_tok = peek();
                                    if (next_tok.line == inner.line && next_tok.column > inner.column + inner.lexeme.length()) {
                                        raw_code += " ";
                                    } else if (next_tok.line > inner.line) {
                                        raw_code += "\n";
                                    }
                                }
                            }

                            // Expect '{' for catch handler body
                            if (check(TokenType::LeftBrace)) {
                                raw_code += " ";
                                raw_code += peek().lexeme;
                                advance();

                                // Find matching closing brace for catch handler
                                int brace_depth = 1;
                                while (!is_at_end() && brace_depth > 0) {
                                    const Token& inner = peek();
                                    if (inner.type == TokenType::LeftBrace) brace_depth++;
                                    else if (inner.type == TokenType::RightBrace) brace_depth--;
                                    raw_code += inner.lexeme;
                                    advance();

                                    // Add whitespace
                                    if (!is_at_end() && current < tokens.size()) {
                                        const Token& next_tok = peek();
                                        if (next_tok.line == inner.line && next_tok.column > inner.column + inner.lexeme.length()) {
                                            raw_code += " ";
                                        } else if (next_tok.line > inner.line) {
                                            raw_code += "\n";
                                        }
                                    }
                                }
                            }
                        }
                        // Continue to check for additional catch handlers
                    } else {
                        // Unexpected token - stop here
                        break;
                    }
                }
            }
            break;
        }
    }

    auto decl = std::make_unique<Cpp1PassthroughDeclaration>(raw_code, tokens[start_pos].line);
    return decl;
}

// Error handling
void Parser::error(const Token& token, [[maybe_unused]] const char* message) {
    if (token.type == TokenType::EndOfFile) {
        error_at(token, " at end");
    } else {
        error_at(token, std::format(" at '{}'", token.lexeme).c_str());
    }
}

void Parser::error_at(const Token& token, const char* message) {
    // During speculative parsing, suppress error output
    if (suppress_errors) {
        panic_mode = true;
        return;
    }
    
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
    error_count++;  // Track error count
}

void Parser::error_at_current(const char* message) {
    error_at(peek(), message);
}

void Parser::collect_markdown_blocks() {
    // Collect all consecutive MarkdownBlock tokens
    while (check(TokenType::MarkdownBlock)) {
        Token block_token = advance();
        std::string_view content = block_token.lexeme;

        // Parse name from content (first word until newline or space)
        std::string name;
        std::string actual_content;
        std::size_t name_end = content.find_first_of(" \n\r\t");

        if (name_end != 0) {
            // Content starts with a name
            name = std::string(content.substr(0, name_end));
            if (name_end < content.length()) {
                actual_content = std::string(content.substr(name_end));
            }
        } else {
            actual_content = std::string(content);
        }

        // Compute SHA256 hash
        std::string sha256 = compute_markdown_hash(actual_content);

        // Create metadata and add to pending blocks
        pending_markdown_blocks.emplace_back(
            std::move(sha256),
            std::string(content),
            std::move(name),
            block_token.line,
            block_token.column
        );
    }
}

void Parser::attach_markdown_blocks(Declaration* decl) {
    if (!decl || pending_markdown_blocks.empty()) {
        return;
    }

    // Move all pending blocks to the declaration
    decl->markdown_blocks = std::move(pending_markdown_blocks);
    pending_markdown_blocks.clear();
}

} // namespace cpp2_transpiler