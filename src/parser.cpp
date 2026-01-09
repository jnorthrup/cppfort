// ============================================================================
// Cpp2 Parser - bbcursive-style approach
// ============================================================================
// All parsers: bool parse(Stream&) - advance stream on success, restore on fail
// No complex Result types, no template explosion

#include "parser.hpp"
#include "core/tokens.hpp"
#include "ast.hpp"
#include <span>
#include <functional>
#include <string_view>

namespace cpp2_transpiler {

// ============================================================================
// Stream - mutable position, like ByteBuffer
// ============================================================================

struct Stream {
    std::span<const Token> tokens;
    size_t pos = 0;
    
    bool empty() const { return pos >= tokens.size(); }
    const Token& peek() const { return tokens[pos]; }
    void advance() { if (!empty()) ++pos; }
    size_t position() const { return pos; }
    void restore(size_t p) { pos = p; }
};

// ============================================================================
// Rule - type-erased parser, stores std::function<bool(Stream&)>
// ============================================================================

using ParseFn = std::function<bool(Stream&)>;

struct Rule {
    ParseFn fn;
    
    bool parse(Stream& s) const { return fn ? fn(s) : false; }
    void set(ParseFn f) { fn = std::move(f); }
    
    // Convenience operator
    bool operator()(Stream& s) const { return parse(s); }
};

// ============================================================================
// Combinators - all return ParseFn
// ============================================================================

// Terminal: match specific token type
inline ParseFn tok(TokenType type) {
    return [type](Stream& s) -> bool {
        if (s.empty() || s.peek().type != type) return false;
        s.advance();
        return true;
    };
}

// Terminal: match identifier with specific lexeme (keyword)
inline ParseFn kw(std::string_view lexeme) {
    return [lexeme](Stream& s) -> bool {
        if (s.empty() || s.peek().type != TokenType::Identifier || s.peek().lexeme != lexeme) return false;
        s.advance();
        return true;
    };
}

// seq: a >> b (sequence)
template<typename... Ps>
inline ParseFn seq(Ps... ps) {
    return [=](Stream& s) -> bool {
        auto start = s.position();
        bool ok = (ps(s) && ...);
        if (!ok) s.restore(start);
        return ok;
    };
}

// alt: a | b (alternative with backtrack)
template<typename... Ps>
inline ParseFn alt(Ps... ps) {
    return [=](Stream& s) -> bool {
        return (ps(s) || ...);
    };
}

// opt: -p (optional)
inline ParseFn opt(ParseFn p) {
    return [p](Stream& s) -> bool {
        p(s);  // Try but always succeed
        return true;
    };
}

// many: *p (zero or more)
inline ParseFn many(ParseFn p) {
    return [p](Stream& s) -> bool {
        while (p(s)) {}
        return true;
    };
}

// some: +p (one or more)
inline ParseFn some(ParseFn p) {
    return [p](Stream& s) -> bool {
        if (!p(s)) return false;
        while (p(s)) {}
        return true;
    };
}

// ref: lazy reference to Rule (for recursion)
inline ParseFn ref(Rule& r) {
    return [&r](Stream& s) -> bool { return r.parse(s); };
}

// ============================================================================
// Forward-declared rules for recursion
// ============================================================================

static Rule expression;
static Rule statement;
static Rule type_specifier;
static Rule decl_rule;
static Rule block_statement;
static Rule param_list;
static Rule template_params;
static Rule template_args;

// ============================================================================
// Grammar initialization
// ============================================================================

static bool g_init = false;

void init_grammar() {
    if (g_init) return;
    g_init = true;
    
    using TT = TokenType;
    
    // Keywords
    auto FUNC = tok(TT::Func);
    auto TYPE = tok(TT::Type);
    auto NAMESPACE = tok(TT::Namespace);
    auto LET = tok(TT::Let);
    auto CONST = tok(TT::Const);
    auto RETURN = tok(TT::Return);
    auto IF = tok(TT::If);
    auto ELSE = tok(TT::Else);
    auto WHILE = tok(TT::While);
    auto FOR = tok(TT::For);
    auto DO = tok(TT::Do);
    auto BREAK = tok(TT::Break);
    auto CONTINUE = tok(TT::Continue);
    auto SWITCH = tok(TT::Switch);
    auto CASE = tok(TT::Case);
    auto DEFAULT = tok(TT::Default);
    auto TRY = tok(TT::Try);
    auto CATCH = tok(TT::Catch);
    auto THROW = tok(TT::Throw);
    auto THROWS = tok(TT::Throws);
    auto NOEXCEPT = tok(TT::Noexcept);
    auto REQUIRES = tok(TT::Requires);
    auto PUBLIC = tok(TT::Public);
    auto PRIVATE = tok(TT::Private);
    auto PROTECTED = tok(TT::Protected);
    auto VIRTUAL = tok(TT::Virtual);
    auto OVERRIDE = tok(TT::Override);
    auto IMPLICIT = tok(TT::Implicit);
    auto THIS = tok(TT::This);
    auto THAT = tok(TT::That);
    auto IN = tok(TT::In);
    auto TRUE = tok(TT::True);
    auto FALSE = tok(TT::False);
    auto AUTO = tok(TT::Auto);
    auto CONCEPT = tok(TT::Concept);
    auto AS = tok(TT::As);
    auto IS = tok(TT::Is);
    auto FINAL = tok(TT::Final);
    auto IMPORT = tok(TT::Import);
    
    // Contextual keywords
    auto COPY = kw("copy");
    auto MOVE = kw("move");
    auto FORWARD = kw("forward");
    auto INOUT = kw("inout");
    auto OUT = kw("out");
    auto PRE = kw("pre");
    auto POST = kw("post");
    auto ASSERT = kw("assert");
    auto USING = kw("using");
    auto TEMPLATE = kw("template");
    auto AWAIT = kw("await");
    
    // Punctuation
    auto LPAREN = tok(TT::LeftParen);
    auto RPAREN = tok(TT::RightParen);
    auto LBRACE = tok(TT::LeftBrace);
    auto RBRACE = tok(TT::RightBrace);
    auto LBRACKET = tok(TT::LeftBracket);
    auto RBRACKET = tok(TT::RightBracket);
    auto LT = tok(TT::LessThan);
    auto GT = tok(TT::GreaterThan);
    auto COLON = tok(TT::Colon);
    auto SEMICOLON = tok(TT::Semicolon);
    auto COMMA = tok(TT::Comma);
    auto DOT = tok(TT::Dot);
    auto ARROW = tok(TT::Arrow);
    auto EQUAL = tok(TT::Equal);
    auto DOUBLE_EQUAL = tok(TT::DoubleEqual);
    auto COLON_EQUAL = tok(TT::ColonEqual);
    auto DOUBLE_COLON = tok(TT::DoubleColon);
    auto AT = tok(TT::At);
    auto ELLIPSIS = tok(TT::Ellipsis);
    auto UNDERSCORE = tok(TT::Underscore);
    auto PLUS = tok(TT::Plus);
    auto MINUS = tok(TT::Minus);
    auto STAR = tok(TT::Asterisk);
    auto SLASH = tok(TT::Slash);
    auto PERCENT = tok(TT::Percent);
    auto AMP = tok(TT::Ampersand);
    auto PIPE = tok(TT::Pipe);
    auto QUESTION = tok(TT::Question);
    auto DOUBLE_AMP = tok(TT::DoubleAmpersand);
    auto DOUBLE_PIPE = tok(TT::DoublePipe);
    auto BANG = tok(TT::Exclamation);
    auto TILDE = tok(TT::Tilde);
    
    // Identifiers & Literals
    auto IDENTIFIER = tok(TT::Identifier);
    auto INTEGER_LIT = tok(TT::IntegerLiteral);
    auto FLOAT_LIT = tok(TT::FloatLiteral);
    auto STRING_LIT = tok(TT::StringLiteral);
    auto CHAR_LIT = tok(TT::CharacterLiteral);
    auto END_OF_FILE = tok(TT::EndOfFile);
    
    // ========================================================================
    // Grammar Rules
    // ========================================================================
    
    // access_specifier = "public" | "private" | "protected"
    auto access_specifier = alt(PUBLIC, PRIVATE, PROTECTED);
    
    // identifier_like
    auto identifier_like = alt(IDENTIFIER, IN, COPY, MOVE, FORWARD, FUNC, TYPE, NAMESPACE, UNDERSCORE, OUT, INOUT);
    
    // param_qualifier = in | copy | inout | out | move | forward | virtual | override | implicit
    auto param_qualifier = alt(IN, COPY, INOUT, OUT, MOVE, FORWARD, VIRTUAL, OVERRIDE, IMPLICIT);
    
    // param_name = IDENTIFIER | "this" | "that" | "_"
    auto param_name = alt(IDENTIFIER, THIS, THAT, UNDERSCORE);
    
    // throws_spec = "throws" | "noexcept"
    auto throws_spec = alt(THROWS, NOEXCEPT);
    
    // literal = true | false | int | float | string | char
    auto literal = alt(TRUE, FALSE, INTEGER_LIT, FLOAT_LIT, STRING_LIT, CHAR_LIT);
    
    // template_params = '<' ids '>'
    template_params.set(seq(LT, opt(seq(IDENTIFIER, many(seq(COMMA, IDENTIFIER)))), GT));
    
    // template_args = '<' exprs '>'
    template_args.set(seq(LT, opt(seq(ref(expression), many(seq(COMMA, ref(expression))))), GT));
    
    // type_specifier
    type_specifier.set(seq(
        alt(IDENTIFIER, AUTO, UNDERSCORE, TYPE),
        opt(ref(template_args)),
        many(seq(DOUBLE_COLON, IDENTIFIER, opt(ref(template_args)))),
        many(alt(STAR, AMP))
    ));
    
    // parameter = { qualifier } name [ ... ] [ : type ] [ = expr ]
    auto parameter = seq(
        many(param_qualifier),
        param_name,
        opt(ELLIPSIS),
        opt(seq(COLON, ref(type_specifier))),
        opt(seq(EQUAL, ref(expression)))
    );
    
    // param_list = '(' [ parameters ] ')'
    param_list.set(seq(LPAREN, opt(seq(parameter, many(seq(COMMA, parameter)), opt(COMMA))), RPAREN));
    
    // return_spec = '->' type
    auto return_spec = seq(ARROW, opt(alt(FORWARD, MOVE)), ref(type_specifier));
    
    // requires_clause = 'requires' expression
    auto requires_clause = seq(REQUIRES, ref(expression));
    
    // contracts
    auto contracts = many(seq(alt(PRE, POST, ASSERT), alt(seq(COLON, ref(expression)), seq(LPAREN, ref(expression), RPAREN))));
    
    // function_body
    auto function_body = alt(
        seq(EQUAL, ref(expression), SEMICOLON),
        seq(DOUBLE_EQUAL, ref(expression), SEMICOLON),
        seq(EQUAL, ref(block_statement)),
        ref(block_statement),
        SEMICOLON
    );
    
    // func_signature
    auto func_signature = seq(
        opt(ref(template_params)),
        ref(param_list),
        opt(throws_spec),
        opt(return_spec),
        opt(requires_clause),
        contracts
    );
    
    // block_statement = '{' { statement } '}'
    block_statement.set(seq(LBRACE, many(ref(statement)), RBRACE));
    
    // type_body
    auto type_body = seq(EQUAL, LBRACE, many(ref(decl_rule)), RBRACE, opt(SEMICOLON));
    
    // namespace_body
    auto namespace_body = seq(opt(EQUAL), LBRACE, many(ref(decl_rule)), RBRACE);
    
    // ========================================================================
    // Expressions
    // ========================================================================
    
    // qualified_name
    auto qualified_name = seq(IDENTIFIER, many(seq(DOUBLE_COLON, IDENTIFIER)));
    
    // primary_expression
    auto primary_expression = alt(
        literal,
        seq(opt(DOUBLE_COLON), qualified_name),
        THIS, THAT, UNDERSCORE,
        seq(LPAREN, ref(expression), RPAREN),
        seq(LBRACKET, opt(seq(ref(expression), many(seq(COMMA, ref(expression))))), RBRACKET)
    );
    
    // postfix_expression
    auto call_op = seq(LPAREN, opt(seq(ref(expression), many(seq(COMMA, ref(expression))))), RPAREN);
    auto member_op = seq(DOT, IDENTIFIER);
    auto subscript_op = seq(LBRACKET, ref(expression), RBRACKET);
    auto postfix_op = alt(call_op, member_op, subscript_op, tok(TT::PlusPlus), tok(TT::MinusMinus));
    auto postfix_expression = seq(primary_expression, many(postfix_op));
    
    // prefix_expression
    auto prefix_op = alt(PLUS, MINUS, BANG, TILDE, tok(TT::PlusPlus), tok(TT::MinusMinus), AMP, STAR);
    auto prefix_expression = seq(many(prefix_op), postfix_expression);
    
    // Binary expressions (flattened precedence for simplicity)
    auto mult_op = alt(STAR, SLASH, PERCENT);
    auto add_op = alt(PLUS, MINUS);
    auto cmp_op = alt(LT, GT, tok(TT::LessThanOrEqual), tok(TT::GreaterThanOrEqual));
    auto eq_op = alt(DOUBLE_EQUAL, tok(TT::NotEqual));
    auto assign_op = alt(EQUAL, tok(TT::PlusEqual), tok(TT::MinusEqual), tok(TT::AsteriskEqual), tok(TT::SlashEqual));
    
    auto mult_expr = seq(prefix_expression, many(seq(mult_op, prefix_expression)));
    auto add_expr = seq(mult_expr, many(seq(add_op, mult_expr)));
    auto cmp_expr = seq(add_expr, many(seq(cmp_op, add_expr)));
    auto eq_expr = seq(cmp_expr, many(seq(eq_op, cmp_expr)));
    auto and_expr = seq(eq_expr, many(seq(DOUBLE_AMP, eq_expr)));
    auto or_expr = seq(and_expr, many(seq(DOUBLE_PIPE, and_expr)));
    auto ternary_expr = seq(or_expr, opt(seq(QUESTION, ref(expression), COLON, or_expr)));
    auto assign_expr = seq(ternary_expr, opt(seq(assign_op, ref(expression))));
    
    expression.set(assign_expr);
    
    // ========================================================================
    // Statements
    // ========================================================================
    
    auto return_stmt = seq(RETURN, opt(ref(expression)), SEMICOLON);
    auto break_stmt = seq(BREAK, opt(IDENTIFIER), SEMICOLON);
    auto continue_stmt = seq(CONTINUE, opt(IDENTIFIER), SEMICOLON);
    auto throw_stmt = seq(THROW, opt(ref(expression)), SEMICOLON);
    auto if_stmt = seq(IF, ref(expression), ref(block_statement), opt(seq(ELSE, alt(ref(block_statement), ref(statement)))));
    auto while_stmt = seq(WHILE, ref(expression), ref(block_statement));
    auto do_stmt = seq(DO, ref(block_statement), WHILE, ref(expression), SEMICOLON);
    auto for_stmt = seq(FOR, ref(expression), DO, LPAREN, parameter, RPAREN, ref(block_statement));
    auto switch_case = alt(seq(CASE, ref(expression), COLON, ref(statement)), seq(DEFAULT, COLON, ref(statement)));
    auto switch_stmt = seq(SWITCH, ref(expression), LBRACE, many(switch_case), RBRACE);
    auto try_stmt = seq(TRY, ref(block_statement), many(seq(CATCH, LPAREN, opt(seq(ref(type_specifier), opt(IDENTIFIER))), RPAREN, ref(block_statement))));
    auto contract_stmt = seq(alt(ASSERT, PRE, POST), ref(expression), SEMICOLON);
    auto expr_stmt = seq(ref(expression), SEMICOLON);
    auto local_decl = alt(
        seq(IDENTIFIER, COLON, ref(type_specifier), opt(seq(alt(EQUAL, DOUBLE_EQUAL), ref(expression))), SEMICOLON),
        seq(IDENTIFIER, COLON_EQUAL, ref(expression), SEMICOLON)
    );
    
    statement.set(alt(
        ref(block_statement), if_stmt, while_stmt, for_stmt, do_stmt, switch_stmt, try_stmt,
        return_stmt, break_stmt, continue_stmt, throw_stmt, contract_stmt, local_decl, expr_stmt, SEMICOLON
    ));
    
    // ========================================================================
    // Declarations
    // ========================================================================
    
    auto let_decl = seq(alt(LET, CONST), IDENTIFIER, opt(seq(COLON, ref(type_specifier))), alt(EQUAL, DOUBLE_EQUAL), ref(expression), SEMICOLON);
    auto func_decl = seq(FUNC, IDENTIFIER, opt(COLON), func_signature, function_body);
    auto type_decl = seq(TYPE, IDENTIFIER, opt(ref(template_params)), opt(requires_clause), type_body);
    auto namespace_decl = seq(NAMESPACE, IDENTIFIER, alt(seq(DOUBLE_EQUAL, qualified_name, SEMICOLON), namespace_body));
    auto import_decl = seq(IMPORT, IDENTIFIER, SEMICOLON);
    auto using_decl = seq(USING, alt(
        seq(NAMESPACE, IDENTIFIER, SEMICOLON),
        seq(IDENTIFIER, EQUAL, qualified_name, SEMICOLON),
        seq(qualified_name, SEMICOLON)
    ));
    
    // Unified declarations
    auto unified_func = seq(func_signature, function_body);
    auto unified_type = seq(many(seq(AT, IDENTIFIER, opt(ref(template_args)))), alt(TYPE, CONCEPT),
                            opt(ref(template_params)), opt(requires_clause),
                            alt(seq(DOUBLE_EQUAL, ref(type_specifier), SEMICOLON),
                                seq(EQUAL, ref(type_specifier), SEMICOLON), type_body));
    auto unified_ns = seq(NAMESPACE, alt(seq(DOUBLE_EQUAL, qualified_name, SEMICOLON), namespace_body));
    auto unified_var = seq(ref(type_specifier), opt(seq(alt(EQUAL, DOUBLE_EQUAL), ref(expression))), SEMICOLON);
    auto unified_decl = seq(identifier_like, COLON, opt(ref(template_params)), alt(unified_func, unified_type, unified_ns, unified_var));
    auto auto_decl = seq(identifier_like, COLON_EQUAL, ref(expression), SEMICOLON);
    
    decl_rule.set(seq(
        opt(access_specifier),
        alt(func_decl, type_decl, namespace_decl, using_decl, import_decl, let_decl, unified_decl, auto_decl, ref(statement))
    ));
}

// ============================================================================
// Parser Public Interface
// ============================================================================

Parser::Parser(std::span<Token> tokens) : tokens(tokens) {
    init_grammar();
}

std::unique_ptr<AST> Parser::parse() {
    auto ast = std::make_unique<AST>();
    Stream s{std::span<const Token>(tokens.data(), tokens.size())};
    
    // Parse translation_unit = { declaration } EOF
    while (!s.empty() && s.peek().type != TokenType::EndOfFile) {
        if (!decl_rule.parse(s)) {
            error_count++;
            s.advance();  // Skip bad token
        }
    }
    
    return ast;
}

} // namespace cpp2_transpiler