// ============================================================================
// CPP2 Parser - Spirit EBNF Mapping with AST Construction
// ============================================================================
// Grammar rules as constexpr static members for zero-cost idempotent access.
// Builds ParseTree via TreeBuilder during parse.
// ============================================================================

#include "combinators/spirit.hpp"
#include "slim_ast.hpp"

namespace cpp2::parser {
namespace { // Internal linkage

using namespace spirit;
using namespace cpp2::ast;

// ============================================================================
// Thread-Local Tree Builder
// ============================================================================



// ============================================================================
// Semantic Token Flyweights (Zero-Cost Constexpr)
// ============================================================================

struct Tok {
    static constexpr auto ID    = tok(TT::Identifier);
    static constexpr auto INT   = tok(TT::IntegerLiteral);
    static constexpr auto FLOAT = tok(TT::FloatLiteral);
    static constexpr auto STR   = tok(TT::StringLiteral);
    static constexpr auto CHR   = tok(TT::CharacterLiteral);
    static constexpr auto END   = tok(TT::EndOfFile);
};

// ============================================================================
// Operator Groups (Idempotent Constexpr Statics)
// ============================================================================

struct Ops {
    static constexpr auto prefix     = lit("+") | "-" | "!" | "~" | "++" | "--" | "&" | "*" | "call";
    static constexpr auto postfix_op = lit("++") | "--" | "*" | "&";
    static constexpr auto assign     = lit("=") | "+=" | "-=" | "*=" | "/=" | "%=";
    static constexpr auto mul        = lit("*") | "/" | "%";
    static constexpr auto add        = lit("+") | "-";
    static constexpr auto shift      = lit("<<") | ">>";
    static constexpr auto cmp        = lit("<") | ">" | "<=" | ">=" | "<=>";
    static constexpr auto eq         = lit("==") | "!=";
    static constexpr auto param_qual = lit("in") | "out" | "inout" | "copy" | "move" | "forward";
    static constexpr auto access     = lit("public") | "private" | "protected";
};

// ============================================================================
// Recursive Parser References (Lazy via std::function)
// ============================================================================
// Uses std::function to defer grammar rule access entirely until parse time.
// No function-local statics for the recursive parsers themselves.

using ParseFn = std::function<ebnf::Result<std::monostate, TokenStream>(TokenStream)>;

// Forward declarations
auto parse_type_specifier(TokenStream) -> ebnf::Result<std::monostate, TokenStream>;
auto parse_expression(TokenStream) -> ebnf::Result<std::monostate, TokenStream>;
auto parse_statement(TokenStream) -> ebnf::Result<std::monostate, TokenStream>;
auto parse_declaration(TokenStream) -> ebnf::Result<std::monostate, TokenStream>;

// Parser struct that calls the parse function
struct FnParser {
    ParseFn fn;
    auto parse(TokenStream input) const -> ebnf::Result<std::monostate, TokenStream> {
        return fn(input);
    }
};

// These return new Proto<FnParser> objects each time - no static storage
inline auto type_spec_parser() { return Proto<FnParser>{{parse_type_specifier}}; }
inline auto expr_parser()      { return Proto<FnParser>{{parse_expression}}; }
inline auto stmt_parser()      { return Proto<FnParser>{{parse_statement}}; }
inline auto decl_parser()      { return Proto<FnParser>{{parse_declaration}}; }

// ============================================================================
// Grammar Rules (Constexpr Static Members)
// ============================================================================

struct Rules {
    // Terminals - include various identifer-like tokens
    static constexpr auto identifier_like = Tok::ID | lit("_") | lit("this") | lit("that") | lit("$");
    static constexpr auto literal = lit("true") | "false" | Tok::INT | Tok::FLOAT | Tok::STR | Tok::CHR;
    
    // Types
    static constexpr auto basic_type = (lit("auto") | "_" | Tok::ID) % with_node(NodeKind::BasicType);
};

// Rules requiring forward refs need function-local statics
inline auto& template_args() { static auto r = (lit("<") >> -(type_spec_parser() % ",") >> ">") % with_node(NodeKind::TemplateArgs); return r; }
inline auto& qualified_type() { 
    // Grammar: [const] basic_type [::ID [<args>]]* [const | * | &]*
    static auto r = (-lit("const") >> Rules::basic_type >> *(lit("::") >> Tok::ID >> -template_args()) >> *(lit("const") | "*" | "&")) % with_node(NodeKind::QualifiedType); 
    return r; 
}
inline auto& type_specifier() { static auto r = qualified_type() % with_node(NodeKind::TypeSpecifier); return r; }

// ============================================================================
// Pratt Parser for Expressions (Iterative - No Stack Explosion)
// ============================================================================
// Replaces 14+ combinator rules with a single iterative precedence parser.

namespace pratt {

// Operator precedence levels (higher = tighter binding)
enum Prec : int {
    NONE = 0,
    ASSIGN = 1,      // = += -= etc
    PIPELINE = 2,    // |>
    TERNARY = 3,     // ?:
    LOR = 4,         // ||
    LAND = 5,        // &&
    BOR = 6,         // |
    BXOR = 7,        // ^
    BAND = 8,        // &
    EQ = 9,          // == !=
    CMP = 10,        // < > <= >= <=>
    SHIFT = 11,      // << >>
    ADD = 12,        // + -
    MUL = 13,        // * / %
    PREFIX = 14,     // unary + - ! ~ ++ -- & *
    POSTFIX = 15,    // () [] . ++ --
};

// Get precedence for binary operator
inline int get_prec(std::string_view op) {
    if (op == "=" || op == "+=" || op == "-=" || op == "*=" || op == "/=" || op == "%=") return ASSIGN;
    if (op == "|>") return PIPELINE;
    if (op == "||") return LOR;
    if (op == "&&") return LAND;
    if (op == "|") return BOR;
    if (op == "^") return BXOR;
    if (op == "&") return BAND;
    if (op == "==" || op == "!=") return EQ;
    if (op == "<" || op == ">" || op == "<=" || op == ">=" || op == "<=>") return CMP;
    if (op == "<<" || op == ">>") return SHIFT;
    if (op == "+" || op == "-") return ADD;
    if (op == "*" || op == "/" || op == "%") return MUL;
    // Range operators: .., ..<, ..=
    if (op == ".." || op == "..<" || op == "..=") return CMP;
    return NONE;
}

// Check if token is a binary operator
inline bool is_binop(const cpp2_transpiler::Token& t) {
    return get_prec(t.lexeme) > NONE;
}

// Check if token is prefix operator
inline bool is_prefix(const cpp2_transpiler::Token& t) {
    auto l = t.lexeme;
    return l == "+" || l == "-" || l == "!" || l == "~" ||
           l == "++" || l == "--" || l == "&" || l == "*";
}

// Check if token starts postfix operation
inline bool is_postfix_start(const cpp2_transpiler::Token& t) {
    auto l = t.lexeme;
    return l == "(" || l == "[" || l == "." || l == "++" || l == "--" || l == "*" || l == "&";
}

// Forward declaration
auto parse_pratt(TokenStream input, int min_prec) -> ebnf::Result<std::monostate, TokenStream>;

// Parse primary expression (literal, identifier, grouped)
auto parse_primary(TokenStream input) -> ebnf::Result<std::monostate, TokenStream> {
    if (input.empty()) return ebnf::Result<std::monostate, TokenStream>::fail(input);

    const auto& tok = input.peek();

    // Grouped expression
    if (tok.lexeme == "(") {
        begin(NodeKind::GroupedExpression, input.pos);
        input = input.next(); // consume (
        auto inner = parse_pratt(input, NONE);
        if (!inner.success()) return ebnf::Result<std::monostate, TokenStream>::fail(input);
        input = inner.remaining();
        if (input.empty() || input.peek().lexeme != ")")
            return ebnf::Result<std::monostate, TokenStream>::fail(input);
        input = input.next(); // consume )
        end(input.pos);
        return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
    }

    // Literals (all types)
    if (tok.type == TT::IntegerLiteral || tok.type == TT::FloatLiteral ||
        tok.type == TT::StringLiteral || tok.type == TT::CharacterLiteral ||
        tok.type == TT::BooleanLiteral || tok.type == TT::InterpolatedRawStringLiteral) {
        begin(NodeKind::Literal, input.pos);
        input = input.next();
        end(input.pos);
        return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
    }

    // Keywords: this, that, true, false, _
    if (tok.lexeme == "this" || tok.lexeme == "that" || tok.lexeme == "_" ||
        tok.lexeme == "true" || tok.lexeme == "false") {
        begin(NodeKind::Identifier, input.pos);
        input = input.next();
        end(input.pos);
        return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
    }

    // Identifier
    if (tok.type == TT::Identifier) {
        begin(NodeKind::Identifier, input.pos);
        input = input.next();
        end(input.pos);
        return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
    }

    return ebnf::Result<std::monostate, TokenStream>::fail(input);
}

// Parse atom: prefix* primary postfix*
auto parse_atom(TokenStream input) -> ebnf::Result<std::monostate, TokenStream> {
    auto start = input.pos;

    // Prefix operators
    if (!input.empty() && is_prefix(input.peek())) {
        begin(NodeKind::PrefixExpression, input.pos);
        
        // Operator node
        begin(NodeKind::PrefixOp, input.pos);
        input = input.next(); // consume operator
        end(input.pos);
        
        // Recurse for operand (which might be another prefix expr)
        auto rhs = parse_atom(input);
        if (!rhs.success()) return rhs; // Fail propagates
        input = rhs.remaining();
        
        end(input.pos); // Close PrefixExpression
        return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
    }

    // Primary expression
    auto primary = parse_primary(input);
    if (!primary.success()) return ebnf::Result<std::monostate, TokenStream>::fail(input);
    input = primary.remaining();

    // Postfix operators
    while (!input.empty() && is_postfix_start(input.peek())) {
        const auto& tok = input.peek();

        if (tok.lexeme == "(") {
            // Function call
            begin(NodeKind::CallOp, input.pos);
            input = input.next(); // consume (
            // Parse arguments (comma-separated expressions)
            if (!input.empty() && input.peek().lexeme != ")") {
                auto arg = parse_pratt(input, NONE);
                if (arg.success()) {
                    input = arg.remaining();
                    while (!input.empty() && input.peek().lexeme == ",") {
                        input = input.next(); // consume ,
                        arg = parse_pratt(input, NONE);
                        if (!arg.success()) break;
                        input = arg.remaining();
                    }
                }
            }
            if (input.empty() || input.peek().lexeme != ")")
                return ebnf::Result<std::monostate, TokenStream>::fail(input);
            input = input.next(); // consume )
            end(input.pos);
        }
        else if (tok.lexeme == "[") {
            // Subscript
            begin(NodeKind::SubscriptOp, input.pos);
            input = input.next(); // consume [
            auto idx = parse_pratt(input, NONE);
            if (!idx.success()) return ebnf::Result<std::monostate, TokenStream>::fail(input);
            input = idx.remaining();
            if (input.empty() || input.peek().lexeme != "]")
                return ebnf::Result<std::monostate, TokenStream>::fail(input);
            input = input.next(); // consume ]
            end(input.pos);
        }
        else if (tok.lexeme == ".") {
            // Member access
            begin(NodeKind::MemberOp, input.pos);
            input = input.next(); // consume .
            if (input.empty() || input.peek().type != TT::Identifier)
                return ebnf::Result<std::monostate, TokenStream>::fail(input);
            input = input.next(); // consume member name
            end(input.pos);
        }
        else if (tok.lexeme == "++" || tok.lexeme == "--") {
            // Unambiguous postfix
            begin(NodeKind::PostfixOp, input.pos);
            input = input.next();
            end(input.pos);
        }
        else if (tok.lexeme == "*" || tok.lexeme == "&") {
            // Ambiguous postfix (vs binary * or &)
            // It is postfix primarily if the NEXT token does NOT look like the start of an operand.
            // If the next token IS an operand (id, literal, prefix op, etc), then this * or & is binary.
            // "operand start" = Identifier, Literal, '(', 'true', 'false', 'this', 'that', '_'
            // OR prefix operator (+, -, !, ~, ++, --, *, &) - but wait! `a * * b` -> `a * (*b)` (bin * prefix)
            // So if next is prefix safe, it's binary.
            
            // Heuristic: If next token is ';', ')', ']', ',', or an operator that cannot be prefix, it is postfix.
            // Simplification: Peek next. If it's a binary-only operator or terminator, it's postfix.
            // Otherwise assume binary (falling through loop).
            
            if (input.empty()) break; // End of file, shouldn't happen if we peeked
            const auto& next = input.peek(1); // peek + 1
            
            bool is_next_operand_start = 
                next.type == TT::Identifier || 
                next.type == TT::IntegerLiteral || 
                next.type == TT::FloatLiteral || 
                next.type == TT::StringLiteral ||
                next.type == TT::CharacterLiteral ||
                next.lexeme == "(" || 
                next.lexeme == "true" || next.lexeme == "false" ||
                next.lexeme == "this" || next.lexeme == "that" || next.lexeme == "_" ||
                is_prefix(next); // e.g. * * p -> mul deref p

            if (!is_next_operand_start) {
                begin(NodeKind::PostfixOp, input.pos);
                input = input.next();
                end(input.pos);
            } else {
                // It's binary (or start of next part of expr), stop postfix processing
                break;
            }
        }
        else {
            break;
        }
    }

    // Close any open prefix expressions
    // (handled implicitly by end positions)

    return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
}

// Pratt parser: iterative precedence climbing
auto parse_pratt(TokenStream input, int min_prec) -> ebnf::Result<std::monostate, TokenStream> {
    // Parse left-hand side (atom)
    auto lhs = parse_atom(input);
    if (!lhs.success()) return lhs;
    input = lhs.remaining();

    // Handle binary operators iteratively
    while (!input.empty()) {
        const auto& tok = input.peek();

        // Ternary operator special case
        if (tok.lexeme == "?" && TERNARY >= min_prec) {
            begin(NodeKind::TernaryExpression, input.pos);
            input = input.next(); // consume ?
            auto then_expr = parse_pratt(input, NONE);
            if (!then_expr.success()) return ebnf::Result<std::monostate, TokenStream>::fail(input);
            input = then_expr.remaining();
            if (input.empty() || input.peek().lexeme != ":")
                return ebnf::Result<std::monostate, TokenStream>::fail(input);
            input = input.next(); // consume :
            auto else_expr = parse_pratt(input, TERNARY);
            if (!else_expr.success()) return ebnf::Result<std::monostate, TokenStream>::fail(input);
            input = else_expr.remaining();
            end(input.pos);
            continue;
        }

        int prec = get_prec(tok.lexeme);
        if (prec == NONE || prec < min_prec) break;

        // Determine associativity (right-assoc for assignment, left for others)
        int next_min = (prec == ASSIGN) ? prec : prec + 1;

        // Build binary expression node based on operator
        // Build binary expression node based on operator
        NodeKind kind;
        NodeKind op_kind = NodeKind::BinaryOp; // Default operator kind

        if (tok.lexeme == "||") kind = NodeKind::LogicalOrExpression;
        else if (tok.lexeme == "&&") kind = NodeKind::LogicalAndExpression;
        else if (tok.lexeme == "|") kind = NodeKind::BitwiseOrExpression;
        else if (tok.lexeme == "^") kind = NodeKind::BitwiseXorExpression;
        else if (tok.lexeme == "&") kind = NodeKind::BitwiseAndExpression;
        else if (tok.lexeme == "==" || tok.lexeme == "!=") kind = NodeKind::EqualityExpression;
        else if (tok.lexeme == "<" || tok.lexeme == ">" || tok.lexeme == "<=" ||
                 tok.lexeme == ">=" || tok.lexeme == "<=>") kind = NodeKind::ComparisonExpression;
        else if (tok.lexeme == "<<" || tok.lexeme == ">>") kind = NodeKind::ShiftExpression;
        else if (tok.lexeme == "+" || tok.lexeme == "-") kind = NodeKind::AdditiveExpression;
        else if (tok.lexeme == "*" || tok.lexeme == "/" || tok.lexeme == "%") kind = NodeKind::MultiplicativeExpression;
        else if (tok.lexeme == "|>") kind = NodeKind::PipelineExpression;
        else if (tok.lexeme == ".." || tok.lexeme == "..<" || tok.lexeme == "..=") kind = NodeKind::RangeExpression;
        else {
            kind = NodeKind::AssignmentExpression;  // Default for assignment ops
            op_kind = NodeKind::AssignmentOp;
        }

        g_builder.start_infix(kind, input.pos);
        
        // Emit formal Operator node
        begin(op_kind, input.pos);
        input = input.next(); // consume operator
        end(input.pos);

        auto rhs = parse_pratt(input, next_min);
        if (!rhs.success()) return ebnf::Result<std::monostate, TokenStream>::fail(input);
        input = rhs.remaining();
        end(input.pos);
    }

    return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
}

} // namespace pratt

// Expression entry point using Pratt parser
auto parse_expr_pratt(TokenStream input) -> ebnf::Result<std::monostate, TokenStream> {
    return pratt::parse_pratt(input, pratt::NONE);
}

// Statements
inline auto& block_stmt() { static auto r = (lit("{") >> *stmt_parser() >> "}") % with_node(NodeKind::BlockStatement); return r; }
inline auto& return_stmt() { static auto r = (lit("return") >> -expr_parser() >> ";") % with_node(NodeKind::ReturnStatement); return r; }
inline auto& break_stmt() { static auto r = lit("break") >> -Rules::identifier_like >> ";"; return r; }
inline auto& continue_stmt() { static auto r = lit("continue") >> -Rules::identifier_like >> ";"; return r; }
inline auto& next_stmt() { static auto r = lit("next") >> -Rules::identifier_like >> ";"; return r; }
inline auto& throw_stmt() { static auto r = lit("throw") >> -expr_parser() >> ";"; return r; }
inline auto& assert_stmt() { static auto r = (lit("assert") | "pre" | "post") >> "(" >> expr_parser() >> -(lit(",") >> expr_parser()) >> ")" >> ";"; return r; }

// Control flow
inline auto& if_stmt() { static auto r = (lit("if") >> expr_parser() >> (block_stmt() | stmt_parser()) >> -(lit("else") >> (block_stmt() | stmt_parser()))) % with_node(NodeKind::IfStatement); return r; }
inline auto& while_stmt() { static auto r = (lit("while") >> expr_parser() >> block_stmt()) % with_node(NodeKind::WhileStatement); return r; }
inline auto& do_while_stmt() { static auto r = lit("do") >> block_stmt() >> "while" >> expr_parser() >> ";"; return r; }
inline auto& for_range_stmt() { 
    // Cpp2: for 0 ..< 10 do (i) { body }
    // OR: for items do (item) { body }
    static auto r = (lit("for") >> expr_parser() >> "do" >> "(" >> Rules::identifier_like >> ")" >> block_stmt()) % with_node(NodeKind::ForStatement);
    return r; 
}

// Try-catch
inline auto& catch_clause() { 
    // catch (name: type) { block }
    static auto r = lit("catch") >> "(" >> *Ops::param_qual >> Rules::identifier_like >> -(lit(":") >> type_specifier()) >> ")" >> block_stmt(); 
    return r; 
}
inline auto& try_stmt() { static auto r = lit("try") >> block_stmt() >> *catch_clause(); return r; }

// Switch / inspect
inline auto& is_pattern()       { static auto r = lit("is") >> (type_specifier() | expr_parser()); return r; }
inline auto& as_pattern()       { static auto r = lit("as") >> type_specifier() >> "=" >> Rules::identifier_like; return r; }
inline auto& pattern()          { static auto r = is_pattern() | as_pattern() | (lit("_") | expr_parser()); return r; }  
inline auto& inspect_arm()      { static auto r = pattern() >> "=" >> (block_stmt() | (expr_parser() >> ";")); return r; }
inline auto& inspect_expr()     { static auto r = lit("inspect") >> expr_parser() >> -(lit("->") >> type_specifier()) >> "{" >> *inspect_arm() >> "}"; return r; }

// Local variable declaration (inside block)
inline auto& local_var_decl()   { static auto r = (Rules::identifier_like >> (
    (":="_l >> expr_parser() >> ";") |
    (lit(":") >> type_specifier() >> -(lit("=") >> expr_parser()) >> ";")
)) % with_node(NodeKind::UnifiedDeclaration); return r; }

// Expression statement
inline auto& expr_stmt()        { static auto r = (expr_parser() >> ";") % with_node(NodeKind::ExpressionStatement); return r; }

// Statement alternatives
inline auto& statement()        { 
    static auto r = (block_stmt() | if_stmt() | while_stmt() | do_while_stmt() | for_range_stmt() | 
                    try_stmt() | return_stmt() | break_stmt() | continue_stmt() | next_stmt() | 
                    throw_stmt() | assert_stmt() | local_var_decl() | expr_stmt() | lit(";")) % with_node(NodeKind::Statement); 
    return r; 
}

// Parameters
inline auto& parameter()        { static auto r = (*Ops::param_qual >> Rules::identifier_like >> -(lit(":") >> type_specifier()) >> -(lit("=") >> expr_parser())) % with_node(NodeKind::Parameter); return r; }
inline auto& param_list()       { static auto r = (lit("(") >> -(parameter() % ",") >> ")") % with_node(NodeKind::ParamList); return r; }

// Declarations
inline auto& return_spec()      { static auto r = (lit("->") >> type_specifier()) % with_node(NodeKind::ReturnSpec); return r; }
inline auto& func_body()        { 
    // func_body = '=' expr ';' | '=' block | ';' (forward decl)
    // Cpp2: f: () = expr;  OR  f: () = { stmts }  OR  f: ();
    static auto r = ((lit("=") >> (block_stmt() | (expr_parser() >> ";"))) | lit(";")) % with_node(NodeKind::FunctionBody); 
    return r; 
}
inline auto& func_suffix()      { static auto r = (param_list() >> -return_spec() >> func_body()) % with_node(NodeKind::FunctionSuffix); return r; }
inline auto& var_suffix()       { 
    static auto r = (
          (type_specifier() >> -(lit("=") >> expr_parser()) >> ";") 
        | (lit("=") >> expr_parser() >> ";")
    ) % with_node(NodeKind::VariableSuffix); 
    return r; 
}
inline auto& type_body()        { static auto r = (lit("=") >> "{" >> *decl_parser() >> "}") % with_node(NodeKind::TypeBody); return r; }
inline auto& type_suffix()      { static auto r = (lit("type") >> -type_body()) % with_node(NodeKind::TypeSuffix); return r; }
inline auto& ns_body()          { static auto r = (-lit("=") >> "{" >> *decl_parser() >> "}") % with_node(NodeKind::NamespaceBody); return r; }
inline auto& ns_suffix()        { static auto r = (lit("namespace") >> ns_body()) % with_node(NodeKind::NamespaceSuffix); return r; }

// Template parameters: <T, U = int, ...Args>
inline auto& template_param()   { static auto r = Rules::identifier_like >> -(lit("...")) >> -(lit("=") >> type_specifier()); return r; }
inline auto& template_params()  { static auto r = (lit("<") >> -(template_param() % ",") >> ">") % with_node(NodeKind::TemplateArgs); return r; }

// Type alias: MyInt: type == int;
inline auto& alias_suffix()     { static auto r = lit("type") >> "==" >> type_specifier() >> ";"; return r; }

// Operator declaration: operator=: (params) -> type = body
inline auto& operator_name()    { static auto r = lit("=") | "[]" | "()" | "++" | "--" | "->" | "<=>" | "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | ">" | "<=" | ">="; return r; }
inline auto& operator_suffix()  { static auto r = lit("operator") >> operator_name() >> ":" >> -template_params() >> func_suffix(); return r; }

// Extends decl_suffix to include templates and aliases
inline auto& decl_suffix()      { 
    static auto r = 
        (template_params() >> (func_suffix() | type_suffix() | var_suffix())) |  // templated declarations
        alias_suffix() |
        operator_suffix() |
        func_suffix() | 
        type_suffix() | 
        ns_suffix() | 
        var_suffix(); 
    return r; 
}
inline auto& unified_decl()     { static auto r = (Rules::identifier_like >> ((":="_l >> expr_parser() >> ";") | (lit(":") >> decl_suffix()))) % with_node(NodeKind::UnifiedDeclaration); return r; }
inline auto& declaration()      { static auto r = (-Ops::access >> unified_decl()) % with_node(NodeKind::Declaration); return r; }
inline auto& translation_unit() { static auto r = *declaration() >> Tok::END; return r; }

// ============================================================================
// Recursive Parse Functions
// ============================================================================
// These are called lazily during parse, not during static init.
// They build AST nodes and delegate to the grammar rules.

// Helper to ensure parser made progress (prevents infinite loops in Many/Some)
template<typename R>
bool made_progress(const TokenStream& before, const R& result) {
    return result.remaining().pos > before.pos;
}

auto parse_type_specifier(TokenStream input) -> ebnf::Result<std::monostate, TokenStream> {
    return type_specifier().parse(input);
}

auto parse_expression(TokenStream input) -> ebnf::Result<std::monostate, TokenStream> {
    auto cp = tree_checkpoint();
    begin(NodeKind::Expression, input.pos);
    auto result = parse_expr_pratt(input);
    if (result.success() && made_progress(input, result)) {
        end(result.remaining().pos);
        return ebnf::Result<std::monostate, TokenStream>::ok({}, result.remaining());
    }
    tree_restore(cp);
    return ebnf::Result<std::monostate, TokenStream>::fail(input);
}

auto parse_statement(TokenStream input) -> ebnf::Result<std::monostate, TokenStream> {
    return statement().parse(input);
}

auto parse_declaration(TokenStream input) -> ebnf::Result<std::monostate, TokenStream> {
    return declaration().parse(input);
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

ParseTree parse(std::span<const cpp2_transpiler::Token> tokens) {
    g_builder = TreeBuilder{};
    TokenStream stream{tokens};
    begin(NodeKind::TranslationUnit, 0);
    auto result = translation_unit().parse(stream);
    if (result.success()) end(result.remaining().pos);
    return g_builder.finish(tokens);
}

} // namespace cpp2::parser
