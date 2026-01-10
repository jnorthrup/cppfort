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

inline thread_local TreeBuilder g_builder;

inline void begin(NodeKind k, std::size_t pos) { g_builder.begin(k, static_cast<uint32_t>(pos)); }
inline void end(std::size_t pos) { g_builder.end(static_cast<uint32_t>(pos)); }

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
    static constexpr auto prefix     = lit("+") | "-" | "!" | "~" | "++" | "--" | "&" | "*";
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
// Forward Declarations (Type-Erased Parsers)
// ============================================================================

struct TypeSpecifierParser { auto parse(TokenStream input) const -> ebnf::Result<std::monostate, TokenStream>; };
struct ExpressionParser    { auto parse(TokenStream input) const -> ebnf::Result<std::monostate, TokenStream>; };
struct StatementParser     { auto parse(TokenStream input) const -> ebnf::Result<std::monostate, TokenStream>; };
struct DeclarationParser   { auto parse(TokenStream input) const -> ebnf::Result<std::monostate, TokenStream>; };

inline auto type_spec_parser() { return lift(TypeSpecifierParser{}); }
inline auto expr_parser()      { return lift(ExpressionParser{}); }
inline auto stmt_parser()      { return lift(StatementParser{}); }
inline auto decl_parser()      { return lift(DeclarationParser{}); }

// ============================================================================
// Grammar Rules (Constexpr Static Members)
// ============================================================================

struct Rules {
    // Terminals
    static constexpr auto identifier_like = Tok::ID | lit("_");
    static constexpr auto literal = lit("true") | "false" | Tok::INT | Tok::FLOAT | Tok::STR | Tok::CHR;
    
    // Types
    static constexpr auto basic_type = lit("auto") | "_" | Tok::ID;
};

// Rules requiring forward refs need function-local statics
inline auto& template_args()    { static auto r = lit("<") >> -(type_spec_parser() % ",") >> ">"; return r; }
inline auto& qualified_type()   { static auto r = Rules::basic_type >> *(lit("::") >> Tok::ID >> -template_args()) >> *(lit("*") | "&"); return r; }
inline auto& type_specifier()   { return qualified_type(); }

// Expressions
inline auto& grouped_expr()     { static auto r = lit("(") >> -expr_parser() >> ")"; return r; }
inline auto& primary_expr()     { static auto r = Rules::literal | Tok::ID | lit("this") | "that" | "_" | grouped_expr(); return r; }
inline auto& call_op()          { static auto r = lit("(") >> -expr_parser() >> ")"; return r; }
inline auto& member_op()        { static auto r = lit(".") >> Tok::ID >> -call_op(); return r; }
inline auto& subscript_op()     { static auto r = lit("[") >> expr_parser() >> "]"; return r; }
inline auto& postfix_op()       { static auto r = call_op() | member_op() | subscript_op() | Ops::postfix_op; return r; }
inline auto& postfix_expr()     { static auto r = primary_expr() >> *postfix_op(); return r; }
inline auto& prefix_expr()      { static auto r = *Ops::prefix >> postfix_expr(); return r; }

// Binary ops - precedence climbing
inline auto& mul_expr()         { static auto r = prefix_expr() >> *(Ops::mul >> prefix_expr()); return r; }
inline auto& add_expr()         { static auto r = mul_expr() >> *(Ops::add >> mul_expr()); return r; }
inline auto& shift_expr()       { static auto r = add_expr() >> *(Ops::shift >> add_expr()); return r; }
inline auto& cmp_expr()         { static auto r = shift_expr() >> *(Ops::cmp >> shift_expr()); return r; }
inline auto& eq_expr()          { static auto r = cmp_expr() >> *(Ops::eq >> cmp_expr()); return r; }
inline auto& band_expr()        { static auto r = eq_expr() >> *(lit("&") >> eq_expr()); return r; }
inline auto& bxor_expr()        { static auto r = band_expr() >> *(lit("^") >> band_expr()); return r; }
inline auto& bor_expr()         { static auto r = bxor_expr() >> *(lit("|") >> bxor_expr()); return r; }
inline auto& land_expr()        { static auto r = bor_expr() >> *(lit("&&") >> bor_expr()); return r; }
inline auto& lor_expr()         { static auto r = land_expr() >> *(lit("||") >> land_expr()); return r; }
inline auto& ternary_expr()     { static auto r = lor_expr() >> -(lit("?") >> expr_parser() >> ":" >> lor_expr()); return r; }
inline auto& pipeline_expr()    { static auto r = ternary_expr() >> *(lit("|>") >> ternary_expr()); return r; }
inline auto& assign_expr()      { static auto r = pipeline_expr() >> *(Ops::assign >> pipeline_expr()); return r; }
inline auto& expression()       { return assign_expr(); }

// Statements
inline auto& block_stmt()       { static auto r = lit("{") >> *stmt_parser() >> "}"; return r; }
inline auto& return_stmt()      { static auto r = lit("return") >> -expr_parser() >> ";"; return r; }
inline auto& if_stmt()          { static auto r = lit("if") >> expr_parser() >> block_stmt() >> -(lit("else") >> block_stmt()); return r; }
inline auto& while_stmt()       { static auto r = lit("while") >> expr_parser() >> block_stmt(); return r; }
inline auto& for_stmt()         { static auto r = lit("for") >> expr_parser() >> "do" >> "(" >> Rules::identifier_like >> ")" >> block_stmt(); return r; }
inline auto& expr_stmt()        { static auto r = expr_parser() >> ";"; return r; }
inline auto& statement()        { static auto r = block_stmt() | if_stmt() | while_stmt() | for_stmt() | return_stmt() | expr_stmt() | lit(";"); return r; }

// Parameters
inline auto& parameter()        { static auto r = *Ops::param_qual >> Rules::identifier_like >> -(lit(":") >> type_specifier()) >> -(lit("=") >> expr_parser()); return r; }
inline auto& param_list()       { static auto r = lit("(") >> -(parameter() % ",") >> ")"; return r; }

// Declarations
inline auto& return_spec()      { static auto r = lit("->") >> type_specifier(); return r; }
inline auto& func_body()        { static auto r = (lit("=") >> expr_parser() >> ";") | block_stmt() | lit(";"); return r; }
inline auto& func_suffix()      { static auto r = param_list() >> -return_spec() >> func_body(); return r; }
inline auto& var_suffix()       { static auto r = type_specifier() >> -(lit("=") >> expr_parser()) >> ";"; return r; }
inline auto& type_body()        { static auto r = lit("=") >> "{" >> *decl_parser() >> "}"; return r; }
inline auto& type_suffix()      { static auto r = lit("type") >> -type_body(); return r; }
inline auto& ns_body()          { static auto r = -lit("=") >> "{" >> *decl_parser() >> "}"; return r; }
inline auto& ns_suffix()        { static auto r = lit("namespace") >> ns_body(); return r; }
inline auto& decl_suffix()      { static auto r = func_suffix() | type_suffix() | ns_suffix() | var_suffix(); return r; }
inline auto& unified_decl()     { static auto r = Rules::identifier_like >> ((":="_l >> expr_parser() >> ";") | (lit(":") >> decl_suffix())); return r; }
inline auto& declaration()      { static auto r = -Ops::access >> unified_decl(); return r; }
inline auto& translation_unit() { static auto r = *declaration() >> Tok::END; return r; }

// ============================================================================
// Type-Erased Parser Implementations
// ============================================================================

auto TypeSpecifierParser::parse(TokenStream input) const -> ebnf::Result<std::monostate, TokenStream> {
    begin(NodeKind::TypeSpecifier, input.pos);
    auto result = type_specifier().parse(input);
    if (result.success()) { end(result.remaining().pos); return ebnf::Result<std::monostate, TokenStream>::ok({}, result.remaining()); }
    return ebnf::Result<std::monostate, TokenStream>::fail(input);
}

auto ExpressionParser::parse(TokenStream input) const -> ebnf::Result<std::monostate, TokenStream> {
    begin(NodeKind::Expression, input.pos);
    auto result = expression().parse(input);
    if (result.success()) { end(result.remaining().pos); return ebnf::Result<std::monostate, TokenStream>::ok({}, result.remaining()); }
    return ebnf::Result<std::monostate, TokenStream>::fail(input);
}

auto StatementParser::parse(TokenStream input) const -> ebnf::Result<std::monostate, TokenStream> {
    begin(NodeKind::Statement, input.pos);
    auto result = statement().parse(input);
    if (result.success()) { end(result.remaining().pos); return ebnf::Result<std::monostate, TokenStream>::ok({}, result.remaining()); }
    return ebnf::Result<std::monostate, TokenStream>::fail(input);
}

auto DeclarationParser::parse(TokenStream input) const -> ebnf::Result<std::monostate, TokenStream> {
    begin(NodeKind::Declaration, input.pos);
    auto result = declaration().parse(input);
    if (result.success()) { end(result.remaining().pos); return ebnf::Result<std::monostate, TokenStream>::ok({}, result.remaining()); }
    return ebnf::Result<std::monostate, TokenStream>::fail(input);
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

inline ParseTree parse(std::span<const cpp2_transpiler::Token> tokens) {
    g_builder = TreeBuilder{};
    TokenStream stream{tokens};
    begin(NodeKind::TranslationUnit, 0);
    auto result = translation_unit().parse(stream);
    if (result.success()) end(result.remaining().pos);
    return g_builder.finish(tokens);
}

} // namespace cpp2::parser
