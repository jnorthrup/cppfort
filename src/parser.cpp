#include "combinators/spirit.hpp"
#include "slim_ast.hpp"
#include <string>
#include <string_view>

namespace cpp2::parser {
using namespace spirit;
using namespace cpp2::ast;

namespace { // Internal linkage

struct Tok {
  static constexpr auto ID = tok(TT::Identifier);
  static constexpr auto INT = tok(TT::IntegerLiteral);
  static constexpr auto FLOAT = tok(TT::FloatLiteral);
  static constexpr auto STR = tok(TT::StringLiteral);
  static constexpr auto CHR = tok(TT::CharacterLiteral);
  static constexpr auto END = tok(TT::EndOfFile);
};

struct Ops {
  static constexpr auto prefix =
      "+"_l | "-" | "!" | "~" | "++" | "--" | "&" | "*" | "call";
  static constexpr auto postfix_op = "++"_l | "--" | "*" | "&";
  static constexpr auto assign = "="_l | "+=" | "-=" | "*=" | "/=" | "%=" |
      "&=" | "|=" | "^=" | ">>=" | "<<=";
  static constexpr auto mul = "*"_l | "/" | "%";
  static constexpr auto add = "+"_l | "-";
  static constexpr auto shift = "<<"_l | ">>";
  static constexpr auto cmp = "<"_l | ">" | "<=" | ">=" | "<=>";
  static constexpr auto eq = "=="_l | "!=";
  static constexpr auto param_qual =
      "in"_l | "out" | "inout" | "copy" | "move" | "forward";
  static constexpr auto access = "public"_l | "private" | "protected";
};

using ParseFn =
    std::function<ebnf::Result<std::monostate, TokenStream>(TokenStream)>;

// Forward declarations
auto parse_type_specifier(TokenStream)
    -> ebnf::Result<std::monostate, TokenStream>;
auto parse_expression(TokenStream) -> ebnf::Result<std::monostate, TokenStream>;
auto parse_statement(TokenStream) -> ebnf::Result<std::monostate, TokenStream>;
auto parse_declaration(TokenStream)
    -> ebnf::Result<std::monostate, TokenStream>;

// Parser struct that calls the parse function
struct FnParser {
  ParseFn fn;
  auto parse(TokenStream input) const
      -> ebnf::Result<std::monostate, TokenStream> {
    return fn(input);
  }
};

inline auto type_spec_parser() {
  return Proto<FnParser>{{parse_type_specifier}};
}
inline auto expr_parser() { return Proto<FnParser>{{parse_expression}}; }

inline auto debug_log(const char *msg) {
  return Proto<FnParser>{
      {[=](TokenStream input) -> ebnf::Result<std::monostate, TokenStream> {
        std::cout << "DEBUG_PARSER: " << msg
                  << " token: " << (input.empty() ? "EOF" : input.peek().lexeme)
                  << "\n";
        return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
      }}};
}

inline auto stmt_parser() { return Proto<FnParser>{{parse_statement}}; }
inline auto decl_parser() { return Proto<FnParser>{{parse_declaration}}; }

// Forward declare helper used in pratt postfix call parsing.
inline Proto<FnParser> initializer_expr();

struct Rules {
  // Cpp2 keywords that can also be used as identifiers in declarations
  // (e.g., `next := ...`, `base: @struct type = { ... }`, `in(min, max)`)
  static constexpr auto identifier_like =
      Tok::ID | lit("_") | lit("this") | lit("that") | lit("$") | lit("next") |
      lit("base") | lit("in") | lit("is") | lit("as") | lit("type") |
      lit("namespace") | lit("import");
  static constexpr auto literal =
      lit("true") | "false" | Tok::INT | Tok::FLOAT | Tok::STR | Tok::CHR;
  static constexpr auto basic_type =
      (lit("auto") | "_" | Tok::ID) % with_node(NodeKind::BasicType);
};

inline auto &template_args() {
  static auto r = (lit("<") >> -(type_spec_parser() % ",") >> ">") %
                  with_binary(NodeKind::TemplateArgs);
  return r;
}
inline auto &qualified_type() {
  static auto r =
      (*(lit("const") | "*") >> Rules::basic_type >>
       *((lit("::") % with_node(NodeKind::BinaryOp)) >>
         (Tok::ID % with_node(NodeKind::Identifier)) >> -template_args()) >>
       *(lit("const") | "*" | "&")) %
      with_node(NodeKind::QualifiedType);
  return r;
}
// Parse decltype(expr) and similar C++ type constructors
inline auto &decltype_type() {
  using TT = cpp2_transpiler::TokenType;
  static auto r =
      (tok(TT::Decltype) >> "(" >> expr_parser() >> ")") |
      (tok(TT::Sizeof) >> "(" >> expr_parser() >> ")") |
      (tok(TT::Typeid) >> "(" >> expr_parser() >> ")") |
      (Tok::ID >> "(" >> expr_parser() >> ")")  // Generic fallback for other type constructors
      ;
  return r;
}

inline auto &type_specifier() {
  static auto r = (decltype_type() | qualified_type()) % with_node(NodeKind::TypeSpecifier);
  return r;
}

auto parse_inspect(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream>;

namespace pratt {

enum Prec : int {
  NONE = 0,
  COMMA = 1,
  ASSIGN = 2,
  PIPELINE = 3,
  TERNARY = 4,
  LOR = 5,
  LAND = 6,
  BOR = 7,
  BXOR = 8,
  BAND = 9,
  EQ = 10,
  CMP = 11,
  SHIFT = 12,
  ADD = 13,
  MUL = 14,
  PREFIX = 15,
  POSTFIX = 16
};

inline int get_prec(std::string_view op) {
  struct Entry {
    std::string_view op;
    int prec;
  };
  static constexpr Entry map[] = {
      {",", COMMA},   {"=", ASSIGN},  {"+=", ASSIGN}, {"-=", ASSIGN},
      {"*=", ASSIGN}, {"/=", ASSIGN}, {"%=", ASSIGN}, {"&=", ASSIGN},
      {"|=", ASSIGN}, {"^=", ASSIGN}, {">>=", ASSIGN}, {"<<=", ASSIGN},
      {"|>", PIPELINE},
      {"||", LOR},    {"&&", LAND},   {"|", BOR},     {"^", BXOR},
      {"&", BAND},    {"==", EQ},     {"!=", EQ},     {"<", CMP},
      {">", CMP},     {"<=", CMP},    {">=", CMP},    {"<=>", CMP},
      {"<<", SHIFT},  {">>", SHIFT},  {"+", ADD},     {"-", ADD},
      {"*", MUL},     {"/", MUL},     {"%", MUL},     {"..", CMP},
      {"..<", CMP},   {"..=", CMP},   {"is", POSTFIX}, {"as", POSTFIX}};
  for (const auto &e : map) {
    if (e.op == op)
      return e.prec;
  }
  return NONE;
}

inline bool is_binop(const cpp2_transpiler::Token &t) {
  return get_prec(t.lexeme) > NONE;
}

inline bool is_prefix(const cpp2_transpiler::Token &t) {
  auto l = t.lexeme;
  return l == "+" || l == "-" || l == "!" || l == "~" || l == "++" ||
         l == "--" || l == "&" || l == "*" || l == "delete" || l == "new" ||
         l == "sizeof" || l == "typeid" || l == "throw";
}

inline bool is_postfix_start(const cpp2_transpiler::Token &t) {
  auto l = t.lexeme;
  return l == "(" || l == "[" || l == "." || l == "++" || l == "--" ||
         l == "*" || l == "&";
}

auto parse_pratt(TokenStream input, int min_prec)
    -> ebnf::Result<std::monostate, TokenStream>;

auto parse_lambda(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream>;

auto parse_primary(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  if (input.empty())
    return ebnf::Result<std::monostate, TokenStream>::fail(input);

  const auto &tok = input.peek();

  // Grouped expression or unit/tuple
  if (tok.lexeme == "(") {
    begin(NodeKind::GroupedExpression, input.pos);
    input = input.next(); // consume (

    // Check for empty ()
    if (!input.empty() && input.peek().lexeme == ")") {
      input = input.next(); // consume )
      end(input.pos);
      return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
    }

    auto inner = parse_pratt(input, NONE);
    if (!inner.success())
      return ebnf::Result<std::monostate, TokenStream>::fail(input);
    input = inner.remaining();
    if (input.empty() || input.peek().lexeme != ")")
      return ebnf::Result<std::monostate, TokenStream>::fail(input);
    input = input.next(); // consume )
    end(input.pos);
    return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
  }

  // C++1 lambda: [capture](params) -> return { body }
  if (tok.lexeme == "[") {
    begin(NodeKind::LambdaExpression, input.pos);
    auto start = input.pos;
    input = input.next(); // consume [

    // Capture list: anything until ]
    while (!input.empty() && input.peek().lexeme != "]") {
      input = input.next();
    }
    if (input.empty() || input.peek().lexeme != "]")
      return ebnf::Result<std::monostate, TokenStream>::fail(input);
    input = input.next(); // consume ]

    // Optional parameters: (params)
    if (!input.empty() && input.peek().lexeme == "(") {
      input = input.next(); // consume (
      int paren_count = 1;
      while (!input.empty() && paren_count > 0) {
        if (input.peek().lexeme == "(") paren_count++;
        if (input.peek().lexeme == ")") paren_count--;
        input = input.next();
      }
      if (paren_count != 0)
        return ebnf::Result<std::monostate, TokenStream>::fail(input);
    }

    // Optional return type: -> type
    if (!input.empty() && input.peek().lexeme == "->") {
      input = input.next(); // consume ->
      // Skip tokens until we hit {
      while (!input.empty() && input.peek().lexeme != "{") {
        input = input.next();
      }
    }

    // Body: { statements } - consume as single unit (doesn't need internal parsing for transpiler)
    if (input.empty() || input.peek().lexeme != "{")
      return ebnf::Result<std::monostate, TokenStream>::fail(input);
    
    input = input.next(); // consume {
    int brace_count = 1;
    while (!input.empty() && brace_count > 0) {
      if (input.peek().lexeme == "{") brace_count++;
      if (input.peek().lexeme == "}") brace_count--;
      input = input.next();
    }

    if (brace_count != 0)
      return ebnf::Result<std::monostate, TokenStream>::fail(input);

    end(input.pos);
    return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
  }

  // Literals (all types)
  if (tok.type == TT::IntegerLiteral || tok.type == TT::FloatLiteral ||
      tok.type == TT::StringLiteral || tok.type == TT::CharacterLiteral ||
      tok.type == TT::BooleanLiteral ||
      tok.type == TT::InterpolatedRawStringLiteral) {
    begin(NodeKind::Literal, input.pos);
    input = input.next();
    end(input.pos);
    return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
  }

  if (tok.lexeme == "this" || tok.lexeme == "that" || tok.lexeme == "_" ||
      tok.lexeme == "true" || tok.lexeme == "false" || tok.type == TT::Dollar) {
    begin(NodeKind::Identifier, input.pos);
    input = input.next();
    end(input.pos);
    return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
  }

  if (tok.type == TT::Inspect) {
    return parse_inspect(input);
  }

  if (tok.type == TT::Identifier) {
    if (tok.lexeme == "inspect") {
      return parse_inspect(input);
    }

    begin(NodeKind::Identifier, input.pos);
    input = input.next();
    end(input.pos);
    return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
  }

  if (tok.lexeme == ":") {
    return parse_lambda(input);
  }

  // Leading :: (global scope) - parse as ScopeOp with empty left side
  if (tok.lexeme == "::") {
    begin(NodeKind::ScopeOp, input.pos);
    input = input.next(); // consume ::

    // Expect identifier after ::
    if (input.empty() || input.peek().type != TT::Identifier) {
      tree_restore(tree_checkpoint());
      return ebnf::Result<std::monostate, TokenStream>::fail(input);
    }

    begin(NodeKind::Identifier, input.pos);
    input = input.next();
    end(input.pos);

    end(input.pos);
    return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
  }

  return ebnf::Result<std::monostate, TokenStream>::fail(input);
}

auto parse_atom(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  auto start = input.pos;

  if (!input.empty() && is_prefix(input.peek())) {
    begin(NodeKind::PrefixExpression, input.pos);

    begin(NodeKind::PrefixOp, input.pos);
    input = input.next();
    end(input.pos);

    auto rhs = parse_atom(input);
    if (!rhs.success())
      return rhs;
    input = rhs.remaining();

    end(input.pos);
    return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
  }

  auto primary = parse_primary(input);
  if (!primary.success())
    return ebnf::Result<std::monostate, TokenStream>::fail(input);
  input = primary.remaining();

  // Combinator-based postfix parsing
  // IMPORTANT: call arguments must not treat ',' as an expression operator,
  // otherwise we consume the entire argument list as a single expression.
  static auto call = (lit("(") >> -(initializer_expr() % ",") >> ")") %
                     with_binary(NodeKind::CallOp);
  static auto subs =
      (lit("[") >> expr_parser() >> "]") % with_binary(NodeKind::SubscriptOp);
  // Member access: . for UFCS, .. for explicit member call (both produce MemberOp)
  static auto mem = ((lit("..") | lit(".")) >> (Tok::ID % with_node(NodeKind::Identifier))) %
                    with_binary(NodeKind::MemberOp);
  static auto scope =
      (lit("::") >> (Tok::ID % with_node(NodeKind::Identifier))) %
      with_binary(NodeKind::ScopeOp); // scope resolution
  static auto inc = (lit("++") | "--" | "~") % with_node(NodeKind::BinaryOp) %
                    with_binary(NodeKind::PostfixOp);

  struct PostfixPtrCheckWrapper {
    ebnf::Result<std::monostate, TokenStream> parse(TokenStream input) const {
      if (input.empty())
        return ebnf::Result<std::monostate, TokenStream>::fail(input);
      const auto &tok = input.peek();
      if (tok.lexeme != "*" && tok.lexeme != "&")
        return ebnf::Result<std::monostate, TokenStream>::fail(input);

      const auto &next = input.peek(1);
      // In Cpp2, postfix * (dereference) binds tighter than binary operators.
      // Only treat * as binary multiplication when followed by a primary
      // expression start (identifier, literal, paren). Do NOT treat +/- as
      // operand starts here - they are binary operators after postfix *.
      bool is_next_operand_start =
          next.type == TT::Identifier || next.type == TT::IntegerLiteral ||
          next.type == TT::FloatLiteral || next.type == TT::StringLiteral ||
          next.type == TT::CharacterLiteral || next.lexeme == "(" ||
          next.lexeme == "true" || next.lexeme == "false" ||
          next.lexeme == "this" || next.lexeme == "that" ||
          next.lexeme == "_";

      if (is_next_operand_start)
        return ebnf::Result<std::monostate, TokenStream>::fail(input);

      return ebnf::Result<std::monostate, TokenStream>::ok({}, input.next());
    }
  };
  static auto ptr = lift(PostfixPtrCheckWrapper{}) %
                    with_node(NodeKind::BinaryOp) %
                    with_binary(NodeKind::PostfixOp);
  static auto dollar = tok(TT::Dollar) % with_node(NodeKind::BinaryOp) %
                       with_binary(NodeKind::PostfixOp);
  static auto postfix_chain =
      *(call | subs | mem | scope | inc | ptr | dollar | template_args());

  auto postfix = postfix_chain.parse(input);
  if (postfix.success()) {
    input = postfix.remaining();
  }

  return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
}

auto parse_pratt(TokenStream input, int min_prec)
    -> ebnf::Result<std::monostate, TokenStream> {
  auto lhs = parse_atom(input);
  if (!lhs.success())
    return lhs;
  input = lhs.remaining();

  while (!input.empty()) {
    const auto &tok = input.peek();

    if (tok.lexeme == "?" && TERNARY >= min_prec) {
      // Ternary: cond ? then : else
      // Use start_infix so the existing LHS becomes the condition child.
      g_builder.start_infix(NodeKind::TernaryExpression, input.pos);
      input = input.next();
      auto then_expr = parse_pratt(input, NONE);
      if (!then_expr.success())
        return ebnf::Result<std::monostate, TokenStream>::fail(input);
      input = then_expr.remaining();
      if (input.empty() || input.peek().lexeme != ":")
        return ebnf::Result<std::monostate, TokenStream>::fail(input);
      input = input.next();
      auto else_expr = parse_pratt(input, TERNARY);
      if (!else_expr.success())
        return ebnf::Result<std::monostate, TokenStream>::fail(input);
      input = else_expr.remaining();
      end(input.pos);
      continue;
    }

    int prec = get_prec(tok.lexeme);
    if (prec == NONE || prec < min_prec)
      break;

    int next_min = (prec == ASSIGN) ? prec : prec + 1;

    NodeKind kind = NodeKind::AssignmentExpression;
    NodeKind op_kind = NodeKind::AssignmentOp;

    struct BinOpEntry {
      std::string_view op;
      NodeKind kind;
    };
    static constexpr BinOpEntry binop_entries[] = {
        {"||", NodeKind::LogicalOrExpression},
        {"&&", NodeKind::LogicalAndExpression},
        {"|", NodeKind::BitwiseOrExpression},
        {"^", NodeKind::BitwiseXorExpression},
        {"&", NodeKind::BitwiseAndExpression},
        {"==", NodeKind::EqualityExpression},
        {"!=", NodeKind::EqualityExpression},
        {"<", NodeKind::ComparisonExpression},
        {">", NodeKind::ComparisonExpression},
        {"<=", NodeKind::ComparisonExpression},
        {">=", NodeKind::ComparisonExpression},
        {"<=>", NodeKind::ComparisonExpression},
        {"<<", NodeKind::ShiftExpression},
        {">>", NodeKind::ShiftExpression},
        {"+", NodeKind::AdditiveExpression},
        {"-", NodeKind::AdditiveExpression},
        {"*", NodeKind::MultiplicativeExpression},
        {"/", NodeKind::MultiplicativeExpression},
        {"%", NodeKind::MultiplicativeExpression},
        {"|>", NodeKind::PipelineExpression},
        {"..", NodeKind::RangeExpression},
        {"..<", NodeKind::RangeExpression},
        {"..=", NodeKind::RangeExpression},
        {"is", NodeKind::IsExpression},
        {"as", NodeKind::AsExpression}};

    for (const auto &e : binop_entries) {
      if (e.op == tok.lexeme) {
        kind = e.kind;
        op_kind = NodeKind::BinaryOp;
        break;
      }
    }

    g_builder.start_infix(kind, input.pos);

    begin(op_kind, input.pos);
    input = input.next();
    end(input.pos);

    auto rhs = parse_pratt(input, next_min);
    if (!rhs.success())
      return ebnf::Result<std::monostate, TokenStream>::fail(input);
    input = rhs.remaining();
    end(input.pos);
  }

  return ebnf::Result<std::monostate, TokenStream>::ok({}, input);
}

} // namespace pratt

auto parse_expr_pratt(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  return pratt::parse_pratt(input, pratt::NONE);
}

inline auto &block_stmt() {
  static auto r =
      (lit("{") >> *stmt_parser() >> "}") % with_node(NodeKind::BlockStatement);
  return r;
}
inline auto &return_stmt() {
  static auto r = (lit("return") >> -expr_parser() >> ";") %
                  with_node(NodeKind::ReturnStatement);
  return r;
}
inline auto &break_stmt() {
  static auto r = lit("break") >> -Rules::identifier_like >> ";";
  return r;
}
inline auto &continue_stmt() {
  static auto r = lit("continue") >> -Rules::identifier_like >> ";";
  return r;
}
inline auto &next_stmt() {
  static auto r = lit("next") >> -Rules::identifier_like >> ";";
  return r;
}
inline auto &next_clause() {
  static auto r = lit("next") >> (block_stmt() | stmt_parser() | expr_parser());
  return r;
}
inline auto &throw_stmt() {
  static auto r = lit("throw") >> -expr_parser() >> ";";
  return r;
}
inline auto &assert_stmt() {
  // assert, pre, post: assert<type_safety, audit>(cond, "msg")
  // Template args are optional: <type> or <type, audit>
  static auto r = ((lit("assert") | "pre" | "post") >>
                   -((lit("<") >> -(Tok::ID % ",") >> ">") % with_binary(NodeKind::TemplateArgs)) >>
                   "(" >> expr_parser() >>
                   -(lit(",") >> expr_parser()) >> ")" >> ";") %
                  with_node(NodeKind::AssertStatement);
  return r;
}
inline auto &unchecked_stmt() {
  static auto r = (lit("unchecked") >> block_stmt()) %
                  with_node(NodeKind::UncheckedStatement);
  return r;
}

// Control flow
inline auto &if_stmt() {
  static auto r =
      (lit("if") >> expr_parser() >> (block_stmt() | stmt_parser()) >>
       -(lit("else") >> (block_stmt() | stmt_parser()))) %
      with_node(NodeKind::IfStatement);
  return r;
}
inline auto &while_stmt() {
  static auto r =
      (lit("while") >> expr_parser() >> -next_clause() >> block_stmt()) %
      with_node(NodeKind::WhileStatement);
  return r;
}
inline auto &do_while_stmt() {
  static auto r = (lit("do") >> block_stmt() >> -next_clause() >> "while" >>
                   expr_parser() >> ";") %
                  with_node(NodeKind::DoWhileStatement);
  return r;
}
inline auto &for_range_stmt() {
  // Cpp2: for range_expr next next_expr? do (var) { body }
  static auto r = (lit("for") >> expr_parser() >> -next_clause() >> "do" >> "(" >>
                   *(Ops::param_qual % with_node(NodeKind::ParamQualifier)) >>
                   (Rules::identifier_like % with_node(NodeKind::Identifier)) >>
                   ")" >> (block_stmt() | stmt_parser())) %
                  with_node(NodeKind::ForStatement);
  return r;
}
inline auto &for_cpp1_range_stmt() {
  static auto r = (lit("for") >> "(" >>
                   (Rules::identifier_like % with_node(NodeKind::Identifier)) >>
                   ":" >> expr_parser() >> ")" >> block_stmt()) %
                  with_node(NodeKind::ForStatement);
  return r;
}
inline auto &catch_clause() {
  static auto r = lit("catch") >> "(" >> *Ops::param_qual >>
                  Rules::identifier_like >> -(lit(":") >> type_specifier()) >>
                  ")" >> block_stmt();
  return r;
}
inline auto &try_stmt() {
  static auto r = lit("try") >> block_stmt() >> *catch_clause();
  return r;
}

// Helper for expressions that shouldn't consume assignment (for patterns)
auto parse_expr_pratt_no_assign(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  return pratt::parse_pratt(input, pratt::PIPELINE);
}
auto parse_expression_no_assign(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  auto cp = tree_checkpoint();
  begin(NodeKind::Expression, input.pos);
  auto result = parse_expr_pratt_no_assign(input);
  if (result.success() && result.remaining().pos > input.pos) {
    end(result.remaining().pos);
    return ebnf::Result<std::monostate, TokenStream>::ok({},
                                                         result.remaining());
  }
  tree_restore(cp);
  return ebnf::Result<std::monostate, TokenStream>::fail(input);
}
inline auto expr_parser_no_assign() {
  return Proto<FnParser>{{parse_expression_no_assign}};
}

// Switch / inspect
inline auto &is_pattern() {
  static auto r = (lit("is") >> (type_specifier() | expr_parser_no_assign())) %
                  with_node(NodeKind::IsPattern);
  return r;
}
inline auto &as_pattern() {
  static auto r =
      (lit("as") >> type_specifier() >> "=" >> Rules::identifier_like) %
      with_node(NodeKind::AsPattern);
  return r;
}
inline auto &pattern() {
  static auto r =
      (is_pattern() | as_pattern() | (lit("_") | expr_parser_no_assign())) %
      with_node(NodeKind::Pattern);
  return r;
}
inline auto &inspect_arm() {
  static auto r =
      (pattern() >> "=" >> (block_stmt() | (expr_parser() >> ";"))) %
      with_node(NodeKind::InspectArm);
  return r;
}
inline auto &inspect_expr() {
  static auto r =
      (lit("inspect") >> expr_parser() >> -(lit("->") >> type_specifier()) >>
       "{" >> *inspect_arm() >> "}") %
      with_node(NodeKind::InspectExpression);
  return r;
}

// Forward declared helper for parse_primary to call
auto parse_inspect(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  return inspect_expr().parse(input);
}

// Local variable declaration (inside block)
inline auto &local_var_decl() {
  static auto r =
      (Rules::identifier_like >> ((":="_l >> expr_parser() >> ";") |
                                  (lit(":") >> type_specifier() >>
                                   -(lit("=") >> expr_parser()) >> ";"))) %
      with_node(NodeKind::UnifiedDeclaration);
  return r;
}

// Expression statement
inline auto &expr_stmt() {
  static auto r =
      (expr_parser() >> ";") % with_node(NodeKind::ExpressionStatement);
  return r;
}
inline auto &labeled_stmt() {
  static auto r = (Rules::identifier_like >> ":" >> stmt_parser());
  return r;
}

auto parse_initializer_expr(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  // Use ASSIGN precedence to stop before comma
  return pratt::parse_pratt(input, pratt::ASSIGN);
}
inline Proto<FnParser> initializer_expr() {
  return Proto<FnParser>{{parse_initializer_expr}};
}

inline auto &parameter() {
  static auto r =
      (*(Ops::param_qual % with_node(NodeKind::ParamQualifier)) >>
       (Rules::identifier_like % with_node(NodeKind::Identifier)) >>
       -(lit(":") >> type_specifier()) >> -(lit("=") >> initializer_expr())) %
      with_node(NodeKind::Parameter);
  return r;
}
inline auto &param_list() {
  static auto r = (lit("(") >> -(parameter() % "," >> -lit(",")) >> lit(")")) %
                  with_node(NodeKind::ParamList);
  return r;
}

inline auto &cpp1_parameter() {
  static auto r =
      (type_specifier() >>
       (Rules::identifier_like % with_node(NodeKind::Identifier)) >>
       -(lit("=") >> initializer_expr())) %
      with_node(NodeKind::Parameter);
  return r;
}

inline auto &cpp1_param_list() {
  static auto r = (lit("(") >> -(cpp1_parameter() % "," >> -lit(",")) >> lit(")")) %
                  with_node(NodeKind::ParamList);
  return r;
}

inline auto &scope_stmt() {
  static auto r =
      (param_list() >> block_stmt()) % with_node(NodeKind::ScopeStatement);
  return r;
}

inline auto &statement() {
  // local_var_decl MUST come before labeled_stmt to prevent "x: int = 42;"
  // from being parsed as label "x:" with body "int = 42;"
  static auto r =
      (local_var_decl() | labeled_stmt() | block_stmt() | unchecked_stmt() |
       scope_stmt() | if_stmt() | while_stmt() | do_while_stmt() |
       for_range_stmt() | for_cpp1_range_stmt() | try_stmt() | return_stmt() |
       break_stmt() | continue_stmt() | next_stmt() | throw_stmt() |
       assert_stmt() | expr_stmt() | lit(";")) %
      with_node(NodeKind::Statement);
  return r;
}

inline auto &return_spec() {
  // Support both "-> type" and "-> (named:type, ...)" for tuple returns
  static auto r = (lit("->") >> (param_list() | type_specifier())) %
                  with_node(NodeKind::ReturnSpec);
  return r;
}
inline auto &func_body() {
  static auto r =
      ((lit("=") >> (block_stmt() | (expr_parser() >> ";"))) | lit(";")) %
      with_node(NodeKind::FunctionBody);
  return r;
}

inline auto &contract_clause() {
  static auto r =
      ((lit("pre") | "post" | "assert") >> "(" >> expr_parser() >> ")") %
      with_node(NodeKind::ContractClause);
  return r;
}
inline auto &requires_clause() {
  static auto r = (lit("requires") >> expr_parser_no_assign()) %
                  with_node(NodeKind::RequiresClause);
  return r;
}
inline auto &func_suffix() {
  static auto r = (param_list() >> -return_spec() >> *contract_clause() >>
                   -requires_clause() >> func_body()) %
                  with_node(NodeKind::FunctionSuffix);
  return r;
}
inline auto &var_suffix() {
  static auto r =
      ((type_specifier() >> -((lit("=") | "==") >> expr_parser()) >> ";") |
       ((lit("=") | "==") >> expr_parser() >> ";")) %
      with_node(NodeKind::VariableSuffix);
  return r;
}
inline auto &type_body() {
  static auto r = (lit("=") >> "{" >> *decl_parser() >> "}") %
                  with_node(NodeKind::TypeBody);
  return r;
}
inline auto &metafunction() {
  // @name or @name<args>
  using TT = cpp2_transpiler::TokenType;
  static auto r =
      ((lit("@") >> (Tok::ID | tok(TT::Enum) | tok(TT::Struct) | tok(TT::Class) |
                    tok(TT::Interface) | tok(TT::Union)) >>
       -template_args()) % with_node(NodeKind::Metafunction)) |
      tok(TT::Final) | tok(TT::Virtual) | tok(TT::Override);
  return r;
}
inline auto &type_suffix() {
  static auto r = (*metafunction() >> lit("type") >> -type_body()) %
                  with_node(NodeKind::TypeSuffix);
  return r;
}
inline auto &ns_body() {
  static auto r = (-lit("=") >> "{" >> *decl_parser() >> "}") %
                  with_node(NodeKind::NamespaceBody);
  return r;
}
inline auto &ns_alias() {
  // namespace == QualifiedName ;
  static auto r = (lit("namespace") >> "==" >> type_specifier() >> ";") %
                  with_node(NodeKind::NamespaceSuffix);
  return r;
}
inline auto &ns_suffix() {
  static auto r =
      (lit("namespace") >> (("==" >> type_specifier() >> ";") | ns_body())) %
      with_node(NodeKind::NamespaceSuffix);
  return r;
}

inline auto &template_param() {
  // Template parameter with optional constraint, variadic, and default:
  // T, T: type, T..., T: type..., T: _ = default, Ts...: type
  static auto r = Rules::identifier_like >> -(lit("...")) >>
                  -(lit(":") >> type_specifier()) >> -(lit("...")) >>
                  -(lit("=") >> type_specifier());
  return r;
}
inline auto &template_params() {
  static auto r = (lit("<") >> -(template_param() % ",") >> ">") %
                  with_node(NodeKind::TemplateArgs);
  return r;
}

inline auto &alias_suffix() {
  static auto r = (lit("type") >> "==" >> type_specifier() >> ";") %
                  with_node(NodeKind::TypeAliasSuffix);
  return r;
}

inline auto &operator_name() {
  static auto r = lit("=") | "[]" | "()" | "++" | "--" | "->" | "<=>" | "+" |
                  "-" | "*" | "/" | "%" | "==" | "!=" | "<" | ">" | "<=" | ">=";
  return r;
}
inline auto &operator_suffix() {
  // Note: "=:" is lexed as a single EqualColon token, so handle it specially
  static auto r = lit("operator") >>
                  (lit("=:") | (operator_name() >> lit(":"))) >>
                  -template_params() >> func_suffix();
  return r;
}

inline auto &concept_suffix() {
  // concept = expression ;
  static auto r = lit("concept") >> "=" >> expr_parser() >> ";";
  return r;
}

inline auto &decl_suffix() {
  static auto r =
      (template_params() >>
       (concept_suffix() | alias_suffix() | func_suffix() | type_suffix() | var_suffix())) |
      alias_suffix() | operator_suffix() | func_suffix() | type_suffix() |
      ns_suffix() | var_suffix();
  return r;
}
inline auto &operator_decl() {
  // operator=: (params) = body — operator as declaration name
  // Note: "=:" is lexed as a single EqualColon token, so handle it specially
  static auto r = (lit("operator") >>
                   (lit("=:") | (operator_name() >> lit(":"))) >>
                   -template_params() >> func_suffix()) %
                  with_node(NodeKind::UnifiedDeclaration);
  return r;
}
inline auto &unified_decl() {
  static auto r = (Rules::identifier_like >> ((":="_l >> expr_parser() >> ";") |
                                              (lit(":") >> decl_suffix()))) %
                  with_node(NodeKind::UnifiedDeclaration);
  return r;
}
inline auto &cpp1_function_decl() {
  // C++1: auto name(params) -> return_type { body }
  static auto r = (lit("auto") >>
                   (Rules::identifier_like % with_node(NodeKind::Identifier)) >>
                   (cpp1_param_list() >> -return_spec() >>
                    ((lit("=") >> (block_stmt() | (expr_parser() >> ";"))) |
                     block_stmt() | (expr_parser() >> ";")) %
                        with_node(NodeKind::FunctionBody)) %
                       with_node(NodeKind::FunctionSuffix)) %
                  with_node(NodeKind::UnifiedDeclaration);
  return r;
}
inline auto &declaration() {
  // Note: operator_decl() is defined but not yet used here — the emitter
  // doesn't properly differentiate (out this) constructors from (inout this)
  // assignment operators yet. Once that's fixed, add operator_decl() back
  // as the first alternative here.
  static auto r =
      (-Ops::access >>
       (unified_decl() | cpp1_function_decl())) %
      with_node(NodeKind::Declaration);
  return r;
}
inline auto &preprocessor_directive() {
  static auto r = tok(TT::Hash) % with_node(NodeKind::Preprocessor);
  return r;
}

inline auto &translation_unit() {
  static auto r = *(preprocessor_directive() | declaration()) >> Tok::END;
  return r;
}

template <typename R>
bool made_progress(const TokenStream &before, const R &result) {
  return result.remaining().pos > before.pos;
}

auto parse_type_specifier(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  return type_specifier().parse(input);
}

auto parse_expression(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  auto cp = tree_checkpoint();
  begin(NodeKind::Expression, input.pos);
  auto result = parse_expr_pratt(input);
  if (result.success() && made_progress(input, result)) {
    end(result.remaining().pos);
    return ebnf::Result<std::monostate, TokenStream>::ok({},
                                                         result.remaining());
  }
  tree_restore(cp);
  return ebnf::Result<std::monostate, TokenStream>::fail(input);
}

auto parse_statement(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  return statement().parse(input);
}

auto parse_declaration(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  return declaration().parse(input);
}
namespace pratt {
auto parse_lambda(TokenStream input)
    -> ebnf::Result<std::monostate, TokenStream> {
  // Lambda starts with :
  if (input.empty() || input.peek().lexeme != ":")
    return ebnf::Result<std::monostate, TokenStream>::fail(input);

  auto cp = tree_checkpoint();
  auto start = input.pos;

  begin(NodeKind::LambdaExpression, start);

  input = input.next(); // consume :

  auto suffix = func_suffix().parse(input);
  if (!suffix.success()) {
    tree_restore(cp);
    return ebnf::Result<std::monostate, TokenStream>::fail(input);
  }

  end(suffix.remaining().pos);
  return ebnf::Result<std::monostate, TokenStream>::ok({}, suffix.remaining());
}
} // namespace pratt

} // anonymous namespace

ParseTree parse(std::span<const cpp2_transpiler::Token> tokens) {
  g_builder = TreeBuilder{};
  TokenStream stream{tokens};
  begin(NodeKind::TranslationUnit, 0);
  auto result = translation_unit().parse(stream);
  if (result.success())
    end(result.remaining().pos);
  return g_builder.finish(tokens);
}

} // namespace cpp2::parser
