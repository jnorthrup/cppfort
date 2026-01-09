// ============================================================================
// Cpp2 Parser - Direct EBNF Translation
// ============================================================================
// Source: grammar/cpp2.ebnf → spirit combinators
// Uses Lazy<> for forward-declared recursive rules

#include "parser.hpp"
#include "combinators/ebnf.hpp"
#include "combinators/spirit.hpp"
#include "ast.hpp"

namespace cpp2_transpiler {

using namespace cpp2::parser::spirit;
namespace ebnf = cpp2::combinators::ebnf;
using TT = TokenType;
using TS = TokenStream;

// ============================================================================
// Forward-declared rules (Lazy for recursion)
// ============================================================================

static ebnf::Lazy<Token, TS> expression;
static ebnf::Lazy<Token, TS> statement;
static ebnf::Lazy<Token, TS> type_specifier;
static ebnf::Lazy<Token, TS> declaration;
static ebnf::Lazy<Token, TS> block_statement;

// ============================================================================
// LEXICAL - Keywords  
// ============================================================================

inline const auto KW_AS = token(TT::As);
inline const auto KW_ASSERT = token(TT::Assert);
inline const auto KW_AUTO = token(TT::Auto);
inline const auto KW_AWAIT = token(TT::Await);
inline const auto KW_BREAK = token(TT::Break);
inline const auto KW_CASE = token(TT::Case);
inline const auto KW_CATCH = token(TT::Catch);
inline const auto KW_CONCEPT = token(TT::Concept);
inline const auto KW_CONST = token(TT::Const);
inline const auto KW_CONTINUE = token(TT::Continue);
inline const auto KW_COPY = token(TT::Copy);
inline const auto KW_DEFAULT = token(TT::Default);
inline const auto KW_DO = token(TT::Do);
inline const auto KW_ELSE = token(TT::Else);
inline const auto KW_FALSE = token(TT::False);
inline const auto KW_FINAL = token(TT::Final);
inline const auto KW_FOR = token(TT::For);
inline const auto KW_FORWARD = token(TT::Forward);
inline const auto KW_FUNC = token(TT::Func);
inline const auto KW_IF = token(TT::If);
inline const auto KW_IMPLICIT = token(TT::Implicit);
inline const auto KW_IMPORT = token(TT::Import);
inline const auto KW_IN = token(TT::In);
inline const auto KW_INOUT = token(TT::Inout);
inline const auto KW_INSPECT = token(TT::Inspect);
inline const auto KW_IS = token(TT::Is);
inline const auto KW_LET = token(TT::Let);
inline const auto KW_MOVE = token(TT::Move);
inline const auto KW_NAMESPACE = token(TT::Namespace);
inline const auto KW_NOEXCEPT = token(TT::Noexcept);
inline const auto KW_OPERATOR = token(TT::Operator);
inline const auto KW_OUT = token(TT::Out);
inline const auto KW_OVERRIDE = token(TT::Override);
inline const auto KW_POST = token(TT::Post);
inline const auto KW_PRE = token(TT::Pre);
inline const auto KW_PRIVATE = token(TT::Private);
inline const auto KW_PROTECTED = token(TT::Protected);
inline const auto KW_PUBLIC = token(TT::Public);
inline const auto KW_REQUIRES = token(TT::Requires);
inline const auto KW_RETURN = token(TT::Return);
inline const auto KW_SWITCH = token(TT::Switch);
inline const auto KW_TEMPLATE = token(TT::Template);
inline const auto KW_THAT = token(TT::That);
inline const auto KW_THIS = token(TT::This);
inline const auto KW_THROW = token(TT::Throw);
inline const auto KW_THROWS = token(TT::Throws);
inline const auto KW_TRUE = token(TT::True);
inline const auto KW_TRY = token(TT::Try);
inline const auto KW_TYPE = token(TT::Type);
inline const auto KW_USING = token(TT::Using);
inline const auto KW_VIRTUAL = token(TT::Virtual);
inline const auto KW_WHILE = token(TT::While);

// ============================================================================
// LEXICAL - Punctuation
// ============================================================================

inline const auto LPAREN = token(TT::LeftParen);
inline const auto RPAREN = token(TT::RightParen);
inline const auto LBRACE = token(TT::LeftBrace);
inline const auto RBRACE = token(TT::RightBrace);
inline const auto LBRACKET = token(TT::LeftBracket);
inline const auto RBRACKET = token(TT::RightBracket);
inline const auto LT = token(TT::LessThan);
inline const auto GT = token(TT::GreaterThan);
inline const auto COLON = token(TT::Colon);
inline const auto SEMICOLON = token(TT::Semicolon);
inline const auto COMMA = token(TT::Comma);
inline const auto DOT = token(TT::Dot);
inline const auto ARROW = token(TT::Arrow);
inline const auto EQUAL = token(TT::Equal);
inline const auto DOUBLE_EQUAL = token(TT::DoubleEqual);
inline const auto COLON_EQUAL = token(TT::ColonEqual);
inline const auto DOUBLE_COLON = token(TT::DoubleColon);
inline const auto AT = token(TT::At);
inline const auto ELLIPSIS = token(TT::Ellipsis);
inline const auto UNDERSCORE = token(TT::Underscore);
inline const auto PLUS = token(TT::Plus);
inline const auto MINUS = token(TT::Minus);
inline const auto STAR = token(TT::Asterisk);
inline const auto SLASH = token(TT::Slash);
inline const auto PERCENT = token(TT::Percent);
inline const auto AMP = token(TT::Ampersand);
inline const auto PIPE = token(TT::Pipe);
inline const auto CARET = token(TT::Caret);
inline const auto TILDE = token(TT::Tilde);
inline const auto BANG = token(TT::Bang);
inline const auto QUESTION = token(TT::Question);
inline const auto DOUBLE_AMP = token(TT::DoubleAmp);
inline const auto DOUBLE_PIPE = token(TT::DoublePipe);

// ============================================================================
// LEXICAL - Identifiers & Literals
// ============================================================================

inline const auto IDENTIFIER = token(TT::Identifier);
inline const auto INTEGER_LIT = token(TT::IntegerLiteral);
inline const auto FLOAT_LIT = token(TT::FloatLiteral);
inline const auto STRING_LIT = token(TT::StringLiteral);
inline const auto CHAR_LIT = token(TT::CharacterLiteral);
inline const auto END_OF_FILE = token(TT::EndOfFile);

// ============================================================================
// TOP LEVEL (line 54-69)
// ============================================================================

// access_specifier = "public" | "private" | "protected"
inline const auto access_specifier = KW_PUBLIC | KW_PRIVATE | KW_PROTECTED;

// ============================================================================
// UNIFIED DECLARATION SYNTAX (line 73-95)
// ============================================================================

// identifier_like = IDENTIFIER | CONTEXTUAL_KEYWORD | "_"
inline const auto identifier_like = IDENTIFIER | KW_IN | KW_OUT | KW_INOUT | KW_COPY | 
                                     KW_MOVE | KW_FORWARD | KW_FUNC | KW_TYPE | 
                                     KW_NAMESPACE | UNDERSCORE;

// ============================================================================  
// TEMPLATE PARAMETERS (line 120-143)
// ============================================================================

// template_param_name = IDENTIFIER | "_"
inline const auto template_param_name = IDENTIFIER | UNDERSCORE;

// type_constraint = type_specifier | "_" | "type"
inline const auto type_constraint = type_specifier | UNDERSCORE | KW_TYPE;

// template_param = template_param_name [ "..." ] [ ":" type_constraint ] [ "=" default_value ]
inline const auto template_param = template_param_name >> -ELLIPSIS >> -(COLON >> type_constraint) >> -(EQUAL >> expression);

// template_param_list = template_param { "," template_param } [ "," ]
inline const auto template_param_list = template_param >> *(COMMA >> template_param) >> -COMMA;

// template_params = "<" [ template_param_list ] ">"
inline const auto template_params = LT >> -template_param_list >> GT;

// template_args = "<" [ template_arg_list ] ">"
inline const auto template_args = LT >> -(expression >> *(COMMA >> expression)) >> GT;

// ============================================================================
// FUNCTIONS (line 144-191)
// ============================================================================

// param_qualifier = "in" | "copy" | "inout" | "out" | "move" | "forward" | "virtual" | "override" | "implicit"
inline const auto param_qualifier = KW_IN | KW_COPY | KW_INOUT | KW_OUT | KW_MOVE | 
                                     KW_FORWARD | KW_VIRTUAL | KW_OVERRIDE | KW_IMPLICIT;

// param_name = IDENTIFIER | "this" | "that" | "_" | CONTEXTUAL_KEYWORD
inline const auto param_name = IDENTIFIER | KW_THIS | KW_THAT | UNDERSCORE;

// parameter = { param_qualifier } param_name [ "..." ] [ ":" type_specifier ] [ "=" expression ]
inline const auto parameter = *param_qualifier >> param_name >> -ELLIPSIS >> -(COLON >> type_specifier) >> -(EQUAL >> expression);

// param_list = "(" [ parameter_list ] ")"
inline const auto param_list = LPAREN >> -(parameter >> *(COMMA >> parameter) >> -COMMA) >> RPAREN;

// throws_spec = "throws" | "noexcept"
inline const auto throws_spec = KW_THROWS | KW_NOEXCEPT;

// return_modifier = "forward" | "move"
inline const auto return_modifier = KW_FORWARD | KW_MOVE;

// return_type = [ return_modifier ] type_specifier | "(" named_return_list ")"
inline const auto return_type = (-return_modifier >> type_specifier) | (LPAREN >> (IDENTIFIER >> COLON >> type_specifier >> *(COMMA >> IDENTIFIER >> COLON >> type_specifier)) >> RPAREN);

// return_spec = "->" return_type | "=:" type_specifier
inline const auto return_spec = (ARROW >> return_type) | (token(TT::ColonEqual) >> type_specifier);

// requires_clause = "requires" requires_expression
inline const auto requires_clause = KW_REQUIRES >> expression;

// contract_clause = ( "pre" | "post" | "assert" ) [ "<" IDENTIFIER ">" ] ( ":" expression | "(" expression [ "," STRING_LITERAL ] ")" )
inline const auto contract_clause = (KW_PRE | KW_POST | KW_ASSERT) >> -(LT >> IDENTIFIER >> GT) >> 
                                     ((COLON >> expression) | (LPAREN >> expression >> -(COMMA >> STRING_LIT) >> RPAREN));

// contracts = { contract_clause }
inline const auto contracts = *contract_clause;

// function_body = "=" expression ";" | "==" expression ";" | "=" block_statement | block_statement | ";"
inline const auto function_body = (EQUAL >> expression >> SEMICOLON) | 
                                   (DOUBLE_EQUAL >> expression >> SEMICOLON) | 
                                   (EQUAL >> block_statement) | 
                                   block_statement | 
                                   SEMICOLON;

// func_signature = [ template_params ] param_list [ throws_spec ] [ return_spec ] [ requires_clause ] [ contracts ]
inline const auto func_signature = -template_params >> param_list >> -throws_spec >> -return_spec >> -requires_clause >> contracts;

// ============================================================================
// TYPES (line 193-208)
// ============================================================================

// metafunction = "@" IDENTIFIER [ template_args ]
inline const auto metafunction = AT >> IDENTIFIER >> -template_args;

// type_body = "=" "{" { type_member } "}" [ ";" ]
inline const auto type_body = EQUAL >> LBRACE >> *declaration >> RBRACE >> -SEMICOLON;

// type_alias = "==" type_specifier ";" | "=" type_specifier ";"
inline const auto type_alias = (DOUBLE_EQUAL >> type_specifier >> SEMICOLON) | (EQUAL >> type_specifier >> SEMICOLON);

// type_body_or_alias = type_alias | type_body
inline const auto type_body_or_alias = type_alias | type_body;

// ============================================================================
// NAMESPACES (line 229-235)
// ============================================================================

// qualified_name = IDENTIFIER { "::" IDENTIFIER }
inline const auto qualified_name = IDENTIFIER >> *(DOUBLE_COLON >> IDENTIFIER);

// namespace_alias = "==" qualified_name ";"
inline const auto namespace_alias = DOUBLE_EQUAL >> qualified_name >> SEMICOLON;

// namespace_body = [ "=" ] "{" { declaration } "}"
inline const auto namespace_body = -EQUAL >> LBRACE >> *declaration >> RBRACE;

// ============================================================================
// USING & IMPORT (line 237-249)
// ============================================================================

// using_alias = IDENTIFIER "=" qualified_name ";"
inline const auto using_alias = IDENTIFIER >> EQUAL >> qualified_name >> SEMICOLON;

// using_path = qualified_name ";"
inline const auto using_path = qualified_name >> SEMICOLON;

// using_namespace = "namespace" IDENTIFIER ";"
inline const auto using_namespace = KW_NAMESPACE >> IDENTIFIER >> SEMICOLON;

// using_declaration = "using" ( using_alias | using_path | using_namespace )
inline const auto using_declaration = KW_USING >> (using_alias | using_path | using_namespace);

// import_declaration = "import" IDENTIFIER ";"
inline const auto import_declaration = KW_IMPORT >> IDENTIFIER >> SEMICOLON;

// ============================================================================
// KEYWORD DECLARATIONS (line 98-118)
// ============================================================================

// let_declaration = ( "let" | "const" ) IDENTIFIER [ ":" type_specifier ] initializer ";"
inline const auto let_declaration = (KW_LET | KW_CONST) >> IDENTIFIER >> -(COLON >> type_specifier) >> (EQUAL | DOUBLE_EQUAL) >> expression >> SEMICOLON;

// func_declaration = "func" IDENTIFIER [ ":" ] func_signature function_body
inline const auto func_declaration = KW_FUNC >> IDENTIFIER >> -COLON >> func_signature >> function_body;

// type_declaration_kw = "type" IDENTIFIER [ template_params ] [ requires_clause ] type_body
inline const auto type_declaration_kw = KW_TYPE >> IDENTIFIER >> -template_params >> -requires_clause >> type_body;

// namespace_declaration_kw = "namespace" IDENTIFIER ( namespace_alias | namespace_body )
inline const auto namespace_declaration_kw = KW_NAMESPACE >> IDENTIFIER >> (namespace_alias | namespace_body);

// ============================================================================
// UNIFIED SUFFIX (line 82-95)
// ============================================================================

// function_suffix = func_signature function_body
inline const auto function_suffix = func_signature >> function_body;

// type_suffix = { metafunction } ( "type" | "concept" ) [ template_params ] [ requires_clause ] type_body_or_alias
inline const auto type_suffix = *metafunction >> (KW_TYPE | KW_CONCEPT) >> -template_params >> -requires_clause >> type_body_or_alias;

// namespace_suffix = "namespace" ( namespace_alias | namespace_body )
inline const auto namespace_suffix = KW_NAMESPACE >> (namespace_alias | namespace_body);

// variable_suffix = [ template_params ] type_specifier [ initializer ] ";"
inline const auto variable_suffix = -template_params >> type_specifier >> -((EQUAL | DOUBLE_EQUAL) >> expression) >> SEMICOLON;

// declaration_suffix = function_suffix | type_suffix | namespace_suffix | variable_suffix
inline const auto declaration_suffix = function_suffix | type_suffix | namespace_suffix | variable_suffix;

// unified_declaration = identifier_like ":" declaration_suffix | identifier_like ":=" expression ";"
inline const auto unified_declaration = (identifier_like >> COLON >> declaration_suffix) | 
                                         (identifier_like >> COLON_EQUAL >> expression >> SEMICOLON);

// ============================================================================
// STATEMENTS (line 251-273)
// ============================================================================

// return_statement = "return" [ expression ] ";"
inline const auto return_statement = KW_RETURN >> -expression >> SEMICOLON;

// break_statement = "break" [ IDENTIFIER ] ";"
inline const auto break_statement = KW_BREAK >> -IDENTIFIER >> SEMICOLON;

// continue_statement = "continue" [ IDENTIFIER ] ";"
inline const auto continue_statement = KW_CONTINUE >> -IDENTIFIER >> SEMICOLON;

// throw_statement = "throw" [ expression ] ";"
inline const auto throw_statement = KW_THROW >> -expression >> SEMICOLON;

// catch_clause = "catch" "(" [ catch_param ] ")" block_statement
inline const auto catch_clause = KW_CATCH >> LPAREN >> -(type_specifier >> -IDENTIFIER) >> RPAREN >> block_statement;

// try_statement = "try" block_statement { catch_clause }
inline const auto try_statement = KW_TRY >> block_statement >> *catch_clause;

// if_statement = "if" expression block_statement [ "else" ( block_statement | if_statement ) ]
inline const auto if_statement = KW_IF >> expression >> block_statement >> -(KW_ELSE >> (block_statement | statement));

// while_statement = "while" expression block_statement
inline const auto while_statement = KW_WHILE >> expression >> block_statement;

// do_while_statement = "do" block_statement "while" expression ";"
inline const auto do_while_statement = KW_DO >> block_statement >> KW_WHILE >> expression >> SEMICOLON;

// for_statement = "for" expression "do" "(" parameter ")" block_statement
inline const auto for_statement = KW_FOR >> expression >> KW_DO >> LPAREN >> parameter >> RPAREN >> block_statement;

// switch_case = "case" expression ":" statement | "default" ":" statement
inline const auto switch_case = (KW_CASE >> expression >> COLON >> statement) | (KW_DEFAULT >> COLON >> statement);

// switch_statement = "switch" expression "{" { switch_case } "}"
inline const auto switch_statement = KW_SWITCH >> expression >> LBRACE >> *switch_case >> RBRACE;

// contract_statement = ( "assert" | "pre" | "post" ) expression ";"
inline const auto contract_statement = (KW_ASSERT | KW_PRE | KW_POST) >> expression >> SEMICOLON;

// expression_statement = expression ";"
inline const auto expression_statement = expression >> SEMICOLON;

// local_declaration = IDENTIFIER ":" type_specifier [ initializer ] ";" | IDENTIFIER ":=" expression ";"
inline const auto local_declaration = (IDENTIFIER >> COLON >> type_specifier >> -((EQUAL | DOUBLE_EQUAL) >> expression) >> SEMICOLON) |
                                       (IDENTIFIER >> COLON_EQUAL >> expression >> SEMICOLON);

// ============================================================================
// EXPRESSIONS (line 371-487) - Precedence hierarchy
// ============================================================================

// literal = "true" | "false" | INTEGER_LITERAL | FLOAT_LITERAL | STRING_LITERAL | CHAR_LITERAL
inline const auto literal = KW_TRUE | KW_FALSE | INTEGER_LIT | FLOAT_LIT | STRING_LIT | CHAR_LIT;

// primary_expression
inline const auto primary_expression = literal | 
                                        (DOUBLE_COLON >> qualified_name) |
                                        qualified_name |
                                        KW_THIS | KW_THAT | UNDERSCORE |
                                        (LPAREN >> expression >> RPAREN) |
                                        (LBRACKET >> -(expression >> *(COMMA >> expression)) >> RBRACKET);

// postfix operators  
inline const auto call_op = LPAREN >> -(expression >> *(COMMA >> expression)) >> RPAREN;
inline const auto member_op = DOT >> IDENTIFIER;
inline const auto subscript_op = LBRACKET >> expression >> RBRACKET;
inline const auto postfix_op = call_op | member_op | subscript_op | token(TT::PlusPlus) | token(TT::MinusMinus);

// postfix_expression = primary_expression { postfix_op }
inline const auto postfix_expression = primary_expression >> *postfix_op;

// prefix operators
inline const auto prefix_op = PLUS | MINUS | BANG | TILDE | token(TT::PlusPlus) | token(TT::MinusMinus) | AMP | STAR;

// prefix_expression = { prefix_op } postfix_expression
inline const auto prefix_expression = *prefix_op >> postfix_expression;

// multiplicative_expression = prefix_expression { ( "*" | "/" | "%" ) prefix_expression }
inline const auto multiplicative_expression = prefix_expression >> *((STAR | SLASH | PERCENT) >> prefix_expression);

// additive_expression = multiplicative_expression { ( "+" | "-" ) multiplicative_expression }
inline const auto additive_expression = multiplicative_expression >> *((PLUS | MINUS) >> multiplicative_expression);

// comparison_expression = additive_expression { ( "<" | ">" | "<=" | ">=" ) additive_expression }
inline const auto comparison_expression = additive_expression >> *((LT | GT | token(TT::LessEqual) | token(TT::GreaterEqual)) >> additive_expression);

// equality_expression = comparison_expression { ( "==" | "!=" ) comparison_expression }
inline const auto equality_expression = comparison_expression >> *((DOUBLE_EQUAL | token(TT::NotEqual)) >> comparison_expression);

// bitwise_and_expression = equality_expression { "&" equality_expression }
inline const auto bitwise_and_expression = equality_expression >> *(AMP >> equality_expression);

// bitwise_xor_expression = bitwise_and_expression { "^" bitwise_and_expression }
inline const auto bitwise_xor_expression = bitwise_and_expression >> *(CARET >> bitwise_and_expression);

// bitwise_or_expression = bitwise_xor_expression { "|" bitwise_xor_expression }
inline const auto bitwise_or_expression = bitwise_xor_expression >> *(PIPE >> bitwise_xor_expression);

// logical_and_expression = bitwise_or_expression { "&&" bitwise_or_expression }
inline const auto logical_and_expression = bitwise_or_expression >> *(DOUBLE_AMP >> bitwise_or_expression);

// logical_or_expression = logical_and_expression { "||" logical_and_expression }
inline const auto logical_or_expression = logical_and_expression >> *(DOUBLE_PIPE >> logical_and_expression);

// ternary_expression = logical_or_expression [ "?" expression ":" ternary_expression ]
inline const auto ternary_expression = logical_or_expression >> -(QUESTION >> expression >> COLON >> logical_or_expression);

// assignment_op = "=" | "+=" | "-=" | "*=" | "/=" | "%="
inline const auto assignment_op = EQUAL | token(TT::PlusEqual) | token(TT::MinusEqual) | token(TT::StarEqual) | token(TT::SlashEqual);

// assignment_expression = ternary_expression [ assignment_op assignment_expression ]
inline const auto assignment_expression = ternary_expression >> -(assignment_op >> expression);

// ============================================================================
// TYPE SPECIFIERS (line 586-613)
// ============================================================================

// basic_type = IDENTIFIER | "auto" | "_" | "type"
inline const auto basic_type = IDENTIFIER | KW_AUTO | UNDERSCORE | KW_TYPE;

// qualified_type = basic_type [ template_args ] { "::" IDENTIFIER [ template_args ] } { "*" | "&" }
inline const auto qualified_type = basic_type >> -template_args >> *(DOUBLE_COLON >> IDENTIFIER >> -template_args) >> *(STAR | AMP);

// type_spec_impl = qualified_type | "const" type_specifier | "*" type_specifier
inline const auto type_spec_impl = qualified_type | (KW_CONST >> type_specifier) | (STAR >> -KW_CONST >> type_specifier);

// ============================================================================
// TOP-LEVEL PRODUCTION RULES
// ============================================================================

// declaration_body
inline const auto declaration_body = unified_declaration | func_declaration | type_declaration_kw | 
                                      namespace_declaration_kw | using_declaration | import_declaration | 
                                      let_declaration | statement;

// declaration = [ access_specifier ] declaration_body
inline const auto declaration_rule = -access_specifier >> declaration_body;

// statement rule
inline const auto statement_rule = block_statement | if_statement | while_statement | for_statement |
                                    do_while_statement | switch_statement | try_statement |
                                    return_statement | break_statement | continue_statement | 
                                    throw_statement | contract_statement |
                                    local_declaration | expression_statement | SEMICOLON;

// block_statement = "{" { statement } "}"
inline const auto block_impl = LBRACE >> *statement >> RBRACE;

// translation_unit = { declaration } EOF
inline const auto translation_unit = *declaration >> END_OF_FILE;

// ============================================================================
// Initialize Lazy Rules
// ============================================================================

static bool g_init = false;

inline void init_grammar() {
    if (g_init) return;
    g_init = true;
    
    expression.parser_fn = [](TS in) { return assignment_expression.parse(in); };
    statement.parser_fn = [](TS in) { return statement_rule.parse(in); };
    type_specifier.parser_fn = [](TS in) { return type_spec_impl.parse(in); };
    declaration.parser_fn = [](TS in) { return declaration_rule.parse(in); };
    block_statement.parser_fn = [](TS in) { return block_impl.parse(in); };
}

// ============================================================================
// Parser Public Interface
// ============================================================================

Parser::Parser(std::span<Token> tokens) : tokens(tokens) {
    init_grammar();
}

std::unique_ptr<AST> Parser::parse() {
    auto ast = std::make_unique<AST>();
    TS stream(std::span<const Token>(tokens.data(), tokens.size()));
    
    auto result = translation_unit.parse(stream);
    if (!result.success()) {
        error_count++;
    }
    
    return ast;
}

} // namespace cpp2_transpiler