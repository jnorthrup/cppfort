#pragma once

#include "ast.hpp"
#include "lexer.hpp"
#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <memory>

namespace cpp2_transpiler {
namespace parser {
namespace grammar {

// ============================================================================
// Parser Grammar (Boost Spirit-style EBNF to AST Mapping)
// ============================================================================
//
// This namespace defines type aliases that map EBNF grammar symbols from
// conductor/PARSER_ORCHESTRATION.md Section 1 to the AST node types they produce.
//
// Purpose:
// 1. Bridge formal grammar (EBNF) and implementation (AST types)
// 2. Self-documenting grammar specification
// 3. Foundation for future combinator-based parser refactoring
//
// Pattern (inspired by Boost Spirit):
//   - Each EBNF symbol has a corresponding type alias
//   - Type alias names match EBNF symbols exactly
//   - Aliases map to AST node types (what the parser produces)
//
// Example:
//   EBNF:     expression ::= assignment_expression
//   Type:     using expression_result = std::unique_ptr<Expression>;
//
//   Future combinator style:
//   using expression = assignment_expression_parser;
//
// ============================================================================

// ----------------------------------------------------------------------------
// Section 1.2: Lexical Structure
// ----------------------------------------------------------------------------

// EBNF: identifier ::= [a-zA-Z_] { [a-zA-Z0-9_] }
using identifier = std::string;

// EBNF: integer_literal ::= ? standard C++ integer literal ?
using integer_literal = std::string;

// EBNF: float_literal ::= ? standard C++ floating point literal ?
using float_literal = std::string;

// EBNF: string_literal ::= ? standard C++ string literal, with (...)$ interpolation ?
using string_literal = std::string;

// EBNF: char_literal ::= ? standard C++ char literal ?
using char_literal = std::string;

// EBNF: user_defined_literal ::= ( integer_literal | float_literal ) identifier
using user_defined_literal = std::string;

// ----------------------------------------------------------------------------
// Section 1.3: Top-Level Structure
// ----------------------------------------------------------------------------

// EBNF: translation_unit ::= { declaration }
using translation_unit_result = std::unique_ptr<AST>;

// EBNF: declaration ::= namespace_declaration | template_declaration | type_declaration
//                     | function_declaration | variable_declaration | using_declaration
//                     | import_declaration | statement
using declaration_result = std::unique_ptr<Declaration>;

// ----------------------------------------------------------------------------
// Section 1.4: Template Parameters
// ----------------------------------------------------------------------------

// EBNF: template_params ::= "<" template_param_list ">"
// Produces: std::vector<TemplateParameter>
using template_params_result = std::vector<std::string>;  // Simplified for now

// EBNF: template_param_list ::= template_param { "," template_param } [ "," ]
using template_param_list_result = std::vector<std::string>;

// EBNF: template_param ::= identifier [ "..." ] [ ":" type_constraint ] [ "=" default_value ]
//                        | "_" [ "..." ]
using template_param_result = std::string;

// ----------------------------------------------------------------------------
// Section 1.5: Function Declarations
// ----------------------------------------------------------------------------

// EBNF: function_declaration ::= identifier ":" function_signature [ contracts ] function_body
using function_declaration_result = std::unique_ptr<FunctionDeclaration>;

// EBNF: parameter ::= [ param_qualifiers ] param_name [ ":" type_specifier ] [ "=" default_arg ]
//                   | [ param_qualifiers ] param_name "..."
//                   | [ param_qualifiers ] "this"
using parameter_result = std::pair<std::string, std::unique_ptr<Type>>;

// EBNF: parameter_list ::= parameter { "," parameter } [ "," ]
using parameter_list_result = std::vector<std::pair<std::string, std::unique_ptr<Type>>>;

// ----------------------------------------------------------------------------
// Section 1.6: Type Declarations
// ----------------------------------------------------------------------------

// EBNF: type_declaration ::= identifier ":" [ metafunctions ] "type" [ template_params ]
//                            [ base_types ] "=" type_body
using type_declaration_result = std::unique_ptr<TypeDeclaration>;

// EBNF: metafunction ::= "@" identifier
using metafunction_result = std::string;

// EBNF: metafunctions ::= { metafunction }
using metafunctions_result = std::vector<std::string>;

// ----------------------------------------------------------------------------
// Section 1.7: Statements
// ----------------------------------------------------------------------------

// EBNF: statement ::= block_statement | if_statement | loop_statement | switch_statement
//                   | inspect_statement | return_statement | throw_statement | try_statement
//                   | contract_statement | expression_statement | declaration | ";"
using statement_result = std::unique_ptr<Statement>;

// EBNF: block_statement ::= "{" { statement } "}"
using block_statement_result = std::unique_ptr<BlockStatement>;

// EBNF: if_statement ::= "if" [ "constexpr" ] expression block_statement
//                        [ "else" ( block_statement | if_statement ) ]
using if_statement_result = std::unique_ptr<IfStatement>;

// EBNF: loop_statement ::= [ identifier ":" ] [ loop_parameters ] loop_kind
using loop_statement_result = std::unique_ptr<Statement>;  // WhileStmt, ForStmt, DoWhileStmt

// EBNF: while_loop ::= "while" expression [ "next" expression ] block_statement
using while_loop_result = std::unique_ptr<WhileStatement>;

// EBNF: for_loop ::= "for" expression [ "next" expression ] "do" "(" parameter ")" block_statement
using for_loop_result = std::unique_ptr<ForStatement>;

// EBNF: do_loop ::= "do" block_statement [ "next" expression ] "while" expression ";"
using do_loop_result = std::unique_ptr<DoWhileStatement>;

// EBNF: return_statement ::= "return" [ expression ] ";"
using return_statement_result = std::unique_ptr<ReturnStatement>;

// ----------------------------------------------------------------------------
// Section 1.8: Expressions
// ----------------------------------------------------------------------------

// EBNF: expression ::= assignment_expression
using expression_result = std::unique_ptr<Expression>;

// EBNF: assignment_expression ::= ternary_expression [ assignment_operator assignment_expression ]
using assignment_expression_result = std::unique_ptr<Expression>;

// EBNF: ternary_expression ::= logical_or_expression [ "?" expression ":" ternary_expression ]
using ternary_expression_result = std::unique_ptr<Expression>;

// EBNF: logical_or_expression ::= logical_and_expression { "||" logical_and_expression }
using logical_or_expression_result = std::unique_ptr<Expression>;

// EBNF: logical_and_expression ::= bitwise_or_expression { "&&" bitwise_or_expression }
using logical_and_expression_result = std::unique_ptr<Expression>;

// EBNF: equality_expression ::= comparison_expression { ( "==" | "!=" ) comparison_expression }
using equality_expression_result = std::unique_ptr<Expression>;

// EBNF: comparison_expression ::= range_expression { ( "<" | ">" | "<=" | ">=" ) range_expression }
using comparison_expression_result = std::unique_ptr<Expression>;

// EBNF: addition_expression ::= multiplication_expression { ( "+" | "-" ) multiplication_expression }
using addition_expression_result = std::unique_ptr<Expression>;

// EBNF: multiplication_expression ::= prefix_expression { ( "*" | "/" | "%" ) prefix_expression }
using multiplication_expression_result = std::unique_ptr<Expression>;

// EBNF: prefix_expression ::= postfix_expression
//                           | "await" prefix_expression
//                           | ( "+" | "-" | "!" | "~" | "++" | "--" | "&" | "*" ) prefix_expression
using prefix_expression_result = std::unique_ptr<Expression>;

// EBNF: postfix_expression ::= primary_expression { postfix_op }
using postfix_expression_result = std::unique_ptr<Expression>;

// EBNF: primary_expression ::= literal | identifier_expression | "this" | "that" | "_"
//                            | "(" expression_or_tuple ")" | "[" list_or_lambda "]"
//                            | "{" struct_initializer "}" | "inspect" inspect_expression
//                            | "@" identifier [ "(" [ argument_list ] ")" ]
//                            | ":" function_expression
using primary_expression_result = std::unique_ptr<Expression>;

// ----------------------------------------------------------------------------
// Section 1.9: Pattern Matching
// ----------------------------------------------------------------------------

// EBNF: inspect_expression ::= expression [ "->" type_specifier ] "{" { inspect_arm } "}"
using inspect_expression_result = std::unique_ptr<InspectExpression>;

// EBNF: inspect_arm ::= pattern "=>" statement
using inspect_arm_result = std::pair<std::unique_ptr<Expression>, std::unique_ptr<Statement>>;

// EBNF: pattern ::= "_" | identifier | identifier ":" type_specifier
//                 | "is" type_specifier | "is" "(" expression ")" | "is" literal | expression
using pattern_result = std::unique_ptr<Expression>;

// ----------------------------------------------------------------------------
// Section 1.10: Type Specifiers
// ----------------------------------------------------------------------------

// EBNF: type_specifier ::= function_type | pointer_type | qualified_type
using type_specifier_result = std::unique_ptr<Type>;

// EBNF: function_type ::= "(" [ param_type_list ] ")" "->" [ return_qualifier ] type_specifier
using function_type_result = std::unique_ptr<Type>;

// EBNF: pointer_type ::= "*" [ "const" ] type_specifier
using pointer_type_result = std::unique_ptr<Type>;

// EBNF: qualified_type ::= basic_type { "::" identifier [ template_args ] } { "*" | "&" }
using qualified_type_result = std::unique_ptr<Type>;

// EBNF: basic_type ::= [ type_modifier { type_modifier } ] type_name [ template_args ]
//                    | "auto" | "_" [ "is" type_constraint ] | "type"
//                    | "decltype" "(" expression ")" | "const" type_specifier
using basic_type_result = std::unique_ptr<Type>;

// ============================================================================
// Combinator Specification Namespace
// ============================================================================
//
// This namespace documents how EBNF productions would map to combinator
// compositions in a future combinator-based parser refactoring.
//
// Combinator primitives (from bytebuffer/combinator library):
//   seq(a, b, ...)      - Sequencing: a then b then ...
//   alt(a, b, ...)      - Alternation: a or b or ...
//   many(p)             - Zero or more: p*
//   many1(p)            - One or more: p+
//   opt(p)              - Optional: p?
//   sep_by(p, delim)    - Separated list: p (delim p)*
//   between(l, p, r)    - Bracketed: l p r
//   token(T)            - Match token type T
//   keyword(K)          - Match keyword K
//   punct(P)            - Match punctuation P
//
// ============================================================================

namespace combinators {

// Example EBNF to combinator mapping (for future refactoring):
//
// EBNF:
//   declaration ::= namespace_declaration
//                 | template_declaration
//                 | type_declaration
//                 | function_declaration
//                 | variable_declaration
//                 | using_declaration
//                 | import_declaration
//                 | statement
//
// Combinator composition:
//   using declaration_parser = alt<
//       namespace_declaration_parser,
//       template_declaration_parser,
//       type_declaration_parser,
//       function_declaration_parser,
//       variable_declaration_parser,
//       using_declaration_parser,
//       import_declaration_parser,
//       statement_parser
//   >;
//
// EBNF:
//   parameter_list ::= parameter { "," parameter } [ "," ]
//
// Combinator composition:
//   using parameter_list_parser = sep_by<parameter_parser, punct<','>>;
//
// EBNF:
//   if_statement ::= "if" [ "constexpr" ] expression block_statement
//                    [ "else" ( block_statement | if_statement ) ]
//
// Combinator composition:
//   using if_statement_parser = seq<
//       keyword<'if'>,
//       opt<keyword<'constexpr'>>,
//       expression_parser,
//       block_statement_parser,
//       opt<seq<keyword<'else'>, alt<block_statement_parser, if_statement_parser>>>
//   >;

// Future parser combinator type aliases would go here
// For now, this serves as documentation of the intended structure

} // namespace combinators

// ============================================================================
// Operator Precedence Table
// ============================================================================
//
// From PARSER_ORCHESTRATION.md Section 1.11
// Maps binary operators to their precedence levels
//
// ============================================================================

namespace operators {

enum class binary_op {
    // Level 3: Multiplicative
    multiply, divide, modulo,

    // Level 4: Additive
    add, subtract,

    // Level 5: Shift
    shift_left, shift_right,

    // Level 6: Range
    range_inclusive, range_exclusive,

    // Level 7: Comparison
    less, greater, less_eq, greater_eq,

    // Level 8: Equality
    equal, not_equal,

    // Level 9-13: Bitwise and logical
    bitwise_and, bitwise_xor, bitwise_or,
    logical_and, logical_or
};

constexpr int precedence(binary_op op) {
    switch (op) {
        case binary_op::multiply:
        case binary_op::divide:
        case binary_op::modulo:
            return 3;

        case binary_op::add:
        case binary_op::subtract:
            return 4;

        case binary_op::shift_left:
        case binary_op::shift_right:
            return 5;

        case binary_op::range_inclusive:
        case binary_op::range_exclusive:
            return 6;

        case binary_op::less:
        case binary_op::greater:
        case binary_op::less_eq:
        case binary_op::greater_eq:
            return 7;

        case binary_op::equal:
        case binary_op::not_equal:
            return 8;

        case binary_op::bitwise_and:
            return 9;

        case binary_op::bitwise_xor:
            return 10;

        case binary_op::bitwise_or:
            return 11;

        case binary_op::logical_and:
            return 12;

        case binary_op::logical_or:
            return 13;

        default:
            return 0;
    }
}

enum class associativity { left, right };

constexpr associativity assoc(binary_op op) {
    // All binary operators in the precedence table are left-associative
    // (Ternary and assignment are right-associative but handled separately)
    return associativity::left;
}

} // namespace operators

} // namespace grammar
} // namespace parser
} // namespace cpp2_transpiler
