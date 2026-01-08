#pragma once

#include "ast.hpp"
#include "lexer.hpp"
#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <memory>
#include <tuple>
#include <type_traits>

namespace cpp2_transpiler {
namespace parser {
namespace grammar {

// ============================================================================
// Parser Grammar Type Specification (Boost Spirit Pattern)
// ============================================================================
//
// Direct type-level encoding of EBNF grammar from PARSER_ORCHESTRATION.md.
// Each EBNF symbol maps to a type representing its combinator structure.
//
// Pattern (EBNF → C++ type encoding):
//   EBNF alternation (|)     → alt<...> = std::variant<...>
//   EBNF sequence (a b)      → seq<...> = std::tuple<...>
//   EBNF optional ([ a ])    → opt<T> = std::optional<T>
//   EBNF repetition ({ a })  → many<T> = std::vector<T>
//
// NOTE: This is a type-level grammar specification. Some types contain
//       circular dependencies (e.g., expression/statement) and cannot be
//       instantiated as values. Use the operators namespace for runtime code.
//
// ============================================================================

// ----------------------------------------------------------------------------
// Primitive Combinator Types
// ----------------------------------------------------------------------------

#include <type_traits>

template<typename T>
using maybe_ptr = ::std::conditional_t<::std::is_class_v<T>, ::std::shared_ptr<T>, T>;

template<typename T>
using opt = ::std::optional<maybe_ptr<T>>;

template<typename T>
using many = ::std::vector<maybe_ptr<T>>;

template<typename... Ts>
using seq = ::std::tuple<maybe_ptr<Ts>...>;

template<typename... Ts>
using alt = ::std::variant<maybe_ptr<Ts>...>;

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
using user_defined_literal = seq<alt<integer_literal, float_literal>, identifier>;

// ----------------------------------------------------------------------------
// Section 1.3: Top-Level Structure
// ----------------------------------------------------------------------------

// Forward declarations
struct namespace_declaration;
struct template_declaration;
struct type_declaration;
struct function_declaration;
struct variable_declaration;
struct using_declaration;
struct import_declaration;
struct statement;

// EBNF: translation_unit ::= { declaration }
struct declaration;
using translation_unit = many<declaration>;

// EBNF: declaration ::= namespace_declaration | template_declaration | type_declaration
//                     | function_declaration | variable_declaration | using_declaration
//                     | import_declaration | statement
struct declaration : alt<
    namespace_declaration,
    template_declaration,
    type_declaration,
    function_declaration,
    variable_declaration,
    using_declaration,
    import_declaration,
    statement
> {};

// ----------------------------------------------------------------------------
// Section 1.4: Template Parameters
// ----------------------------------------------------------------------------

struct type_constraint;
struct default_value;

// EBNF: template_param ::= identifier [ "..." ] [ ":" type_constraint ] [ "=" default_value ]
//                        | "_" [ "..." ]
struct wildcard {};
struct template_param : seq<
    alt<identifier, wildcard>,
    opt<bool>,                      // is_variadic
    opt<type_constraint>,
    opt<default_value>
> {};

// EBNF: template_param_list ::= template_param { "," template_param } [ "," ]
using template_param_list = many<template_param>;

// EBNF: template_params ::= "<" template_param_list ">"
using template_params = template_param_list;

// EBNF: type_constraint ::= type_specifier | "_" | "type"
struct type_specifier;
struct type_keyword {};
struct type_constraint : alt<type_specifier, wildcard, type_keyword> {};

// EBNF: default_value ::= type_specifier | expression
struct expression;
struct default_value : alt<type_specifier, expression> {};

// ----------------------------------------------------------------------------
// Section 1.5: Function Declarations
// ----------------------------------------------------------------------------

// EBNF: param_qualifier ::= "in" | "copy" | "inout" | "out" | "move" | "forward"
//                         | "in_ref" | "forward_ref" | "virtual" | "override" | "implicit"
enum class param_qualifier {
    in, copy, inout, out, move, forward,
    in_ref, forward_ref, virtual_, override_, implicit
};

// EBNF: param_qualifiers ::= { param_qualifier }
using param_qualifiers = many<param_qualifier>;

// EBNF: parameter ::= [ param_qualifiers ] param_name [ ":" type_specifier ] [ "=" default_arg ]
//                   | [ param_qualifiers ] param_name "..."
//                   | [ param_qualifiers ] "this"
struct this_param {};
struct parameter : seq<
    opt<param_qualifiers>,
    alt<identifier, this_param>,
    opt<bool>,                  // is_variadic
    opt<type_specifier>,
    opt<expression>             // default_arg
> {};

// EBNF: parameter_list ::= parameter { "," parameter } [ "," ]
using parameter_list = many<parameter>;

// EBNF: named_return ::= identifier ":" type_specifier
using named_return = seq<identifier, type_specifier>;

// EBNF: return_list ::= return_type | "(" named_return { "," named_return } ")"
using return_list = alt<type_specifier, many<named_return>>;

// EBNF: return_spec ::= "->" return_list
using return_spec = return_list;

// EBNF: throws_spec ::= "throws" { type_specifier }
using throws_spec = many<type_specifier>;

// EBNF: requires_clause ::= "requires" expression
using requires_clause = expression;

// EBNF: function_signature ::= [ template_params ] "(" [ parameter_list ] ")"
//                              [ throws_spec ] [ return_spec ] [ requires_clause ]
using function_signature = seq<
    opt<template_params>,
    parameter_list,
    opt<throws_spec>,
    opt<return_spec>,
    opt<requires_clause>
>;

// EBNF: contracts ::= { contract_expression }
using contracts = many<expression>;

// EBNF: function_body ::= block_statement | expression
struct block_statement;
using function_body = alt<block_statement, expression>;

// EBNF: function_declaration ::= identifier ":" function_signature [ contracts ] function_body
struct function_declaration : seq<
    identifier,
    function_signature,
    opt<contracts>,
    function_body
> {};

// ----------------------------------------------------------------------------
// Section 1.6: Type Declarations
// ----------------------------------------------------------------------------

// EBNF: metafunction ::= "@" identifier
using metafunction = identifier;

// EBNF: metafunctions ::= { metafunction }
using metafunctions = many<metafunction>;

// EBNF: base_types ::= ":" type_specifier { "," type_specifier }
using base_types = many<type_specifier>;

// EBNF: type_body ::= "{" { type_member } "}"
struct type_member;
using type_body = many<type_member>;

// EBNF: type_member ::= function_declaration | variable_declaration | type_declaration
struct type_member : alt<
    function_declaration,
    variable_declaration,
    type_declaration
> {};

// EBNF: type_declaration ::= identifier ":" [ metafunctions ] "type" [ template_params ]
//                            [ base_types ] "=" type_body
struct type_declaration : seq<
    identifier,
    opt<metafunctions>,
    opt<template_params>,
    opt<base_types>,
    type_body
> {};

// ----------------------------------------------------------------------------
// Section 1.7: Statements
// ----------------------------------------------------------------------------

// EBNF: block_statement ::= "{" { statement } "}"
struct block_statement : many<statement> {};

// EBNF: if_statement ::= "if" [ "constexpr" ] expression block_statement
//                        [ "else" ( block_statement | if_statement ) ]
struct if_statement;
struct if_statement_rec : seq<
    opt<bool>,              // is_constexpr
    expression,
    block_statement,
    opt<alt<block_statement, if_statement>>
> {};
struct if_statement : if_statement_rec {};

// EBNF: loop_parameters ::= "(" param_declaration { "," param_declaration } ")"
using loop_parameters = many<parameter>;

// EBNF: while_loop ::= "while" expression [ "next" expression ] block_statement
using while_loop = seq<expression, opt<expression>, block_statement>;

// EBNF: for_loop ::= "for" expression [ "next" expression ] "do" "(" parameter ")" block_statement
using for_loop = seq<expression, opt<expression>, parameter, block_statement>;

// EBNF: do_loop ::= "do" block_statement [ "next" expression ] "while" expression ";"
using do_loop = seq<block_statement, opt<expression>, expression>;

// EBNF: loop_kind ::= while_loop | for_loop | do_loop
using loop_kind = alt<while_loop, for_loop, do_loop>;

// EBNF: loop_statement ::= [ identifier ":" ] [ loop_parameters ] loop_kind
using loop_statement = seq<opt<identifier>, opt<loop_parameters>, loop_kind>;

// EBNF: return_statement ::= "return" [ expression ] ";"
using return_statement = opt<expression>;

// EBNF: throw_statement ::= "throw" [ expression ] ";"
using throw_statement = opt<expression>;

// EBNF: switch_statement ::= "switch" expression "{" { switch_case } "}"
using switch_statement = seq<expression, many<seq<opt<expression>, many<statement>>>>;

// EBNF: switch_case ::= "case" expression ":" { statement }
//                     | "default" ":" { statement }
using switch_case = seq<opt<expression>, many<statement>>;

// EBNF: inspect_expression ::= ...
struct inspect_expression;

// EBNF: inspect_statement ::= "inspect" inspect_expression
using inspect_statement = inspect_expression;

// EBNF: try_statement ::= "try" block_statement { catch_clause }
using try_statement = seq<block_statement, many<seq<parameter, block_statement>>>;

// EBNF: catch_clause ::= "catch" "(" parameter ")" block_statement

// EBNF: contract_statement ::= "pre" expression | "post" expression | "assert" expression
enum class contract_kind { pre, post, assert_ };
using contract_statement = seq<contract_kind, expression, opt<string_literal>>;

// EBNF: expression_statement ::= expression ";"
using expression_statement = expression;

// EBNF: statement ::= block_statement | if_statement | loop_statement | switch_statement
//                   | inspect_statement | return_statement | throw_statement | try_statement
//                   | contract_statement | expression_statement | declaration | ";"
struct statement : alt<
    block_statement,
    if_statement,
    loop_statement,
    switch_statement,
    inspect_statement,
    return_statement,
    throw_statement,
    try_statement,
    contract_statement,
    expression_statement,
    declaration
> {};

// ----------------------------------------------------------------------------
// Section 1.8: Expressions
// ----------------------------------------------------------------------------

// EBNF: assignment_operator ::= "=" | "+=" | "-=" | "*=" | "/=" | "%="
//                             | "<<=" | ">>=" | "&=" | "|=" | "^="
enum class assignment_operator {
    assign, add_assign, sub_assign, mul_assign, div_assign, mod_assign,
    shl_assign, shr_assign, and_assign, or_assign, xor_assign
};

// EBNF: binary_operator (precedence levels)
enum class binary_operator {
    multiply, divide, modulo,                    // Level 3
    add, subtract,                                // Level 4
    shift_left, shift_right,                      // Level 5
    range_inclusive, range_exclusive,             // Level 6
    less, greater, less_eq, greater_eq,           // Level 7
    equal, not_equal,                             // Level 8
    bitwise_and,                                  // Level 9
    bitwise_xor,                                  // Level 10
    bitwise_or,                                   // Level 11
    logical_and,                                  // Level 12
    logical_or                                    // Level 13
};

// EBNF: unary_operator
enum class unary_operator {
    await_, launch, select_, move_, forward_, copy_,
    plus, minus, logical_not, bitwise_not,
    pre_increment, pre_decrement, address_of, dereference
};

// EBNF: postfix_op ::= "(" [ argument_list ] ")" | "<" template_arg_list ">" | ...
enum class postfix_op_kind {
    call, template_call, member_access, ufcs_call,
    subscript, pointer_deref, address_of,
    post_increment, post_decrement, dollar, ellipsis,
    as_cast, is_check, scope_resolution
};

struct postfix_op;

// EBNF: primary_expression ::= literal | identifier_expression | "this" | "that" | "_"
//                            | "(" expression_or_tuple ")" | "[" list_or_lambda "]"
//                            | "{" struct_initializer "}" | "inspect" inspect_expression
//                            | "@" identifier [ "(" [ argument_list ] ")" ]
//                            | ":" function_expression
struct this_expr {};
struct that_expr {};
struct literal : alt<integer_literal, float_literal, string_literal, char_literal> {};
struct grouped_expression;
struct list_literal;
struct struct_initializer;
struct metafunction_call;
struct function_expression;

using primary_expression = alt<
    literal,
    identifier,
    this_expr,
    that_expr,
    wildcard,
    grouped_expression,
    list_literal,
    struct_initializer,
    inspect_expression,
    metafunction_call,
    function_expression
>;

// EBNF: postfix_expression ::= primary_expression { postfix_op }
struct postfix_expression : seq<primary_expression, many<postfix_op>> {};

// EBNF: prefix_expression ::= postfix_expression | unary_operator prefix_expression
struct prefix_expression;
struct prefix_expression_rec : alt<
    postfix_expression,
    seq<unary_operator, prefix_expression>
> {};
struct prefix_expression : prefix_expression_rec {};

// EBNF: Binary expression (left-associative precedence climbing)
struct binary_expression : seq<prefix_expression, many<seq<binary_operator, prefix_expression>>> {};

// EBNF: ternary_expression ::= binary_expression [ "?" expression ":" ternary_expression ]
struct ternary_expression;
struct ternary_expression_rec : alt<
    binary_expression,
    seq<binary_expression, expression, ternary_expression>
> {};
struct ternary_expression : ternary_expression_rec {};

// EBNF: assignment_expression ::= ternary_expression [ assignment_operator assignment_expression ]
struct assignment_expression;
struct assignment_expression_rec : alt<
    ternary_expression,
    seq<ternary_expression, assignment_operator, assignment_expression>
> {};
struct assignment_expression : assignment_expression_rec {};

// EBNF: pipeline_expression ::= assignment_expression { "|>" assignment_expression }
using pipeline_expression = seq<assignment_expression, many<assignment_expression>>;

// EBNF: expression ::= pipeline_expression
struct expression : pipeline_expression {};

// ----------------------------------------------------------------------------
// Section 1.9: Pattern Matching
// ----------------------------------------------------------------------------

// EBNF: pattern ::= "_" | identifier | identifier ":" type_specifier
//                 | "is" type_specifier | "is" "(" expression ")" | "is" literal | expression
using is_type_pattern = type_specifier;
using is_value_pattern = expression;
struct named_pattern : seq<identifier, type_specifier> {};

using pattern = alt<
    wildcard,
    identifier,
    named_pattern,
    is_type_pattern,
    is_value_pattern,
    expression
>;

// EBNF: inspect_arm ::= pattern "=>" statement
using inspect_arm = seq<pattern, statement>;

// EBNF: inspect_expression ::= expression [ "->" type_specifier ] "{" { inspect_arm } "}"
struct inspect_expression : seq<
    expression,
    opt<type_specifier>,
    many<inspect_arm>
> {};

// ----------------------------------------------------------------------------
// Section 1.10: Type Specifiers
// ----------------------------------------------------------------------------

// EBNF: function_type ::= "(" [ param_type_list ] ")" "->" [ return_qualifier ] type_specifier
struct function_type;
struct function_type_rec : seq<
    many<type_specifier>,       // param_type_list
    opt<param_qualifier>,       // return_qualifier
    type_specifier
> {};
struct function_type : function_type_rec {};

// EBNF: pointer_type ::= "*" [ "const" ] type_specifier
struct pointer_type : seq<opt<bool>, type_specifier> {};  // is_const

// EBNF: basic_type ::= [ type_modifier { type_modifier } ] type_name [ template_args ]
//                    | "auto" | "_" [ "is" type_constraint ] | "type"
//                    | "decltype" "(" expression ")" | "const" type_specifier
enum class type_modifier { const_, volatile_, mutable_ };
struct auto_type {};
struct wildcard_type : opt<type_constraint> {};
struct decltype_type : expression {};
using template_args = many<alt<type_specifier, expression>>;

struct basic_type : seq<
    many<type_modifier>,
    alt<identifier, auto_type, wildcard_type, type_keyword, decltype_type>,
    opt<template_args>
> {};

// EBNF: qualified_type ::= basic_type { "::" identifier [ template_args ] } { "*" | "&" }
using qualified_type = seq<
    basic_type,
    many<seq<identifier, opt<template_args>>>,  // scope qualifiers
    many<char>                                    // pointer/ref qualifiers
>;

// EBNF: type_specifier ::= function_type | pointer_type | qualified_type
struct type_specifier : alt<function_type, pointer_type, qualified_type, basic_type> {};

// ----------------------------------------------------------------------------
// Additional Constructs
// ----------------------------------------------------------------------------

// EBNF: namespace_declaration ::= "namespace" identifier "{" { declaration } "}"
struct namespace_declaration : seq<identifier, many<declaration>> {};

// EBNF: using_declaration ::= "using" identifier "=" type_specifier ";"
struct using_declaration : seq<identifier, type_specifier> {};

// EBNF: import_declaration ::= "import" { identifier "::" } identifier ";"
struct import_declaration : many<identifier> {};

// EBNF: template_declaration ::= template_params declaration
struct template_declaration : seq<template_params, declaration> {};

// EBNF: variable_declaration ::= identifier [ ":" type_specifier ] "=" expression ";"
struct variable_declaration : seq<
    identifier,
    opt<type_specifier>,
    opt<expression>,        // initializer
    opt<param_qualifiers>   // qualifiers (in, inout, etc.)
> {};

// EBNF: metafunction_call ::= "@" identifier [ "(" { expression } ")" ]
struct metafunction_call : seq<identifier, opt<many<expression>>> {};

// EBNF: function_expression ::= ":" function_signature function_body
struct function_expression : seq<function_signature, function_body> {};

// EBNF: grouped_expression ::= "(" expression ")"
struct grouped_expression : expression {};

// EBNF: list_literal ::= "[" { expression } "]"
struct list_literal : many<expression> {};

// EBNF: struct_initializer ::= "{" { identifier ":" expression } "}"
struct struct_initializer : many<seq<identifier, expression>> {};

// EBNF: postfix_op (variant of different postfix operations)
struct postfix_op : alt<
    many<expression>,           // call arguments
    many<type_specifier>,       // template arguments
    identifier,                 // member name
    expression,                 // subscript
    type_specifier,             // as/is type
    pattern                     // is pattern
> {};

// ============================================================================
// Operator Precedence Table (constexpr utilities)
// ============================================================================

namespace operators {

constexpr int precedence(binary_operator op) {
    switch (op) {
        case binary_operator::multiply:
        case binary_operator::divide:
        case binary_operator::modulo:
            return 3;
        case binary_operator::add:
        case binary_operator::subtract:
            return 4;
        case binary_operator::shift_left:
        case binary_operator::shift_right:
            return 5;
        case binary_operator::range_inclusive:
        case binary_operator::range_exclusive:
            return 6;
        case binary_operator::less:
        case binary_operator::greater:
        case binary_operator::less_eq:
        case binary_operator::greater_eq:
            return 7;
        case binary_operator::equal:
        case binary_operator::not_equal:
            return 8;
        case binary_operator::bitwise_and:
            return 9;
        case binary_operator::bitwise_xor:
            return 10;
        case binary_operator::bitwise_or:
            return 11;
        case binary_operator::logical_and:
            return 12;
        case binary_operator::logical_or:
            return 13;
        default:
            return 0;
    }
}

enum class associativity { left, right };

using binary_op = binary_operator;

constexpr associativity assoc(binary_operator op) {
    return associativity::left;
}

} // namespace operators

} // namespace grammar
} // namespace parser
} // namespace cpp2_transpiler
