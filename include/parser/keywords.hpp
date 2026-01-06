#pragma once

// ============================================================================
// Hierarchical Keyword Enums for Cpp2 Parser
// ============================================================================
//
// This file organizes Cpp2 keywords into hierarchical enum classes matching
// the EBNF grammar structure. Each category represents a distinct syntactic
// context where keywords appear.
//
// Hierarchy:
//   Declaration - namespace, type, function, variable, using, import
//   Statement   - control flow and other statement keywords
//   Expression  - expression precedence levels
//   Literal     - literal kinds
//   Operator    - all operators by category
//   Qualifier   - type and parameter qualifiers
//
// ============================================================================

#ifndef CPP2_PARSER_KEYWORDS_HPP
#define CPP2_PARSER_KEYWORDS_HPP

#include <string_view>

namespace cpp2::parser::keywords {

// ============================================================================
// Declaration Keywords
// ============================================================================
// Keywords that begin or modify declarations

enum class Declaration {
    // Primary declaration keywords
    namespace_,     // namespace Name { ... }
    type,           // type Name = ...
    func,           // func name(...) = ...
    let,            // let x = ...
    const_,         // const x = ...
    
    // Declaration modifiers
    using_,         // using Name = ...
    import_,        // import std;
    template_,      // template<...>
    
    // Type-related
    struct_,        // type X = struct { ... }
    class_,         // type X = class { ... }
    union_,         // type X = union { ... }
    enum_,          // type X = enum { ... }
    concept_,       // type C = concept { ... }
    interface_,     // type I = interface { ... }
    
    // Access specifiers
    public_,
    private_,
    protected_,
    
    // Function modifiers
    virtual_,
    override_,
    final_,
    explicit_,
    implicit_,
    static_assert_,
    
    // Operator declaration
    operator_,
};

// ============================================================================
// Statement Keywords
// ============================================================================
// Keywords that begin statements

enum class Statement {
    // Block statements
    block,          // { ... }
    
    // Conditional
    if_,
    else_,
    switch_,
    case_,
    default_,
    
    // Loops
    while_,
    for_,
    do_,
    
    // Control flow
    return_,
    break_,
    continue_,
    throw_,
    try_,
    catch_,
    
    // Pattern matching
    inspect_,
    
    // Contracts
    pre_,           // [[pre: ...]]
    post_,          // [[post: ...]]
    assert_,        // assert(...)
    
    // Concurrency
    await_,
    launch_,
    coroutine_scope_,
    parallel_for_,
    select_,
    channel_,
};

// ============================================================================
// Expression Precedence Levels
// ============================================================================
// Categories matching expression grammar precedence

enum class ExpressionLevel {
    // Lowest precedence
    assignment,         // = += -= etc.
    ternary,            // ? :
    logical_or,         // ||
    logical_and,        // &&
    bitwise_or,         // |
    bitwise_xor,        // ^
    bitwise_and,        // &
    equality,           // == !=
    comparison,         // < > <= >= <=>
    shift,              // << >>
    additive,           // + -
    multiplicative,     // * / %
    prefix,             // ++ -- ! ~ - + * &
    postfix,            // () [] . -> ++ --
    primary,            // literals, identifiers, ( )
    // Highest precedence
};

// ============================================================================
// Literal Types
// ============================================================================

enum class Literal {
    integer,
    float_,
    string,
    char_,
    boolean,
    nullptr_,
};

// ============================================================================
// Operator Categories
// ============================================================================

enum class Operator {
    // Logical
    logical_and,        // &&
    logical_or,         // ||
    logical_not,        // !
    
    // Bitwise
    bit_and,            // &
    bit_or,             // |
    bit_xor,            // ^
    bit_not,            // ~
    shift_left,         // <<
    shift_right,        // >>
    
    // Comparison
    eq,                 // ==
    ne,                 // !=
    lt,                 // <
    gt,                 // >
    le,                 // <=
    ge,                 // >=
    spaceship,          // <=>
    
    // Arithmetic
    add,                // +
    sub,                // -
    mul,                // *
    div,                // /
    mod,                // %
    
    // Unary
    unary_plus,         // +
    unary_minus,        // -
    deref,              // *
    addr_of,            // &
    increment,          // ++
    decrement,          // --
    
    // Assignment
    assign,             // =
    add_assign,         // +=
    sub_assign,         // -=
    mul_assign,         // *=
    div_assign,         // /=
    mod_assign,         // %=
    and_assign,         // &=
    or_assign,          // |=
    xor_assign,         // ^=
    shl_assign,         // <<=
    shr_assign,         // >>=
    
    // Special
    pipeline,           // |>
    ternary,            // ? :
    arrow,              // ->
    scope,              // ::
    member,             // .
    range_incl,         // ..=
    range_excl,         // ..<
};

// ============================================================================
// Parameter Qualifiers (Cpp2-specific)
// ============================================================================

enum class ParamQualifier {
    none,
    in_,                // in (default, read-only)
    out_,               // out (output)
    inout_,             // inout (read-write)
    copy_,              // copy (by value)
    move_,              // move
    forward_,           // forward (perfect forwarding)
    in_ref_,            // in_ref
    forward_ref_,       // forward_ref
};

// ============================================================================
// Type Qualifiers
// ============================================================================

enum class TypeQualifier {
    const_,
    volatile_,
    mutable_,
    constexpr_,
    noexcept_,
};

// ============================================================================
// String Conversion Functions
// ============================================================================

[[nodiscard]] constexpr std::string_view to_string(Declaration d) {
    switch (d) {
        case Declaration::namespace_:   return "namespace";
        case Declaration::type:         return "type";
        case Declaration::func:         return "func";
        case Declaration::let:          return "let";
        case Declaration::const_:       return "const";
        case Declaration::using_:       return "using";
        case Declaration::import_:      return "import";
        case Declaration::template_:    return "template";
        case Declaration::struct_:      return "struct";
        case Declaration::class_:       return "class";
        case Declaration::union_:       return "union";
        case Declaration::enum_:        return "enum";
        case Declaration::concept_:     return "concept";
        case Declaration::interface_:   return "interface";
        case Declaration::public_:      return "public";
        case Declaration::private_:     return "private";
        case Declaration::protected_:   return "protected";
        case Declaration::virtual_:     return "virtual";
        case Declaration::override_:    return "override";
        case Declaration::final_:       return "final";
        case Declaration::explicit_:    return "explicit";
        case Declaration::implicit_:    return "implicit";
        case Declaration::static_assert_: return "static_assert";
        case Declaration::operator_:    return "operator";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(Statement s) {
    switch (s) {
        case Statement::block:          return "block";
        case Statement::if_:            return "if";
        case Statement::else_:          return "else";
        case Statement::switch_:        return "switch";
        case Statement::case_:          return "case";
        case Statement::default_:       return "default";
        case Statement::while_:         return "while";
        case Statement::for_:           return "for";
        case Statement::do_:            return "do";
        case Statement::return_:        return "return";
        case Statement::break_:         return "break";
        case Statement::continue_:      return "continue";
        case Statement::throw_:         return "throw";
        case Statement::try_:           return "try";
        case Statement::catch_:         return "catch";
        case Statement::inspect_:       return "inspect";
        case Statement::pre_:           return "pre";
        case Statement::post_:          return "post";
        case Statement::assert_:        return "assert";
        case Statement::await_:         return "await";
        case Statement::launch_:        return "launch";
        case Statement::coroutine_scope_: return "coroutineScope";
        case Statement::parallel_for_:  return "parallel_for";
        case Statement::select_:        return "select";
        case Statement::channel_:       return "channel";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(ExpressionLevel e) {
    switch (e) {
        case ExpressionLevel::assignment:     return "assignment";
        case ExpressionLevel::ternary:        return "ternary";
        case ExpressionLevel::logical_or:     return "logical_or";
        case ExpressionLevel::logical_and:    return "logical_and";
        case ExpressionLevel::bitwise_or:     return "bitwise_or";
        case ExpressionLevel::bitwise_xor:    return "bitwise_xor";
        case ExpressionLevel::bitwise_and:    return "bitwise_and";
        case ExpressionLevel::equality:       return "equality";
        case ExpressionLevel::comparison:     return "comparison";
        case ExpressionLevel::shift:          return "shift";
        case ExpressionLevel::additive:       return "additive";
        case ExpressionLevel::multiplicative: return "multiplicative";
        case ExpressionLevel::prefix:         return "prefix";
        case ExpressionLevel::postfix:        return "postfix";
        case ExpressionLevel::primary:        return "primary";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(Literal l) {
    switch (l) {
        case Literal::integer:  return "integer";
        case Literal::float_:   return "float";
        case Literal::string:   return "string";
        case Literal::char_:    return "char";
        case Literal::boolean:  return "boolean";
        case Literal::nullptr_: return "nullptr";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(ParamQualifier q) {
    switch (q) {
        case ParamQualifier::none:        return "";
        case ParamQualifier::in_:         return "in";
        case ParamQualifier::out_:        return "out";
        case ParamQualifier::inout_:      return "inout";
        case ParamQualifier::copy_:       return "copy";
        case ParamQualifier::move_:       return "move";
        case ParamQualifier::forward_:    return "forward";
        case ParamQualifier::in_ref_:     return "in_ref";
        case ParamQualifier::forward_ref_: return "forward_ref";
    }
    return "unknown";
}

} // namespace cpp2::parser::keywords

#endif // CPP2_PARSER_KEYWORDS_HPP
