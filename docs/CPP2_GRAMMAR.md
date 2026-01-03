# Cpp2 Grammar Specification

This document defines the formal grammar for Cpp2 (cppfront syntax) as implemented by cppfort.

**See also:**
- `grammar/cpp2.combinators.md` — Orthogonal combinator specification (authoritative for parser implementation)
- `grammar/cpp2.ebnf` — Formal EBNF grammar

## Overview

Cpp2 uses a **Unified Declaration Syntax**: `Name : Kind = Value`

This pattern applies consistently to variables, functions, types, and namespaces.

---

## Lexical Structure

```ebnf
(* ========================================================================== *)
(* LEXICAL STRUCTURE                                                          *)
(* ========================================================================== *)

identifier          ::= [a-zA-Z_] { [a-zA-Z0-9_] } ;
integer_literal     ::= ? standard C++ integer literal ? ;
float_literal       ::= ? standard C++ floating point literal ? ;
string_literal      ::= ? standard C++ string literal, with (...)$ interpolation ? ;
char_literal        ::= ? standard C++ char literal ? ;

(* User-Defined Literals: 10s, 1.0ms, etc. - lexer yields separate tokens *)
user_defined_literal ::= ( integer_literal | float_literal ) identifier ;
```

### String Interpolation

Inside string literals, `(expression)$` is replaced with the stringified value:

```cpp2
name := "world";
std::cout << "Hello (name)$!\n";  // prints: Hello world!
```

---

## Top Level

```ebnf
(* ========================================================================== *)
(* TOP LEVEL                                                                  *)
(* ========================================================================== *)

translation_unit    ::= { declaration } ;

declaration         ::= namespace_declaration
                      | template_declaration  
                      | type_declaration
                      | function_declaration
                      | variable_declaration
                      | using_declaration
                      | import_declaration
                      | statement
                      ;

(* Unified Declaration Syntax: Name : Kind = Value *)
(* The parser uses lookahead to distinguish declaration forms *)
name_declaration    ::= identifier [ template_params ] ":" declaration_body ;

declaration_body    ::= namespace_body
                      | type_body  
                      | function_body
                      | variable_body
                      ;
```

### Template Parameters

```ebnf
template_params     ::= "<" template_param_list ">" ;
template_param_list ::= template_param { "," template_param } [ "," ] ;  (* trailing comma allowed *)
template_param      ::= identifier [ "..." ] [ ":" type_constraint ] [ "=" default_value ] 
                      | "_" [ "..." ]  (* anonymous/placeholder parameter *)
                      ;

type_constraint     ::= type_specifier | "_" | "type" ;

default_value       ::= type_specifier   (* for type params *)
                      | expression       (* for NTTP params - complex expressions tracked for balance *)
                      ;
```

### Variable Declarations

```ebnf
variable_declaration ::= identifier ":" variable_spec ";" 
                       | identifier ":=" expression ";"   (* type-deduced shorthand *)
                       | "let" identifier [ ":" type_specifier ] "=" expression ";"
                       | "const" identifier [ ":" type_specifier ] ( "=" | "==" ) expression ";"
                       ;

variable_spec       ::= [ template_params ] type_specifier [ "=" expression ]
                      | [ template_params ] type_specifier "==" expression  (* compile-time const *)
                      | "_" "=" expression                                  (* deduced type *)
                      ;
```

### Examples

```cpp2
// Variable declarations
x: int = 42;
y := 3.14;                    // deduced type (: _ =)
name: std::string = "hello";
PI: f64 == 3.14159;           // compile-time constant

// Type alias
MyInt: type == int;

// Namespace alias  
io: namespace == std::io;
```

---

## Functions

```ebnf
(* ========================================================================== *)
(* FUNCTIONS                                                                  *)
(* ========================================================================== *)

function_declaration ::= identifier ":" function_signature [ contracts ] function_body ;

function_signature  ::= [ template_params ] "(" [ parameter_list ] ")" 
                        [ throws_spec ] [ return_spec ] [ requires_clause ] ;

parameter_list      ::= parameter { "," parameter } [ "," ] ;  (* trailing comma allowed *)

parameter           ::= [ param_qualifiers ] param_name [ ":" type_specifier ] [ "=" default_arg ]
                      | [ param_qualifiers ] param_name "..."                (* variadic *)
                      | [ param_qualifiers ] "this"                          (* member function *)
                      ;

param_name          ::= identifier | "_" ;

param_qualifiers    ::= { param_qualifier } ;
param_qualifier     ::= "in" | "copy" | "inout" | "out" | "move" | "forward"
                      | "in_ref" | "forward_ref"
                      | "virtual" | "override" | "implicit"
                      ;

throws_spec         ::= "throws" ;

return_spec         ::= "->" return_list ;

return_list         ::= return_type
                      | "(" named_return { "," named_return } ")"  (* named returns *)
                      ;

return_type         ::= [ "forward" | "move" ] type_specifier ;

named_return        ::= identifier ":" type_specifier ;

requires_clause     ::= "requires" requires_expression ;
(* Note: requires_expression stops at '=' and '==' which are body separators *)

contracts           ::= { contract_clause } ;
contract_clause     ::= ( "pre" | "post" | "assert" ) [ "<" identifier ">" ] contract_body ;
contract_body       ::= ":" expression
                      | "(" expression [ "," string_literal ] ")"
                      ;

function_body       ::= "=" expression_body
                      | "==" expression_body     (* constexpr *)
                      | block_statement
                      | ";"                      (* forward declaration *)
                      ;

expression_body     ::= expression ";" | block_statement ;
```

### Parameter Passing Semantics

| Mode          | C++ Equivalent           | Semantics                    |
|---------------|--------------------------|------------------------------|
| `in`          | `T const&`               | Read-only (default)          |
| `copy`        | `T`                      | By value                     |
| `inout`       | `T&`                     | Read-write reference         |
| `out`         | `T&` (uninitialized)     | Must be assigned before use  |
| `move`        | `T&&`                    | Takes ownership              |
| `forward`     | `T&&` (forwarding ref)   | Perfect forwarding           |
| `in_ref`      | `T const&` (explicit)    | Reference In                 |
| `forward_ref` | `T&&` (explicit)         | Explicit forward             |

### Examples

```cpp2
// Basic function
add: (a: int, b: int) -> int = a + b;

// With parameter modes
swap: (inout a: int, inout b: int) = {
    tmp := a;
    a = b;
    b = tmp;
}

// Template function with requires clause
max: <T> (a: T, b: T) -> T 
    requires std::totally_ordered<T>
= if a > b then a else b;

// Named return values
divide: (a: int, b: int) -> (quotient: int, remainder: int) = {
    quotient = a / b;
    remainder = a % b;
}

// With contracts
sqrt: (x: f64) -> f64
    pre(x >= 0.0)
    post(r: r * r == x)
= /* implementation */;

// Member function with 'this'
print: (this) = std::cout << data << "\n";

// Const member function
get: (in this) -> int = data;

// Constexpr function
square: (x: int) -> int == x * x;
```

---

## Types & Namespaces

```ebnf
(* ========================================================================== *)
(* TYPES & NAMESPACES                                                         *)
(* ========================================================================== *)

type_declaration    ::= identifier ":" [ metafunctions ] "type" [ template_params ]
                        [ base_types ] "=" type_body ;

metafunctions       ::= { metafunction } ;
metafunction        ::= "@" identifier ;

base_types          ::= ":" type_specifier { "," type_specifier } ;

type_body           ::= "{" { type_member } "}" ;

type_member         ::= [ access_specifier ] declaration ;

access_specifier    ::= "public" | "protected" | "private" ;

namespace_declaration ::= identifier ":" "namespace" "=" namespace_body ;
namespace_body      ::= "{" { declaration } "}" ;
```

### Operator Declarations (inside types) 

```ebnf
operator_declaration ::= "operator" operator_name ":" function_signature function_body ;

operator_name       ::= "=" | "<=>" | "==" | "!=" | "<" | ">" | "<=" | ">="
                      | "+" | "-" | "*" | "/" | "%" | "+=" | "-=" | "*=" | "/=" | "%="
                      | "<<" | ">>" | "<<=" | ">>="
                      | "&" | "|" | "^" | "&=" | "|=" | "^="
                      | "~" | "!" | "&&" | "||"
                      | "++" | "--"
                      | "[]" | "()" | "->" | "->*"
                      | "new" | "delete" | "new[]" | "delete[]"
                      | "co_await"
                      | string_literal   (* user-defined literal operator *)
                      ;
```

### Metafunctions

| Metafunction       | Effect                                      |
|--------------------|---------------------------------------------|
| `@value`           | Regular value type (default)                |
| `@struct`          | All members public                          |
| `@interface`       | Pure virtual base class                     |
| `@polymorphic_base`| Virtual destructor                          |
| `@ordered`         | Generates comparison operators              |
| `@copyable`        | Generates copy operations                   |
| `@enum`            | Strongly-typed enumeration                  |
| `@flag_enum`       | Bitmask enumeration                         |
| `@union`           | Discriminated union                         |

### Examples

```cpp2
// Basic type with constructors
Point: type = {
    x: int = 0;
    y: int = 0;
    
    // Default constructor
    operator=: (out this) = { }
    
    // Parameterized constructor
    operator=: (out this, x_: int, y_: int) = {
        x = x_;
        y = y_;
    }
    
    // Copy constructor/assignment
    operator=: (out this, that) = {
        x = that.x;
        y = that.y;
    }
}

// With metafunction
Color: @enum type = {
    red; green; blue;
}

// Interface (pure virtual)
Drawable: @interface type = {
    draw: (this);
}

// Inheritance
Circle: type : Drawable = {
    radius: f64 = 0.0;
    draw: (override this) = { /* ... */ }
}
```

---

## Statements

```ebnf
(* ========================================================================== *)
(* STATEMENTS                                                                 *)
(* ========================================================================== *)

statement           ::= block_statement
                      | if_statement
                      | loop_statement
                      | switch_statement
                      | inspect_statement
                      | return_statement
                      | throw_statement
                      | try_statement
                      | contract_statement
                      | expression_statement
                      | declaration
                      | ";" 
                      ;

block_statement     ::= "{" { statement } "}" ;
```

### If Statement

```ebnf
if_statement        ::= "if" [ "constexpr" ] expression block_statement 
                        [ "else" ( block_statement | if_statement ) ] ;

(* Note: if-expression syntax in expressions: if a > b then a else b *)
```

### Loop Statements

```ebnf
(* Loops support labeled breaks/continues and statement-local parameters *)
loop_statement      ::= [ identifier ":" ]   (* label *)
                        [ loop_parameters ]  (* statement-local vars *)
                        loop_kind ;

loop_parameters     ::= "(" param_declaration { "," param_declaration } ")" ;
param_declaration   ::= [ "copy" | "move" ] identifier [ ":=" expression ] ;

loop_kind           ::= while_loop | for_loop | do_loop ;

while_loop          ::= "while" expression [ "next" expression ] block_statement ;

for_loop            ::= "for" expression [ "next" expression ] "do" "(" parameter ")" block_statement ;

do_loop             ::= "do" block_statement [ "next" expression ] "while" expression ";" ;
```

### Switch Statement

```ebnf
switch_statement    ::= "switch" expression "{" { switch_case } "}" ;
switch_case         ::= "case" expression ":" statement
                      | "default" ":" statement
                      ;
```

### Control Flow

```ebnf
return_statement    ::= "return" [ expression ] ";" ;

throw_statement     ::= "throw" [ expression ] ";" ;

try_statement       ::= "try" block_statement { catch_clause } ;
catch_clause        ::= "catch" "(" [ type_specifier [ identifier ] | "..." ] ")" block_statement ;

contract_statement  ::= ( "assert" | "pre" | "post" ) contract_body ";" ;
```

### Examples

```cpp2
// If statement (braces required)
if x > 0 {
    std::cout << "positive\n";
} else if x < 0 {
    std::cout << "negative\n";
} else {
    std::cout << "zero\n";
}

// While with next clause and statement-local parameter
(copy i := 0) while i < 10 next i++ {
    std::cout << i << "\n";
}

// For-range loop (Cpp2 syntax)
for vec next idx++ do (item) {
    std::cout << idx << ": " << item << "\n";
}

// Labeled loop with break
outer: while true {
    inner: while true {
        break outer;
    }
}

// Switch
switch value {
    case 1: handle_one();
    case 2: handle_two();
    default: handle_default();
}
```

---

## Expressions

```ebnf
(* ========================================================================== *)
(* EXPRESSIONS                                                                *)
(* ========================================================================== *)

expression          ::= assignment_expression ;

assignment_expression 
                    ::= ternary_expression 
                        [ assignment_operator assignment_expression ] ;

assignment_operator ::= "=" | "+=" | "-=" | "*=" | "/=" | "%=" 
                      | "<<=" | ">>=" | "&=" | "|=" | "^=" ;
```

### Ternary Expression

```ebnf
ternary_expression  ::= logical_or_expression 
                        [ "?" expression ":" ternary_expression ] ;
```

### Logical Expressions

```ebnf
logical_or_expression
                    ::= logical_and_expression { "||" logical_and_expression } ;

logical_and_expression
                    ::= bitwise_or_expression { "&&" bitwise_or_expression } ;
```

### Bitwise Expressions

```ebnf
bitwise_or_expression
                    ::= bitwise_xor_expression { "|" bitwise_xor_expression } ;

bitwise_xor_expression
                    ::= bitwise_and_expression { "^" bitwise_and_expression } ;

bitwise_and_expression
                    ::= equality_expression { "&" equality_expression } ;
```

### Comparison Expressions

```ebnf
equality_expression ::= comparison_expression { ( "==" | "!=" ) comparison_expression } ;

comparison_expression
                    ::= range_expression { ( "<" | ">" | "<=" | ">=" ) range_expression } ;
```

### Range and Shift Expressions

```ebnf
range_expression    ::= shift_expression { ( "..=" | "..<" ) shift_expression } ;

shift_expression    ::= addition_expression { ( "<<" | ">>" ) addition_expression } ;
```

### Arithmetic Expressions

```ebnf
addition_expression ::= multiplication_expression { ( "+" | "-" ) multiplication_expression } ;

multiplication_expression
                    ::= prefix_expression { ( "*" | "/" | "%" ) prefix_expression } ;
```

### Prefix Expressions

```ebnf
prefix_expression   ::= postfix_expression
                      | "await" prefix_expression
                      | "launch" prefix_expression      (* fire-and-forget coroutine *)
                      | "select" select_body            (* channel select *)
                      | ( "move" | "forward" | "copy" ) prefix_expression
                      | ( "+" | "-" | "!" | "~" | "++" | "--" | "&" | "*" ) prefix_expression
                      ;
```

### Postfix Expressions

```ebnf
postfix_expression  ::= primary_expression { postfix_op } ;

postfix_op          ::= "(" [ argument_list ] ")"                    (* call *)
                      | "<" template_arg_list ">" [ "(" [ argument_list ] ")" ]  (* template call *)
                      | "." member_access                            (* member/UFCS *)
                      | ".." identifier [ "(" [ argument_list ] ")" ] (* explicit non-UFCS *)
                      | "[" expression "]"                           (* subscript *)
                      | "*"                                          (* postfix deref *)
                      | "&"                                          (* postfix address-of *)
                      | "++" | "--"                                  (* postfix inc/dec *)
                      | "$"                                          (* capture operator *)
                      | "..."                                        (* pack expansion *)
                      | "as" type_specifier                          (* safe cast *)
                      | "is" is_pattern                              (* type/value test *)
                      | "::" identifier                              (* scope resolution *)
                      ;

member_access       ::= identifier [ "::" identifier ]* [ "(" [ argument_list ] ")" ] ;

is_pattern          ::= type_specifier                               (* type pattern *)
                      | "(" expression ")"                           (* value predicate *)
                      | literal                                      (* literal pattern *)
                      ;

argument_list       ::= argument { "," argument } [ "," ] ;
argument            ::= [ argument_qualifier ] expression ;
argument_qualifier  ::= "out" | "inout" | "move" | "forward" | "in_ref" | "forward_ref" ;
```

### Primary Expressions

```ebnf
primary_expression  ::= literal
                      | identifier_expression
                      | "this"
                      | "_"                                  (* wildcard/discard *)
                      | "(" expression_or_tuple ")"
                      | "[" list_or_lambda "]"
                      | "{" struct_initializer "}"
                      | "inspect" inspect_expression
                      | "@" identifier [ "(" [ argument_list ] ")" ]  (* metafunction call *)
                      | ":" function_expression              (* lambda *)
                      | ":" type_specifier "=" expression    (* typed construction *)
                      ;

literal             ::= "true" | "false"
                      | integer_literal [ identifier ]       (* with optional UDL suffix *)
                      | float_literal [ identifier ]
                      | string_literal
                      | char_literal
                      ;

identifier_expression
                    ::= [ "::" ] identifier { "::" identifier } ;
                    (* Also allows contextual keywords as identifiers:
                       func, type, namespace, in, out, inout, copy, move, forward *)

expression_or_tuple ::= (* empty - () for default construction *)
                      | expression
                      | expression "," expression { "," expression }  (* tuple *)
                      | fold_expression
                      | qualified_argument_list                       (* (out y) constructor call *)
                      ;

qualified_argument_list
                    ::= argument_qualifier expression { "," [ argument_qualifier ] expression } ;

fold_expression     ::= "..." fold_operator expression                (* left fold *)
                      | expression fold_operator "..." fold_operator expression  (* binary fold *)
                      ;

fold_operator       ::= "+" | "-" | "*" | "/" | "%" | "^" | "&" | "|"
                      | "&&" | "||" | "," | "<" | ">" | "<=" | ">="
                      | "==" | "!=" | "<<" | ">>"
                      ;

list_or_lambda      ::= [ expression { "," expression } ]            (* list literal *)
                      | cpp1_lambda                                   (* [captures](params){} *)
                      ;

struct_initializer  ::= identifier "{" field_init { "," field_init } "}" ;
field_init          ::= identifier ":" expression ;
```

### Function Expression (Lambda)

```ebnf
function_expression ::= [ "<" template_param_list ">" ]              (* templated lambda *)
                        "(" [ parameter_list ] ")"
                        [ "->" type_specifier ]
                        function_body ;

function_body       ::= "=" expression                               (* expression body *)
                      | "==" expression                              (* constexpr expression *)
                      | "=" block_statement
                      | block_statement
                      ;

(* C++1-style lambda for interop *)
cpp1_lambda         ::= "[" [ capture_list ] "]" "(" [ cpp1_param_list ] ")"
                        [ "->" type_specifier ] block_statement ;

capture_list        ::= capture { "," capture } ;
capture             ::= "=" | "&" | "this" | identifier | "&" identifier ;

cpp1_param_list     ::= cpp1_param { "," cpp1_param } ;
cpp1_param          ::= [ "auto" | type_specifier ] identifier ;
```

### Operator Precedence (highest to lowest)

| Level | Operators                              | Associativity |
|-------|----------------------------------------|---------------|
| 1     | `()` `[]` `.` `..` `::` `++` `--` `*` `&` `$` `...` `as` `is` | Left |
| 2     | Prefix: `++` `--` `+` `-` `!` `~` `*` `&` `await` `move` `forward` `copy` | Right |
| 3     | `*` `/` `%`                            | Left          |
| 4     | `+` `-`                                | Left          |
| 5     | `<<` `>>`                              | Left          |
| 6     | `..=` `..<`                            | Left          |
| 7     | `<` `>` `<=` `>=`                      | Left          |
| 8     | `==` `!=`                              | Left          |
| 9     | `&` (bitwise)                          | Left          |
| 10    | `^`                                    | Left          |
| 11    | `\|`                                   | Left          |
| 12    | `&&`                                   | Left          |
| 13    | `\|\|`                                 | Left          |
| 14    | `?` `:`                                | Right         |
| 15    | `=` `+=` `-=` `*=` `/=` etc.           | Right         |

---

## Pattern Matching (inspect)

```ebnf
inspect_expression  ::= expression [ "->" type_specifier ] 
                        "{" { inspect_arm } "}" ;

inspect_arm         ::= pattern "=>" statement ;

pattern             ::= "_"                                  (* wildcard *)
                      | identifier                           (* binding *)
                      | identifier ":" type_specifier        (* typed binding *)
                      | "is" type_specifier                  (* type pattern *)
                      | "is" "(" expression ")"              (* value predicate *)
                      | "is" literal                         (* literal pattern *)
                      | expression                           (* value pattern *)
                      ;
```

### Examples

```cpp2
// Type-based matching
result := inspect shape -> std::string {
    is Circle    => "circle";
    is Rectangle => "rectangle";
    is _         => "unknown";
};

// Value-based matching
name := inspect n {
    is 0 => "zero";
    is 1 => "one";
    is 2 => "two";
    is _ => "many";
};

// With predicate guards
classify := inspect x {
    is (:(v) = v < 0;) => "negative";
    is (:(v) = v > 0;) => "positive";
    is _               => "zero";
};
```

---

## Type Specifiers

```ebnf
(* ========================================================================== *)
(* TYPE SPECIFIERS                                                            *)
(* ========================================================================== *)

type_specifier      ::= function_type
                      | pointer_type
                      | qualified_type
                      ;

function_type       ::= "(" [ param_type_list ] ")" "->" [ return_qualifier ] type_specifier ;

param_type_list     ::= [ param_kind ] type_specifier { "," [ param_kind ] type_specifier } ;
param_kind          ::= "in" | "inout" | "out" | "copy" | "move" | "forward" ;

return_qualifier    ::= "forward" | "move" ;

pointer_type        ::= "*" [ "const" ] type_specifier ;     (* prefix pointer *)

qualified_type      ::= basic_type { "::" identifier [ template_args ] } 
                        { "*" | "&" } ;                      (* postfix pointer/ref *)

basic_type          ::= [ type_modifier { type_modifier } ] type_name [ template_args ]
                      | "auto"
                      | "_" [ "is" type_constraint ]
                      | "type"
                      | "decltype" "(" expression ")"
                      | "const" type_specifier
                      ;

type_modifier       ::= "unsigned" | "signed" | "short" | "long" ;

type_name           ::= identifier | "int" | "char" | "double" | "float" | "void" | "bool" ;

template_args       ::= "<" [ template_arg { "," template_arg } ] ">" ;
template_arg        ::= type_specifier
                      | expression                           (* NTTP *)
                      | identifier "(" expression ")"        (* CPP2_TYPEOF etc. *)
                      ;
```

### Built-in Type Aliases

| Cpp2 Type | C++ Equivalent    |
|-----------|-------------------|
| `i8`      | `std::int8_t`     |
| `i16`     | `std::int16_t`    |
| `i32`     | `std::int32_t`    |
| `i64`     | `std::int64_t`    |
| `u8`      | `std::uint8_t`    |
| `u16`     | `std::uint16_t`   |
| `u32`     | `std::uint32_t`   |
| `u64`     | `std::uint64_t`   |
| `f32`     | `float`           |
| `f64`     | `double`          |

---

## Special Member Functions

The `operator=:` pattern unifies construction, assignment, and destruction:

```cpp2
MyType: type = {
    data: int;
    
    // Default constructor (out this = uninitialized)
    operator=: (out this) = { data = 0; }
    
    // Parameterized constructor
    operator=: (out this, value: int) = { data = value; }
    
    // Copy constructor/assignment (that = source)
    operator=: (out this, that) = { data = that.data; }
    
    // Move constructor/assignment
    operator=: (out this, move that) = { data = that.data; }
    
    // Destructor (move this = ending lifetime)
    operator=: (move this) = { /* cleanup */ }
}
```

---

## UFCS (Uniform Function Call Syntax)

Any free function `f(x, args...)` can be called as `x.f(args...)`:

```cpp2
// These are equivalent:
result1 := std::ssize(vec);
result2 := vec.ssize();

// Enables fluent APIs:
result := data
    .filter(:(x) = x > 0)
    .map(:(x) = x * 2)
    .sum();

// Explicit non-UFCS with ..
ptr..reset();  // Forces member call, no UFCS rewrite
```

---

## Concurrency (Kotlin-style)

```ebnf
(* Concurrency expressions - async/await patterns *)
await_expression    ::= "await" expression ;
spawn_expression    ::= "launch" expression ;

select_body         ::= "{" { select_case } [ default_case ] "}" ;
select_case         ::= "onSend" "(" identifier "," expression ")" block_statement
                      | "onRecv" "(" identifier ")" block_statement
                      ;
default_case        ::= "default" "=>" expression ;

channel_send        ::= identifier "<-" expression ;
channel_recv        ::= "<-" identifier ;
```

---

## Disambiguation Notes

The parser uses the following strategies for ambiguous constructs:

### `<` Disambiguation
- When `<` follows an identifier, heuristics determine template vs. comparison
- Template parsing is speculative with rollback on failure

### `:` Disambiguation
- `identifier:` at statement start → labeled statement or declaration
- `? expr :` → ternary expression
- `case expr:` → switch case
- Otherwise → declaration separator

### Postfix `*` and `&`
Postfix dereference/address-of is recognized when followed by:
- `;` `,` `)` `]` `}` `.` `++` `--` `$` `*` `&`
- Binary operators: `+` `-` `/` `=` `==` `<` `>` `<<` `>>`
- Compound assignment: `+=` `-=` etc.
- `is` `as` keywords

---

## Syntactic Notes

- **Unified Syntax**: `name : kind = value` is the core pattern
- **Postfix Operators**: `*` (deref) and `&` (address) are postfix: `ptr*` not `*ptr`
- **UFCS**: `x.f()` tries member first, then free function `f(x)`
- **Implicit/Contextual Keywords**: `virtual`, `override`, `in`, `out` etc. are identifiers in most contexts
- **Braces Required**: Control flow bodies always need `{}`
- **Contracts**: `pre()` `post()` `assert()` with optional template parameter like `<bounds_safety>`
- **Captures**: `val$` captures value at evaluation point

---

## Version

This grammar specification corresponds to cppfort v0.1.0 and is extracted from the actual
parser implementation. It accurately reflects all parsing rules including:

- Ternary operator `?:`
- All bitwise operators (`&`, `|`, `^`, `<<`, `>>`)
- Range expressions (`..=`, `..<`)
- Postfix `is` and `as` operators
- Statement-local parameters `(copy i := 0) while ...`
- Named return values `-> (name: type)`
- Contract clauses with template parameters
- C++1 lambda syntax `[captures](params){}`
- Concurrency primitives (await, launch, select)
- Fold expressions
