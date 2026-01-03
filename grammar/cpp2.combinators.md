# Cpp2 Grammar: Orthogonal Combinator Specification

This grammar defines Cpp2 syntax using orthogonal parser combinators.
Each rule is a composition of primitive combinators that can be directly
implemented with the `include/combinators/` infrastructure.

---

## Primitive Combinators

```
// Sequencing
seq(a, b, ...)      → a then b then ...
                    → Parser<std::tuple<A, B, ...>>

// Alternation  
alt(a, b, ...)      → a or b or ...
                    → Parser<std::variant<A, B, ...>>

// Repetition
many(p)             → zero or more p
                    → Parser<std::vector<T>>

many1(p)            → one or more p
                    → Parser<std::vector<T>>

opt(p)              → zero or one p
                    → Parser<std::optional<T>>

// Lookahead
peek(p)             → match p without consuming
                    → Parser<bool>

not_peek(p)         → fail if p matches
                    → Parser<()>

// Terminals
token(T)            → match token type T
                    → Parser<Token>

keyword(K)          → match keyword K
                    → Parser<Token>

punct(P)            → match punctuation P
                    → Parser<Token>

// Transformations
map(p, f)           → apply f to result of p
                    → Parser<U> where f: T -> U

filter(p, pred)     → fail if pred(result) is false
                    → Parser<T>

// Combinators
sep_by(p, delim)    → p (delim p)*
                    → Parser<std::vector<T>>

sep_by1(p, delim)   → p (delim p)+
                    → Parser<std::vector<T>>

between(l, p, r)    → l p r, return p
                    → Parser<T>

// Error handling
try_(p)             → backtrack on failure
                    → Parser<T>

label(p, msg)       → replace error message
                    → Parser<T>

recover(p, sync)    → on failure, skip to sync point
                    → Parser<T>
```

---

## Lexical Combinators

```
identifier = token(Identifier)
           | filter(token(Keyword), is_contextual_keyword)

integer    = token(IntegerLiteral)
float_     = token(FloatLiteral)
string     = token(StringLiteral)
char_      = token(CharLiteral)

// User-defined literal: 10s, 1.5ms
udl        = seq(alt(integer, float_), opt(identifier))
           |> map(concat_lexemes)
```

---

## Top-Level Structure

```
translation_unit = many(declaration)

declaration = alt(
    namespace_decl,
    type_decl,
    function_decl,
    variable_decl,
    using_decl,
    import_decl,
    cpp1_passthrough,
    statement
)

// Unified declaration: name : kind = value
unified_decl = seq(
    identifier,
    punct(':'),
    declaration_body
)
```

---

## Variable Declarations

```
variable_decl = alt(
    // Unified: name : type = value
    seq(identifier, punct(':'), type_spec, opt(initializer), punct(';')),
    
    // Type-deduced: name := value  
    seq(identifier, punct(':='), expression, punct(';')),
    
    // Keyword: let/const name : type = value
    seq(alt(keyword('let'), keyword('const')), 
        identifier, 
        opt(seq(punct(':'), type_spec)),
        alt(punct('='), punct('==')),
        expression,
        punct(';'))
)

initializer = alt(
    seq(punct('='), expression),
    seq(punct('=='), expression)   // compile-time
)
```

---

## Function Declarations

```
function_decl = seq(
    identifier,
    punct(':'),
    opt(template_params),
    param_list,
    opt(throws_spec),
    opt(return_spec),
    opt(requires_clause),
    opt(contracts),
    function_body
)

param_list = between(
    punct('('),
    sep_by(parameter, punct(',')),
    punct(')')
)

parameter = seq(
    many(param_qualifier),
    param_name,
    opt(seq(punct(':'), type_spec)),
    opt(seq(punct('='), expression))
)

param_qualifier = alt(
    keyword('in'),
    keyword('copy'),
    keyword('inout'),
    keyword('out'),
    keyword('move'),
    keyword('forward'),
    keyword('in_ref'),
    keyword('forward_ref'),
    keyword('virtual'),
    keyword('override'),
    keyword('implicit')
)

param_name = alt(identifier, keyword('this'), keyword('that'), punct('_'))

throws_spec = keyword('throws')

return_spec = seq(
    punct('->'),
    alt(
        // Named returns: -> (x: int, y: int)
        between(punct('('), sep_by1(named_return, punct(',')), punct(')')),
        // Simple return
        seq(opt(alt(keyword('forward'), keyword('move'))), type_spec)
    )
)

named_return = seq(identifier, punct(':'), type_spec, opt(seq(punct('='), expression)))

requires_clause = seq(
    keyword('requires'),
    requires_expr   // stops at '=' or '=='
)

contracts = many1(contract_clause)

contract_clause = seq(
    alt(keyword('pre'), keyword('post'), keyword('assert')),
    opt(between(punct('<'), identifier, punct('>'))),
    contract_body
)

contract_body = alt(
    seq(punct(':'), expression),
    between(punct('('), seq(expression, opt(seq(punct(','), string))), punct(')'))
)

function_body = alt(
    seq(punct('='), expression, punct(';')),
    seq(punct('=='), expression, punct(';')),
    seq(punct('='), block_stmt),
    block_stmt,
    punct(';')   // forward declaration
)
```

---

## Type Declarations

```
type_decl = seq(
    identifier,
    punct(':'),
    many(metafunction),
    keyword('type'),
    opt(template_params),
    opt(base_types),
    opt(requires_clause),
    punct('='),
    type_body
)

metafunction = seq(punct('@'), identifier, opt(template_args))

base_types = seq(punct(':'), sep_by1(type_spec, punct(',')))

type_body = between(
    punct('{'),
    many(type_member),
    punct('}')
)

type_member = seq(
    opt(access_spec),
    alt(
        base_class_decl,
        operator_decl,
        function_decl,
        variable_decl
    )
)

access_spec = alt(keyword('public'), keyword('protected'), keyword('private'))

base_class_decl = seq(
    keyword('this'),
    punct(':'),
    type_spec,
    opt(initializer),
    punct(';')
)
```

---

## Operator Declarations

```
operator_decl = seq(
    keyword('operator'),
    operator_name,
    punct(':'),
    opt(template_params),
    param_list,
    opt(return_spec),
    opt(requires_clause),
    function_body
)

operator_name = alt(
    punct('='),
    punct('<=>'),
    punct('=='), punct('!='),
    punct('<'), punct('>'), punct('<='), punct('>='),
    punct('+'), punct('-'), punct('*'), punct('/'), punct('%'),
    punct('+='), punct('-='), punct('*='), punct('/='), punct('%='),
    punct('<<'), punct('>>'), punct('<<='), punct('>>='),
    punct('&'), punct('|'), punct('^'),
    punct('&='), punct('|='), punct('^='),
    punct('~'), punct('!'),
    punct('&&'), punct('||'),
    punct('++'), punct('--'),
    seq(punct('['), punct(']')),
    seq(punct('('), punct(')')),
    punct('->'), punct('->*'),
    keyword('new'), keyword('delete'),
    string   // user-defined literal
)
```

---

## Namespace Declarations

```
namespace_decl = seq(
    identifier,
    punct(':'),
    keyword('namespace'),
    alt(
        // Alias: N: namespace == std::io;
        seq(punct('=='), qualified_name, punct(';')),
        // Body: N: namespace = { ... }
        seq(punct('='), between(punct('{'), many(declaration), punct('}')))
    )
)
```

---

## Using & Import

```
using_decl = seq(
    keyword('using'),
    alt(
        // Alias: using name = target;
        seq(identifier, punct('='), qualified_name, punct(';')),
        // Path: using std::cout;
        seq(qualified_name, punct(';')),
        // Namespace: using namespace std;
        seq(keyword('namespace'), identifier, punct(';'))
    )
)

import_decl = seq(keyword('import'), identifier, punct(';'))
```

---

## Template Parameters

```
template_params = between(
    punct('<'),
    sep_by(template_param, punct(',')),
    punct('>')
)

template_param = seq(
    alt(identifier, punct('_')),
    opt(punct('...')),
    opt(seq(punct(':'), type_constraint)),
    opt(seq(punct('='), alt(type_spec, expression)))
)

type_constraint = alt(type_spec, punct('_'), keyword('type'))

template_args = between(
    punct('<'),
    sep_by(template_arg, punct(',')),
    punct('>')
)

template_arg = alt(
    try_(type_spec),
    expression
)
```

---

## Statements

```
statement = alt(
    block_stmt,
    if_stmt,
    while_stmt,
    for_stmt,
    do_stmt,
    switch_stmt,
    inspect_stmt,
    return_stmt,
    break_stmt,
    continue_stmt,
    try_stmt,
    throw_stmt,
    contract_stmt,
    static_assert_stmt,
    labeled_stmt,
    local_decl_stmt,
    expr_stmt
)

block_stmt = between(punct('{'), many(statement), punct('}'))

// Local variable in statement position
local_decl_stmt = alt(
    seq(identifier, punct(':'), type_spec, opt(initializer), punct(';')),
    seq(identifier, punct(':='), expression, punct(';'))
)

expr_stmt = seq(expression, punct(';'))
```

---

## Control Flow Statements

```
if_stmt = seq(
    keyword('if'),
    opt(keyword('constexpr')),
    expression,
    block_stmt,
    opt(seq(keyword('else'), alt(block_stmt, if_stmt)))
)

while_stmt = seq(
    opt(loop_label),
    opt(loop_params),
    keyword('while'),
    expression,
    opt(seq(keyword('next'), expression)),
    block_stmt
)

for_stmt = seq(
    opt(loop_label),
    opt(loop_params),
    keyword('for'),
    expression,
    opt(seq(keyword('next'), expression)),
    keyword('do'),
    between(punct('('), parameter, punct(')')),
    block_stmt
)

do_stmt = seq(
    opt(loop_label),
    opt(loop_params),
    keyword('do'),
    block_stmt,
    opt(seq(keyword('next'), expression)),
    keyword('while'),
    expression,
    punct(';')
)

loop_label = seq(identifier, punct(':'))

loop_params = between(
    punct('('),
    sep_by1(loop_param, punct(',')),
    punct(')')
)

loop_param = seq(
    opt(alt(keyword('copy'), keyword('move'))),
    identifier,
    opt(seq(punct(':='), expression))
)

switch_stmt = seq(
    keyword('switch'),
    expression,
    between(punct('{'), many(switch_case), punct('}'))
)

switch_case = alt(
    seq(keyword('case'), expression, punct(':'), statement),
    seq(keyword('default'), punct(':'), statement)
)

return_stmt = seq(keyword('return'), opt(expression), punct(';'))

break_stmt = seq(keyword('break'), opt(identifier), punct(';'))

continue_stmt = seq(keyword('continue'), opt(identifier), punct(';'))

throw_stmt = seq(keyword('throw'), opt(expression), punct(';'))

try_stmt = seq(
    keyword('try'),
    block_stmt,
    many(catch_clause)
)

catch_clause = seq(
    keyword('catch'),
    between(punct('('), catch_param, punct(')')),
    block_stmt
)

catch_param = alt(
    punct('...'),
    seq(type_spec, opt(identifier))
)

contract_stmt = seq(
    alt(keyword('assert'), keyword('pre'), keyword('post')),
    opt(between(punct('<'), identifier, punct('>'))),
    expression,
    punct(';')
)

static_assert_stmt = seq(
    keyword('static_assert'),
    between(punct('('), seq(expression, opt(seq(punct(','), string))), punct(')')),
    punct(';')
)

labeled_stmt = seq(identifier, punct(':'), alt(while_stmt, for_stmt, do_stmt))
```

---

## Expressions (Precedence Climbing)

Expression parsing uses precedence climbing with these levels:

```
expression = assignment_expr

// Level 15: Assignment (right-associative)
assignment_expr = seq(
    pipeline_expr,
    opt(seq(assignment_op, assignment_expr))
)

assignment_op = alt(
    punct('='),
    punct('+='), punct('-='), punct('*='), punct('/='), punct('%='),
    punct('<<='), punct('>>='),
    punct('&='), punct('|='), punct('^=')
)

// Level 14: Pipeline (left-associative)
pipeline_expr = seq(
    ternary_expr,
    many(seq(punct('|>'), ternary_expr))
) |> map(fold_left_to_pipeline)

// Level 13: Ternary (right-associative)
ternary_expr = seq(
    logical_or_expr,
    opt(seq(punct('?'), expression, punct(':'), ternary_expr))
)

// Level 12: Logical OR (left-associative)
logical_or_expr = sep_by1(logical_and_expr, punct('||'))
                |> map(fold_left_to_binary)

// Level 11: Logical AND (left-associative)
logical_and_expr = sep_by1(bitwise_or_expr, punct('&&'))
                 |> map(fold_left_to_binary)

// Level 10: Bitwise OR (left-associative)
bitwise_or_expr = sep_by1(bitwise_xor_expr, punct('|'))
                |> map(fold_left_to_binary)

// Level 9: Bitwise XOR (left-associative)
bitwise_xor_expr = sep_by1(bitwise_and_expr, punct('^'))
                 |> map(fold_left_to_binary)

// Level 8: Bitwise AND (left-associative)
bitwise_and_expr = sep_by1(equality_expr, punct('&'))
                 |> map(fold_left_to_binary)

// Level 7: Equality (left-associative)
equality_expr = seq(
    comparison_expr,
    many(seq(alt(punct('=='), punct('!=')), comparison_expr))
) |> map(fold_left_to_binary)

// Level 6: Comparison (left-associative)
comparison_expr = seq(
    range_expr,
    many(seq(comparison_op, range_expr))
) |> map(fold_left_to_binary)

comparison_op = alt(punct('<'), punct('>'), punct('<='), punct('>='), punct('<=>'))

// Level 5: Range (left-associative)
range_expr = seq(
    shift_expr,
    many(seq(alt(punct('..='), punct('..<')), shift_expr))
) |> map(fold_left_to_range)

// Level 4: Shift (left-associative)
shift_expr = seq(
    additive_expr,
    many(seq(alt(punct('<<'), punct('>>')), additive_expr))
) |> map(fold_left_to_binary)

// Level 3: Additive (left-associative)
additive_expr = seq(
    multiplicative_expr,
    many(seq(alt(punct('+'), punct('-')), multiplicative_expr))
) |> map(fold_left_to_binary)

// Level 2: Multiplicative (left-associative)
multiplicative_expr = seq(
    prefix_expr,
    many(seq(alt(punct('*'), punct('/'), punct('%')), prefix_expr))
) |> map(fold_left_to_binary)

// Level 1: Prefix (right-associative)
prefix_expr = alt(
    seq(keyword('await'), prefix_expr)   |> map(to_await),
    seq(keyword('launch'), prefix_expr)  |> map(to_spawn),
    seq(keyword('select'), select_body)  |> map(to_select),
    seq(keyword('move'), prefix_expr)    |> map(to_move),
    seq(keyword('forward'), prefix_expr) |> map(to_forward),
    seq(keyword('copy'), prefix_expr)    |> map(to_copy),
    seq(prefix_op, prefix_expr)          |> map(to_unary_prefix),
    postfix_expr
)

prefix_op = alt(
    punct('+'), punct('-'), punct('!'), punct('~'),
    punct('++'), punct('--'), punct('&'), punct('*')
)

// Level 0: Postfix (left-associative)
postfix_expr = seq(primary_expr, many(postfix_op))
             |> map(fold_left_to_postfix)

postfix_op = alt(
    // Call: (args)
    between(punct('('), sep_by(argument, punct(',')), punct(')'))
    |> map(to_call),
    
    // Template instantiation: <types>(args)?
    seq(template_args, opt(between(punct('('), sep_by(argument, punct(',')), punct(')'))))
    |> map(to_template_call),
    
    // Member access: .member
    seq(punct('.'), identifier, opt(call_args))
    |> map(to_member),
    
    // Explicit non-UFCS: ..member
    seq(punct('..'), identifier, opt(call_args))
    |> map(to_explicit_member),
    
    // Subscript: [index]
    between(punct('['), expression, punct(']'))
    |> map(to_subscript),
    
    // Postfix operators
    punct('*')   |> map(to_postfix_deref),
    punct('&')   |> map(to_postfix_addr),
    punct('++')  |> map(to_postfix_inc),
    punct('--')  |> map(to_postfix_dec),
    punct('$')   |> map(to_capture),
    punct('...') |> map(to_pack_expand),
    
    // Type operations
    seq(keyword('as'), type_spec) |> map(to_as_cast),
    seq(keyword('is'), is_pattern) |> map(to_is_test),
    
    // Scope resolution
    seq(punct('::'), identifier) |> map(to_scope_access)
)

is_pattern = alt(
    type_spec,
    between(punct('('), expression, punct(')')),
    literal
)

argument = seq(opt(argument_qualifier), expression)

argument_qualifier = alt(
    keyword('out'), keyword('inout'), keyword('move'),
    keyword('forward'), keyword('in_ref'), keyword('forward_ref')
)

call_args = between(punct('('), sep_by(argument, punct(',')), punct(')'))
```

---

## Primary Expressions

```
primary_expr = alt(
    literal,
    identifier_expr,
    keyword('this'),
    keyword('that'),
    punct('_'),
    grouped_expr,
    list_expr,
    struct_init,
    inspect_expr,
    metafunction_call,
    lambda_expr,
    cpp1_lambda
)

literal = alt(
    keyword('true')  |> map(to_bool_lit),
    keyword('false') |> map(to_bool_lit),
    udl              |> map(to_number_lit),
    string           |> map(to_string_lit),
    char_            |> map(to_char_lit)
)

identifier_expr = seq(
    opt(punct('::')),
    sep_by1(identifier, punct('::'))
) |> map(to_qualified_name)

grouped_expr = between(
    punct('('),
    alt(
        // Empty: ()
        peek(punct(')')) |> map(to_unit),
        // Tuple: (a, b, c)
        seq(expression, punct(','), sep_by1(expression, punct(','))) |> map(to_tuple),
        // Fold: (... op expr) or (expr op ... op expr)
        fold_expr,
        // Qualified args: (out x)
        seq(argument_qualifier, expression) |> map(to_qualified_arg),
        // Simple grouping
        expression
    ),
    punct(')')
)

fold_expr = alt(
    // Unary left fold: (... op pack)
    seq(punct('...'), fold_op, expression) |> map(to_left_fold),
    // Unary right fold: (pack op ...)
    seq(expression, fold_op, punct('...')) |> map(to_right_fold),
    // Binary fold: (init op ... op pack)
    seq(expression, fold_op, punct('...'), fold_op, expression) |> map(to_binary_fold)
)

fold_op = alt(
    punct('+'), punct('-'), punct('*'), punct('/'), punct('%'),
    punct('^'), punct('&'), punct('|'),
    punct('&&'), punct('||'), punct(','),
    punct('<'), punct('>'), punct('<='), punct('>='),
    punct('=='), punct('!='), punct('<<'), punct('>>')
)

list_expr = between(
    punct('['),
    alt(
        cpp1_lambda_body,
        sep_by(expression, punct(','))
    ),
    punct(']')
)

struct_init = between(
    punct('{'),
    sep_by(field_init, punct(',')),
    punct('}')
)

field_init = seq(identifier, punct(':'), expression)
```

---

## Lambda Expressions

```
// Cpp2 lambda: :(params) -> ret = body
lambda_expr = seq(
    punct(':'),
    opt(template_params),
    param_list,
    opt(return_spec),
    function_body
) |> map(to_lambda)

// C++1 lambda: [captures](params) -> ret { body }
cpp1_lambda = seq(
    between(punct('['), capture_list, punct(']')),
    between(punct('('), sep_by(cpp1_param, punct(',')), punct(')')),
    opt(seq(punct('->'), type_spec)),
    block_stmt
) |> map(to_cpp1_lambda)

capture_list = sep_by(capture, punct(','))

capture = alt(
    punct('='),
    punct('&'),
    keyword('this'),
    seq(punct('&'), identifier),
    identifier
)

cpp1_param = seq(opt(alt(keyword('auto'), type_spec)), identifier)

cpp1_lambda_body = seq(
    capture_list,
    punct(']'),
    between(punct('('), sep_by(cpp1_param, punct(',')), punct(')')),
    opt(seq(punct('->'), type_spec)),
    block_stmt
)
```

---

## Inspect (Pattern Matching)

```
inspect_stmt = seq(
    keyword('inspect'),
    expression,
    opt(seq(punct('->'), type_spec)),
    between(punct('{'), many(inspect_arm), punct('}'))
)

inspect_expr = seq(
    keyword('inspect'),
    expression,
    opt(seq(punct('->'), type_spec)),
    between(punct('{'), many(inspect_arm), punct('}'))
)

inspect_arm = seq(pattern, punct('=>'), statement)

pattern = alt(
    punct('_')                                    |> map(to_wildcard),
    seq(identifier, punct(':'), type_spec)        |> map(to_typed_binding),
    seq(keyword('is'), type_spec)                 |> map(to_type_pattern),
    seq(keyword('is'), between(punct('('), expression, punct(')'))) |> map(to_predicate),
    seq(keyword('is'), literal)                   |> map(to_literal_pattern),
    identifier                                    |> map(to_binding),
    expression                                    |> map(to_value_pattern)
)
```

---

## Type Specifiers

```
type_spec = alt(
    function_type,
    pointer_type,
    qualified_type
)

function_type = seq(
    between(punct('('), sep_by(param_type, punct(',')), punct(')')),
    punct('->'),
    opt(alt(keyword('forward'), keyword('move'))),
    type_spec
) |> map(to_function_type)

param_type = seq(opt(param_qualifier), type_spec)

// Prefix pointer: *T, *const T
pointer_type = seq(
    punct('*'),
    opt(keyword('const')),
    type_spec
) |> map(to_pointer_type)

qualified_type = seq(
    basic_type,
    many(seq(punct('::'), identifier, opt(template_args))),
    many(alt(punct('*'), punct('&')))
) |> map(to_qualified_type)

basic_type = alt(
    // Multi-word C types: unsigned long long
    seq(many1(type_modifier), opt(base_type_name)) |> map(to_builtin),
    
    // Simple type with optional template args
    seq(type_name, opt(template_args)) |> map(to_named_type),
    
    // Auto
    keyword('auto') |> map(to_auto),
    
    // Deduced with constraint: _ is Constraint
    seq(punct('_'), opt(seq(keyword('is'), type_spec))) |> map(to_deduced),
    
    // Type keyword
    keyword('type') |> map(to_type_type),
    
    // Decltype
    seq(keyword('decltype'), between(punct('('), expression, punct(')'))) |> map(to_decltype),
    
    // Const prefix
    seq(keyword('const'), type_spec) |> map(to_const)
)

type_modifier = alt(
    keyword('unsigned'), keyword('signed'), keyword('short'), keyword('long')
)

base_type_name = alt(
    keyword('int'), keyword('char'), keyword('double'), 
    keyword('float'), keyword('void'), keyword('bool')
)

type_name = alt(base_type_name, identifier)

qualified_name = seq(opt(punct('::')), sep_by1(identifier, punct('::')))
```

---

## Concurrency Combinators

```
select_body = between(
    punct('{'),
    seq(many(select_case), opt(default_case)),
    punct('}')
)

select_case = alt(
    seq(keyword('onSend'), between(punct('('), seq(identifier, punct(','), expression), punct(')')), block_stmt),
    seq(keyword('onRecv'), between(punct('('), identifier, punct(')')), block_stmt)
)

default_case = seq(keyword('default'), punct('=>'), expression)

channel_send = seq(identifier, punct('<-'), expression)
channel_recv = seq(punct('<-'), identifier)
```

---

## Semantic Actions (Map Functions)

```
// Binary expression folder
fold_left_to_binary = (parts) => {
    let [first, rest] = parts;
    return rest.fold(first, (acc, [op, rhs]) => BinaryExpr(acc, op, rhs));
}

// Pipeline folder: a |> f |> g => PipelineExpr
fold_left_to_pipeline = (parts) => {
    let [first, rest] = parts;
    return rest.fold(first, (acc, [_, rhs]) => PipelineExpr(acc, rhs));
}

// Postfix folder
fold_left_to_postfix = (parts) => {
    let [base, ops] = parts;
    return ops.fold(base, (acc, op) => apply_postfix(acc, op));
}

// Range folder
fold_left_to_range = (parts) => {
    let [first, rest] = parts;
    return rest.fold(first, (acc, [op, rhs]) => 
        RangeExpr(acc, rhs, op == '..='));
}

// AST constructors
to_call = (args) => (callee) => CallExpr(callee, args)
to_member = ([_, name, args]) => (obj) => MemberExpr(obj, name, args)
to_subscript = (index) => (arr) => SubscriptExpr(arr, index)
to_as_cast = ([_, type]) => (expr) => AsExpr(expr, type)
to_is_test = ([_, pattern]) => (expr) => IsExpr(expr, pattern)
to_unary_prefix = ([op, operand]) => UnaryExpr(op, operand, false)
to_postfix_deref = (_) => (expr) => UnaryExpr('*', expr, true)
to_postfix_addr = (_) => (expr) => UnaryExpr('&', expr, true)
```

---

## Disambiguation Strategies

### Template vs Comparison (`<`)

```
// Use try_ with backtracking
template_or_comparison = alt(
    try_(seq(
        template_args,
        peek(alt(punct('('), punct('::'), punct('>')))
    )) |> map(to_template),
    
    comparison_expr
)
```

### Colon Disambiguation (`:`)

```
// Context-dependent parsing
colon_context = alt(
    // After identifier at statement start: declaration or label
    seq(identifier, punct(':'), 
        alt(
            peek(alt(keyword('while'), keyword('for'), keyword('do'))) |> map(to_label),
            declaration_body |> map(to_declaration)
        )),
    
    // In ternary: expr ? expr : expr
    seq(punct('?'), expression, punct(':')) |> map(to_ternary_else),
    
    // In case: case expr:
    seq(keyword('case'), expression, punct(':')) |> map(to_case_label)
)
```

### Postfix `*` and `&`

```
// Lookahead determines postfix vs binary
postfix_deref_or_mult = alt(
    seq(
        punct('*'),
        peek(alt(
            punct(';'), punct(','), punct(')'), punct(']'), punct('}'),
            punct('.'), punct('++'), punct('--'), punct('$'),
            punct('*'), punct('&'), punct('+'), punct('-'), punct('/'),
            punct('='), punct('=='), punct('<'), punct('>'),
            keyword('is'), keyword('as')
        ))
    ) |> map(to_postfix_deref),
    
    seq(punct('*'), multiplicative_expr) |> map(to_binary_mult)
)
```

---

## Error Recovery

```
// Synchronization points
sync_declaration = skip_until(alt(
    punct(';'),
    keyword('func'),
    keyword('type'),
    keyword('namespace'),
    keyword('let'),
    keyword('const')
))

sync_statement = skip_until(alt(
    punct(';'),
    punct('}'),
    keyword('if'),
    keyword('while'),
    keyword('for'),
    keyword('return')
))

// Recoverable parsers
declaration_with_recovery = recover(declaration, sync_declaration)
statement_with_recovery = recover(statement, sync_statement)
```

---

## Combinator Implementation Mapping

| Grammar Combinator | Implementation                          |
|-------------------|-----------------------------------------|
| `seq(a, b, ...)`  | `combinators::seq<A, B, ...>`           |
| `alt(a, b, ...)`  | `combinators::alt<A, B, ...>`           |
| `many(p)`         | `combinators::many<P>`                  |
| `many1(p)`        | `combinators::many1<P>`                 |
| `opt(p)`          | `combinators::opt<P>`                   |
| `sep_by(p, d)`    | `combinators::sep_by<P, D>`             |
| `between(l, p, r)`| `combinators::between<L, P, R>`         |
| `try_(p)`         | `combinators::try_<P>`                  |
| `map(p, f)`       | `p \|> combinators::map(f)`             |
| `filter(p, pred)` | `p \|> combinators::filter(pred)`       |
| `peek(p)`         | `combinators::lookahead<P>`             |
| `not_peek(p)`     | `combinators::not_followed_by<P>`       |
| `token(T)`        | `combinators::token<T>`                 |
| `punct(P)`        | `combinators::punct<P>`                 |
| `keyword(K)`      | `combinators::keyword<K>`               |
| `label(p, msg)`   | `combinators::labeled<P>(msg)`          |
| `recover(p, s)`   | `combinators::recover<P, S>`            |

---

## Version

Combinator grammar specification v0.1.0
Corresponds to cppfort parser implementation as of 2025-01-02.
