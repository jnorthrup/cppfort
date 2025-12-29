#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <span>

#include "lexer.hpp"

namespace cpp2_transpiler {

// Forward declarations
struct Expression;
struct Statement;
struct Declaration;

// Markdown block metadata for CAS-linked modules
struct MarkdownBlockAttr {
    std::string sha256;      // 64-char lowercase hex SHA256 hash
    std::string content;     // Original markdown content
    std::string name;        // Optional identifier from ```name
    std::size_t line;        // Source location
    std::size_t column;

    MarkdownBlockAttr() = default;
    MarkdownBlockAttr(std::string s, std::string c, std::string n, std::size_t l, std::size_t col)
        : sha256(std::move(s)), content(std::move(c)), name(std::move(n)), line(l), column(col) {}
};

// Cpp2-specific parameter qualifiers (derived from corpus analysis)
enum class ParameterQualifier {
    None,
    In,         // in - immutable borrow (const T&)
    InOut,      // inout - mutable borrow with read/write (T&)
    Out,        // out - mutable borrow with write-before-return (T&)
    Move,       // move - ownership transfer (T&&)
    Forward,    // forward - perfect forwarding (T&&)
    Virtual,    // virtual - for polymorphic types
    Override    // override - for overriding base functions
};

// Type system
struct Type {
    enum class Kind {
        Builtin,
        UserDefined,
        Template,
        Pointer,
        Reference,
        Array,
        Function,
        Auto,
        Deduced,
        TemplateParameter,
        Optional,   // std::optional<T>
        Variant     // std::variant<Ts...>
    };

    // Null safety annotation (from null_safety analysis)
    enum class NullAnnotation {
        Unknown,        // Not yet determined
        Nullable,       // May be null (default for pointers)
        NonNull,        // Guaranteed not null
        NullChecked     // Null check has been performed
    };

    Kind kind;
    std::string name;
    std::vector<std::unique_ptr<Type>> template_args;
    std::unique_ptr<Type> pointee;
    std::shared_ptr<Expression> size;
    bool is_const = false;
    bool is_mut = false;

    // Optional and Variant type support
    std::unique_ptr<Type> base_type;                           // For Optional<T>
    std::vector<std::unique_ptr<Type>> alternatives;           // For Variant<Ts...>

    // Semantic attributes from Clang analysis
    NullAnnotation null_annotation = NullAnnotation::Unknown;  // Null safety
    bool requires_null_check = false;                          // Requires runtime null check
    bool is_view = false;                                       // Is a view/reference type

    Type(Kind k) : kind(k) {}
    ~Type();
};

// Expressions
struct Expression {
    enum class Kind {
        Literal,
        Identifier,
        Binary,
        Unary,
        Call,
        MemberAccess,
        Subscript,
        Ternary,
        Lambda,
        Is,
        As,
        StringInterpolation,
        Range,
        List,
        StructInitializer,
        MetafunctionCall,
        ContractExpression,
        InspectExpr,  // Cpp2 inspect expression
        // Concurrency expressions
        Await,
        Spawn,
        ChannelSend,
        ChannelRecv,
        ChannelSelect
    };

    Kind kind;
    std::size_t line;

    // Semantic attributes from Clang analysis
    bool is_definite_last_use = false;           // Marked for move optimization
    bool has_null_check = false;                  // Null check was inserted
    bool has_bounds_check = false;                // Bounds check was inserted
    std::string inferred_type;                    // Type deduced by Clang
    std::string source_location;                  // Source location for diagnostics

    Expression(Kind k, std::size_t l) : kind(k), line(l) {}
    virtual ~Expression() = default;
};

struct LiteralExpression : Expression {
    std::variant<int64_t, double, bool, std::string, char> value;
    std::unique_ptr<Type> type;

    LiteralExpression(int64_t v, std::size_t l)
        : Expression(Kind::Literal, l), value(v) {}
    LiteralExpression(double v, std::size_t l)
        : Expression(Kind::Literal, l), value(v) {}
    LiteralExpression(bool v, std::size_t l)
        : Expression(Kind::Literal, l), value(v) {}
    LiteralExpression(std::string v, std::size_t l)
        : Expression(Kind::Literal, l), value(std::move(v)) {}
    LiteralExpression(char v, std::size_t l)
        : Expression(Kind::Literal, l), value(v) {}
};

struct IdentifierExpression : Expression {
    std::string name;

    IdentifierExpression(std::string n, std::size_t l)
        : Expression(Kind::Identifier, l), name(std::move(n)) {}
};

struct BinaryExpression : Expression {
    std::unique_ptr<Expression> left;
    TokenType op;
    std::unique_ptr<Expression> right;

    BinaryExpression(std::unique_ptr<Expression> lhs, TokenType o,
                    std::unique_ptr<Expression> rhs, std::size_t l)
        : Expression(Kind::Binary, l), left(std::move(lhs)),
          op(o), right(std::move(rhs)) {}
};

struct UnaryExpression : Expression {
    TokenType op;
    std::unique_ptr<Expression> operand;
    bool is_postfix = false;

    UnaryExpression(TokenType o, std::unique_ptr<Expression> expr,
                    std::size_t l, bool postfix = false)
        : Expression(Kind::Unary, l), op(o),
          operand(std::move(expr)), is_postfix(postfix) {}
};

struct CallExpression : Expression {
    std::unique_ptr<Expression> callee;
    std::vector<std::unique_ptr<Expression>> args;
    bool is_ufcs = false;  // Cpp2 UFCS (Unified Function Call Syntax)

    CallExpression(std::unique_ptr<Expression> c, std::size_t l)
        : Expression(Kind::Call, l), callee(std::move(c)) {}
};

struct MemberAccessExpression : Expression {
    std::unique_ptr<Expression> object;
    std::string member;

    MemberAccessExpression(std::unique_ptr<Expression> obj, std::string m, std::size_t l)
        : Expression(Kind::MemberAccess, l), object(std::move(obj)), member(std::move(m)) {}
};

struct SubscriptExpression : Expression {
    std::unique_ptr<Expression> array;
    std::unique_ptr<Expression> index;
    bool has_bounds_check = false;  // Cpp2: automatic bounds safety

    SubscriptExpression(std::unique_ptr<Expression> a, std::unique_ptr<Expression> i, std::size_t l)
        : Expression(Kind::Subscript, l), array(std::move(a)), index(std::move(i)) {}
};

// Corpus-derived: Bounds check expression (CPP2_ASSERT_IN_BOUNDS_LITERAL)
struct BoundsCheckExpression : Expression {
    std::unique_ptr<Expression> container;
    std::unique_ptr<Expression> index;

    BoundsCheckExpression(std::unique_ptr<Expression> c, std::unique_ptr<Expression> i, std::size_t l)
        : Expression(Kind::Unary, l), container(std::move(c)), index(std::move(i)) {}
};

struct TernaryExpression : Expression {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Expression> then_expr;
    std::unique_ptr<Expression> else_expr;

    TernaryExpression(std::unique_ptr<Expression> cond,
                     std::unique_ptr<Expression> then_e,
                     std::unique_ptr<Expression> else_e, std::size_t l)
        : Expression(Kind::Ternary, l),
          condition(std::move(cond)),
          then_expr(std::move(then_e)),
          else_expr(std::move(else_e)) {}
};

struct LambdaExpression : Expression {
    struct Parameter {
        std::string name;
        std::unique_ptr<Type> type;
        std::unique_ptr<Expression> default_value;
        std::vector<ParameterQualifier> qualifiers;  // Cpp2: inout, out, etc.
    };

    std::vector<Parameter> parameters;
    std::unique_ptr<Type> return_type;
    std::vector<std::shared_ptr<Statement>> body;
    std::vector<std::unique_ptr<Type>> template_params;

    LambdaExpression(std::size_t l) : Expression(Kind::Lambda, l) {}
    ~LambdaExpression();
};

struct IsExpression : Expression {
    std::unique_ptr<Expression> expr;
    std::unique_ptr<Type> type;

    IsExpression(std::unique_ptr<Expression> e, std::unique_ptr<Type> t, std::size_t l)
        : Expression(Kind::Is, l), expr(std::move(e)), type(std::move(t)) {}
};

struct AsExpression : Expression {
    std::unique_ptr<Expression> expr;
    std::unique_ptr<Type> type;

    AsExpression(std::unique_ptr<Expression> e, std::unique_ptr<Type> t, std::size_t l)
        : Expression(Kind::As, l), expr(std::move(e)), type(std::move(t)) {}
};

struct StringInterpolationExpression : Expression {
    std::vector<std::variant<std::string, std::unique_ptr<Expression>>> parts;

    StringInterpolationExpression(std::size_t l) : Expression(Kind::StringInterpolation, l) {}
};

struct RangeExpression : Expression {
    std::unique_ptr<Expression> start;
    std::unique_ptr<Expression> end;
    bool inclusive;

    RangeExpression(std::unique_ptr<Expression> s, std::unique_ptr<Expression> e,
                   bool inc, std::size_t l)
        : Expression(Kind::Range, l), start(std::move(s)), end(std::move(e)), inclusive(inc) {}
};

struct ListExpression : Expression {
    std::vector<std::unique_ptr<Expression>> elements;

    ListExpression(std::size_t l) : Expression(Kind::List, l) {}
};

struct StructInitializerExpression : Expression {
    std::unique_ptr<Expression> type;
    std::vector<std::pair<std::string, std::unique_ptr<Expression>>> fields;

    StructInitializerExpression(std::unique_ptr<Expression> t, std::size_t l)
        : Expression(Kind::StructInitializer, l), type(std::move(t)) {}
};

struct MetafunctionCallExpression : Expression {
    std::string name;
    std::vector<std::unique_ptr<Expression>> args;

    MetafunctionCallExpression(std::string n, std::size_t l)
        : Expression(Kind::MetafunctionCall, l), name(std::move(n)) {}
};

// Contract categories (from cpp2 safety categories)
enum class ContractCategory {
    TypeSafety,         // type_safety - type correctness
    BoundsSafety,       // bounds_safety - array bounds checking
    NullSafety,         // null_safety - null pointer checking
    LifetimeSafety,     // lifetime_safety - object lifetime
    InitializationSafety, // initialization_safety - definite initialization
    ArithmeticSafety,   // arithmetic_safety - overflow/underflow
    Unevaluated         // unevaluated - compile-time only
};

struct ContractExpression : Expression {
    enum class ContractKind { Pre, Post, Assert };
    ContractKind kind;
    std::unique_ptr<Expression> condition;
    std::optional<std::string> message;
    std::vector<std::string> captures;

    // Semantic attributes from Clang analysis
    std::vector<ContractCategory> categories;  // Contract categories
    bool audit = false;                          // Audit-mode contract
    bool has_handler = false;                   // Whether a handler is installed
    std::string source_location;                // Source location for diagnostics

    ContractExpression(ContractKind k, std::unique_ptr<Expression> cond, std::size_t l)
        : Expression(Kind::ContractExpression, l), kind(k), condition(std::move(cond)) {}
};

// ============================================================================
// Concurrency Expressions (Kotlin-style)
// ============================================================================

// Await expression - suspend until value is ready
struct AwaitExpression : Expression {
    std::unique_ptr<Expression> value;
    std::string channel;  // Optional: await from specific channel

    AwaitExpression(std::unique_ptr<Expression> v, std::size_t l)
        : Expression(Kind::Await, l), value(std::move(v)) {}
};

// Spawn expression - launch async task
struct SpawnExpression : Expression {
    std::unique_ptr<Expression> task;
    std::string result_channel;  // Optional: send result to this channel

    SpawnExpression(std::unique_ptr<Expression> t, std::size_t l)
        : Expression(Kind::Spawn, l), task(std::move(t)) {}
};

// Channel send expression
struct ChannelSendExpression : Expression {
    std::string channel;
    std::unique_ptr<Expression> value;

    ChannelSendExpression(std::string ch, std::unique_ptr<Expression> v, std::size_t l)
        : Expression(Kind::ChannelSend, l), channel(std::move(ch)), value(std::move(v)) {}
};

// Channel receive expression
struct ChannelRecvExpression : Expression {
    std::string channel;
    bool non_blocking = false;

    ChannelRecvExpression(std::string ch, std::size_t l, bool nb = false)
        : Expression(Kind::ChannelRecv, l), channel(std::move(ch)), non_blocking(nb) {}
};

// Channel select expression (Kotlin-style select)
struct ChannelSelectExpression : Expression {
    struct SelectCase {
        std::string channel;
        enum class Kind { Send, Recv } kind;
        std::unique_ptr<Expression> value;  // For Send case
        std::unique_ptr<Expression> action;  // What to do when selected
    };

    std::vector<SelectCase> cases;
    std::unique_ptr<Expression> default_case;  // Optional: what to do if none ready

    ChannelSelectExpression(std::vector<SelectCase> c, std::size_t l)
        : Expression(Kind::ChannelSelect, l), cases(std::move(c)) {}
};

// Inspect expression - cpp2 pattern matching that returns a value
struct InspectExpression : Expression {
    struct Arm {
        enum class PatternKind { Value, Wildcard, Type };
        PatternKind pattern_kind;
        std::unique_ptr<Expression> pattern_value;
        std::unique_ptr<Type> pattern_type;
        std::unique_ptr<Expression> result_value;
    };

    std::unique_ptr<Expression> value;
    std::unique_ptr<Type> result_type;  // Optional: inspect value -> type
    std::vector<Arm> arms;

    InspectExpression(std::unique_ptr<Expression> v, std::size_t l)
        : Expression(Kind::InspectExpr, l), value(std::move(v)) {}
};

// Statements
struct Statement {
    enum class Kind {
        Expression,
        Declaration,
        Block,
        If,
        While,
        For,
        ForRange,
        Switch,
        Inspect,
        Return,
        Break,
        Continue,
        Try,
        Throw,
        Contract,
        Template,
        StaticAssert,
        // Concurrency statements
        ChannelDecl,
        CoroutineScope,
        ParallelFor
    };

    Kind kind;
    std::size_t line;

    Statement(Kind k, std::size_t l) : kind(k), line(l) {}
    virtual ~Statement() = default;
};

struct ExpressionStatement : Statement {
    std::unique_ptr<Expression> expr;

    ExpressionStatement(std::unique_ptr<Expression> e, std::size_t l)
        : Statement(Kind::Expression, l), expr(std::move(e)) {}
};

// Forward declaration - defined after Declaration is complete
struct DeclarationStatement;

struct BlockStatement : Statement {
    std::vector<std::unique_ptr<Statement>> statements;

    BlockStatement(std::size_t l) : Statement(Kind::Block, l) {}
};

struct IfStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> then_stmt;
    std::unique_ptr<Statement> else_stmt;

    IfStatement(std::unique_ptr<Expression> cond,
               std::unique_ptr<Statement> then_s,
               std::unique_ptr<Statement> else_s, std::size_t l)
        : Statement(Kind::If, l),
          condition(std::move(cond)),
          then_stmt(std::move(then_s)),
          else_stmt(std::move(else_s)) {}
};

struct WhileStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Expression> increment; // Cpp2 'next' clause
    std::unique_ptr<Statement> body;

    WhileStatement(std::unique_ptr<Expression> cond,
                  std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::While, l),
          condition(std::move(cond)),
          body(std::move(b)) {}

    WhileStatement(std::unique_ptr<Expression> cond,
                  std::unique_ptr<Expression> inc,
                  std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::While, l),
          condition(std::move(cond)),
          increment(std::move(inc)),
          body(std::move(b)) {}
};

struct ForStatement : Statement {
    std::unique_ptr<Statement> init;
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Expression> increment;
    std::unique_ptr<Statement> body;

    ForStatement(std::unique_ptr<Statement> i,
                std::unique_ptr<Expression> cond,
                std::unique_ptr<Expression> inc,
                std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::For, l),
          init(std::move(i)),
          condition(std::move(cond)),
          increment(std::move(inc)),
          body(std::move(b)) {}
};

struct ForRangeStatement : Statement {
    std::string variable;
    std::unique_ptr<Type> var_type;
    std::unique_ptr<Expression> range;
    std::unique_ptr<Statement> body;

    ForRangeStatement(std::string var, std::unique_ptr<Type> t,
                     std::unique_ptr<Expression> r,
                     std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::ForRange, l),
          variable(std::move(var)),
          var_type(std::move(t)),
          range(std::move(r)),
          body(std::move(b)) {}
};

struct SwitchStatement : Statement {
    std::unique_ptr<Expression> value;
    std::vector<std::pair<std::unique_ptr<Expression>, std::unique_ptr<Statement>>> cases;
    std::unique_ptr<Statement> default_case;

    SwitchStatement(std::unique_ptr<Expression> v, std::size_t l)
        : Statement(Kind::Switch, l), value(std::move(v)) {}
};

struct InspectStatement : Statement {
    struct Pattern {
        enum class Kind { Value, Type, Wildcard, Binding };
        Kind kind;
        std::unique_ptr<Expression> value;
        std::unique_ptr<Type> type;
        std::string binding_name;
    };

    std::unique_ptr<Expression> value;
    std::vector<std::pair<Pattern, std::unique_ptr<Statement>>> arms;
    std::unique_ptr<Statement> else_arm;

    InspectStatement(std::unique_ptr<Expression> v, std::size_t l)
        : Statement(Kind::Inspect, l), value(std::move(v)) {}
};

struct ReturnStatement : Statement {
    std::unique_ptr<Expression> value;

    ReturnStatement(std::unique_ptr<Expression> v, std::size_t l)
        : Statement(Kind::Return, l), value(std::move(v)) {}
};

struct BreakStatement : Statement {
    BreakStatement(std::size_t l) : Statement(Kind::Break, l) {}
};

struct ContinueStatement : Statement {
    ContinueStatement(std::size_t l) : Statement(Kind::Continue, l) {}
};

struct TryStatement : Statement {
    std::unique_ptr<Statement> try_block;
    std::vector<std::pair<std::string, std::unique_ptr<Statement>>> catch_blocks;

    TryStatement(std::unique_ptr<Statement> tb, std::size_t l)
        : Statement(Kind::Try, l), try_block(std::move(tb)) {}
};

struct ThrowStatement : Statement {
    std::unique_ptr<Expression> value;

    ThrowStatement(std::unique_ptr<Expression> v, std::size_t l)
        : Statement(Kind::Throw, l), value(std::move(v)) {}
};

struct ContractStatement : Statement {
    std::unique_ptr<ContractExpression> contract;

    ContractStatement(std::unique_ptr<ContractExpression> c, std::size_t l)
        : Statement(Kind::Contract, l), contract(std::move(c)) {}
};

struct TemplateStatement : Statement {
    std::vector<std::string> parameters;
    std::unique_ptr<Statement> statement;

    TemplateStatement(std::vector<std::string> params, std::unique_ptr<Statement> stmt, std::size_t l)
        : Statement(Kind::Template, l),
          parameters(std::move(params)),
          statement(std::move(stmt)) {}
};

struct StaticAssertStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::optional<std::string> message;

    StaticAssertStatement(std::unique_ptr<Expression> cond, std::size_t l)
        : Statement(Kind::StaticAssert, l), condition(std::move(cond)) {}
};

// ============================================================================
// Concurrency Statements (Kotlin-style + Hardware Kernel)
// ============================================================================

// Channel declaration
struct ChannelDeclarationStatement : Statement {
    std::string name;
    std::unique_ptr<Type> element_type;
    size_t buffer_size = 0;  // 0 = unbuffered (rendezvous)

    ChannelDeclarationStatement(std::string n, std::unique_ptr<Type> elem, std::size_t l)
        : Statement(Kind::ChannelDecl, l), name(std::move(n)), element_type(std::move(elem)) {}
};

// Coroutine scope - structured concurrency region
struct CoroutineScopeStatement : Statement {
    std::unique_ptr<Statement> body;

    CoroutineScopeStatement(std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::CoroutineScope, l), body(std::move(b)) {}
};

// Parallel for loop - for kernel conversion
struct ParallelForStatement : Statement {
    std::string loop_variable;
    std::unique_ptr<Expression> lower_bound;
    std::unique_ptr<Expression> upper_bound;
    std::unique_ptr<Expression> step;
    std::string mapping;  // "global_x", "global_y", "local_x", etc.
    std::unique_ptr<Statement> body;

    ParallelForStatement(std::string var, std::unique_ptr<Expression> lb,
                         std::unique_ptr<Expression> ub, std::unique_ptr<Expression> st,
                         std::string map, std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::ParallelFor, l), loop_variable(std::move(var)),
          lower_bound(std::move(lb)), upper_bound(std::move(ub)), step(std::move(st)),
          mapping(std::move(map)), body(std::move(b)) {}
};

// Declarations
struct Declaration {
    enum class Kind {
        Variable,
        Function,
        Type,
        Namespace,
        Operator,
        Template,
        Using,
        Contract,
        Import,
        Export
    };

    Kind kind;
    std::size_t line;
    std::string name;

    // Markdown blocks attached to this declaration (CAS-linked modules)
    std::vector<MarkdownBlockAttr> markdown_blocks;

    Declaration(Kind k, std::string n, std::size_t l)
        : kind(k), line(l), name(std::move(n)) {}
    virtual ~Declaration() = default;
};

struct VariableDeclaration : Declaration {
    // Initialization state (from definite initialization analysis)
    enum class InitState {
        Unknown,            // State not yet determined
        Uninitialized,      // Declared without initializer
        DefinitelyAssigned, // Assigned on all paths
        PotentiallyUninitialized, // Not assigned on some paths
        OutParameter        // 'out' parameter - will be definitely assigned
    };

    std::unique_ptr<Type> type;
    std::unique_ptr<Expression> initializer;
    bool is_const = false;
    bool is_mut = false;
    bool is_compile_time = false;
    std::vector<ParameterQualifier> qualifiers;  // Cpp2: inout, out, etc.

    // Semantic attributes from Clang analysis
    InitState init_state = InitState::Unknown;    // Initialization safety state
    bool requires_definite_assignment = false;    // Requires 'out' semantics
    bool is_definite_last_use = false;            // Marked for move optimization

    VariableDeclaration(std::string n, std::size_t l)
        : Declaration(Kind::Variable, std::move(n), l) {}
};

struct FunctionDeclaration : Declaration {
    // This-qualifier for member functions (from Clang CXXMethodDecl)
    enum class ThisQualifier {
        None,       // Regular function or non-qualified member
        In,         // 'in this' - read-only access (const this)
        InOut,      // 'inout this' - mutable access
        Move,       // 'move this' - rvalue reference (&&)
        Forward     // 'forward this' - forwarding reference
    };

    struct Parameter {
        std::string name;
        std::unique_ptr<Type> type;
        std::unique_ptr<Expression> default_value;
        std::vector<ParameterQualifier> qualifiers;  // Cpp2: inout, out, move, forward
    };

    std::vector<Parameter> parameters;
    std::unique_ptr<Type> return_type;
    std::unique_ptr<Statement> body;
    std::vector<std::string> contract_groups;
    std::vector<std::string> template_parameters;  // Template parameters for generic functions
    bool is_virtual = false;
    bool is_override = false;
    bool is_final = false;
    bool is_explicit = false;

    // Semantic attributes from Clang analysis
    ThisQualifier this_qualifier = ThisQualifier::None;  // this-qualifier
    bool is_constexpr = false;                            // constexpr function
    bool is_noexcept = false;                              // noexcept function

    // Concurrency attributes (Kotlin-style + GPU kernel)
    bool is_async = false;              // async function (uses suspend/await)
    bool is_kernel = false;             // GPU kernel candidate
    std::string launch_config;          // e.g., "grid(256,256) block(32)"
    std::string memory_policy;          // External memory pipeline policy
    std::vector<std::string> channels;  // Channels used by this function

    FunctionDeclaration(std::string n, std::size_t l)
        : Declaration(Kind::Function, std::move(n), l) {}
};

struct TypeDeclaration : Declaration {
    enum class TypeKind {
        Struct,
        Class,
        Interface,
        Enum,
        Union,
        Alias
    };

    TypeKind type_kind;
    std::vector<std::unique_ptr<Declaration>> members;
    std::unique_ptr<Type> underlying_type;
    std::vector<std::string> metafunctions;

    TypeDeclaration(std::string n, TypeKind tk, std::size_t l)
        : Declaration(Kind::Type, std::move(n), l), type_kind(tk) {}
};

struct NamespaceDeclaration : Declaration {
    std::vector<std::unique_ptr<Declaration>> members;

    NamespaceDeclaration(std::string n, std::size_t l)
        : Declaration(Kind::Namespace, std::move(n), l) {}
};

struct OperatorDeclaration : Declaration {
    std::unique_ptr<Type> return_type;
    std::vector<std::unique_ptr<FunctionDeclaration::Parameter>> parameters;
    std::unique_ptr<Statement> body;

    OperatorDeclaration(std::string n, std::size_t l)
        : Declaration(Kind::Operator, std::move(n), l) {}
};

struct UsingDeclaration : Declaration {
    std::string target;

    UsingDeclaration(std::string n, std::string t, std::size_t l)
        : Declaration(Kind::Using, std::move(n), l), target(std::move(t)) {}
};

struct ContractDeclaration : Declaration {
    enum class ContractKind { Pre, Post, Assert };
    ContractKind contract_kind;
    std::unique_ptr<Expression> condition;
    std::optional<std::string> message;

    ContractDeclaration(std::string n, ContractKind k, std::size_t l)
        : Declaration(Kind::Contract, std::move(n), l), contract_kind(k) {}
};

struct ImportDeclaration : Declaration {
    std::string module_name;

    ImportDeclaration(std::string m, std::size_t l)
        : Declaration(Kind::Import, "", l), module_name(std::move(m)) {}
};

// Wrapper to use a Declaration as a Statement (e.g., variable declarations in function bodies)
// Defined here after Declaration is complete to avoid incomplete type errors
struct DeclarationStatement : Statement {
    std::unique_ptr<Declaration> declaration;

    DeclarationStatement(std::unique_ptr<Declaration> d, std::size_t l)
        : Statement(Kind::Declaration, l), declaration(std::move(d)) {}
};

// AST root
struct AST {
    std::vector<std::unique_ptr<Declaration>> declarations;
};

    // Define destructors after dependent types are complete so unique_ptr can
    // delete the referenced types without triggering "delete of incomplete
    // type" errors.
    inline Type::~Type() = default;
    inline LambdaExpression::~LambdaExpression() = default;
} // namespace cpp2_transpiler