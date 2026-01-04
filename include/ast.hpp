#pragma once

// ============================================================================
// Cpp2 AST - Abstract Syntax Tree Node Types
// ============================================================================
//
// This file defines all AST node types used by the parser. The AST represents
// the parsed structure of Cpp2 source code before code generation.
//
// Hierarchy:
//   Node (base)
//     ├── Declaration
//     │   ├── VariableDeclaration
//     │   ├── FunctionDeclaration
//     │   ├── TypeDeclaration
//     │   ├── NamespaceDeclaration
//     │   ├── OperatorDeclaration
//     │   ├── ImportDeclaration
//     │   ├── UsingDeclaration
//     │   └── Cpp1PassthroughDeclaration
//     ├── Statement
//     │   ├── BlockStatement
//     │   ├── ExpressionStatement
//     │   ├── DeclarationStatement
//     │   ├── IfStatement
//     │   ├── WhileStatement
//     │   ├── ForStatement
//     │   ├── ReturnStatement
//     │   └── ... (more control flow)
//     ├── Expression
//     │   ├── LiteralExpression
//     │   ├── IdentifierExpression
//     │   ├── BinaryExpression
//     │   ├── UnaryExpression
//     │   ├── CallExpression
//     │   └── ... (more expression types)
//     └── Type
//
// ============================================================================

#include "lexer.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <optional>
#include <variant>

namespace cpp2_transpiler {

// Forward declarations
struct Declaration;
struct Statement;
struct Expression;
struct Type;
struct BlockStatement;

// ============================================================================
// AST Root
// ============================================================================

struct AST {
    std::vector<std::unique_ptr<Declaration>> declarations;
};

// ============================================================================
// Semantic Analysis - Escape Analysis
// ============================================================================
// (Defined early so VariableDeclaration can reference it)

// Categorizes how a value escapes its local scope
enum class EscapeKind {
    NoEscape,          // Value stays local (stack)
    EscapeToHeap,      // Stored in heap-allocated object
    EscapeToReturn,    // Returned from function
    EscapeToParam,     // Stored via pointer/reference parameter
    EscapeToGlobal,    // Stored in global variable
    EscapeToChannel,   // Sent through channel
    EscapeToGPU,       // Transferred to GPU memory
    EscapeToDMA        // Transferred via DMA buffer
};

// Tracks escape information for a variable
struct EscapeInfo {
    EscapeKind kind = EscapeKind::NoEscape;
    std::vector<void*> escape_points;  // ASTNode* escape points (void* to avoid forward decl)
    bool needs_lifetime_extension = false;
};

// ============================================================================
// Type Representation
// ============================================================================

struct Type {
    enum class Kind {
        Builtin,        // int, double, bool, etc.
        UserDefined,    // MyClass, std::vector
        Auto,           // auto / deduced type
        Deduced,        // Deduction in progress
        Pointer,        // T*
        Reference,      // T&
        Template,       // T<Args...>
        Function,       // () -> T
        FunctionType,   // Function signature as type
    };
    
    Kind kind = Kind::Builtin;
    std::string name;
    bool is_const = false;
    bool is_volatile = false;
    bool is_mutable = false;
    bool is_constexpr = false;
    
    // For Pointer/Reference
    std::unique_ptr<Type> pointee;
    
    // For Template
    std::vector<std::unique_ptr<Type>> template_args;
    
    // For Function types
    std::vector<std::unique_ptr<Type>> parameter_types;
    std::unique_ptr<Type> return_type;
    
    Type() = default;
    explicit Type(Kind k) : kind(k) {}
    Type(Kind k, std::string n) : kind(k), name(std::move(n)) {}
};

// ============================================================================
// Base Node
// ============================================================================

struct Node {
    std::uint32_t line = 0;
    std::uint32_t column = 0;
    
    virtual ~Node() = default;
    
protected:
    Node() = default;
    explicit Node(std::uint32_t ln) : line(ln) {}
    Node(std::uint32_t ln, std::uint32_t col) : line(ln), column(col) {}
};

// ============================================================================
// Declaration Base
// ============================================================================

struct Declaration : Node {
    std::string name;
    std::vector<std::string> decorators;        // @value, @interface, etc.
    std::vector<std::string> markdown_blocks;   // Associated documentation
    std::string access_specifier;               // public, private, protected
    
    Declaration() = default;
    explicit Declaration(std::uint32_t ln) : Node(ln) {}
    Declaration(std::string n, std::uint32_t ln) : Node(ln), name(std::move(n)) {}
};

// ============================================================================
// Statement Base
// ============================================================================

struct Statement : Node {
    Statement() = default;
    explicit Statement(std::uint32_t ln) : Node(ln) {}
};

// BlockStatement defined here to avoid incomplete type errors in FunctionDeclaration
struct BlockStatement : Statement {
    std::vector<std::unique_ptr<Statement>> statements;
    bool is_function_body = false;

    BlockStatement() = default;
    explicit BlockStatement(std::uint32_t ln) : Statement(ln) {}
};

// ============================================================================
// Expression Base
// ============================================================================

struct Expression : Node {
    std::unique_ptr<Type> inferred_type;
    
    Expression() = default;
    explicit Expression(std::uint32_t ln) : Node(ln) {}
};

// ============================================================================
// Declarations
// ============================================================================

// Parameter qualifier for function parameters
enum class ParameterQualifier {
    None,
    In,         // in (default, read-only)
    Out,        // out (output parameter)
    Inout,      // inout (read-write)
    Copy,       // copy (by value copy)
    Move,       // move (by value move)
    Forward,    // forward (perfect forwarding)
    InRef,      // in_ref (read-only reference)
    ForwardRef, // forward_ref (forwarding reference)
};

struct Parameter {
    std::string name;
    std::unique_ptr<Type> type;
    ParameterQualifier qualifier = ParameterQualifier::None;
    std::unique_ptr<Expression> default_value;
    bool is_variadic = false;
};

struct NamedReturn {
    std::string name;
    std::unique_ptr<Type> type;
};

struct VariableDeclaration : Declaration {
    std::unique_ptr<Type> type;
    std::unique_ptr<Expression> initializer;
    bool is_const = false;
    bool is_let = false;
    bool is_constexpr = false;
    std::optional<EscapeInfo> escape_info;  // Filled by semantic analysis

    VariableDeclaration() = default;
    VariableDeclaration(std::string name, std::uint32_t ln)
        : Declaration(std::move(name), ln) {}
};

struct FunctionDeclaration : Declaration {
    std::vector<Parameter> parameters;
    std::unique_ptr<Type> return_type;
    std::unique_ptr<BlockStatement> body;
    std::vector<NamedReturn> named_returns;
    bool throws = false;
    bool is_const = false;
    bool is_constexpr = false;
    bool is_virtual = false;
    bool is_override = false;
    bool is_final = false;
    bool is_noexcept = false;
    std::vector<std::string> template_params;
    std::string requires_clause;
    std::vector<std::unique_ptr<Expression>> pre_conditions;
    std::vector<std::unique_ptr<Expression>> post_conditions;
    
    FunctionDeclaration() = default;
    FunctionDeclaration(std::string name, std::uint32_t ln)
        : Declaration(std::move(name), ln) {}
};

struct TypeDeclaration : Declaration {
    enum class TypeKind {
        Type,       // type Name = ...
        Struct,     // type Name = struct ...
        Class,      // type Name = class ...
        Union,      // type Name = union ...
        Enum,       // type Name = enum ...
        Concept,    // type Name = concept ...
        Interface,  // type Name = interface ...
    };
    
    TypeKind type_kind = TypeKind::Type;
    std::vector<std::unique_ptr<Declaration>> members;
    std::vector<std::string> base_types;
    std::vector<std::string> template_params;
    std::string requires_clause;
    std::unique_ptr<Type> alias_type;   // For type aliases
    
    TypeDeclaration() = default;
    TypeDeclaration(std::string name, TypeKind kind, std::uint32_t ln)
        : Declaration(std::move(name), ln), type_kind(kind) {}
};

struct NamespaceDeclaration : Declaration {
    std::vector<std::unique_ptr<Declaration>> declarations;
    bool is_alias = false;
    std::string alias_target;
    std::vector<std::string> qualified_name;  // For nested namespaces
    
    NamespaceDeclaration() = default;
    NamespaceDeclaration(std::string name, std::uint32_t ln)
        : Declaration(std::move(name), ln) {}
};

struct OperatorDeclaration : Declaration {
    TokenType operator_token;
    std::vector<Parameter> parameters;
    std::unique_ptr<Type> return_type;
    std::unique_ptr<BlockStatement> body;
    std::vector<NamedReturn> named_returns;
    bool is_const = false;
    
    OperatorDeclaration() = default;
    OperatorDeclaration(TokenType op, std::uint32_t ln)
        : Declaration(ln), operator_token(op) {}
};

struct ImportDeclaration : Declaration {
    std::string module_name;
    bool is_std = false;
    std::vector<std::string> items;  // Specific imports if any
    
    ImportDeclaration() = default;
    ImportDeclaration(std::string module, std::uint32_t ln)
        : Declaration(ln), module_name(std::move(module)) {}
};

struct UsingDeclaration : Declaration {
    std::string target;
    bool is_namespace = false;
    
    UsingDeclaration() = default;
    UsingDeclaration(std::string tgt, std::uint32_t ln)
        : Declaration(ln), target(std::move(tgt)) {}
};

struct Cpp1PassthroughDeclaration : Declaration {
    std::string content;
    
    Cpp1PassthroughDeclaration() = default;
    Cpp1PassthroughDeclaration(std::string text, std::uint32_t ln)
        : Declaration(ln), content(std::move(text)) {}
};

// ============================================================================
// Statements
// ============================================================================
// (BlockStatement defined earlier to avoid forward declaration issues)

struct ExpressionStatement : Statement {
    std::unique_ptr<Expression> expression;
    
    ExpressionStatement() = default;
    ExpressionStatement(std::unique_ptr<Expression> expr, std::uint32_t ln)
        : Statement(ln), expression(std::move(expr)) {}
};

struct DeclarationStatement : Statement {
    std::unique_ptr<Declaration> declaration;
    
    DeclarationStatement() = default;
    DeclarationStatement(std::unique_ptr<Declaration> decl, std::uint32_t ln)
        : Statement(ln), declaration(std::move(decl)) {}
};

struct ReturnStatement : Statement {
    std::unique_ptr<Expression> value;
    
    ReturnStatement() = default;
    ReturnStatement(std::unique_ptr<Expression> val, std::uint32_t ln)
        : Statement(ln), value(std::move(val)) {}
};

struct IfStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> then_branch;
    std::unique_ptr<Statement> else_branch;
    std::unique_ptr<Statement> init_statement;  // if constexpr (init; cond)
    bool is_constexpr = false;
    
    IfStatement() = default;
    explicit IfStatement(std::uint32_t ln) : Statement(ln) {}
};

struct WhileStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> body;
    std::unique_ptr<Statement> next_clause;  // Cpp2 next clause
    
    WhileStatement() = default;
    explicit WhileStatement(std::uint32_t ln) : Statement(ln) {}
};

struct DoWhileStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> body;
    
    DoWhileStatement() = default;
    explicit DoWhileStatement(std::uint32_t ln) : Statement(ln) {}
};

struct LoopInitializer {
    std::string name;
    std::unique_ptr<Expression> init_value;
    std::unique_ptr<Expression> next_value;
};

struct ForStatement : Statement {
    std::vector<LoopInitializer> initializers;
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Expression> increment;
    std::unique_ptr<Statement> body;
    std::unique_ptr<Statement> next_clause;
    
    ForStatement() = default;
    explicit ForStatement(std::uint32_t ln) : Statement(ln) {}
};

struct ForRangeStatement : Statement {
    std::string variable;
    std::unique_ptr<Expression> range;
    std::unique_ptr<Statement> body;
    std::unique_ptr<Statement> next_clause;
    
    ForRangeStatement() = default;
    explicit ForRangeStatement(std::uint32_t ln) : Statement(ln) {}
};

struct SwitchStatement : Statement {
    std::unique_ptr<Expression> expression;
    struct Case {
        std::unique_ptr<Expression> value;  // nullptr for default
        std::unique_ptr<Statement> body;
        bool is_default = false;
    };
    std::vector<Case> cases;
    
    SwitchStatement() = default;
    explicit SwitchStatement(std::uint32_t ln) : Statement(ln) {}
};

struct BreakStatement : Statement {
    BreakStatement() = default;
    explicit BreakStatement(std::uint32_t ln) : Statement(ln) {}
};

struct ContinueStatement : Statement {
    ContinueStatement() = default;
    explicit ContinueStatement(std::uint32_t ln) : Statement(ln) {}
};

struct ThrowStatement : Statement {
    std::unique_ptr<Expression> expression;
    
    ThrowStatement() = default;
    ThrowStatement(std::unique_ptr<Expression> expr, std::uint32_t ln)
        : Statement(ln), expression(std::move(expr)) {}
};

struct TryStatement : Statement {
    std::unique_ptr<Statement> try_block;
    struct CatchClause {
        std::string exception_name;
        std::unique_ptr<Type> exception_type;
        std::unique_ptr<Statement> body;
    };
    std::vector<CatchClause> catch_clauses;
    
    TryStatement() = default;
    explicit TryStatement(std::uint32_t ln) : Statement(ln) {}
};

struct InspectStatement : Statement {
    std::unique_ptr<Expression> expression;
    struct Arm {
        std::unique_ptr<Expression> pattern;
        std::unique_ptr<Statement> body;
        bool is_wildcard = false;
    };
    std::vector<Arm> arms;
    
    InspectStatement() = default;
    explicit InspectStatement(std::uint32_t ln) : Statement(ln) {}
};

struct ContractStatement : Statement {
    enum class Kind { Pre, Post, Assert };
    Kind kind;
    std::unique_ptr<Expression> condition;
    std::string message;
    
    ContractStatement() = default;
    ContractStatement(Kind k, std::unique_ptr<Expression> cond, std::uint32_t ln)
        : Statement(ln), kind(k), condition(std::move(cond)) {}
};

struct StaticAssertStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::string message;
    
    StaticAssertStatement() = default;
    explicit StaticAssertStatement(std::uint32_t ln) : Statement(ln) {}
};

struct ScopeBlockStatement : Statement {
    std::unique_ptr<Statement> body;
    
    ScopeBlockStatement() = default;
    explicit ScopeBlockStatement(std::uint32_t ln) : Statement(ln) {}
};

// Concurrency statements
struct CoroutineScopeStatement : Statement {
    std::unique_ptr<Statement> body;
    
    CoroutineScopeStatement() = default;
    explicit CoroutineScopeStatement(std::uint32_t ln) : Statement(ln) {}
};

struct ParallelForStatement : Statement {
    std::string variable;
    std::unique_ptr<Expression> range;
    std::unique_ptr<Statement> body;
    
    ParallelForStatement() = default;
    explicit ParallelForStatement(std::uint32_t ln) : Statement(ln) {}
};

struct ChannelDeclarationStatement : Statement {
    std::string name;
    std::unique_ptr<Type> element_type;
    
    ChannelDeclarationStatement() = default;
    explicit ChannelDeclarationStatement(std::uint32_t ln) : Statement(ln) {}
};

// ============================================================================
// Expressions
// ============================================================================

struct LiteralExpression : Expression {
    enum class LiteralKind {
        Integer,
        Float,
        String,
        Character,
        Boolean,
        Nullptr,
    };
    
    LiteralKind kind;
    std::string value;
    
    LiteralExpression() = default;
    LiteralExpression(LiteralKind k, std::string val, std::uint32_t ln)
        : Expression(ln), kind(k), value(std::move(val)) {}
};

struct IdentifierExpression : Expression {
    std::string name;
    std::vector<std::string> qualifiers;  // For qualified names: std::vector
    
    IdentifierExpression() = default;
    IdentifierExpression(std::string n, std::uint32_t ln)
        : Expression(ln), name(std::move(n)) {}
};

struct BinaryExpression : Expression {
    TokenType op;
    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;
    
    BinaryExpression() = default;
    BinaryExpression(TokenType o, std::unique_ptr<Expression> l,
                    std::unique_ptr<Expression> r, std::uint32_t ln)
        : Expression(ln), op(o), left(std::move(l)), right(std::move(r)) {}
};

struct UnaryExpression : Expression {
    TokenType op;
    std::unique_ptr<Expression> operand;
    bool is_prefix = true;
    
    UnaryExpression() = default;
    UnaryExpression(TokenType o, std::unique_ptr<Expression> operand, 
                   bool prefix, std::uint32_t ln)
        : Expression(ln), op(o), operand(std::move(operand)), is_prefix(prefix) {}
};

struct CallExpression : Expression {
    std::unique_ptr<Expression> callee;
    std::vector<std::unique_ptr<Expression>> arguments;
    
    CallExpression() = default;
    CallExpression(std::unique_ptr<Expression> func, std::uint32_t ln)
        : Expression(ln), callee(std::move(func)) {}
};

struct MemberAccessExpression : Expression {
    std::unique_ptr<Expression> object;
    std::string member;
    bool is_arrow = false;  // -> vs .
    
    MemberAccessExpression() = default;
    MemberAccessExpression(std::unique_ptr<Expression> obj, std::string mem,
                          bool arrow, std::uint32_t ln)
        : Expression(ln), object(std::move(obj)), member(std::move(mem)), is_arrow(arrow) {}
};

struct SubscriptExpression : Expression {
    std::unique_ptr<Expression> object;
    std::unique_ptr<Expression> index;
    
    SubscriptExpression() = default;
    SubscriptExpression(std::unique_ptr<Expression> obj, std::unique_ptr<Expression> idx,
                       std::uint32_t ln)
        : Expression(ln), object(std::move(obj)), index(std::move(idx)) {}
};

struct TernaryExpression : Expression {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Expression> then_expr;
    std::unique_ptr<Expression> else_expr;
    
    TernaryExpression() = default;
    explicit TernaryExpression(std::uint32_t ln) : Expression(ln) {}
};

struct LambdaExpression : Expression {
    std::vector<Parameter> parameters;
    std::unique_ptr<Type> return_type;
    std::unique_ptr<Statement> body;
    std::vector<std::string> captures;
    
    LambdaExpression() = default;
    explicit LambdaExpression(std::uint32_t ln) : Expression(ln) {}
};

struct InspectExpression : Expression {
    std::unique_ptr<Expression> expression;
    struct Arm {
        std::unique_ptr<Expression> pattern;
        std::unique_ptr<Expression> result;
        bool is_wildcard = false;
    };
    std::vector<Arm> arms;
    
    InspectExpression() = default;
    explicit InspectExpression(std::uint32_t ln) : Expression(ln) {}
};

struct IsExpression : Expression {
    std::unique_ptr<Expression> expression;
    std::unique_ptr<Type> type;
    
    IsExpression() = default;
    explicit IsExpression(std::uint32_t ln) : Expression(ln) {}
};

struct AsExpression : Expression {
    std::unique_ptr<Expression> expression;
    std::unique_ptr<Type> type;
    
    AsExpression() = default;
    explicit AsExpression(std::uint32_t ln) : Expression(ln) {}
};

struct MoveExpression : Expression {
    std::unique_ptr<Expression> expression;
    
    MoveExpression() = default;
    MoveExpression(std::unique_ptr<Expression> expr, std::uint32_t ln)
        : Expression(ln), expression(std::move(expr)) {}
};

struct ListExpression : Expression {
    std::vector<std::unique_ptr<Expression>> elements;
    
    ListExpression() = default;
    explicit ListExpression(std::uint32_t ln) : Expression(ln) {}
};

struct StructInitializerExpression : Expression {
    std::vector<std::pair<std::string, std::unique_ptr<Expression>>> fields;
    
    StructInitializerExpression() = default;
    explicit StructInitializerExpression(std::uint32_t ln) : Expression(ln) {}
};

struct RangeExpression : Expression {
    std::unique_ptr<Expression> start;
    std::unique_ptr<Expression> end;
    bool inclusive = false;
    
    RangeExpression() = default;
    explicit RangeExpression(std::uint32_t ln) : Expression(ln) {}
};

struct PipelineExpression : Expression {
    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;
    
    PipelineExpression() = default;
    PipelineExpression(std::unique_ptr<Expression> l, std::unique_ptr<Expression> r,
                      std::uint32_t ln)
        : Expression(ln), left(std::move(l)), right(std::move(r)) {}
};

struct ContractExpression : Expression {
    ContractStatement::Kind kind;
    std::unique_ptr<Expression> condition;
    
    ContractExpression() = default;
    explicit ContractExpression(std::uint32_t ln) : Expression(ln) {}
};

struct MetafunctionCallExpression : Expression {
    std::string metafunction;
    std::unique_ptr<Expression> argument;
    
    MetafunctionCallExpression() = default;
    explicit MetafunctionCallExpression(std::uint32_t ln) : Expression(ln) {}
};

struct StringInterpolationExpression : Expression {
    std::vector<std::variant<std::string, std::unique_ptr<Expression>>> parts;
    
    StringInterpolationExpression() = default;
    explicit StringInterpolationExpression(std::uint32_t ln) : Expression(ln) {}
};

// Concurrency expressions
struct AwaitExpression : Expression {
    std::unique_ptr<Expression> expression;
    
    AwaitExpression() = default;
    AwaitExpression(std::unique_ptr<Expression> expr, std::uint32_t ln)
        : Expression(ln), expression(std::move(expr)) {}
};

struct SpawnExpression : Expression {
    std::unique_ptr<Expression> expression;
    
    SpawnExpression() = default;
    SpawnExpression(std::unique_ptr<Expression> expr, std::uint32_t ln)
        : Expression(ln), expression(std::move(expr)) {}
};

struct ChannelSendExpression : Expression {
    std::unique_ptr<Expression> channel;
    std::unique_ptr<Expression> value;
    
    ChannelSendExpression() = default;
    explicit ChannelSendExpression(std::uint32_t ln) : Expression(ln) {}
};

struct ChannelRecvExpression : Expression {
    std::unique_ptr<Expression> channel;
    
    ChannelRecvExpression() = default;
    explicit ChannelRecvExpression(std::uint32_t ln) : Expression(ln) {}
};

struct ChannelSelectExpression : Expression {
    struct SelectCase {
        std::unique_ptr<Expression> channel;
        std::unique_ptr<Statement> body;
        bool is_send = false;
    };
    std::vector<SelectCase> cases;

    ChannelSelectExpression() = default;
    explicit ChannelSelectExpression(std::uint32_t ln) : Expression(ln) {}
};

} // namespace cpp2_transpiler
