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
        TemplateParameter
    };

    Kind kind;
    std::string name;
    std::vector<std::unique_ptr<Type>> template_args;
    std::unique_ptr<Type> pointee;
    std::shared_ptr<Expression> size;
    bool is_const = false;
    bool is_mut = false;

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
        ContractExpression
    };

    Kind kind;
    std::size_t line;

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

    SubscriptExpression(std::unique_ptr<Expression> a, std::unique_ptr<Expression> i, std::size_t l)
        : Expression(Kind::Subscript, l), array(std::move(a)), index(std::move(i)) {}
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

struct ContractExpression : Expression {
    enum class ContractKind { Pre, Post, Assert };
    ContractKind kind;
    std::unique_ptr<Expression> condition;
    std::optional<std::string> message;
    std::vector<std::string> captures;

    ContractExpression(ContractKind k, std::unique_ptr<Expression> cond, std::size_t l)
        : Expression(Kind::ContractExpression, l), kind(k), condition(std::move(cond)) {}
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
        StaticAssert
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
    std::unique_ptr<Statement> body;

    WhileStatement(std::unique_ptr<Expression> cond,
                  std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::While, l),
          condition(std::move(cond)),
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

    Declaration(Kind k, std::string n, std::size_t l)
        : kind(k), line(l), name(std::move(n)) {}
    virtual ~Declaration() = default;
};

struct VariableDeclaration : Declaration {
    std::unique_ptr<Type> type;
    std::unique_ptr<Expression> initializer;
    bool is_const = false;
    bool is_mut = false;
    bool is_compile_time = false;

    VariableDeclaration(std::string n, std::size_t l)
        : Declaration(Kind::Variable, std::move(n), l) {}
};

struct FunctionDeclaration : Declaration {
    struct Parameter {
        std::string name;
        std::unique_ptr<Type> type;
        std::unique_ptr<Expression> default_value;
    };

    std::vector<Parameter> parameters;
    std::unique_ptr<Type> return_type;
    std::unique_ptr<Statement> body;
    std::vector<std::string> contract_groups;
    bool is_virtual = false;
    bool is_override = false;
    bool is_final = false;
    bool is_explicit = false;

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