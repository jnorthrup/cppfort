#pragma once
// Unified Intermediate Representation for C/C++/CPP2 n-way compiler
// Supports bidirectional transpilation with semantic preservation

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <unordered_map>

namespace ir {

// Forward declarations
struct Type;
struct Expression;
struct Statement;
struct Declaration;
struct Function;
struct Module;

using TypePtr = std::shared_ptr<Type>;
using ExprPtr = std::shared_ptr<Expression>;
using StmtPtr = std::shared_ptr<Statement>;
using DeclPtr = std::shared_ptr<Declaration>;
using FuncPtr = std::shared_ptr<Function>;

// Source language tracking for semantic preservation
enum class SourceLang {
    C,          // C11/C17
    CPP,        // C++17/20
    CPP2        // Cpp2
};

// Type system - unified across all languages
enum class TypeKind {
    Void,
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Char, WChar,
    Pointer,
    Reference,
    Array,
    Function,
    Struct,
    Union,
    Enum,
    Template,
    Auto,       // C++ auto / CPP2 inference
    Contract    // CPP2 contracts
};

struct Type {
    TypeKind kind;
    SourceLang source;

    // For composite types
    TypePtr pointee;                    // Pointer/Reference target
    std::vector<TypePtr> params;        // Function parameters
    TypePtr return_type;                // Function return
    size_t array_size;                   // Array dimension

    // For named types
    std::string name;
    std::vector<std::pair<std::string, TypePtr>> fields;  // Struct/Union fields

    // C++ specific
    bool is_const = false;
    bool is_volatile = false;
    bool is_mutable = false;
    bool is_rvalue_ref = false;

    // CPP2 specific
    std::optional<std::string> contract_pre;
    std::optional<std::string> contract_post;
    bool is_move = false;
    bool is_forward = false;

    // Type qualifiers and attributes
    std::vector<std::string> attributes;
};

// Expression nodes
enum class ExprKind {
    // Literals
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    CharLiteral,
    BoolLiteral,
    NullLiteral,

    // Variables and members
    Identifier,
    MemberAccess,
    ArraySubscript,

    // Operators
    UnaryOp,
    BinaryOp,
    TernaryOp,
    Assignment,

    // Function calls
    Call,
    MethodCall,
    Constructor,
    Destructor,

    // Casts
    Cast,
    ImplicitCast,
    StaticCast,
    DynamicCast,
    ReinterpretCast,
    ConstCast,

    // C++ specific
    Lambda,
    New,
    Delete,
    Throw,
    SizeOf,
    AlignOf,
    TypeId,

    // CPP2 specific
    Inspect,        // Pattern matching
    Is,            // Type testing
    As,            // Safe casting
    Contract,      // Contract expressions
    Interpolation  // String interpolation
};

struct Expression {
    ExprKind kind;
    TypePtr type;
    SourceLang source;

    // Node data
    std::variant<
        int64_t,                                    // IntLiteral
        double,                                     // FloatLiteral
        std::string,                                // StringLiteral, Identifier
        char,                                       // CharLiteral
        bool,                                       // BoolLiteral
        std::pair<ExprPtr, std::string>,          // MemberAccess
        std::pair<ExprPtr, ExprPtr>,              // ArraySubscript, BinaryOp
        std::tuple<ExprPtr, ExprPtr, ExprPtr>,    // TernaryOp
        std::vector<ExprPtr>                       // Call arguments
    > data;

    // Operator info
    std::string op;  // For UnaryOp, BinaryOp, Assignment

    // CPP2 inspect patterns
    struct Pattern {
        std::string match_expr;
        ExprPtr guard;
        ExprPtr result;
    };
    std::vector<Pattern> patterns;

    // Location info for error reporting
    size_t line = 0;
    size_t column = 0;
};

// Statement nodes
enum class StmtKind {
    Expression,
    Block,
    If,
    While,
    DoWhile,
    For,
    RangeFor,      // C++11 range-based for
    Switch,
    Case,
    Default,
    Break,
    Continue,
    Return,
    Goto,
    Label,

    // C++ specific
    Try,
    Catch,

    // CPP2 specific
    Defer,         // Defer statement
    Assert,        // Contract assertion
    Unsafe         // Unsafe block
};

struct Statement {
    StmtKind kind;
    SourceLang source;

    // Control flow
    ExprPtr condition;                    // If, While, DoWhile, For
    StmtPtr then_stmt;                   // If
    StmtPtr else_stmt;                   // If
    std::vector<StmtPtr> body;          // Block, Loop body

    // For loop
    StmtPtr init;                        // For init
    ExprPtr update;                      // For update

    // Switch
    std::vector<std::pair<ExprPtr, StmtPtr>> cases;

    // Jump targets
    std::string label;                   // Goto, Label

    // Return value
    ExprPtr return_value;

    // Exception handling
    std::string exception_type;          // Catch
    std::string exception_var;           // Catch variable

    // CPP2 contracts
    ExprPtr contract_expr;               // Assert condition
    std::string contract_msg;            // Assert message
};

// Declaration nodes
enum class DeclKind {
    Variable,
    Function,
    Type,
    Namespace,
    Using,

    // C specific
    Typedef,

    // C++ specific
    Class,
    Template,

    // CPP2 specific
    Concept,
    Module
};

struct Declaration {
    DeclKind kind;
    SourceLang source;
    std::string name;
    TypePtr type;

    // Variable initialization
    ExprPtr initializer;

    // Storage and linkage
    bool is_static = false;
    bool is_extern = false;
    bool is_inline = false;
    bool is_constexpr = false;

    // C++ access control
    enum Access { Public, Protected, Private };
    Access access = Public;

    // CPP2 parameter passing
    enum ParamMode { In, Out, InOut, Move, Forward };
    ParamMode mode = In;

    // Attributes
    std::vector<std::string> attributes;
};

// Function definition
struct Function {
    std::string name;
    TypePtr return_type;
    std::vector<std::pair<std::string, TypePtr>> params;
    std::vector<StmtPtr> body;
    SourceLang source;

    // Function qualifiers
    bool is_inline = false;
    bool is_constexpr = false;
    bool is_noexcept = false;
    bool is_virtual = false;
    bool is_override = false;
    bool is_final = false;
    bool is_deleted = false;
    bool is_default = false;

    // CPP2 specific
    std::optional<std::string> requires_clause;
    std::optional<std::string> ensures_clause;
    std::vector<std::string> captures;  // For function expressions

    // Templates
    std::vector<std::string> template_params;

    // Attributes and metadata
    std::vector<std::string> attributes;
    std::string mangled_name;  // For C++ name mangling
};

// Translation unit / Module
struct Module {
    std::string filename;
    SourceLang source;

    // Preprocessor (C/C++)
    std::vector<std::string> includes;
    std::vector<std::pair<std::string, std::string>> defines;

    // Declarations
    std::vector<DeclPtr> declarations;
    std::vector<FuncPtr> functions;

    // Namespaces (C++/CPP2)
    std::unordered_map<std::string, std::vector<DeclPtr>> namespaces;

    // CPP2 imports/exports
    std::vector<std::string> imports;
    std::vector<std::string> exports;

    // Metadata for attestation
    std::string source_hash;     // SHA3-512 of source
    std::string ir_hash;         // SHA3-512 of IR
    uint64_t timestamp;          // Build timestamp
    std::string compiler_version;
};

// IR Builder interface
class IRBuilder {
public:
    // Type construction
    TypePtr makeVoidType();
    TypePtr makeIntType(int bits, bool is_signed);
    TypePtr makeFloatType(int bits);
    TypePtr makePointerType(TypePtr pointee);
    TypePtr makeArrayType(TypePtr element, size_t size);
    TypePtr makeFunctionType(TypePtr ret, std::vector<TypePtr> params);

    // Expression construction
    ExprPtr makeIntLiteral(int64_t val, TypePtr type);
    ExprPtr makeStringLiteral(const std::string& str);
    ExprPtr makeIdentifier(const std::string& name, TypePtr type);
    ExprPtr makeBinaryOp(const std::string& op, ExprPtr left, ExprPtr right);
    ExprPtr makeCall(ExprPtr func, std::vector<ExprPtr> args);

    // Statement construction
    StmtPtr makeExprStmt(ExprPtr expr);
    StmtPtr makeBlock(std::vector<StmtPtr> stmts);
    StmtPtr makeIf(ExprPtr cond, StmtPtr then_stmt, StmtPtr else_stmt = nullptr);
    StmtPtr makeWhile(ExprPtr cond, StmtPtr body);
    StmtPtr makeReturn(ExprPtr value = nullptr);

    // Declaration construction
    DeclPtr makeVariable(const std::string& name, TypePtr type, ExprPtr init = nullptr);
    FuncPtr makeFunction(const std::string& name, TypePtr ret_type,
                        std::vector<std::pair<std::string, TypePtr>> params,
                        std::vector<StmtPtr> body);

    // Module construction
    std::shared_ptr<Module> makeModule(const std::string& filename, SourceLang lang);

private:
    // Type caching for efficiency
    std::unordered_map<std::string, TypePtr> type_cache;
};

// Semantic analyzer for IR validation
class SemanticAnalyzer {
public:
    bool analyze(const Module& module);
    std::vector<std::string> getErrors() const { return errors; }

private:
    std::vector<std::string> errors;

    bool checkType(const Type& type);
    bool checkExpression(const Expression& expr);
    bool checkStatement(const Statement& stmt);
    bool checkFunction(const Function& func);
};

} // namespace ir