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
    Override,   // override - for overriding base functions
    Implicit    // implicit - for implicit conversion operators
};

// ============================================================================
// Semantic Analysis: Escape Analysis and Borrowing
// ============================================================================

// Escape analysis: Track where values escape local scope
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

struct EscapeInfo {
    EscapeKind kind = EscapeKind::NoEscape;
    std::vector<void*> escape_points;  // ASTNode* escape points (void* to avoid forward declaration)
    bool needs_lifetime_extension = false;
};

// Ownership tracking: Rust-like ownership semantics
enum class OwnershipKind {
    Owned,      // Unique owner (value semantics)
    Borrowed,   // Immutable borrow (shared reference)
    MutBorrowed,// Mutable borrow (exclusive reference)
    Moved       // Ownership transferred
};

// Forward declaration for LifetimeRegion
struct LifetimeRegion;

struct BorrowInfo {
    OwnershipKind kind = OwnershipKind::Owned;
    void* owner = nullptr;  // ASTNode* owner if borrowed (void* to avoid forward declaration)
    std::vector<void*> active_borrows;  // ASTNode* active borrows (void* to avoid forward declaration)
    LifetimeRegion* lifetime = nullptr;
};

// Lifetime region: Scope-based lifetime bounds
struct LifetimeRegion {
    void* scope_start = nullptr;  // ASTNode* (void* to avoid forward declaration)
    void* scope_end = nullptr;    // ASTNode* (void* to avoid forward declaration)
    std::vector<LifetimeRegion*> nested_regions;

    // Check if this region outlives another region
    bool outlives(const LifetimeRegion& other) const {
        // Simplified implementation: a region outlives another if the other is nested within it
        for (const auto* nested : nested_regions) {
            if (nested == &other) {
                return true;
            }
            if (nested->outlives(other)) {
                return true;
            }
        }
        return false;
    }
};

// External Memory: Memory region tracking for GPU/DMA transfers
struct MemoryRegion {
    std::string name;              // Region identifier (e.g., "gpu_global", "host", "gpu_shared")
    std::size_t size_bytes = 0;    // Region size in bytes
    bool is_device_memory = false; // True for GPU/DMA, false for host

    MemoryRegion() = default;
    MemoryRegion(std::string n, std::size_t size, bool is_device)
        : name(std::move(n)), size_bytes(size), is_device_memory(is_device) {}
};

// External Memory: Track GPU/DMA transfers with escape analysis integration
struct MemoryTransfer {
    EscapeKind escape_kind = EscapeKind::NoEscape;
    MemoryRegion* source_region = nullptr;
    MemoryRegion* dest_region = nullptr;
    bool is_async = false;                          // True for DMA transfers
    std::vector<void*> transferred_vars;            // VarDecl* (void* to avoid forward declaration)
    LifetimeRegion transfer_lifetime;               // Lifetime bounds for the transfer

    MemoryTransfer() = default;
};

// Channelized Concurrency: Track ownership transfer through channel operations (Phase 4)
struct ChannelTransfer {
    EscapeKind escape_kind = EscapeKind::EscapeToChannel;
    void* send_point = nullptr;                     // ASTNode* where send occurs
    void* recv_point = nullptr;                     // ASTNode* where recv occurs (may be unknown)
    OwnershipKind ownership_transfer = OwnershipKind::Moved;  // Default: move semantics
    LifetimeRegion channel_lifetime;                // Lifetime bounds for the channel
    std::string channel_name;                       // Channel identifier
    std::string value_type;                         // Type being transferred
    bool is_buffered = false;                       // Buffered vs unbuffered channel
    std::size_t buffer_size = 0;                    // Buffer capacity (0 for unbuffered)
    bool transfer_complete = false;                 // Whether recv has matched send
    
    ChannelTransfer() = default;
    ChannelTransfer(std::string name, std::string type, bool buffered = false, std::size_t buf_size = 0)
        : channel_name(std::move(name)), value_type(std::move(type)), 
          is_buffered(buffered), buffer_size(buf_size) {}
};

// ============================================================================
// Safety Contract (for contract annotations) - Phase 5
// ============================================================================

struct SafetyContract {
    enum class Kind {
        Precondition,   // [[expects: condition]]
        Postcondition,  // [[ensures: condition]]
        Assertion,      // [[assert: condition]]
        Invariant       // Type invariant
    };
    
    Kind kind;
    std::string condition;      // The condition expression as string
    std::string message;        // Optional message
    bool is_audit = false;      // Audit-level contract (may be expensive)
    
    SafetyContract() = default;
    SafetyContract(Kind k, std::string cond, std::string msg = "")
        : kind(k), condition(std::move(cond)), message(std::move(msg)) {}
};

// ============================================================================
// Kernel Launch Context (for GPU operations) - Phase 5
// ============================================================================

struct KernelLaunch {
    std::string kernel_name;
    std::string grid_dims;      // e.g., "256,256"
    std::string block_dims;     // e.g., "32"
    std::string memory_policy;  // "coherent", "streaming", "private"
    bool is_active = false;
    
    KernelLaunch() = default;
};

// ============================================================================
// Unified Semantic Info - Phase 5: Single representation for all semantics
// ============================================================================

// Arena Region: JIT memory management scope (Phase 7)
struct ArenaRegion {
    std::size_t scope_id = 0;
    LifetimeRegion* associated_lifetime = nullptr;
    
    ArenaRegion() = default;
    ArenaRegion(std::size_t id, LifetimeRegion* life) 
        : scope_id(id), associated_lifetime(life) {}
};

struct SemanticInfo {
    // Ownership and borrowing
    BorrowInfo borrow;
    
    // Escape analysis
    EscapeInfo escape;
    
    // Memory location
    std::optional<MemoryRegion> memory_region;
    std::optional<MemoryTransfer> active_transfer;
    
    // Concurrency
    std::optional<ChannelTransfer> channel_transfer;
    std::optional<KernelLaunch> kernel_context;
    
    // Arena allocation (Phase 7)
    std::optional<ArenaRegion> arena;

    // Lifetime bounds
    LifetimeRegion lifetime;
    std::vector<LifetimeRegion*> must_outlive;
    
    // Safety contracts
    std::vector<SafetyContract> contracts;
    
    // Constructors
    SemanticInfo() = default;
    
    /// Returns true if this semantic info indicates safe operation
    bool is_safe() const {
        // Not safe if moved (use-after-move)
        if (borrow.kind == OwnershipKind::Moved) {
            return false;
        }
        // Not safe if mutable borrow with existing borrows (aliasing)
        if (borrow.kind == OwnershipKind::MutBorrowed && 
            !borrow.active_borrows.empty()) {
            return false;
        }
        // Async transfers need NoEscape for safety
        if (active_transfer && active_transfer->is_async) {
            return escape.kind == EscapeKind::NoEscape;
        }
        return true;
    }
    
    /// Returns true if this value can be optimized away
    bool can_optimize_away() const {
        // Cannot optimize away if escapes to external memory
        if (escape.kind == EscapeKind::EscapeToGPU ||
            escape.kind == EscapeKind::EscapeToDMA ||
            escape.kind == EscapeKind::EscapeToChannel) {
            return false;
        }
        // Can optimize away if NoEscape
        if (escape.kind == EscapeKind::NoEscape) {
            return true;
        }
        // Can optimize if no transfers and owned with no borrows
        if (!active_transfer && !channel_transfer) {
            if (borrow.kind == OwnershipKind::Owned && 
                borrow.active_borrows.empty()) {
                return true;
            }
        }
        return false;
    }
    
    /// Returns human-readable explanation of semantics
    std::string explain_semantics() const {
        std::string result;
        
        // Ownership
        switch (borrow.kind) {
            case OwnershipKind::Owned: result += "Owned"; break;
            case OwnershipKind::Borrowed: result += "Borrowed"; break;
            case OwnershipKind::MutBorrowed: result += "MutBorrowed"; break;
            case OwnershipKind::Moved: result += "Moved"; break;
        }
        
        // Escape
        result += " | ";
        switch (escape.kind) {
            case EscapeKind::NoEscape: result += "NoEscape"; break;
            case EscapeKind::EscapeToHeap: result += "EscapeToHeap"; break;
            case EscapeKind::EscapeToReturn: result += "EscapeToReturn"; break;
            case EscapeKind::EscapeToParam: result += "EscapeToParam"; break;
            case EscapeKind::EscapeToGlobal: result += "EscapeToGlobal"; break;
            case EscapeKind::EscapeToChannel: result += "EscapeToChannel"; break;
            case EscapeKind::EscapeToGPU: result += "EscapeToGPU"; break;
            case EscapeKind::EscapeToDMA: result += "EscapeToDMA"; break;
        }
        
        // Memory region (uses string name, not enum)
        if (memory_region) {
            result += " | Region[" + memory_region->name + "]";
            if (memory_region->is_device_memory) {
                result += "(device)";
            }
        }
        
        // Concurrency
        if (channel_transfer) {
            result += " | Channel[" + channel_transfer->channel_name + "]";
        }
            if (kernel_context && kernel_context->is_active) {
                result += " | GPU[" + kernel_context->kernel_name + "]";
            }
        
            // Arena
            if (arena) {
                result += " | Arena[" + std::to_string(arena->scope_id) + "]";
            }
            
                // Safety
            
                if (!is_safe()) {
            
                    result += " | UNSAFE";
            
                }
        
        return result;
    }
    
    /// Generate MLIR attributes for this semantic info
    std::string to_mlir_attributes() const {
        std::string attrs;
        
        // Escape kind attribute
        attrs += "escape_kind = #cpp2fir.escape<";
        switch (escape.kind) {
            case EscapeKind::NoEscape: attrs += "no_escape"; break;
            case EscapeKind::EscapeToHeap: attrs += "heap"; break;
            case EscapeKind::EscapeToReturn: attrs += "return"; break;
            case EscapeKind::EscapeToParam: attrs += "param"; break;
            case EscapeKind::EscapeToGlobal: attrs += "global"; break;
            case EscapeKind::EscapeToChannel: attrs += "channel"; break;
            case EscapeKind::EscapeToGPU: attrs += "gpu"; break;
            case EscapeKind::EscapeToDMA: attrs += "dma"; break;
        }
        attrs += ">";
        
        // Memory region if present (uses string name)
        if (memory_region) {
            attrs += ", memory_region = \"" + memory_region->name + "\"";
            if (memory_region->is_device_memory) {
                attrs += ", is_device = true";
            }
        }
        
        // Ownership
        attrs += ", ownership = \"";
        switch (borrow.kind) {
            case OwnershipKind::Owned: attrs += "owned"; break;
            case OwnershipKind::Borrowed: attrs += "borrowed"; break;
            case OwnershipKind::MutBorrowed: attrs += "mut_borrowed"; break;
            case OwnershipKind::Moved: attrs += "moved"; break;
        }
        attrs += "\"";
        
            // Transfer state (MemoryTransfer doesn't have .kind - uses escape_kind)
            if (active_transfer) {
                if (active_transfer->is_async) {
                    attrs += ", async_transfer = true";
                }
            }
        
            // Arena scope
            if (arena) {
                attrs += ", arena_scope = " + std::to_string(arena->scope_id);
            }
            
            return attrs;
        }};

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
        FunctionType,   // Cpp2 function type: (params) -> return_type
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
        Cpp1Lambda,   // C++1 style lambda: [capture](params) { body }
        Is,
        As,
        StringInterpolation,
        Range,
        List,
        StructInitializer,
        MetafunctionCall,
        ContractExpression,
        InspectExpr,  // Cpp2 inspect expression
        Move,         // Cpp2 move/forward/copy expression
        Pipeline,     // Cpp2 pipeline operator |>
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

// Cpp2 pipeline operator expression (|>)
// left |> right is equivalent to right(left) syntactically
// Enables left-to-right function composition
struct PipelineExpression : Expression {
    std::unique_ptr<Expression> left;   // Input value/expression
    std::unique_ptr<Expression> right;  // Function to apply

    PipelineExpression(std::unique_ptr<Expression> lhs,
                      std::unique_ptr<Expression> rhs, std::size_t l)
        : Expression(Kind::Pipeline, l), left(std::move(lhs)),
          right(std::move(rhs)) {}
};

// Cpp2 move/forward/copy expressions
struct MoveExpression : Expression {
    TokenType op;  // Move, Forward, or Copy
    std::unique_ptr<Expression> operand;

    MoveExpression(TokenType o, std::unique_ptr<Expression> expr, std::size_t l)
        : Expression(Kind::Move, l), op(o), operand(std::move(expr)) {}
};

struct CallExpression : Expression {
    struct Argument {
        std::unique_ptr<Expression> expr;
        ParameterQualifier qualifier = ParameterQualifier::None;  // out, inout, move, forward
    };
    
    std::unique_ptr<Expression> callee;
    std::vector<Argument> arguments;  // Arguments with qualifiers
    std::vector<std::unique_ptr<Expression>> args;  // DEPRECATED: plain args for backward compat
    std::vector<std::unique_ptr<Type>> template_args;  // Template arguments for template function calls
    bool is_ufcs = false;  // Cpp2 UFCS (Unified Function Call Syntax)
    bool is_template_instantiation = false;  // True for template instantiation without call (e.g., std::integral<T>)

    CallExpression(std::unique_ptr<Expression> c, std::size_t l)
        : Expression(Kind::Call, l), callee(std::move(c)) {}
};

struct MemberAccessExpression : Expression {
    std::unique_ptr<Expression> object;
    std::string member;
    bool explicit_non_ufcs = false;  // true for obj..member (explicit non-UFCS syntax)

    MemberAccessExpression(std::unique_ptr<Expression> obj, std::string m, std::size_t l, bool non_ufcs = false)
        : Expression(Kind::MemberAccess, l), object(std::move(obj)), member(std::move(m)), explicit_non_ufcs(non_ufcs) {}
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
    std::vector<std::string> template_params;
    bool is_constexpr = false;  // Cpp2: == for compile-time function expressions

    LambdaExpression(std::size_t l) : Expression(Kind::Lambda, l) {}
    ~LambdaExpression();
};

// C++1 style lambda: [capture](params) -> return_type { body }
struct Cpp1LambdaExpression : Expression {
    struct Capture {
        enum class Mode {
            ByCopy,       // x
            ByRef,        // &x
            DefaultCopy,  // =
            DefaultRef,   // &
            This          // this
        };
        Mode mode = Mode::ByCopy;
        std::string name;  // Empty for default captures
    };
    
    struct Parameter {
        std::string name;
        std::string type_str;  // C++1 type as string
    };
    
    std::vector<Capture> captures;
    std::vector<Parameter> parameters;
    std::unique_ptr<Type> return_type;
    std::vector<std::shared_ptr<Statement>> body;
    
    Cpp1LambdaExpression(std::size_t l) : Expression(Kind::Cpp1Lambda, l) {}
    ~Cpp1LambdaExpression();
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
    std::optional<std::string> annotation;  // Contract annotation level (e.g., "bounds_safety")
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
        DoWhile,
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
        ScopeBlock,  // Cpp2: (copy x := value) { body }
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
    std::optional<std::size_t> arena_scope_id;

    BlockStatement(std::size_t l) : Statement(Kind::Block, l) {}
};

// Loop initializer for Cpp2: (copy i:=0) while/for/do or (copy x := value) { body }
struct LoopInitializer {
    std::string name;           // Variable name
    std::string param_kind;     // "copy", "move", "inout", or empty
    std::unique_ptr<Type> type; // Variable type (auto if type-deduced)
    std::unique_ptr<Expression> initializer; // Initial value

    LoopInitializer() = default;
    LoopInitializer(std::string n, std::string pk, std::unique_ptr<Type> t, std::unique_ptr<Expression> init)
        : name(std::move(n)), param_kind(std::move(pk)), type(std::move(t)), initializer(std::move(init)) {}
};

// Cpp2 scope block with initializer: (copy x := value) { body }
struct ScopeBlockStatement : Statement {
    std::vector<std::unique_ptr<LoopInitializer>> initializers;
    std::unique_ptr<Statement> body;

    ScopeBlockStatement(std::vector<std::unique_ptr<LoopInitializer>> inits, std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::ScopeBlock, l), initializers(std::move(inits)), body(std::move(b)) {}
};

struct IfStatement : Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> then_stmt;
    std::unique_ptr<Statement> else_stmt;
    bool is_constexpr = false;  // 'if constexpr' for compile-time branching

    IfStatement(std::unique_ptr<Expression> cond,
               std::unique_ptr<Statement> then_s,
               std::unique_ptr<Statement> else_s, std::size_t l,
               bool constexpr_if = false)
        : Statement(Kind::If, l),
          condition(std::move(cond)),
          then_stmt(std::move(then_s)),
          else_stmt(std::move(else_s)),
          is_constexpr(constexpr_if) {}
};

struct WhileStatement : Statement {
    std::string label;  // Optional label for labeled break/continue
    std::vector<std::unique_ptr<LoopInitializer>> loop_inits;  // Cpp2 loop initializers: (copy i:=0, copy t:=0.0) while ...
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

    WhileStatement(std::string label,
                  std::unique_ptr<Expression> cond,
                  std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::While, l),
          label(std::move(label)),
          condition(std::move(cond)),
          body(std::move(b)) {}

    WhileStatement(std::string label,
                  std::unique_ptr<Expression> cond,
                  std::unique_ptr<Expression> inc,
                  std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::While, l),
          label(std::move(label)),
          condition(std::move(cond)),
          increment(std::move(inc)),
          body(std::move(b)) {}
};

struct DoWhileStatement : Statement {
    std::string label;  // Optional label for labeled break/continue
    std::vector<std::unique_ptr<LoopInitializer>> loop_inits;  // Cpp2 loop initializers: (copy i:=0, copy t:=0.0) do ...
    std::unique_ptr<Statement> body;
    std::unique_ptr<Expression> increment; // Cpp2 'next' clause (between body and while)
    std::unique_ptr<Expression> condition;

    DoWhileStatement(std::unique_ptr<Statement> b,
                    std::unique_ptr<Expression> cond, std::size_t l)
        : Statement(Kind::DoWhile, l),
          body(std::move(b)),
          condition(std::move(cond)) {}

    DoWhileStatement(std::string label,
                    std::unique_ptr<Statement> b,
                    std::unique_ptr<Expression> cond, std::size_t l)
        : Statement(Kind::DoWhile, l),
          label(std::move(label)),
          body(std::move(b)),
          condition(std::move(cond)) {}

    DoWhileStatement(std::unique_ptr<Statement> b,
                    std::unique_ptr<Expression> inc,
                    std::unique_ptr<Expression> cond, std::size_t l)
        : Statement(Kind::DoWhile, l),
          body(std::move(b)),
          increment(std::move(inc)),
          condition(std::move(cond)) {}

    DoWhileStatement(std::string label,
                    std::unique_ptr<Statement> b,
                    std::unique_ptr<Expression> inc,
                    std::unique_ptr<Expression> cond, std::size_t l)
        : Statement(Kind::DoWhile, l),
          label(std::move(label)),
          body(std::move(b)),
          increment(std::move(inc)),
          condition(std::move(cond)) {}
};

struct ForStatement : Statement {
    std::string label;  // Optional label for labeled break/continue
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

    ForStatement(std::string label,
                std::unique_ptr<Statement> i,
                std::unique_ptr<Expression> cond,
                std::unique_ptr<Expression> inc,
                std::unique_ptr<Statement> b, std::size_t l)
        : Statement(Kind::For, l),
          label(std::move(label)),
          init(std::move(i)),
          condition(std::move(cond)),
          increment(std::move(inc)),
          body(std::move(b)) {}
};

struct ForRangeStatement : Statement {
    std::string label;  // Optional label for labeled break/continue
    std::vector<std::unique_ptr<LoopInitializer>> loop_inits;  // Cpp2 loop initializers: (copy i:=0, copy t:=0.0) for ...
    std::string variable;
    std::string var_qualifier;  // Parameter kind: in, inout, out, copy, move, forward
    std::unique_ptr<Type> var_type;
    std::unique_ptr<Expression> range;
    std::unique_ptr<Expression> next_clause;  // Optional: for range next expr do (var)
    std::unique_ptr<Statement> body;

    ForRangeStatement(std::string var, std::unique_ptr<Type> t,
                     std::unique_ptr<Expression> r,
                     std::unique_ptr<Statement> b, std::size_t l,
                     std::string qualifier = "")
        : Statement(Kind::ForRange, l),
          variable(std::move(var)),
          var_qualifier(std::move(qualifier)),
          var_type(std::move(t)),
          range(std::move(r)),
          body(std::move(b)) {}

    ForRangeStatement(std::string label, std::string var, std::unique_ptr<Type> t,
                     std::unique_ptr<Expression> r,
                     std::unique_ptr<Statement> b, std::size_t l,
                     std::string qualifier = "")
        : Statement(Kind::ForRange, l),
          label(std::move(label)),
          variable(std::move(var)),
          var_qualifier(std::move(qualifier)),
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
    std::string label;  // Optional label for breaking to specific loop

    BreakStatement(std::size_t l) : Statement(Kind::Break, l) {}
    BreakStatement(std::string label, std::size_t l) : Statement(Kind::Break, l), label(std::move(label)) {}
};

struct ContinueStatement : Statement {
    std::string label;  // Optional label for continuing to specific loop

    ContinueStatement(std::size_t l) : Statement(Kind::Continue, l) {}
    ContinueStatement(std::string label, std::size_t l) : Statement(Kind::Continue, l), label(std::move(label)) {}
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
        Export,
        Cpp1Passthrough  // C++1 code to pass through unchanged
    };

    Kind kind;
    std::size_t line;
    std::string name;

    // Markdown blocks attached to this declaration (CAS-linked modules)
    std::vector<MarkdownBlockAttr> markdown_blocks;
    
    // Unified semantic information (Phase 5) - consolidated ownership, escape,
    // memory, concurrency, lifetime, and contract information
    std::unique_ptr<SemanticInfo> semantic_info;

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

    // Escape analysis integration (Phase 3: External Memory)
    std::unique_ptr<EscapeInfo> escape_info;      // Escape analysis results
    std::unique_ptr<MemoryTransfer> memory_transfer;  // GPU/DMA transfer tracking
    
    // Channelized concurrency integration (Phase 4)
    std::unique_ptr<ChannelTransfer> channel_transfer;  // Channel send/recv tracking

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

    // Named return parameter (e.g., "-> (result: int)" or "-> (result: int = 0)")
    struct NamedReturn {
        std::string name;
        std::unique_ptr<Type> type;
        std::unique_ptr<Expression> default_value;  // Optional default value
    };

    std::vector<Parameter> parameters;
    std::unique_ptr<Type> return_type;
    std::vector<NamedReturn> named_returns;  // Cpp2 named return parameters
    std::unique_ptr<Statement> body;
    std::vector<std::string> contract_groups;
    std::vector<std::string> template_parameters;  // Template parameters for generic functions
    bool is_virtual = false;
    bool is_override = false;
    bool is_final = false;
    bool is_explicit = false;
    bool is_forward_return = false;  // forward return type (pass-through semantics)

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

    struct BaseClass {
        std::unique_ptr<Type> type;
        std::unique_ptr<Expression> initializer;  // Optional: = ()
    };

    TypeKind type_kind;
    std::vector<std::unique_ptr<Declaration>> members;
    std::vector<BaseClass> base_classes;  // For "this: BaseType = ();" syntax
    std::unique_ptr<Type> underlying_type;
    std::vector<std::string> metafunctions;
    std::vector<std::string> template_parameters;  // Template parameters for generic types
    std::unique_ptr<Expression> requires_clause;   // Optional requires constraint

    TypeDeclaration(std::string n, TypeKind tk, std::size_t l)
        : Declaration(Kind::Type, std::move(n), l), type_kind(tk) {}
};

struct NamespaceDeclaration : Declaration {
    std::vector<std::unique_ptr<Declaration>> members;
    std::string alias_target;  // For namespace aliases: "N1: namespace == N;"

    NamespaceDeclaration(std::string n, std::size_t l)
        : Declaration(Kind::Namespace, std::move(n), l) {}
};

struct OperatorDeclaration : Declaration {
    std::unique_ptr<Type> return_type;
    std::vector<std::unique_ptr<FunctionDeclaration::Parameter>> parameters;
    std::unique_ptr<Statement> body;
    bool is_forward_return = false;  // For -> forward T syntax
    bool is_constexpr = false;       // For == (compile-time) syntax
    std::vector<std::string> template_parameters;  // Template parameters for generic operators
    std::vector<std::pair<std::string, std::unique_ptr<Type>>> named_returns;  // Named return values

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

// C++1 passthrough declaration - stores raw C++1 code to emit unchanged
struct Cpp1PassthroughDeclaration : Declaration {
    std::string raw_code;  // Raw C++1 source code

    Cpp1PassthroughDeclaration(std::string code, std::size_t l)
        : Declaration(Kind::Cpp1Passthrough, "", l), raw_code(std::move(code)) {}
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
    inline Cpp1LambdaExpression::~Cpp1LambdaExpression() = default;
} // namespace cpp2_transpiler