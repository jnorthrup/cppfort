#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

// -----------------------------------------------------------------------------
// CRTP Enum Helper (Java‑style enum pattern)
// -----------------------------------------------------------------------------
// Allows enum classes to expose a static list of their values and utility functions
// such as conversion to underlying integer and back, without needing macro magic.
// Example usage:
//
//   enum class Color { Red, Green, Blue };
//   using ColorEnum = EnumBase<Color>;
//
template <typename Enum>
struct EnumBase {
    static_assert(std::is_enum<Enum>::value, "Enum must be an enum type");
    using underlying_t = typename std::underlying_type<Enum>::type;
    static constexpr std::size_t size = []{
        // compile‑time count of enum values; works for contiguous enums starting at 0
        Enum max = Enum::Red; // dummy to avoid unused warnings; will be overwritten
        (void)max;
        return static_cast<std::size_t>(Enum::Count);
    }();

    // Provide an array of all enum values (requires the enum to define Count as last)
    static constexpr std::array<Enum, size> values = []{
        std::array<Enum, size> arr{};
        for (std::size_t i = 0; i < size; ++i) {
            arr[i] = static_cast<Enum>(i);
        }
        return arr;
    }();

    static constexpr underlying_t toUnderlying(Enum e) noexcept {
        return static_cast<underlying_t>(e);
    }

    static constexpr Enum fromUnderlying(underlying_t u) noexcept {
        return static_cast<Enum>(u);
    }
};

// -----------------------------------------------------------------------------
// NodeKind Enum - Band 5: Comprehensive node classification for pattern matching
// -----------------------------------------------------------------------------
enum class NodeKind {
    // ============================================================================
    // Control Flow Nodes (Band 1-2) - Range: 0-99
    // ============================================================================
    CFG_START = 0,

    START = CFG_START,
    REGION,
    PHI,
    PROJ,
    IF,
    LOOP,
    RETURN,
    CALL,
    CALL_END,
    THROW,
    CATCH,
    SWITCH,
    BREAK,
    CONTINUE,
    GOTO,

    CFG_END = 99,

    // ============================================================================
    // Data Flow Nodes (Band 2) - Range: 100-199
    // ============================================================================
    DATA_START = 100,

    LOAD = DATA_START,
    STORE,
    ALLOC,
    NEW = ALLOC,  // Alias for ALLOC (Chapter 10/19 compat)
    FREE,
    ARRAY_LOAD,
    ARRAY_STORE,
    FIELD_LOAD,
    FIELD_STORE,
    CAST,
    CONVERT,

    DATA_END = 199,

    // ============================================================================
    // Arithmetic Nodes (Band 1) - Range: 200-299
    // ============================================================================
    ARITH_START = 200,

    ADD = ARITH_START,
    SUB,
    MUL,
    DIV,
    MOD,
    NEG,
    ABS,
    MIN,
    MAX,
    SQRT,
    POW,

    ARITH_END = 299,

    // ============================================================================
    // Bitwise Operation Nodes (Band 5 - Chapter 16) - Range: 300-399
    // ============================================================================
    BITWISE_START = 300,

    AND = BITWISE_START,
    OR,
    XOR,
    SHL,
    ASHR,
    LSHR,
    NOT,
    BIT_COUNT,
    BIT_REVERSE,
    ROTATE_LEFT,
    ROTATE_RIGHT,

    BITWISE_END = 399,

    // ============================================================================
    // Comparison Nodes (Band 1) - Range: 400-499
    // ============================================================================
    COMPARE_START = 400,

    EQ = COMPARE_START,
    NE,
    LT,
    LE,
    GT,
    GE,
    CMP,

    COMPARE_END = 499,

    // ============================================================================
    // Boolean Operation Nodes (Band 5) - Range: 500-599
    // ============================================================================
    BOOL_START = 500,

    BOOL_AND = BOOL_START,
    BOOL_OR,
    BOOL_NOT,
    BOOL_XOR,

    BOOL_END = 599,

    // ============================================================================
    // Floating Point Operation Nodes (Band 4) - Range: 600-699
    // ============================================================================
    FLOAT_START = 600,

    FADD = FLOAT_START,
    FSUB,
    FMUL,
    FDIV,
    FNEG,
    FABS,
    FMIN,
    FMAX,
    FSQRT,
    FPOW,
    FCMP,

    FLOAT_END = 699,

    // ============================================================================
    // Memory Operation Nodes (Band 2+) - Range: 700-799
    // ============================================================================
    MEMORY_START = 700,

    MEMCPY = MEMORY_START,
    MEMSET,
    MEMCMP,
    MALLOC,
    REALLOC,
    CALLOC,

    MEMORY_END = 799,

    // ============================================================================
    // Constants and Parameters - Range: 800-899
    // ============================================================================
    CONSTANT_START = 800,

    CONSTANT = CONSTANT_START,
    PARAMETER,
    PARM = PARAMETER,  // Alias for PARAMETER (Chapter 18/19 compat)
    FUNCTION,
    FUN = FUNCTION,    // Alias for FUNCTION (Chapter 18/19 compat)
    GLOBAL_VAR,
    STRING_LITERAL,
    NULL_PTR,

    CONSTANT_END = 899,

    // ============================================================================
    // C Language-Specific Nodes - Range: 900-999
    // ============================================================================
    C_START = 900,

    POINTER_DEREF = C_START,
    ADDRESS_OF,
    ARRAY_SUBSCRIPT,
    COMPOUND_LITERAL,
    DESIGNATED_INIT,
    STRUCT_MEMBER,
    UNION_MEMBER,
    SIZEOF_EXPR,
    ALIGNOF_EXPR,
    OFFSETOF_EXPR,
    VA_START,
    VA_ARG,
    VA_END,
    FLEXIBLE_ARRAY,

    C_END = 999,

    // ============================================================================
    // C++ Language-Specific Nodes - Range: 1000-1099
    // ============================================================================
    CPP_START = 1000,

    VIRTUAL_CALL = CPP_START,
    MEMBER_ACCESS,
    NEW_EXPR,
    DELETE_EXPR,
    DELETE_ARRAY_EXPR,
    THROW_EXPR,
    CATCH_HANDLER,
    TRY_BLOCK,
    DYNAMIC_CAST,
    STATIC_CAST,
    REINTERPRET_CAST,
    CONST_CAST,
    TYPEID_EXPR,
    THIS_EXPR,
    LAMBDA_EXPR,
    TEMPLATE_INSTANTIATION,
    OPERATOR_OVERLOAD,
    RVALUE_REF,
    MOVE_EXPR,
    PERFECT_FORWARD,

    CPP_END = 1099,

    // ============================================================================
    // CPP2 Safety/Contract Nodes - Range: 1100-1199
    // ============================================================================
    CPP2_START = 1100,

    CONTRACT_PRE = CPP2_START,
    CONTRACT_POST,
    CONTRACT_ASSERT,
    BOUNDS_CHECK,
    NULL_CHECK,
    DEFINITE_INIT_CHECK,
    PARAM_IN,
    PARAM_OUT,
    PARAM_INOUT,
    PARAM_MOVE,
    PARAM_FORWARD,
    LIFETIME_CAPTURE,
    LIFETIME_EXTEND,
    UNSAFE_REGION,
    INSPECT_EXPR,
    IS_EXPR,
    AS_EXPR,

    CPP2_END = 1199,

    // ============================================================================
    // Attestation & Security Nodes - Range: 1200-1299
    // ============================================================================
    ATTESTATION_START = 1200,

    MERKLE_CHECKPOINT = ATTESTATION_START,
    HASH_WITNESS,
    SIGNATURE_POINT,
    TAMPER_GUARD,
    INJECTION_BARRIER,
    DETERMINISM_FENCE,

    ATTESTATION_END = 1299,

    // ============================================================================
    // Sentinel - Must be last
    // ============================================================================
    Count
};

using NodeKindEnum = EnumBase<NodeKind>;

// -----------------------------------------------------------------------------
// NodeKind String Conversion - Band 5
// -----------------------------------------------------------------------------
inline const char* nodeKindToString(NodeKind kind) {
    switch (kind) {
        // Control Flow
        case NodeKind::START: return "START";
        case NodeKind::REGION: return "REGION";
        case NodeKind::PHI: return "PHI";
        case NodeKind::PROJ: return "PROJ";
        case NodeKind::IF: return "IF";
        case NodeKind::LOOP: return "LOOP";
        case NodeKind::RETURN: return "RETURN";
        case NodeKind::CALL: return "CALL";
        case NodeKind::CALL_END: return "CALL_END";
        case NodeKind::THROW: return "THROW";
        case NodeKind::CATCH: return "CATCH";
        case NodeKind::SWITCH: return "SWITCH";
        case NodeKind::BREAK: return "BREAK";
        case NodeKind::CONTINUE: return "CONTINUE";
        case NodeKind::GOTO: return "GOTO";

        // Data Flow
        case NodeKind::LOAD: return "LOAD";
        case NodeKind::STORE: return "STORE";
        case NodeKind::ALLOC: return "ALLOC";
        case NodeKind::FREE: return "FREE";
        case NodeKind::ARRAY_LOAD: return "ARRAY_LOAD";
        case NodeKind::ARRAY_STORE: return "ARRAY_STORE";
        case NodeKind::FIELD_LOAD: return "FIELD_LOAD";
        case NodeKind::FIELD_STORE: return "FIELD_STORE";
        case NodeKind::CAST: return "CAST";
        case NodeKind::CONVERT: return "CONVERT";

        // Arithmetic
        case NodeKind::ADD: return "ADD";
        case NodeKind::SUB: return "SUB";
        case NodeKind::MUL: return "MUL";
        case NodeKind::DIV: return "DIV";
        case NodeKind::MOD: return "MOD";
        case NodeKind::NEG: return "NEG";
        case NodeKind::ABS: return "ABS";
        case NodeKind::MIN: return "MIN";
        case NodeKind::MAX: return "MAX";
        case NodeKind::SQRT: return "SQRT";
        case NodeKind::POW: return "POW";

        // Bitwise
        case NodeKind::AND: return "AND";
        case NodeKind::OR: return "OR";
        case NodeKind::XOR: return "XOR";
        case NodeKind::SHL: return "SHL";
        case NodeKind::ASHR: return "ASHR";
        case NodeKind::LSHR: return "LSHR";
        case NodeKind::NOT: return "NOT";
        case NodeKind::BIT_COUNT: return "BIT_COUNT";
        case NodeKind::BIT_REVERSE: return "BIT_REVERSE";
        case NodeKind::ROTATE_LEFT: return "ROTATE_LEFT";
        case NodeKind::ROTATE_RIGHT: return "ROTATE_RIGHT";

        // Comparison
        case NodeKind::EQ: return "EQ";
        case NodeKind::NE: return "NE";
        case NodeKind::LT: return "LT";
        case NodeKind::LE: return "LE";
        case NodeKind::GT: return "GT";
        case NodeKind::GE: return "GE";
        case NodeKind::CMP: return "CMP";

        // Boolean
        case NodeKind::BOOL_AND: return "BOOL_AND";
        case NodeKind::BOOL_OR: return "BOOL_OR";
        case NodeKind::BOOL_NOT: return "BOOL_NOT";
        case NodeKind::BOOL_XOR: return "BOOL_XOR";

        // Float
        case NodeKind::FADD: return "FADD";
        case NodeKind::FSUB: return "FSUB";
        case NodeKind::FMUL: return "FMUL";
        case NodeKind::FDIV: return "FDIV";
        case NodeKind::FNEG: return "FNEG";
        case NodeKind::FABS: return "FABS";
        case NodeKind::FMIN: return "FMIN";
        case NodeKind::FMAX: return "FMAX";
        case NodeKind::FSQRT: return "FSQRT";
        case NodeKind::FPOW: return "FPOW";
        case NodeKind::FCMP: return "FCMP";

        // Memory
        case NodeKind::MEMCPY: return "MEMCPY";
        case NodeKind::MEMSET: return "MEMSET";
        case NodeKind::MEMCMP: return "MEMCMP";
        case NodeKind::MALLOC: return "MALLOC";
        case NodeKind::REALLOC: return "REALLOC";
        case NodeKind::CALLOC: return "CALLOC";

        // Constants
        case NodeKind::CONSTANT: return "CONSTANT";
        case NodeKind::PARAMETER: return "PARAMETER";
        case NodeKind::FUNCTION: return "FUNCTION";
        case NodeKind::GLOBAL_VAR: return "GLOBAL_VAR";
        case NodeKind::STRING_LITERAL: return "STRING_LITERAL";
        case NodeKind::NULL_PTR: return "NULL_PTR";

        // C-specific
        case NodeKind::POINTER_DEREF: return "POINTER_DEREF";
        case NodeKind::ADDRESS_OF: return "ADDRESS_OF";
        case NodeKind::ARRAY_SUBSCRIPT: return "ARRAY_SUBSCRIPT";
        case NodeKind::COMPOUND_LITERAL: return "COMPOUND_LITERAL";
        case NodeKind::DESIGNATED_INIT: return "DESIGNATED_INIT";
        case NodeKind::STRUCT_MEMBER: return "STRUCT_MEMBER";
        case NodeKind::UNION_MEMBER: return "UNION_MEMBER";
        case NodeKind::SIZEOF_EXPR: return "SIZEOF_EXPR";
        case NodeKind::ALIGNOF_EXPR: return "ALIGNOF_EXPR";
        case NodeKind::OFFSETOF_EXPR: return "OFFSETOF_EXPR";
        case NodeKind::VA_START: return "VA_START";
        case NodeKind::VA_ARG: return "VA_ARG";
        case NodeKind::VA_END: return "VA_END";
        case NodeKind::FLEXIBLE_ARRAY: return "FLEXIBLE_ARRAY";

        // C++-specific
        case NodeKind::VIRTUAL_CALL: return "VIRTUAL_CALL";
        case NodeKind::MEMBER_ACCESS: return "MEMBER_ACCESS";
        case NodeKind::NEW_EXPR: return "NEW_EXPR";
        case NodeKind::DELETE_EXPR: return "DELETE_EXPR";
        case NodeKind::DELETE_ARRAY_EXPR: return "DELETE_ARRAY_EXPR";
        case NodeKind::THROW_EXPR: return "THROW_EXPR";
        case NodeKind::CATCH_HANDLER: return "CATCH_HANDLER";
        case NodeKind::TRY_BLOCK: return "TRY_BLOCK";
        case NodeKind::DYNAMIC_CAST: return "DYNAMIC_CAST";
        case NodeKind::STATIC_CAST: return "STATIC_CAST";
        case NodeKind::REINTERPRET_CAST: return "REINTERPRET_CAST";
        case NodeKind::CONST_CAST: return "CONST_CAST";
        case NodeKind::TYPEID_EXPR: return "TYPEID_EXPR";
        case NodeKind::THIS_EXPR: return "THIS_EXPR";
        case NodeKind::LAMBDA_EXPR: return "LAMBDA_EXPR";
        case NodeKind::TEMPLATE_INSTANTIATION: return "TEMPLATE_INSTANTIATION";
        case NodeKind::OPERATOR_OVERLOAD: return "OPERATOR_OVERLOAD";
        case NodeKind::RVALUE_REF: return "RVALUE_REF";
        case NodeKind::MOVE_EXPR: return "MOVE_EXPR";
        case NodeKind::PERFECT_FORWARD: return "PERFECT_FORWARD";

        // CPP2 safety/contracts
        case NodeKind::CONTRACT_PRE: return "CONTRACT_PRE";
        case NodeKind::CONTRACT_POST: return "CONTRACT_POST";
        case NodeKind::CONTRACT_ASSERT: return "CONTRACT_ASSERT";
        case NodeKind::BOUNDS_CHECK: return "BOUNDS_CHECK";
        case NodeKind::NULL_CHECK: return "NULL_CHECK";
        case NodeKind::DEFINITE_INIT_CHECK: return "DEFINITE_INIT_CHECK";
        case NodeKind::PARAM_IN: return "PARAM_IN";
        case NodeKind::PARAM_OUT: return "PARAM_OUT";
        case NodeKind::PARAM_INOUT: return "PARAM_INOUT";
        case NodeKind::PARAM_MOVE: return "PARAM_MOVE";
        case NodeKind::PARAM_FORWARD: return "PARAM_FORWARD";
        case NodeKind::LIFETIME_CAPTURE: return "LIFETIME_CAPTURE";
        case NodeKind::LIFETIME_EXTEND: return "LIFETIME_EXTEND";
        case NodeKind::UNSAFE_REGION: return "UNSAFE_REGION";
        case NodeKind::INSPECT_EXPR: return "INSPECT_EXPR";
        case NodeKind::IS_EXPR: return "IS_EXPR";
        case NodeKind::AS_EXPR: return "AS_EXPR";

        // Attestation
        case NodeKind::MERKLE_CHECKPOINT: return "MERKLE_CHECKPOINT";
        case NodeKind::HASH_WITNESS: return "HASH_WITNESS";
        case NodeKind::SIGNATURE_POINT: return "SIGNATURE_POINT";
        case NodeKind::TAMPER_GUARD: return "TAMPER_GUARD";
        case NodeKind::INJECTION_BARRIER: return "INJECTION_BARRIER";
        case NodeKind::DETERMINISM_FENCE: return "DETERMINISM_FENCE";

        default: return "UNKNOWN";
    }
}

// -----------------------------------------------------------------------------
// MultiIndex2D – O(1) 2‑dimensional container
// -----------------------------------------------------------------------------
// Provides random access via (row, col) as well as a row‑wise view.
// The container stores data in a flat contiguous vector, guaranteeing O(1)
// access for element lookup and row iteration.
//
// Template parameters:
//   T      – element type
//   Rows   – number of rows (must be >0)
//   Cols   – number of columns (must be >0)
//
template <typename T, std::size_t Rows, std::size_t Cols>
requires (Rows > 0 && Cols > 0)
class MultiIndex2D {
public:
    using value_type      = T;
    using size_type       = std::size_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;

    // Default construction zero‑initializes storage
    constexpr MultiIndex2D() : _data{} {}

    // Construct from initializer list of rows (each row is an initializer list of Cols)
    constexpr MultiIndex2D(std::initializer_list<std::initializer_list<T>> init) {
        std::size_t r = 0;
        for (auto row_it = init.begin(); row_it != init.end() && r < Rows; ++row_it, ++r) {
            std::size_t c = 0;
            for (auto col_it = row_it->begin(); col_it != row_it->end() && c < Cols; ++col_it, ++c) {
                (*this)(r, c) = *col_it;
            }
        }
    }

    // Element access (row, col) – O(1)
    [[nodiscard]] constexpr reference       operator()(size_type row, size_type col) noexcept {
        return _data[row * Cols + col];
    }
    [[nodiscard]] constexpr const_reference operator()(size_type row, size_type col) const noexcept {
        return _data[row * Cols + col];
    }

    // Row view – returns a mutable span representing the whole row
    [[nodiscard]] constexpr std::span<T, Cols> row(size_type row) noexcept {
        return std::span<T, Cols>(&_data[row * Cols], Cols);
    }
    [[nodiscard]] constexpr std::span<const T, Cols> row(size_type row) const noexcept {
        return std::span<const T, Cols>(&_data[row * Cols], Cols);
    }

    // Simple query helpers
    static constexpr size_type rows() noexcept { return Rows; }
    static constexpr size_type cols() noexcept { return Cols; }
    static constexpr size_type size() noexcept { return Rows * Cols; }

    // Fill whole container with a value
    constexpr void fill(const T& value) noexcept {
        for (auto& elem : _data) elem = value;
    }

    // Direct raw access – if you need the underlying array
    [[nodiscard]] constexpr std::array<T, Rows * Cols>& data() noexcept { return _data; }
    [[nodiscard]] constexpr const std::array<T, Rows * Cols>& data() const noexcept { return _data; }

private:
    std::array<T, Rows * Cols> _data;
};

 