#pragma once

#include <string>

namespace cppfort::ir {

// Centralized enumeration of all node kinds used by the Sea-of-Nodes IR.
// Values are arranged in contiguous ranges so that predicates in
// `node_category.h` (e.g. CFG_START..CFG_END) work correctly.
enum class NodeKind {
    // Control-flow kinds (CFG range)
    CFG_START,
    START,
    PROJ,           // projection (data/ctrl)
    IF,
    REGION,
    LOOP,
    PHI,
    RETURN,
    STOP,
    CALL,
    CALL_END,
    BREAK,
    CONTINUE,
    TAMPER_GUARD,
    MERKLE_CHECKPOINT,
    SIGNATURE_POINT,
    INJECTION_BARRIER,
    DETERMINISM_FENCE,
    // End of CFG kinds
    CFG_END,

    // General data/node kinds (DATA range)
    DATA_START,

    // Constants & witnesses
    CONSTANT_START,
    CONSTANT,
    HASH_WITNESS,
    CONSTANT_END,

    // Arithmetic operations (ARITH range)
    ARITH_START,
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    ARITH_END,

    // Bitwise operations (BITWISE range)
    BITWISE_START,
    AND,
    OR,
    XOR,
    SHL,
    LSHR,
    ASHR,
    BIT_COUNT,
    BIT_REVERSE,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    BITWISE_END,

    // Comparison ops (COMPARE range)
    COMPARE_START,
    EQ,
    LT,
    GT,
    LE,
    GE,
    NE,
    COMPARE_END,

    // Boolean ops (BOOL range)
    BOOL_START,
    BOOL_AND,
    BOOL_OR,
    NOT,
    BOOL_END,

    // Floating point ops (FLOAT range)
    FLOAT_START,
    FADD,
    FMUL,
    FNEG,
    FABS,
    FLOAT_END,

    // Memory operations (MEMORY range)
    MEMORY_START,
    ALLOC,
    LOAD,
    STORE,
    NEWARRAY,
    ARRAY_LOAD,
    ARRAY_STORE,
    MEMORY_END,

    // Casts, functions, parameters and others
    CAST,
    FUNCTION,
    PARAMETER,
    // Generic placeholder for unknown/extension kinds
    UNKNOWN,
};

// Helper: convert NodeKind to human-readable string
inline const char* nodeKindToString(NodeKind k) {
    switch (k) {
        case NodeKind::START: return "START";
        case NodeKind::PROJ: return "PROJ";
        case NodeKind::IF: return "IF";
        case NodeKind::REGION: return "REGION";
        case NodeKind::LOOP: return "LOOP";
        case NodeKind::PHI: return "PHI";
        case NodeKind::RETURN: return "RETURN";
        case NodeKind::STOP: return "STOP";
        case NodeKind::CALL: return "CALL";
        case NodeKind::CALL_END: return "CALL_END";
        case NodeKind::BREAK: return "BREAK";
        case NodeKind::CONTINUE: return "CONTINUE";
        case NodeKind::TAMPER_GUARD: return "TAMPER_GUARD";
        case NodeKind::MERKLE_CHECKPOINT: return "MERKLE_CHECKPOINT";
        case NodeKind::SIGNATURE_POINT: return "SIGNATURE_POINT";
        case NodeKind::INJECTION_BARRIER: return "INJECTION_BARRIER";
        case NodeKind::DETERMINISM_FENCE: return "DETERMINISM_FENCE";

        case NodeKind::CONSTANT: return "CONSTANT";
        case NodeKind::HASH_WITNESS: return "HASH_WITNESS";

        case NodeKind::ADD: return "ADD";
        case NodeKind::SUB: return "SUB";
        case NodeKind::MUL: return "MUL";
        case NodeKind::DIV: return "DIV";
        case NodeKind::NEG: return "NEG";

        case NodeKind::AND: return "AND";
        case NodeKind::OR: return "OR";
        case NodeKind::XOR: return "XOR";
        case NodeKind::SHL: return "SHL";
        case NodeKind::LSHR: return "LSHR";
        case NodeKind::ASHR: return "ASHR";
        case NodeKind::BIT_COUNT: return "BIT_COUNT";
        case NodeKind::BIT_REVERSE: return "BIT_REVERSE";
        case NodeKind::ROTATE_LEFT: return "ROTATE_LEFT";
        case NodeKind::ROTATE_RIGHT: return "ROTATE_RIGHT";

        case NodeKind::EQ: return "EQ";
        case NodeKind::LT: return "LT";
        case NodeKind::GT: return "GT";
        case NodeKind::LE: return "LE";
        case NodeKind::GE: return "GE";
        case NodeKind::NE: return "NE";

        case NodeKind::BOOL_AND: return "BOOL_AND";
        case NodeKind::BOOL_OR: return "BOOL_OR";
        case NodeKind::NOT: return "NOT";

        case NodeKind::FADD: return "FADD";
        case NodeKind::FMUL: return "FMUL";
        case NodeKind::FNEG: return "FNEG";
        case NodeKind::FABS: return "FABS";

        case NodeKind::ALLOC: return "ALLOC";
        case NodeKind::LOAD: return "LOAD";
        case NodeKind::STORE: return "STORE";
        case NodeKind::NEWARRAY: return "NEWARRAY";
        case NodeKind::ARRAY_LOAD: return "ARRAY_LOAD";
        case NodeKind::ARRAY_STORE: return "ARRAY_STORE";

        case NodeKind::CAST: return "CAST";
        case NodeKind::FUNCTION: return "FUNCTION";
        case NodeKind::PARAMETER: return "PARAMETER";
        case NodeKind::UNKNOWN: return "UNKNOWN";

        default: return "<NodeKind:OTHER>";
    }
}

} // namespace cppfort::ir
