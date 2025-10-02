# Developer Handoff: Band 5 Implementation Tasks

## Overview

Band 5 implements n-way pattern matching infrastructure built on Band 4's type enum foundation. This document provides concrete implementation tasks for developers.

## Prerequisites

- Band 1-4 implementation complete
- Understanding of Sea of Nodes concepts
- Familiarity with TableGen (LLVM pattern matching DSL)
- C++20 knowledge (concepts, ranges, modules)

## Architecture Documents

Required reading:
- `/Users/jim/work/cppfort/docs/band5_pattern_matching.md` - Band 5 architecture
- `/Users/jim/work/cppfort/docs/architecture/nway-induction-enum-strategy.md` - Enum-based induction
- `/Users/jim/work/cppfort/docs/architecture/divergence-son-simple-n-way.md` - N-way conversion rationale
- `/Users/jim/work/cppfort/docs/architecture/subsumption-engine.md` - Subsumption architecture

## Task Breakdown

### Task 1: Define NodeKind Enum (2-3 days)

**File:** `/Users/jim/work/cppfort/src/stage0/node.h`

**Add NodeKind enum:**

```cpp
// After existing includes, before Node class definition

/**
 * Band 5: NodeKind enum for pattern-based dispatch
 *
 * Classifies all node types into ranges for inductive pattern matching.
 * Each range represents a category of operations (arithmetic, bitwise, etc.)
 */
enum class NodeKind : uint16_t {
    // ========================================================================
    // Control Flow Nodes (0-99)
    // ========================================================================
    CFG_START = 0,
    START = 0,
    STOP = 1,
    RETURN = 2,
    IF = 3,
    REGION = 4,
    LOOP = 5,
    CPROJ = 6,
    CFG_END = 99,

    // ========================================================================
    // Data Nodes (100-199)
    // ========================================================================
    DATA_START = 100,
    CONSTANT = 100,
    PHI = 101,
    PROJ = 102,
    SCOPE = 103,
    DATA_END = 199,

    // ========================================================================
    // Integer Arithmetic (200-299)
    // ========================================================================
    ARITH_START = 200,
    ADD = 200,
    SUB = 201,
    MUL = 202,
    DIV = 203,
    MINUS = 204,  // Unary minus
    ARITH_END = 299,

    // ========================================================================
    // Bitwise Operations (300-399) - Chapter 16
    // ========================================================================
    BITWISE_START = 300,
    AND = 300,     // &
    OR = 301,      // |
    XOR = 302,     // ^
    SHL = 303,     // <<
    ASHR = 304,    // >> (arithmetic shift right, sign extend)
    LSHR = 305,    // >>> (logical shift right, zero extend)
    BITWISE_END = 399,

    // ========================================================================
    // Float Arithmetic (400-499) - Band 4
    // ========================================================================
    FLOAT_START = 400,
    FADD = 400,
    FSUB = 401,
    FMUL = 402,
    FDIV = 403,
    FLOAT_END = 499,

    // ========================================================================
    // Memory Operations (500-599)
    // ========================================================================
    MEMORY_START = 500,
    NEW = 500,
    LOAD = 501,
    STORE = 502,
    NEW_ARRAY = 503,
    ARRAY_LOAD = 504,
    ARRAY_STORE = 505,
    ARRAY_LENGTH = 506,
    MEMORY_END = 599,

    // ========================================================================
    // Type Conversions (600-699) - Band 4
    // ========================================================================
    CONVERSION_START = 600,
    CAST = 600,
    CONVERSION_END = 699,

    // ========================================================================
    // Comparison Operations (700-799)
    // ========================================================================
    CMP_START = 700,
    EQ = 700,      // ==
    NE = 701,      // !=
    LT = 702,      // <
    LE = 703,      // <=
    GT = 704,      // >
    GE = 705,      // >=
    CMP_END = 799,

    // ========================================================================
    // Boolean Operations (800-899)
    // ========================================================================
    BOOL_START = 800,
    BOOL_AND = 800,  // && (logical AND, short-circuit)
    BOOL_OR = 801,   // || (logical OR, short-circuit)
    BOOL_NOT = 802,  // ! (logical NOT)
    BOOL_END = 899
};

// String representation for debugging
inline const char* nodeKindToString(NodeKind kind) {
    switch (kind) {
        case NodeKind::START: return "Start";
        case NodeKind::STOP: return "Stop";
        case NodeKind::RETURN: return "Return";
        case NodeKind::IF: return "If";
        case NodeKind::REGION: return "Region";
        case NodeKind::LOOP: return "Loop";
        case NodeKind::CPROJ: return "CProj";
        case NodeKind::CONSTANT: return "Constant";
        case NodeKind::PHI: return "Phi";
        case NodeKind::PROJ: return "Proj";
        case NodeKind::SCOPE: return "Scope";
        case NodeKind::ADD: return "Add";
        case NodeKind::SUB: return "Sub";
        case NodeKind::MUL: return "Mul";
        case NodeKind::DIV: return "Div";
        case NodeKind::MINUS: return "Minus";
        case NodeKind::AND: return "And";
        case NodeKind::OR: return "Or";
        case NodeKind::XOR: return "Xor";
        case NodeKind::SHL: return "Shl";
        case NodeKind::ASHR: return "AShr";
        case NodeKind::LSHR: return "LShr";
        case NodeKind::FADD: return "FAdd";
        case NodeKind::FSUB: return "FSub";
        case NodeKind::FMUL: return "FMul";
        case NodeKind::FDIV: return "FDiv";
        case NodeKind::NEW: return "New";
        case NodeKind::LOAD: return "Load";
        case NodeKind::STORE: return "Store";
        case NodeKind::NEW_ARRAY: return "NewArray";
        case NodeKind::ARRAY_LOAD: return "ArrayLoad";
        case NodeKind::ARRAY_STORE: return "ArrayStore";
        case NodeKind::ARRAY_LENGTH: return "ArrayLength";
        case NodeKind::CAST: return "Cast";
        case NodeKind::EQ: return "Eq";
        case NodeKind::NE: return "Ne";
        case NodeKind::LT: return "Lt";
        case NodeKind::LE: return "Le";
        case NodeKind::GT: return "Gt";
        case NodeKind::GE: return "Ge";
        case NodeKind::BOOL_AND: return "BoolAnd";
        case NodeKind::BOOL_OR: return "BoolOr";
        case NodeKind::BOOL_NOT: return "BoolNot";
        default: return "Unknown";
    }
}
```

**Add getKind() to Node base class:**

```cpp
class Node {
public:
    /**
     * Get the kind of this node for pattern matching.
     * Band 5: Enables enum-based dispatch for n-way lowering.
     */
    virtual NodeKind getKind() const = 0;

    // ... existing methods
};
```

**Implementation checklist:**
- [ ] Add NodeKind enum before Node class
- [ ] Add nodeKindToString() helper function
- [ ] Add getKind() pure virtual method to Node
- [ ] Verify enum ranges don't overlap
- [ ] Test enum range queries compile

### Task 2: Implement getKind() for All Nodes (3-4 days)

**Files to modify:**
- `/Users/jim/work/cppfort/src/stage0/node.h`
- `/Users/jim/work/cppfort/src/stage0/node.cpp`

**For each existing node class, add:**

```cpp
// Example: AddNode
class AddNode : public Node {
public:
    NodeKind getKind() const override { return NodeKind::ADD; }
    // ... existing methods
};

// Example: StartNode (CFGNode)
class StartNode : public CFGNode {
public:
    NodeKind getKind() const override { return NodeKind::START; }
    // ... existing methods
};

// Example: LoadNode
class LoadNode : public Node {
public:
    NodeKind getKind() const override { return NodeKind::LOAD; }
    // ... existing methods
};
```

**Nodes to update (from Bands 1-4):**
- [ ] StartNode → NodeKind::START
- [ ] StopNode → NodeKind::STOP
- [ ] ReturnNode → NodeKind::RETURN
- [ ] IfNode → NodeKind::IF
- [ ] RegionNode → NodeKind::REGION
- [ ] LoopNode → NodeKind::LOOP
- [ ] CProjNode → NodeKind::CPROJ
- [ ] ConstantNode → NodeKind::CONSTANT
- [ ] PhiNode → NodeKind::PHI
- [ ] ProjNode → NodeKind::PROJ
- [ ] ScopeNode → NodeKind::SCOPE
- [ ] AddNode → NodeKind::ADD
- [ ] SubNode → NodeKind::SUB
- [ ] MulNode → NodeKind::MUL
- [ ] DivNode → NodeKind::DIV
- [ ] MinusNode → NodeKind::MINUS
- [ ] NewNode → NodeKind::NEW
- [ ] LoadNode → NodeKind::LOAD
- [ ] StoreNode → NodeKind::STORE
- [ ] NewArrayNode → NodeKind::NEW_ARRAY (Band 4)
- [ ] ArrayLoadNode → NodeKind::ARRAY_LOAD (Band 4)
- [ ] ArrayStoreNode → NodeKind::ARRAY_STORE (Band 4)
- [ ] ArrayLengthNode → NodeKind::ARRAY_LENGTH (Band 4)
- [ ] CastNode → NodeKind::CAST (Band 4)

**Validation:**
- [ ] All concrete node classes implement getKind()
- [ ] No two nodes return the same NodeKind
- [ ] Compile without errors
- [ ] Basic test: `node->getKind() == NodeKind::ADD`

### Task 3: Create NodeCategory Helper Class (1 day)

**File:** `/Users/jim/work/cppfort/src/stage0/node_category.h` (new file)

```cpp
#ifndef CPPFORT_NODE_CATEGORY_H
#define CPPFORT_NODE_CATEGORY_H

#include "node.h"

namespace cppfort::ir {

/**
 * Band 5: Category predicates for inductive pattern matching.
 *
 * Enables queries like "all arithmetic operations" or "all bitwise operations"
 * without enumerating every specific node type.
 */
class NodeCategory {
public:
    /**
     * Check if node kind is a control flow operation.
     */
    static constexpr bool isCFG(NodeKind k) {
        return k >= NodeKind::CFG_START && k < NodeKind::CFG_END;
    }

    /**
     * Check if node kind is an integer arithmetic operation.
     */
    static constexpr bool isArithmetic(NodeKind k) {
        return k > NodeKind::ARITH_START && k < NodeKind::ARITH_END;
    }

    /**
     * Check if node kind is a bitwise operation (Chapter 16).
     */
    static constexpr bool isBitwise(NodeKind k) {
        return k > NodeKind::BITWISE_START && k < NodeKind::BITWISE_END;
    }

    /**
     * Check if node kind is a floating-point arithmetic operation (Band 4).
     */
    static constexpr bool isFloatOp(NodeKind k) {
        return k > NodeKind::FLOAT_START && k < NodeKind::FLOAT_END;
    }

    /**
     * Check if node kind is a memory operation.
     */
    static constexpr bool isMemoryOp(NodeKind k) {
        return k > NodeKind::MEMORY_START && k < NodeKind::MEMORY_END;
    }

    /**
     * Check if node kind is a comparison operation.
     */
    static constexpr bool isComparison(NodeKind k) {
        return k > NodeKind::CMP_START && k < NodeKind::CMP_END;
    }

    /**
     * Check if node kind is a boolean operation.
     */
    static constexpr bool isBoolOp(NodeKind k) {
        return k > NodeKind::BOOL_START && k < NodeKind::BOOL_END;
    }

    /**
     * Check if node kind is a data operation (not CFG).
     */
    static constexpr bool isDataOp(NodeKind k) {
        return !isCFG(k);
    }

    /**
     * Check if node kind is a numeric operation (int or float).
     */
    static constexpr bool isNumericOp(NodeKind k) {
        return isArithmetic(k) || isFloatOp(k);
    }

    /**
     * Check if node kind is commutative (a op b == b op a).
     */
    static constexpr bool isCommutative(NodeKind k) {
        return k == NodeKind::ADD ||
               k == NodeKind::MUL ||
               k == NodeKind::FADD ||
               k == NodeKind::FMUL ||
               k == NodeKind::AND ||
               k == NodeKind::OR ||
               k == NodeKind::XOR ||
               k == NodeKind::EQ ||
               k == NodeKind::NE;
    }

    /**
     * Check if node kind has side effects (cannot be eliminated if unused).
     */
    static constexpr bool hasSideEffects(NodeKind k) {
        return k == NodeKind::STORE ||
               k == NodeKind::ARRAY_STORE ||
               k == NodeKind::RETURN;
    }

    /**
     * Check if node kind is pure (no side effects, result depends only on inputs).
     */
    static constexpr bool isPure(NodeKind k) {
        return !hasSideEffects(k) && !isCFG(k);
    }
};

} // namespace cppfort::ir

#endif // CPPFORT_NODE_CATEGORY_H
```

**Implementation checklist:**
- [ ] Create node_category.h
- [ ] Implement all category predicates
- [ ] Add unit tests for each predicate
- [ ] Verify constexpr evaluation works

### Task 4: Add Chapter 16 Bitwise Operation Nodes (3-4 days)

**File:** `/Users/jim/work/cppfort/src/stage0/node.h`

**Add node declarations after existing arithmetic nodes:**

```cpp
// ============================================================================
// Band 5: Chapter 16 - Bitwise Operations
// ============================================================================

/**
 * Bitwise AND operation (a & b).
 */
class AndNode : public Node {
public:
    AndNode(Node* lhs, Node* rhs);

    NodeKind getKind() const override { return NodeKind::AND; }
    std::string label() const override { return "&"; }
    Type* compute() override;
    Node* peephole() override;
};

/**
 * Bitwise OR operation (a | b).
 */
class OrNode : public Node {
public:
    OrNode(Node* lhs, Node* rhs);

    NodeKind getKind() const override { return NodeKind::OR; }
    std::string label() const override { return "|"; }
    Type* compute() override;
    Node* peephole() override;
};

/**
 * Bitwise XOR operation (a ^ b).
 */
class XorNode : public Node {
public:
    XorNode(Node* lhs, Node* rhs);

    NodeKind getKind() const override { return NodeKind::XOR; }
    std::string label() const override { return "^"; }
    Type* compute() override;
    Node* peephole() override;
};

/**
 * Shift left operation (a << b).
 */
class ShlNode : public Node {
public:
    ShlNode(Node* val, Node* shift);

    NodeKind getKind() const override { return NodeKind::SHL; }
    std::string label() const override { return "<<"; }
    Type* compute() override;
    Node* peephole() override;
};

/**
 * Arithmetic shift right operation (a >> b).
 * Sign extends on right shift.
 */
class AShrNode : public Node {
public:
    AShrNode(Node* val, Node* shift);

    NodeKind getKind() const override { return NodeKind::ASHR; }
    std::string label() const override { return ">>"; }
    Type* compute() override;
    Node* peephole() override;
};

/**
 * Logical shift right operation (a >>> b).
 * Zero extends on right shift.
 */
class LShrNode : public Node {
public:
    LShrNode(Node* val, Node* shift);

    NodeKind getKind() const override { return NodeKind::LSHR; }
    std::string label() const override { return ">>>"; }
    Type* compute() override;
    Node* peephole() override;
};
```

**File:** `/Users/jim/work/cppfort/src/stage0/node.cpp`

**Implement bitwise node methods:**

```cpp
// ============================================================================
// Band 5: Bitwise Operations
// ============================================================================

AndNode::AndNode(Node* lhs, Node* rhs) : Node() {
    setInput(0, lhs);
    setInput(1, rhs);
    _type = compute();
}

Type* AndNode::compute() {
    // Bitwise AND requires integer types
    Type* t1 = in(0)->_type;
    Type* t2 = in(1)->_type;

    if (!t1 || !t2) return Type::BOTTOM;

    // Both must be integers
    auto* i1 = dynamic_cast<TypeInteger*>(t1);
    auto* i2 = dynamic_cast<TypeInteger*>(t2);
    if (!i1 || !i2) return Type::BOTTOM;

    // If both are constants, fold
    if (i1->isConstant() && i2->isConstant()) {
        long result = i1->value() & i2->value();
        return TypeInteger::constant(result);
    }

    // Otherwise return integer bottom
    return TypeInteger::bottom();
}

Node* AndNode::peephole() {
    // x & 0 → 0
    if (in(1)->_type->isConstant()) {
        auto* c = dynamic_cast<TypeInteger*>(in(1)->_type);
        if (c && c->value() == 0) {
            return in(1);
        }
    }

    // x & -1 → x (all bits set)
    if (in(1)->_type->isConstant()) {
        auto* c = dynamic_cast<TypeInteger*>(in(1)->_type);
        if (c && c->value() == -1) {
            return in(0);
        }
    }

    // x & x → x (idempotent)
    if (in(0) == in(1)) {
        return in(0);
    }

    return this;
}

// Implement OrNode, XorNode, ShlNode, AShrNode, LShrNode similarly
// (See full implementations in band5_pattern_matching.md)
```

**Implementation checklist:**
- [ ] Add AndNode, OrNode, XorNode declarations
- [ ] Add ShlNode, AShrNode, LShrNode declarations
- [ ] Implement constructors
- [ ] Implement compute() for each (constant folding)
- [ ] Implement peephole() for each (algebraic simplification)
- [ ] Add unit tests for each operation
- [ ] Test constant folding: `5 & 3 → 1`
- [ ] Test peepholes: `x & 0 → 0`, `x | 0 → x`, `x ^ x → 0`

### Task 5: Update Parser for Bitwise Operations (2 days)

**File:** `/Users/jim/work/cppfort/src/stage0/parser.cpp`

**Add bitwise operator precedence (after comparison, before addition):**

```cpp
// Add to parseExpression() or add new parseBitwiseExpression()
Node* Parser::parseBitwiseExpression() {
    Node* lhs = parseComparisonExpression();

    while (match(TokenType::AMPERSAND) ||      // &
           match(TokenType::PIPE) ||            // |
           match(TokenType::CARET)) {           // ^
        Token op = previous();
        Node* rhs = parseComparisonExpression();

        switch (op.type) {
            case TokenType::AMPERSAND:
                lhs = new AndNode(lhs, rhs);
                break;
            case TokenType::PIPE:
                lhs = new OrNode(lhs, rhs);
                break;
            case TokenType::CARET:
                lhs = new XorNode(lhs, rhs);
                break;
        }
    }

    return lhs;
}

Node* Parser::parseShiftExpression() {
    Node* lhs = parseBitwiseExpression();

    while (match(TokenType::LSHIFT) ||         // <<
           match(TokenType::RSHIFT) ||         // >>
           match(TokenType::LRSHIFT)) {        // >>>
        Token op = previous();
        Node* rhs = parseBitwiseExpression();

        switch (op.type) {
            case TokenType::LSHIFT:
                lhs = new ShlNode(lhs, rhs);
                break;
            case TokenType::RSHIFT:
                lhs = new AShrNode(lhs, rhs);
                break;
            case TokenType::LRSHIFT:
                lhs = new LShrNode(lhs, rhs);
                break;
        }
    }

    return lhs;
}
```

**Update operator precedence hierarchy:**
```
Primary (literals, identifiers)
  ↓
Unary (!, -)
  ↓
Multiplicative (*, /)
  ↓
Additive (+, -)
  ↓
Shift (<<, >>, >>>)      ← NEW
  ↓
Comparison (==, !=, <, >, <=, >=)
  ↓
Bitwise (&, |, ^)        ← NEW
  ↓
Boolean (&&, ||)
```

**Implementation checklist:**
- [ ] Add TokenType::AMPERSAND, PIPE, CARET to lexer
- [ ] Add TokenType::LSHIFT, RSHIFT, LRSHIFT to lexer
- [ ] Implement parseBitwiseExpression()
- [ ] Implement parseShiftExpression()
- [ ] Update parseExpression() to call new methods
- [ ] Test parsing: `x & y`, `x | y`, `x ^ y`
- [ ] Test parsing: `x << 2`, `x >> 3`, `x >>> 1`
- [ ] Test precedence: `x + y << 2` parses as `(x + y) << 2`

### Task 6: Create TableGen Pattern Specifications (3-4 days)

**File:** `/Users/jim/work/cppfort/src/stage0/nway_patterns.td` (new file)

```tablegen
// ============================================================================
// Band 5: N-Way Lowering Patterns
// ============================================================================
//
// Declarative patterns for lowering Sea of Nodes to multiple target languages.
// Each pattern specifies how a SoN node maps to C, C++, CPP2, and MLIR.

// Base pattern class
class NWayPattern<NodeKind kind, string desc> {
  NodeKind Kind = kind;
  string Description = desc;
}

// Target language emission specifications
class CEmit<string code> {
  string Code = code;
}

class CPPEmit<string code> {
  string Code = code;
}

class CPP2Emit<string code> {
  string Code = code;
}

class MLIREmit<string dialect, string op> {
  string Dialect = dialect;
  string Operation = op;
}

// ============================================================================
// Integer Arithmetic Patterns
// ============================================================================

def AddPattern : NWayPattern<NodeKind::ADD, "Integer addition"> {
  CEmit C = CEmit<"$lhs + $rhs">;
  CPPEmit CPP = CPPEmit<"$lhs + $rhs">;
  CPP2Emit CPP2 = CPP2Emit<"$lhs + $rhs">;
  MLIREmit MLIR = MLIREmit<"arith", "addi">;
}

def SubPattern : NWayPattern<NodeKind::SUB, "Integer subtraction"> {
  CEmit C = CEmit<"$lhs - $rhs">;
  CPPEmit CPP = CPPEmit<"$lhs - $rhs">;
  CPP2Emit CPP2 = CPP2Emit<"$lhs - $rhs">;
  MLIREmit MLIR = MLIREmit<"arith", "subi">;
}

def MulPattern : NWayPattern<NodeKind::MUL, "Integer multiplication"> {
  CEmit C = CEmit<"$lhs * $rhs">;
  CPPEmit CPP = CPPEmit<"$lhs * $rhs">;
  CPP2Emit CPP2 = CPP2Emit<"$lhs * $rhs">;
  MLIREmit MLIR = MLIREmit<"arith", "muli">;
}

def DivPattern : NWayPattern<NodeKind::DIV, "Integer division"> {
  CEmit C = CEmit<"$lhs / $rhs">;
  CPPEmit CPP = CPPEmit<"$lhs / $rhs">;
  CPP2Emit CPP2 = CPP2Emit<"$lhs / $rhs">;
  MLIREmit MLIR = MLIREmit<"arith", "divsi">;
}

// ============================================================================
// Bitwise Operation Patterns (Chapter 16)
// ============================================================================

def AndPattern : NWayPattern<NodeKind::AND, "Bitwise AND"> {
  CEmit C = CEmit<"$lhs & $rhs">;
  CPPEmit CPP = CPPEmit<"$lhs & $rhs">;
  CPP2Emit CPP2 = CPP2Emit<"$lhs & $rhs">;
  MLIREmit MLIR = MLIREmit<"arith", "andi">;
}

def OrPattern : NWayPattern<NodeKind::OR, "Bitwise OR"> {
  CEmit C = CEmit<"$lhs | $rhs">;
  CPPEmit CPP = CPPEmit<"$lhs | $rhs">;
  CPP2Emit CPP2 = CPP2Emit<"$lhs | $rhs">;
  MLIREmit MLIR = MLIREmit<"arith", "ori">;
}

def XorPattern : NWayPattern<NodeKind::XOR, "Bitwise XOR"> {
  CEmit C = CEmit<"$lhs ^ $rhs">;
  CPPEmit CPP = CPPEmit<"$lhs ^ $rhs">;
  CPP2Emit CPP2 = CPP2Emit<"$lhs ^ $rhs">;
  MLIREmit MLIR = MLIREmit<"arith", "xori">;
}

def ShlPattern : NWayPattern<NodeKind::SHL, "Shift left"> {
  CEmit C = CEmit<"$val << $shift">;
  CPPEmit CPP = CPPEmit<"$val << $shift">;
  CPP2Emit CPP2 = CPP2Emit<"$val << $shift">;
  MLIREmit MLIR = MLIREmit<"arith", "shli">;
}

def AShrPattern : NWayPattern<NodeKind::ASHR, "Arithmetic shift right"> {
  CEmit C = CEmit<"$val >> $shift">;
  CPPEmit CPP = CPPEmit<"$val >> $shift">;
  CPP2Emit CPP2 = CPP2Emit<"$val >> $shift">;
  MLIREmit MLIR = MLIREmit<"arith", "shrsi">;
}

def LShrPattern : NWayPattern<NodeKind::LSHR, "Logical shift right"> {
  CEmit C = CEmit<"(unsigned)$val >> $shift">;
  CPPEmit CPP = CPPEmit<"static_cast<unsigned>($val) >> $shift">;
  CPP2Emit CPP2 = CPP2Emit<"($val as unsigned) >> $shift">;
  MLIREmit MLIR = MLIREmit<"arith", "shrui">;
}

// ============================================================================
// Float Arithmetic Patterns (Band 4 integration)
// ============================================================================

def FAddPattern : NWayPattern<NodeKind::FADD, "Float addition"> {
  CEmit C = CEmit<"$lhs + $rhs">;
  CPPEmit CPP = CPPEmit<"$lhs + $rhs">;
  CPP2Emit CPP2 = CPP2Emit<"$lhs + $rhs">;
  MLIREmit MLIR = MLIREmit<"arith", "addf">;
}

// Add similar patterns for FSUB, FMUL, FDIV

// ============================================================================
// Control Flow Patterns
// ============================================================================

def IfPattern : NWayPattern<NodeKind::IF, "Conditional branch"> {
  CEmit C = CEmit<"if ($cond) { $true } else { $false }">;
  CPPEmit CPP = CPPEmit<"if ($cond) { $true } else { $false }">;
  CPP2Emit CPP2 = CPP2Emit<"if ($cond) { $true } else { $false }">;
  MLIREmit MLIR = MLIREmit<"scf", "if">;
}

def LoopPattern : NWayPattern<NodeKind::LOOP, "Loop construct"> {
  CEmit C = CEmit<"while ($cond) { $body }">;
  CPPEmit CPP = CPPEmit<"while ($cond) { $body }">;
  CPP2Emit CPP2 = CPP2Emit<"while ($cond) { $body }">;
  MLIREmit MLIR = MLIREmit<"scf", "while">;
}

// ============================================================================
// Memory Operation Patterns
// ============================================================================

def LoadPattern : NWayPattern<NodeKind::LOAD, "Memory load"> {
  CEmit C = CEmit<"*$ptr">;
  CPPEmit CPP = CPPEmit<"*$ptr">;
  CPP2Emit CPP2 = CPP2Emit<"*$ptr">;
  MLIREmit MLIR = MLIREmit<"memref", "load">;
}

def StorePattern : NWayPattern<NodeKind::STORE, "Memory store"> {
  CEmit C = CEmit<"*$ptr = $value">;
  CPPEmit CPP = CPPEmit<"*$ptr = $value">;
  CPP2Emit CPP2 = CPP2Emit<"*$ptr = $value">;
  MLIREmit MLIR = MLIREmit<"memref", "store">;
}

def ArrayLoadPattern : NWayPattern<NodeKind::ARRAY_LOAD, "Array element load"> {
  CEmit C = CEmit<"$array[$index]">;
  CPPEmit CPP = CPPEmit<"$array[$index]">;
  CPP2Emit CPP2 = CPP2Emit<"$array[$index]">;
  MLIREmit MLIR = MLIREmit<"memref", "load">;
}

// ============================================================================
// Pattern Generation Instructions
// ============================================================================

// This TableGen file will be processed to generate:
// - Pattern matcher C++ code
// - Target-specific emitters
// - Pattern validation tests

// Usage: llvm-tblgen -gen-pattern-matcher nway_patterns.td -o nway_patterns_generated.cpp
```

**Implementation checklist:**
- [ ] Create nway_patterns.td file
- [ ] Define all arithmetic patterns
- [ ] Define all bitwise patterns
- [ ] Define control flow patterns
- [ ] Define memory operation patterns
- [ ] Set up TableGen generation in CMakeLists.txt
- [ ] Generate C++ pattern matcher code
- [ ] Verify generated code compiles

### Task 7: Implement Pattern Matcher Infrastructure (4-5 days)

**File:** `/Users/jim/work/cppfort/src/stage0/pattern_matcher.h` (new file)

```cpp
#ifndef CPPFORT_PATTERN_MATCHER_H
#define CPPFORT_PATTERN_MATCHER_H

#include "node.h"
#include "node_category.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace cppfort::ir {

// Target language enum
enum class TargetLanguage {
    C,
    CPP,
    CPP2,
    MLIR
};

// Forward declarations
class CEmitter;
class CPPEmitter;
class CPP2Emitter;
class MLIREmitter;

/**
 * Pattern represents a lowering rule from SoN node to target code.
 */
class Pattern {
public:
    using MatchFunc = std::function<bool(Node*)>;
    using EmitFunc = std::function<void(Node*, std::ostream&)>;

private:
    NodeKind _kind;
    MatchFunc _match;
    std::unordered_map<TargetLanguage, EmitFunc> _emitters;

public:
    Pattern(NodeKind kind, MatchFunc match = nullptr)
        : _kind(kind), _match(match) {}

    // Check if pattern matches node
    bool matches(Node* node) const {
        if (node->getKind() != _kind) return false;
        if (_match) return _match(node);
        return true;
    }

    // Register emitter for target
    void addEmitter(TargetLanguage target, EmitFunc emitter) {
        _emitters[target] = emitter;
    }

    // Emit code for node in target language
    void emit(Node* node, TargetLanguage target, std::ostream& out) const {
        auto it = _emitters.find(target);
        if (it != _emitters.end()) {
            it->second(node, out);
        }
    }

    NodeKind kind() const { return _kind; }
};

/**
 * PatternMatcher dispatches nodes to patterns based on kind.
 */
class PatternMatcher {
private:
    // Pattern table indexed by NodeKind
    std::unordered_map<NodeKind, std::vector<Pattern>> _patterns;

public:
    // Register pattern for specific node kind
    void registerPattern(Pattern pattern) {
        _patterns[pattern.kind()].push_back(std::move(pattern));
    }

    // Find and apply matching pattern
    bool match(Node* node, TargetLanguage target, std::ostream& out) {
        auto it = _patterns.find(node->getKind());
        if (it == _patterns.end()) return false;

        for (const Pattern& pat : it->second) {
            if (pat.matches(node)) {
                pat.emit(node, target, out);
                return true;
            }
        }

        return false;
    }

    // Register all built-in patterns
    void registerBuiltinPatterns();
};

} // namespace cppfort::ir

#endif // CPPFORT_PATTERN_MATCHER_H
```

**File:** `/Users/jim/work/cppfort/src/stage0/pattern_matcher.cpp` (new file)

```cpp
#include "pattern_matcher.h"
#include <sstream>

namespace cppfort::ir {

void PatternMatcher::registerBuiltinPatterns() {
    // ========================================================================
    // Arithmetic Patterns
    // ========================================================================

    // ADD pattern
    {
        Pattern addPat(NodeKind::ADD);

        // C emitter
        addPat.addEmitter(TargetLanguage::C, [](Node* n, std::ostream& out) {
            out << "(";
            // Emit lhs (recursively)
            out << " + ";
            // Emit rhs (recursively)
            out << ")";
        });

        // C++ emitter (same as C)
        addPat.addEmitter(TargetLanguage::CPP, [](Node* n, std::ostream& out) {
            out << "(";
            out << " + ";
            out << ")";
        });

        // MLIR emitter
        addPat.addEmitter(TargetLanguage::MLIR, [](Node* n, std::ostream& out) {
            out << "arith.addi ";
            // Emit operands
        });

        registerPattern(std::move(addPat));
    }

    // Repeat for SUB, MUL, DIV, AND, OR, XOR, etc.

    // ========================================================================
    // Bitwise Patterns
    // ========================================================================

    // AND pattern
    {
        Pattern andPat(NodeKind::AND);

        andPat.addEmitter(TargetLanguage::C, [](Node* n, std::ostream& out) {
            out << "( & )";
        });

        andPat.addEmitter(TargetLanguage::MLIR, [](Node* n, std::ostream& out) {
            out << "arith.andi ";
        });

        registerPattern(std::move(andPat));
    }

    // Add patterns for OR, XOR, SHL, ASHR, LSHR
}

} // namespace cppfort::ir
```

**Implementation checklist:**
- [ ] Create pattern_matcher.h and .cpp
- [ ] Implement Pattern class
- [ ] Implement PatternMatcher class
- [ ] Implement registerBuiltinPatterns()
- [ ] Add patterns for all arithmetic ops
- [ ] Add patterns for all bitwise ops
- [ ] Add unit tests for pattern matching
- [ ] Test pattern dispatch: `matcher.match(addNode, C)`

### Task 8: Update CMakeLists.txt (1 day)

**File:** `/Users/jim/work/cppfort/src/stage0/CMakeLists.txt`

**Add new source files:**

```cmake
# Band 5: Pattern matching infrastructure
set(STAGE0_BAND5_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pattern_matcher.cpp
)

# Add to stage0 library
target_sources(stage0 PRIVATE
    ${STAGE0_BAND5_SOURCES}
)

# Add header include for node_category.h
target_include_directories(stage0 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optional: TableGen integration (if using LLVM TableGen)
# find_package(LLVM REQUIRED CONFIG)
# llvm_tablegen(nway_patterns_generated.cpp
#     nway_patterns.td
#     -gen-pattern-matcher
# )
# target_sources(stage0 PRIVATE
#     ${CMAKE_CURRENT_BINARY_DIR}/nway_patterns_generated.cpp
# )
```

**Implementation checklist:**
- [ ] Add pattern_matcher.cpp to build
- [ ] Verify build succeeds
- [ ] Run existing tests to ensure no regressions
- [ ] Add Band 5 tests to test suite

### Task 9: Create Test Suite for Band 5 (2-3 days)

**File:** `/Users/jim/work/cppfort/tests/test_band5.cpp` (new file)

```cpp
#include "../src/stage0/node.h"
#include "../src/stage0/node_category.h"
#include "../src/stage0/pattern_matcher.h"
#include <gtest/gtest.h>
#include <sstream>

using namespace cppfort::ir;

// ============================================================================
// NodeKind Tests
// ============================================================================

TEST(Band5, NodeKindEnumRanges) {
    // Verify enum ranges don't overlap
    EXPECT_LT(NodeKind::CFG_END, NodeKind::DATA_START);
    EXPECT_LT(NodeKind::DATA_END, NodeKind::ARITH_START);
    EXPECT_LT(NodeKind::ARITH_END, NodeKind::BITWISE_START);
    EXPECT_LT(NodeKind::BITWISE_END, NodeKind::FLOAT_START);
}

TEST(Band5, NodeKindGetKind) {
    auto* add = new AddNode(nullptr, nullptr);
    EXPECT_EQ(add->getKind(), NodeKind::ADD);

    auto* andNode = new AndNode(nullptr, nullptr);
    EXPECT_EQ(andNode->getKind(), NodeKind::AND);

    delete add;
    delete andNode;
}

// ============================================================================
// NodeCategory Tests
// ============================================================================

TEST(Band5, CategoryPredicates) {
    EXPECT_TRUE(NodeCategory::isArithmetic(NodeKind::ADD));
    EXPECT_TRUE(NodeCategory::isArithmetic(NodeKind::SUB));
    EXPECT_FALSE(NodeCategory::isArithmetic(NodeKind::AND));

    EXPECT_TRUE(NodeCategory::isBitwise(NodeKind::AND));
    EXPECT_TRUE(NodeCategory::isBitwise(NodeKind::OR));
    EXPECT_FALSE(NodeCategory::isBitwise(NodeKind::ADD));

    EXPECT_TRUE(NodeCategory::isFloatOp(NodeKind::FADD));
    EXPECT_FALSE(NodeCategory::isFloatOp(NodeKind::ADD));
}

TEST(Band5, CategoryCommutative) {
    EXPECT_TRUE(NodeCategory::isCommutative(NodeKind::ADD));
    EXPECT_TRUE(NodeCategory::isCommutative(NodeKind::MUL));
    EXPECT_FALSE(NodeCategory::isCommutative(NodeKind::SUB));
    EXPECT_FALSE(NodeCategory::isCommutative(NodeKind::DIV));
}

// ============================================================================
// Bitwise Operation Tests
// ============================================================================

TEST(Band5, BitwiseAndConstantFolding) {
    auto* a = ConstantNode::make(5);
    auto* b = ConstantNode::make(3);
    auto* andNode = new AndNode(a, b);

    // 5 & 3 = 1
    EXPECT_TRUE(andNode->_type->isConstant());
    auto* result = dynamic_cast<TypeInteger*>(andNode->_type);
    EXPECT_EQ(result->value(), 1);

    delete andNode;
    delete b;
    delete a;
}

TEST(Band5, BitwiseAndPeephole) {
    auto* x = ConstantNode::make(42);
    auto* zero = ConstantNode::make(0);
    auto* allOnes = ConstantNode::make(-1);

    // x & 0 → 0
    auto* and0 = new AndNode(x, zero);
    Node* opt0 = and0->peephole();
    EXPECT_EQ(opt0, zero);

    // x & -1 → x
    auto* andAll = new AndNode(x, allOnes);
    Node* optAll = andAll->peephole();
    EXPECT_EQ(optAll, x);

    // x & x → x
    auto* andSelf = new AndNode(x, x);
    Node* optSelf = andSelf->peephole();
    EXPECT_EQ(optSelf, x);

    delete andSelf;
    delete andAll;
    delete and0;
    delete allOnes;
    delete zero;
    delete x;
}

TEST(Band5, BitwiseShiftConstantFolding) {
    auto* val = ConstantNode::make(8);
    auto* shift = ConstantNode::make(2);

    // 8 << 2 = 32
    auto* shl = new ShlNode(val, shift);
    EXPECT_TRUE(shl->_type->isConstant());
    auto* result = dynamic_cast<TypeInteger*>(shl->_type);
    EXPECT_EQ(result->value(), 32);

    delete shl;
    delete shift;
    delete val;
}

// ============================================================================
// Pattern Matching Tests
// ============================================================================

TEST(Band5, PatternMatcherBasic) {
    PatternMatcher matcher;
    matcher.registerBuiltinPatterns();

    auto* add = new AddNode(ConstantNode::make(5), ConstantNode::make(3));

    std::ostringstream out;
    bool matched = matcher.match(add, TargetLanguage::C, out);

    EXPECT_TRUE(matched);
    EXPECT_THAT(out.str(), testing::HasSubstr("+"));

    delete add;
}

TEST(Band5, PatternMatcherMLIR) {
    PatternMatcher matcher;
    matcher.registerBuiltinPatterns();

    auto* add = new AddNode(ConstantNode::make(5), ConstantNode::make(3));

    std::ostringstream out;
    bool matched = matcher.match(add, TargetLanguage::MLIR, out);

    EXPECT_TRUE(matched);
    EXPECT_THAT(out.str(), testing::HasSubstr("arith.addi"));

    delete add;
}

TEST(Band5, PatternMatcherBitwise) {
    PatternMatcher matcher;
    matcher.registerBuiltinPatterns();

    auto* andNode = new AndNode(ConstantNode::make(5), ConstantNode::make(3));

    std::ostringstream out;
    bool matched = matcher.match(andNode, TargetLanguage::C, out);

    EXPECT_TRUE(matched);
    EXPECT_THAT(out.str(), testing::HasSubstr("&"));

    delete andNode;
}

// ============================================================================
// N-Way Lowering Integration Tests
// ============================================================================

TEST(Band5, NWayArithmeticLowering) {
    // Build graph: return 5 + 3
    auto* start = new StartNode();
    auto* a = ConstantNode::make(5);
    auto* b = ConstantNode::make(3);
    auto* add = new AddNode(a, b);
    auto* ret = new ReturnNode(start, add);

    PatternMatcher matcher;
    matcher.registerBuiltinPatterns();

    // Lower to C
    std::ostringstream c_out;
    matcher.match(add, TargetLanguage::C, c_out);
    EXPECT_THAT(c_out.str(), testing::HasSubstr("+"));

    // Lower to MLIR
    std::ostringstream mlir_out;
    matcher.match(add, TargetLanguage::MLIR, mlir_out);
    EXPECT_THAT(mlir_out.str(), testing::HasSubstr("arith.addi"));

    delete ret;
    delete add;
    delete b;
    delete a;
    delete start;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

**Implementation checklist:**
- [ ] Create test_band5.cpp
- [ ] Add tests for NodeKind enum
- [ ] Add tests for NodeCategory predicates
- [ ] Add tests for bitwise operations
- [ ] Add tests for constant folding
- [ ] Add tests for peephole optimizations
- [ ] Add tests for pattern matching
- [ ] Add tests for n-way lowering
- [ ] All tests pass

**Regression Testing Requirements:**

Band 5 requires -Og baseline capture for differential pattern tracking (Stage2 goal). See [Stage2 Disasm→TableGen Differential](architecture/stage2-disasm-tblgen-differential.md).

**Add to regression harness:**
```bash
# regression-tests/run_with_baseline.sh
for test in regression-tests/*.cpp2; do
    base=$(basename $test .cpp2)

    # Capture -Og baseline (preserves SON IR structure in assembly)
    ./build/stage1_cli -Og $test -o build/${base}_Og.out

    # Extract CFG with Ghidra (better basic blocks than objdump)
    analyzeHeadless /tmp ghidra_${base}_Og \
        -import build/${base}_Og.out \
        -postScript extract_cfg.py \
        -scriptPath ./scripts/ghidra

    # Fallback: objdump for raw disassembly
    objdump -d build/${base}_Og.out > build/${base}_Og.asm

    # Dump IR for pattern baseline
    ./build/stage1_cli -Og --dump-ir $test > build/${base}.ir
done
```

**Rationale:** Stochastic patterns (those SON optimizes to nothing at -O2+) require -Og baseline to capture. Differential tracking measures cost reduction: `dC/dO = -α*C`.

**Implementation checklist (regression):**
- [ ] Add -Og baseline capture to regression harness
- [ ] Generate .asm dumps with objdump
- [ ] Dump IR at -Og for pattern extraction
- [ ] Validate baseline patterns captured

### Task 10: Integration and Documentation (2 days)

**Create integration documentation:**

**File:** `/Users/jim/work/cppfort/docs/INTEGRATION_BAND5.md`

Document how Band 5 integrates with:
- Band 1-4 existing infrastructure
- Parser updates for bitwise operators
- Pattern matching in pipeline
- Future Band 6+ optimizations

**Update README:**

Add Band 5 status to project README.

**Create migration guide:**

Document how to add new node types with pattern matching support.

**Implementation checklist:**
- [ ] Create integration documentation
- [ ] Update main README.md
- [ ] Document pattern matching API
- [ ] Create developer migration guide
- [ ] Add examples of adding new patterns

## Timeline

**Total estimated time: 4-5 weeks**

- Week 1: Tasks 1-3 (Enum infrastructure)
- Week 2: Task 4 (Bitwise operations)
- Week 3: Tasks 5-6 (Parser + TableGen)
- Week 4: Tasks 7-8 (Pattern matcher)
- Week 5: Tasks 9-10 (Testing + Documentation)

## Success Criteria

Band 5 implementation is complete when:

1. ✅ All existing tests pass (Bands 1-4)
2. ✅ NodeKind enum covers all node types
3. ✅ All nodes implement getKind()
4. ✅ NodeCategory predicates work correctly
5. ✅ Bitwise operations parse and optimize
6. ✅ Pattern matcher dispatches correctly
7. ✅ N-way lowering generates valid code for C/C++/MLIR
8. ✅ All Band 5 tests pass
9. ✅ Documentation is complete
10. ✅ No regressions from previous bands

## Next Steps After Band 5

With Band 5 complete, the foundation for n-way meta-transpilation is established. Future bands will add:

- **Band 6:** Escape analysis and borrow checking (Chapters 17-18)
- **Band 7:** Function inlining and specialization (Chapters 19-20)
- **Band 8:** Full code generation (Chapters 21-23)
- **Band 9+:** Advanced optimizations (vectorization, GPU lowering, etc.)

## Questions or Issues

If you encounter issues during implementation:
1. Check architecture documents for clarification
2. Review Simple compiler chapters for reference
3. Examine existing band implementations for patterns
4. Consult subsumption engine docs for query examples

Good luck with the implementation!
