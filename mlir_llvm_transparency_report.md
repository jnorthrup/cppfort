# MLIR/LLVM Graph Components and IR Features Transparency Report

## Executive Summary

This report analyzes the cppfort codebase's integration with MLIR and LLVM graph components, identifying elements that can be made fully transparent back to source code and associated support libraries. The analysis reveals a sophisticated multi-level IR architecture with Sea of Nodes at its core, extensive MLIR dialect integration, and a pattern-based lowering system that enables n-way code generation.

## Current MLIR/LLVM Integration Architecture

### 1. Sea of Nodes IR with MLIR Emission

**Core Components:**
- **MLIREmitter** (`src/stage0/mlir_emitter.h/cpp`): Converts unscheduled Sea of Nodes graphs to scheduled MLIR operations
- **SoNScheduler**: Implements reverse postorder traversal for linearizing data dependencies
- **Region+Phi Pattern Conversion**: Transforms Sea of Nodes control flow to MLIR block arguments

**Key Features:**
```cpp
// MLIR emission strategy
mlir::ModuleOp* MLIREmitter::emit(Node* entry) {
    // 1. Run instruction selection to target MLIR dialects
    InstructionSelection isel({"mlir-arith", "mlir-func", "mlir-memref", "mlir-cf"});
    Node* selectedEntry = isel.selectInstructions(entry);
    
    // 2. Schedule and emit operations
    auto* startBlock = emitControlNode(selectedEntry);
    // 3. Process control flow graph with worklist algorithm
}
```

### 2. MLIR Dialect Integration

**Supported Dialects:**
- **arith**: Arithmetic operations (add, sub, mul, div, comparisons)
- **cf**: Control flow (branches, conditional branches)
- **scf**: Structured control flow (if, for, while loops)
- **memref**: Memory references (loads, stores, allocations)
- **func**: Function operations (calls, returns, definitions)

**Custom Dialect:**
```tablegen
// src/stage0/mlir/MyDialect.td
def MyDialect : Dialect {
  let name = "my";
  let cppNamespace = "::my";
}

def MyAddOp : My_Op<"add"> {
  let summary = "My custom add operation";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$res);
}
```

### 3. Instruction Selection and Pattern Matching

**Target Language Abstraction:**
```cpp
enum class TargetLanguage {
    MLIR_ARITH, MLIR_CF, MLIR_SCF, MLIR_MEMREF, MLIR_FUNC
};
```

**Pattern-Based Lowering:**
```cpp
// Band 5: N-way pattern matching infrastructure
class PatternMatcher {
    std::unordered_map<PatternKey, std::vector<Pattern>> _registry;
    
    // Register patterns for multi-target emission
    void registerPattern(NodeKind kind, TargetLanguage target, 
                        std::function<std::string(Node*)> rewrite);
};
```

## Transparency Gaps and Required Transformations

### 1. Cpp2 Language Features Requiring MLIR Transparency

**Smart Pointer Syntax:**
```cpp
// cpp2 syntax (needs transformation)
auto up = unique.new<int>(1);   // ✗ Invalid C++
auto sp = shared.new<int>(2);   // ✗ Invalid C++

// Must emit as:
auto up = std::make_unique<int>(1);   // ✓ Valid C++/MLIR
auto sp = std::make_shared<int>(2);   // ✓ Valid C++/MLIR
```

**Unified Function Call Syntax (UFCS):**
```cpp
// cpp2 UFCS
log(v.ssize());  // Calls std::ssize(v) if no member function

// Must be transparent to MLIR function call emission
```

**Suffix Dereference Operator:**
```cpp
std::cout << vec*.ssize();  // cpp2: * is SUFFIX operator
// Must emit as:
std::cout << (*vec).ssize();  // C++: * is PREFIX operator
```

**Optional/Expected Dereference:**
```cpp
return op*;  // cpp2 operator* on optional
// Must emit as:
return op.value();  // C++ explicit access
```

### 2. Control Flow Transparency

**Inspect Pattern Matching:**
```cpp
inspect x -> std::string {
    is 0 = "zero";
    is std::string = x as std::string;
    is _ = "(no match)";
}
// Must lower to MLIR cf/scf operations with proper phi nodes
```

**Range Operators:**
```cpp
for 1 ..= 2 do (e) { /* loop body */ }
// Must emit as MLIR scf.for with range bounds
```

### 3. Memory Model Transparency

**Guaranteed Initialization:**
```cpp
buf: std::array<std::byte, 1024>;  // uninitialized
buf = some_value;                  // definite first use = construction
// Must ensure MLIR memref operations respect initialization semantics
```

## Band Structure IR Alignment

### Band 1-2: Foundation (SSA + CFG)
**MLIR Transparency:** Direct mapping to arith/cf dialects
```mlir
// Sea of Nodes → MLIR
%1 = arith.addi %a, %b
%2 = arith.cmpi "eq", %1, %c
cf.cond_br %2, ^true_block, ^false_block
```

### Band 3: Global Code Motion
**MLIR Integration:** Dominance-based scheduling
```cpp
// GCM uses MLIR dominance info
class GlobalCodeMotion {
    void scheduleLate() {
        // Use MLIR DominanceInfo for legality checks
        if (dom.dominates(defBlock, useBlock)) {
            // Legal to schedule here
        }
    }
};
```

### Band 4: Types + Lattice Analysis
**MLIR Type System:** Rich type integration
```mlir
// MLIR type constraints
!range_int = !int<range=[0,100]>      // Range-constrained
!nullable_ptr = !ptr<nullable>         // Nullable types
!array = !tensor<10xi32>               // Fixed arrays
```

### Band 5: N-Way Pattern Matching
**Subsumption Queries:** Cross-band optimization
```cpp
// Query operations suitable for vectorization
auto vectorizable = subsumption.query()
    .whereKind({ADD, MUL, FADD, FMUL})
    .whereCFG(inLoop())
    .whereType(isNumeric())
    .execute();
// Lower to MLIR vector dialect
```

## Required Support Library Enhancements

### 1. MLIR Dialect Extensions

**Enhanced Arith Dialect Support:**
```cpp
// Need patterns for cpp2-specific operations
def CPP2_AddInt : Pat<
    (SON_AddNode IntType:$lhs, IntType:$rhs),
    (MLIR_Arith_AddIOp $lhs $rhs)
>;
```

**Smart Pointer Dialect:**
```tablegen
// Proposed: smart pointer dialect
def SmartPtrDialect : Dialect {
  let name = "smartptr";
}

def MakeUniqueOp : SmartPtr_Op<"make_unique"> {
  let arguments = (ins Type:$elementType);
  let results = (outs UniquePtrType:$result);
}
```

### 2. LLVM IR Lowering Enhancements

**Memory Model Integration:**
```cpp
// Lower MLIR memref to LLVM with cpp2 semantics
void lowerMemRefToLLVM(mlir::ModuleOp module) {
    // Ensure guaranteed initialization
    // Handle smart pointer operations
    // Preserve lifetime semantics
}
```

**Exception Handling:**
```cpp
// cpp2 contracts → LLVM exception handling
void lowerContractsToEH(Node* contractNode) {
    // Preconditions → LLVM assume intrinsics
    // Postconditions → LLVM assertion calls
}
```

## Implementation Roadmap

### Phase 1: Core MLIR Transparency (Week 1-2)
1. **Complete Dialect Integration:** All MLIR dialects fully supported
2. **Basic Pattern Matching:** Core arithmetic/control flow patterns
3. **Type System Alignment:** MLIR types match cpp2 semantics

### Phase 2: Cpp2 Feature Transparency (Week 3-4)
1. **Smart Pointer Lowering:** `unique.new`/`shared.new` → MLIR operations
2. **UFCS Resolution:** Function call patterns with member/non-member dispatch
3. **Operator Transformations:** Suffix `*`, `is`/`as` operators

### Phase 3: Advanced IR Features (Week 5-6)
1. **Subsumption Engine:** Cross-band optimization queries
2. **Vectorization Support:** MLIR vector dialect integration
3. **GPU Lowering:** MLIR GPU/spirv dialect support

### Phase 4: LLVM Backend Integration (Week 7-8)
1. **Full LLVM Lowering:** MLIR → LLVM IR with cpp2 semantics
2. **Optimization Pipeline:** Leverage LLVM passes for cpp2 code
3. **Debug Info Generation:** Source-level debugging support

## Success Metrics

### Transparency Metrics
- **Syntax Coverage:** 95% of cpp2 features lower correctly to MLIR
- **Semantic Preservation:** All cpp2 guarantees maintained in MLIR representation
- **Performance Parity:** MLIR/LLVM optimizations match hand-written C++

### IR Quality Metrics
- **Dialect Coverage:** All major MLIR dialects utilized appropriately
- **Optimization Opportunities:** Subsumption queries find 80%+ of optimization candidates
- **Lowering Efficiency:** N-way emission maintains <5% overhead vs single-target

## Conclusion

The cppfort codebase demonstrates sophisticated MLIR/LLVM integration with Sea of Nodes IR at its core. Key transparency gaps exist in cpp2-specific syntax (smart pointers, UFCS, operators) that require enhanced pattern matching and dialect support. The band-structured architecture provides an excellent foundation for multi-level lowering, with Band 5's subsumption engine enabling powerful cross-IR optimizations.

Successful implementation will make cpp2 semantics fully transparent through MLIR dialects to LLVM IR, enabling the full power of modern compiler infrastructure while preserving cpp2's safety and expressiveness guarantees.</content>
<parameter name="filePath">mlir_llvm_transparency_report.md