# MLIR Migration: From Orbits to Blocks and Regions

## Core Abstractions

### ConfixOrbit → Block
Current `ConfixOrbit` represents a delimiter-delimited region:
- `open_char` / `close_char` → Block boundaries
- `start_pos` / `end_pos` → Block range in source
- `depth_counter_` → Block nesting level
- `selected_pattern_` → Operation type

### OrbitFragment → Region content
Fragments represent code spans within blocks:
- Contain references to child orbits (nested blocks)
- Map to MLIR Block::getOperations()

### PatternData → Operation definitions
Patterns define structural operations:
- `alternating_anchors` → Operand boundaries
- `evidence_types` → Operand types (parameters, body, etc.)
- `transformation_templates` → Lowering to C++

## Evidence Span Behaviors Required

### 1. Subexpression Parsing
**Behavior**: Parse expressions within evidence spans recursively
**Input**: Evidence span containing expression: `a + b * c`
**Output**: Expression tree with operator precedence
**Implementation**:
- Use shunting-yard or Pratt parser on span text
- Generate nested Block structure for subexpressions
- Symbol table tracks identifiers across spans

### 2. Function Parameter Quantification
**Behavior**: Extract and type-check function parameters
**Input**: Evidence span: `(x: int, y: std::string)`
**Output**: Parameter list with types and identifiers
**Implementation**:
- Split on comma at depth=0 (top-level)
- For each parameter: `name : type` or `inout name : type`
- Quantify parameters: count, types, qualifiers
- Build ParameterOp with type operands

### 3. Signature Node Quantification
**Behavior**: Parse function signature components
**Input**: Full function orbit: `name: (params) -> return_type = body`
**Output**: Structured signature with quantified components
**Implementation**:
- Evidence span 0: function name (identifier)
- Evidence span 1: parameters (ParameterOp list)
- Evidence span 2: return type (TypeOp)
- Evidence span 3: body (Block with nested operations)
- Each span quantified: presence, type, constraint checks

### 4. Recursive Cutouts with Symbol Dictionaries
**Behavior**: Nested regions access parent symbol tables
**Input**: Function with nested lambda capturing outer variable
```cpp
outer: () = {
    x := 42;
    inner: () = { print(x); };
}
```
**Output**: Symbol dictionary propagation
**Implementation**:
- Each Block has `symbol_dictionary` (string → SymbolOp)
- Parent Block symbols visible in child Block
- Capture analysis: child references parent symbols
- Symbol dictionary passed during recursive transforms

## MLIR Operation Design

### Core Operations

#### FunctionOp
```mlir
%func = cpp2.function "main" {
  (%params = cpp2.parameters {
    // empty for main: ()
  })
  (%rettype = cpp2.type "int")
  (%body = cpp2.region {
    // function body
  })
} : (() -> int)
```

#### VariableOp
```mlir
%var = cpp2.variable "s1" {
  (%init = cpp2.init {
    %lit = cpp2.literal "u\"u\\\"\"" : string
  })
} : auto
```

#### ParameterOp
```mlir
%param = cpp2.parameter "x" {
  (%type = cpp2.type "int")
} : inout?
```

### Type System
- `cpp2.type` operation wraps C++ type strings
- Type checking happens during pattern matching
- Evidence spans provide type quantification:
  - Check type presence
  - Validate type structure
  - Verify constraints (const, &, *, &&)

### Recursive Lowering
1. Top-level ModuleOp contains functions
2. Each FunctionOp has 3 regions: params, rettype, body
3. Body region contains statements (VariableOp, CallOp, etc.)
4. Expressions recursively parsed within statement ops
5. Symbol dictionaries propagate through region hierarchy

## Migration Steps

### Phase 1: Formalize Evidence Spans
- Document span extraction behaviors (this doc)
- Add span quantification APIs: `span.is_function()`, `span.parameter_count()`
- Implement recursive span parsing for expressions

### Phase 2: Introduce MLIR Operations
- Create mlir::Operation subclasses for CPP2 ops
- Map ConfixOrbit → Block during scan
- Build Operation tree from orbit tree
- Maintain symbol dictionaries per region

### Phase 3: Pattern Matching on Operations
- Replace string patterns with Operation matchers
- Pattern: `FunctionOp with params.empty() and rettype=="int"`
- Lowering: FunctionOp → C++ function declaration
- ExpressionOps → C++ expressions

### Phase 4: Delete Old Abstractions
- Remove ConfixOrbit, OrbitFragment, OrbitIterator
- Keep WideScanner (delimiter detection)
- OrbitRing becomes Block list
- PatternData becomes Operation matcher

## Implementation Priority

1. **Define span behaviors** (subexpression parsing) - 2 hours
2. **Symbol dictionary propagation** - 3 hours
3. **FunctionOp structure** - 2 hours
4. **VariableOp + ParameterOp** - 2 hours
5. **Expression parsing** - 4 hours
6. **Pattern matching on ops** - 3 hours
7. **Migration testing** - iterative
