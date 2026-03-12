include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

//===----------------------------------------------------------------------===//
// Sea of Nodes Dialect Definition
// Full MLIR transcription from SeaOfNodes/Simple Chapter 24 Java implementation
//===----------------------------------------------------------------------===//

def Cpp2SON_Dialect : Dialect {
  let name = "sond";
  let summary = "Sea of Nodes MLIR Dialect";
  let description = [{
    Complete MLIR dialect for Sea of Nodes intermediate representation,
    faithfully transcribed from the SeaOfNodes/Simple Java implementation.
    
    Key concepts from Chapter 24:
    - Nodes represent operations with use-def edges (inputs) and def-use edges (outputs)
    - CFGNodes handle control flow (Start, Stop, Region, Loop, If, CProj)
    - Data nodes represent computations (arithmetic, memory, function calls)
    - Phi nodes merge values at control flow join points
    - Types form a lattice for monotone analysis framework (SCCP)
    - The graph supports both pessimistic and optimistic type inference

    Sea of Nodes unifies control and data flow:
    - Control tokens flow through CFG nodes
    - Data values flow through data nodes  
    - Phi nodes connect CFG regions to data merges
    - All nodes carry type information for optimization
  }];
  let cppNamespace = "::mlir::sond";
  // Types and attributes use generated parsers/printers but we provide the dialect methods
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
  let hasConstantMaterializer = 0;
}

class Cpp2SON_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cpp2SON_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Sea of Nodes Type Definitions (from Type.java hierarchy)
//===----------------------------------------------------------------------===//

class Cpp2SON_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Cpp2SON_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Control type for CFG nodes
def SON_ControlTypeDef : Cpp2SON_Type<"Control", "ctrl"> {
  let summary = "Control flow type (Type.CONTROL/XCONTROL)";
  let description = [{
    Represents live or dead control flow in the Sea of Nodes graph.
    Part of the control lattice: TOP > XCONTROL > CONTROL > BOTTOM
  }];
  let parameters = (ins 
    DefaultValuedParameter<"bool", "false">:$is_dead
  );
  let assemblyFormat = "(`dead` $is_dead^)?";
}

// Memory type for alias analysis
def SON_MemoryTypeDef : Cpp2SON_Type<"Memory", "mem"> {
  let summary = "Memory slice type (TypeMem)";
  let description = [{
    Represents a slice of memory with a specific alias class.
  }];
  let parameters = (ins 
    "int64_t":$alias
  );
  let assemblyFormat = "`<` $alias `>`";
}

// Integer type with range for constant propagation
def SON_IntTypeDef : Cpp2SON_Type<"Int", "int"> {
  let summary = "Integer type with range (TypeInteger)";
  let parameters = (ins 
    DefaultValuedParameter<"int64_t", "0">:$min,
    DefaultValuedParameter<"int64_t", "0">:$max,
    DefaultValuedParameter<"int", "64">:$bit_width
  );
  let assemblyFormat = "`<` $min `,` $max (`,` `bits` `=` $bit_width^)? `>`";
}

// Tuple type for multi-value returns
def SON_TupleTypeDef : Cpp2SON_Type<"Tuple", "tuple"> {
  let summary = "Tuple type (TypeTuple)";
  let parameters = (ins 
    ArrayRefParameter<"Type">:$types
  );
  let assemblyFormat = "`<` $types `>`";
}

// Field attribute for structs
def SON_FieldAttr : AttrDef<Cpp2SON_Dialect, "Field"> {
  let mnemonic = "field";
  let summary = "Struct field descriptor";
  let parameters = (ins
    StringRefParameter<>:$name,
    "Type":$type,
    DefaultValuedParameter<"int64_t", "0">:$alias
  );
  let assemblyFormat = "`<` $name `:` $type (`,` `alias` `=` $alias^)? `>`";
}

//===----------------------------------------------------------------------===//
// Java Memory Model (JMM) Attributes
//===----------------------------------------------------------------------===//

// JMM happens-before relationship attribute
def SON_JMMHappensBeforeAttr : AttrDef<Cpp2SON_Dialect, "JMMHappensBefore"> {
  let mnemonic = "jmm_happens_before";
  let summary = "JMM happens-before edge between operations";
  let description = [{
    Establishes a happens-before relationship for Java Memory Model compliance.
    Used to enforce memory ordering guarantees in concurrent operations.

    Maps to C++ memory_order_acquire/release for synchronization.
  }];
  let parameters = (ins
    ArrayRefParameter<"StringRef">:$predecessors,
    DefaultValuedParameter<"bool", "false">:$is_transitive
  );
  let assemblyFormat = "`<` $predecessors (`,` `transitive` `=` $is_transitive^)? `>`";
}

// JMM volatile attribute (sequential consistency)
def SON_JMMVolatileAttr : AttrDef<Cpp2SON_Dialect, "JMMVolatile"> {
  let mnemonic = "jmm_volatile";
  let summary = "JMM volatile (sequential consistency) marker";
  let description = [{
    Marks an operation as requiring sequential consistency semantics.
    Per Java Memory Model, volatile reads/writes have SC total ordering.

    Maps to C++ memory_order_seq_cst.
  }];
  let parameters = (ins
    DefaultValuedParameter<"bool", "true">:$is_volatile
  );
  let assemblyFormat = "`<` $is_volatile `>`";
}

// JMM final field freeze attribute
def SON_JMMFinalFieldAttr : AttrDef<Cpp2SON_Dialect, "JMMFinalField"> {
  let mnemonic = "jmm_final_field";
  let summary = "JMM final field freeze marker";
  let description = [{
    Marks a field as final, requiring freeze semantics.
    Final fields must be safely published before object becomes visible.

    Maps to C++ atomic_thread_fence(memory_order_release) after constructor.
  }];
  let parameters = (ins
    StringRefParameter<>:$constructor_op,
    DefaultValuedParameter<"bool", "false">:$is_frozen
  );
  let assemblyFormat = "`<` `ctor` `=` $constructor_op (`,` `frozen` `=` $is_frozen^)? `>`";
}

// JMM visibility attribute (thread-local vs shared)
def SON_JMMVisibilityAttr : AttrDef<Cpp2SON_Dialect, "JMMVisibility"> {
  let mnemonic = "jmm_visibility";
  let summary = "JMM memory visibility (thread-local or shared)";
  let description = [{
    Tracks whether a value is thread-local or shared across threads.
    Shared values require atomic operations and memory barriers.

    Maps to: std::atomic<T> for shared, plain T for thread-local.
  }];
  let parameters = (ins
    StringRefParameter<>:$visibility_kind  // "thread_local" or "shared"
  );
  let assemblyFormat = "`<` $visibility_kind `>`";
}

// Type constraint for control flow
def SON_ControlType : Type<CPred<"$_self.isa<::mlir::sond::ControlType>()">,
                           "Sea of Nodes control type">;

// Type constraint for memory
def SON_MemoryType : Type<CPred<"$_self.isa<::mlir::sond::MemoryType>()">,
                          "Sea of Nodes memory type">;

//===----------------------------------------------------------------------===//
// Constant and Value Nodes
//===----------------------------------------------------------------------===//

def Cpp2SON_ConstantOp : Cpp2SON_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Constant node in Sea of Nodes graph";
  let description = [{
    Represents a constant value as a node in the computation graph.
    This is a pure data node with no dependencies.
    
    From ConstantNode.java:
    - Constants have Start as an input to enable forward graph walk
    - The edge carries no semantic meaning, only for visitation
    - compute() returns _con (the constant type itself)
    - idealize() returns null (no further optimization)
  }];
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Type":$type, "::mlir::Attribute":$value),
    [{
      $_state.addTypes(type);
      $_state.addAttribute("value", value);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Arithmetic Nodes (from ArithNode.java and subclasses)
//===----------------------------------------------------------------------===//

def Cpp2SON_AddOp : Cpp2SON_Op<"add", [Pure, Commutative]> {
  let summary = "Addition node (AddNode.java)";
  let description = [{
    Represents binary addition in the Sea of Nodes graph.
    Takes two value dependencies and produces their sum.
    Supports integer and floating-point addition.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Cpp2SON_SubOp : Cpp2SON_Op<"sub", [Pure]> {
  let summary = "Subtraction node (SubNode.java)";
  let description = [{
    Represents binary subtraction in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Cpp2SON_MulOp : Cpp2SON_Op<"mul", [Pure, Commutative]> {
  let summary = "Multiplication node (MulNode.java)";
  let description = [{
    Represents binary multiplication in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Cpp2SON_DivOp : Cpp2SON_Op<"div", [Pure]> {
  let summary = "Division node (DivNode.java)";
  let description = [{
    Represents binary division in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_MinusOp : Cpp2SON_Op<"minus", [Pure]> {
  let summary = "Unary minus node (MinusNode.java)";
  let description = [{
    Represents unary negation in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Floating Point Arithmetic (from ArithFNode.java subclasses)
//===----------------------------------------------------------------------===//

def Cpp2SON_AddFOp : Cpp2SON_Op<"addf", [Pure, Commutative]> {
  let summary = "Float addition node (AddFNode.java)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_SubFOp : Cpp2SON_Op<"subf", [Pure]> {
  let summary = "Float subtraction node (SubFNode.java)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_MulFOp : Cpp2SON_Op<"mulf", [Pure, Commutative]> {
  let summary = "Float multiplication node (MulFNode.java)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_DivFOp : Cpp2SON_Op<"divf", [Pure]> {
  let summary = "Float division node (DivFNode.java)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_MinusFOp : Cpp2SON_Op<"minusf", [Pure]> {
  let summary = "Float unary minus node (MinusFNode.java)";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_RoundF32Op : Cpp2SON_Op<"roundf32", [Pure]> {
  let summary = "Round to float32 (RoundF32Node.java)";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Cpp2SON_ToFloatOp : Cpp2SON_Op<"tofloat", [Pure]> {
  let summary = "Convert to float (ToFloatNode.java)";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Bitwise/Logical Nodes (from BoolNode.java subclasses)
//===----------------------------------------------------------------------===//

def Cpp2SON_AndOp : Cpp2SON_Op<"and", [Pure, Commutative]> {
  let summary = "Boolean/bitwise AND node (AndNode.java)";
  let description = [{
    Represents boolean AND or bitwise AND in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_OrOp : Cpp2SON_Op<"or", [Pure, Commutative]> {
  let summary = "Boolean/bitwise OR node (OrNode.java)";
  let description = [{
    Represents boolean OR or bitwise OR in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_XorOp : Cpp2SON_Op<"xor", [Pure, Commutative]> {
  let summary = "Bitwise XOR node (XorNode.java)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_NotOp : Cpp2SON_Op<"not", [Pure]> {
  let summary = "Boolean NOT node (NotNode.java)";
  let description = [{
    Represents boolean NOT in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_ShlOp : Cpp2SON_Op<"shl", [Pure]> {
  let summary = "Shift left node (ShlNode.java)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_ShrOp : Cpp2SON_Op<"shr", [Pure]> {
  let summary = "Logical shift right node (ShrNode.java)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_SarOp : Cpp2SON_Op<"sar", [Pure]> {
  let summary = "Arithmetic shift right node (SarNode.java)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Comparison Nodes (from BoolNode.java)
//===----------------------------------------------------------------------===//

def Cpp2SON_CmpOp : Cpp2SON_Op<"cmp", [Pure]> {
  let summary = "Integer comparison node";
  let description = [{
    Represents a comparison (e.g. lt/le/gt/ge/eq/ne). 
    From BoolNode.java - the predicate determines comparison semantics.
    
    Chapter 24 supports chained relationals: (min <= x < max)
    All operators must "point" the same direction for chaining.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CFG Control Flow Nodes (from CFGNode.java and subclasses)
//===----------------------------------------------------------------------===//

def Cpp2SON_StartOp : Cpp2SON_Op<"start", [NoTerminator, IsolatedFromAbove]> {
  let summary = "Start node - function entry point (StartNode.java)";
  let description = [{
    The Start node represents the start of the function.
    From StartNode.java:
    - Extends LoopNode (for loop tree building)
    - Has initial control, memory, and arguments
    - idepth() returns 0 (root of dominator tree)
    - Contains all constants and trailing FunNodes
  }];
  let arguments = (ins OptionalAttr<TypeAttr>:$arg_type);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def Cpp2SON_StopOp : Cpp2SON_Op<"stop", [Terminator]> {
  let summary = "Stop node - function exit aggregator (StopNode.java)";
  let description = [{
    The Stop node aggregates all function exit points.
    All return nodes flow into Stop for unified exit handling.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let assemblyFormat = "($inputs^ `:` type($inputs))? attr-dict";
}

def Cpp2SON_RegionOp : Cpp2SON_Op<"region", [NoTerminator]> {
  let summary = "Region node - control flow merge (RegionNode.java)";
  let description = [{
    Block head that merges control flow from multiple predecessors.
    From RegionNode.java:
    - Has Phis for merged values
    - compute() meets all input control types
    - idealize() removes dead paths, collapses single-input regions
    - Supports flattening stacked regions
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "($inputs^ `:` type($inputs))? attr-dict-with-keyword $body";
}

def Cpp2SON_LoopOp : Cpp2SON_Op<"loop", [NoTerminator]> {
  let summary = "Loop node - loop header (LoopNode.java)";
  let description = [{
    Loop node extends Region for loop-specific semantics.
    From LoopNode.java:
    - entry() is cfg(1), back() is cfg(2)
    - compute() returns entry control type
    - Supports loop tree building for scheduling
    - forceExit() handles infinite loops
  }];
  let arguments = (ins 
    AnyType:$entry,
    Optional<AnyType>:$back
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$entry `:` type($entry) (`,` $back `:` type($back)^)? attr-dict-with-keyword $body";
}

def Cpp2SON_IfOp : Cpp2SON_Op<"if", [Terminator]> {
  let summary = "If node - conditional branch (IfNode.java)";
  let description = [{
    Conditional branch node producing two control outputs.
    From IfNode.java:
    - ctrl() is input 0, pred() is input 1 (predicate)
    - compute() returns IF_NEITHER/IF_FALSE/IF_TRUE/IF_BOTH tuple
    - High predicate types mean neither side is reachable
    - Constant predicates select one side only
    - idealize() dominance-based branch folding
  }];
  let arguments = (ins AnyType:$ctrl, AnyType:$pred);
  let results = (outs AnyType:$true_ctrl, AnyType:$false_ctrl);
  let assemblyFormat = "$ctrl `,` $pred attr-dict `:` type($ctrl) `,` type($pred) `->` type($true_ctrl) `,` type($false_ctrl)";
}

def Cpp2SON_CProjOp : Cpp2SON_Op<"cproj"> {
  let summary = "Control projection from multi-output CFG node (CProjNode.java)";
  let description = [{
    Projects a single control output from a multi-output CFG node (like If).
    Similar to ProjNode but specifically for control flow.
  }];
  let arguments = (ins AnyType:$input, I32Attr:$index, OptionalAttr<StrAttr>:$label);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `[` $index `]` (`(` $label^ `)`)? attr-dict `:` type($input) `->` type($result)";
}

def Cpp2SON_NeverOp : Cpp2SON_Op<"never", [Terminator]> {
  let summary = "Never node - unreachable branch (NeverNode.java)";
  let description = [{
    An If that ideally never branches (infinite loop exit).
    Used to force exits from infinite loops.
  }];
  let arguments = (ins AnyType:$ctrl);
  let results = (outs AnyType:$true_ctrl, AnyType:$false_ctrl);
  let assemblyFormat = "$ctrl attr-dict `:` type($ctrl) `->` type($true_ctrl) `,` type($false_ctrl)";
}

def Cpp2SON_XCtrlOp : Cpp2SON_Op<"xctrl", [Pure]> {
  let summary = "Dead/unreachable control (XCtrlNode.java)";
  let description = [{
    Represents unreachable control flow (Type.XCONTROL).
    Used for dead code paths and optimization.
  }];
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Cpp2SON_CtrlOp : Cpp2SON_Op<"ctrl", [Pure]> {
  let summary = "Live control constant (CtrlNode.java)";
  let description = [{
    Represents live control flow (Type.CONTROL).
  }];
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Phi and Data Merge Nodes
//===----------------------------------------------------------------------===//

def Cpp2SON_PhiOp : Cpp2SON_Op<"phi", [Pure]> {
  let summary = "Phi node - SSA value merge (PhiNode.java)";
  let description = [{
    Represents an SSA merge at control flow join points.
    From PhiNode.java:
    - region() is input 0, values are inputs 1..N
    - _minType constrains the phi type domain
    - compute() meets all input types from live paths
    - idealize() handles single-unique-input collapse
    - Supports loop widening for integer types
    
    For Sea-of-Nodes, this is used for data merges.
    Control merges use Region nodes.
  }];
  let arguments = (ins 
    AnyType:$region,
    Variadic<AnyType>:$values,
    OptionalAttr<StrAttr>:$label,
    OptionalAttr<TypeAttr>:$min_type
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "`[` $region `]` `(` $values `)` attr-dict `:` type($region) `,` type($values) `->` type($result)";
}

def Cpp2SON_ParmOp : Cpp2SON_Op<"parm", [Pure]> {
  let summary = "Parameter node - function argument (ParmNode.java)";
  let description = [{
    Extends PhiNode for function parameters.
    Merges argument values from all call sites.
  }];
  let arguments = (ins 
    AnyType:$fun,
    I32Attr:$index,
    OptionalAttr<StrAttr>:$label
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$fun `[` $index `]` (`(` $label^ `)`)? attr-dict `:` type($fun) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Function Nodes (from FunNode.java, ReturnNode.java, CallNode.java)
//===----------------------------------------------------------------------===//

def Cpp2SON_FunOp : Cpp2SON_Op<"fun", [
    FunctionOpInterface, Symbol, IsolatedFromAbove
  ]> {
  let summary = "Function definition node (FunNode.java)";
  let description = [{
    Extends RegionNode for function-specific semantics.
    From FunNode.java:
    - _sig is the function signature (TypeFunPtr)
    - _ret links to the ReturnNode
    - Supports interprocedural SCCP via call graph linking
    - unknownCallers() tracks if Start is still connected
  }];
  let arguments = (ins 
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<ArrayAttr>:$arg_attrs,
    OptionalAttr<ArrayAttr>:$res_attrs,
    OptionalAttr<I64Attr>:$fidx  // Function index for call graph
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name `:` $function_type attr-dict-with-keyword $body";
  
  let extraClassDeclaration = [{
    // FunctionOpInterface methods
    ::mlir::Region *getCallableRegion() { return &getBody(); }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
    void setFunctionType(::mlir::Type type) { setFunctionTypeAttr(::mlir::TypeAttr::get(type)); }
  }];
}

def Cpp2SON_ReturnOp : Cpp2SON_Op<"return", [Terminator, ReturnLike, AttrSizedOperandSegments]> {
  let summary = "Return node - function exit (ReturnNode.java)";
  let description = [{
    From ReturnNode.java:
    - ctrl() input 0, mem() input 1, expr() input 2, rpc() input 3
    - Returns control, memory, and result value
    - Links to parent FunNode via _fun
    - compute() returns TypeTuple of ctrl/mem/expr types
  }];
  let arguments = (ins 
    Optional<AnyType>:$ctrl,
    Optional<AnyType>:$mem,
    Variadic<AnyType>:$values,
    Optional<AnyType>:$rpc
  );
  let assemblyFormat = "attr-dict (`ctrl` $ctrl^ `:` type($ctrl))? (`mem` $mem^ `:` type($mem))? (`values` $values^ `:` type($values))? (`rpc` $rpc^ `:` type($rpc))?";
}

def Cpp2SON_CallOp : Cpp2SON_Op<"call"> {
  let summary = "Call node - function invocation (CallNode.java)";
  let description = [{
    From CallNode.java:
    - ctrl() input 0, mem() input 1, args 2..N-1, fptr() last
    - Links to CallEndNode for results
    - Supports interprocedural optimization via call graph
  }];
  let arguments = (ins 
    AnyType:$ctrl,
    AnyType:$mem,
    Variadic<AnyType>:$args,
    AnyType:$fptr,
    OptionalAttr<StrAttr>:$callee
  );
  let results = (outs AnyType:$call_result);
  let assemblyFormat = "$callee `(` $args `)` `fptr` $fptr `ctrl` $ctrl `mem` $mem attr-dict `:` type($ctrl) `,` type($mem) `,` type($args) `,` type($fptr) `->` type($call_result)";
}

def Cpp2SON_CallEndOp : Cpp2SON_Op<"call_end"> {
  let summary = "CallEnd node - call result aggregator (CallEndNode.java)";
  let description = [{
    Block head that follows a Call node.
    Aggregates results from all called functions' returns.
    Followed by CProj and Proj nodes to extract results.
  }];
  let arguments = (ins AnyType:$call);
  let results = (outs AnyType:$ctrl, AnyType:$mem, AnyType:$result);
  let assemblyFormat = "$call attr-dict `:` type($call) `->` type($ctrl) `,` type($mem) `,` type($result)";
}

def Cpp2SON_ExternOp : Cpp2SON_Op<"extern", [Pure]> {
  let summary = "External function reference (ExternNode.java)";
  let description = [{
    Reference to an externally defined function.
  }];
  let arguments = (ins StrAttr:$name, TypeAttr:$signature);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name `:` $signature attr-dict `->` type($result)";
}

def Cpp2SON_FRefOp : Cpp2SON_Op<"fref", [Pure]> {
  let summary = "Function reference (FRefNode.java)";
  let description = [{
    A reference to a function, producing a function pointer.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$callee attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Memory Operations (from MemOpNode.java, LoadNode.java, StoreNode.java)
//===----------------------------------------------------------------------===//

def Cpp2SON_LoadOp : Cpp2SON_Op<"load"> {
  let summary = "Load node - memory read (LoadNode.java)";
  let description = [{
    Load a value from memory through a pointer.
    From LoadNode.java:
    - mem() input 1, ptr() input 2, off() input 3
    - _name is the field name, _alias is memory alias
    - compute() checks ptr type and memory type
    - idealize() handles load-after-store, load-after-new

    JMM Support:
    - jmm_happens_before: Enforce happens-before ordering
    - jmm_volatile: Sequential consistency for volatile loads
    - jmm_visibility: Thread-local or shared access
  }];
  let arguments = (ins
    AnyType:$mem,
    AnyType:$ptr,
    Optional<Index>:$offset,
    StrAttr:$name,
    I32Attr:$alias,
    TypeAttr:$declared_type,
    OptionalAttr<SON_JMMHappensBeforeAttr>:$jmm_happens_before,
    OptionalAttr<SON_JMMVolatileAttr>:$jmm_volatile,
    OptionalAttr<SON_JMMVisibilityAttr>:$jmm_visibility
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$mem `[` $ptr (`+` $offset^)? `]` `as` $name attr-dict `:` type($mem) `,` type($ptr) `->` type($result)";
}

def Cpp2SON_StoreOp : Cpp2SON_Op<"store"> {
  let summary = "Store node - memory write (StoreNode.java)";
  let description = [{
    Store a value to memory through a pointer.
    From StoreNode.java:
    - mem() input 1, ptr() input 2, off() input 3, val() input 4
    - _init flag for initializing writes (allow null)
    - compute() returns TypeMem with alias and value type
    - idealize() handles store-after-store elimination

    JMM Support:
    - jmm_happens_before: Enforce happens-before ordering
    - jmm_volatile: Sequential consistency for volatile stores
    - jmm_visibility: Thread-local or shared access
    - jmm_final_field: Final field freeze semantics
  }];
  let arguments = (ins
    AnyType:$mem,
    AnyType:$ptr,
    Optional<Index>:$offset,
    AnyType:$value,
    StrAttr:$name,
    I32Attr:$alias,
    TypeAttr:$declared_type,
    UnitAttr:$init,
    OptionalAttr<SON_JMMHappensBeforeAttr>:$jmm_happens_before,
    OptionalAttr<SON_JMMVolatileAttr>:$jmm_volatile,
    OptionalAttr<SON_JMMVisibilityAttr>:$jmm_visibility,
    OptionalAttr<SON_JMMFinalFieldAttr>:$jmm_final_field
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$mem `[` $ptr (`+` $offset^)? `]` `as` $name `=` $value attr-dict `:` type($mem) `,` type($ptr) `,` type($value) `->` type($result)";
}

def Cpp2SON_NewOp : Cpp2SON_Op<"new"> {
  let summary = "New node - heap allocation (NewNode.java)";
  let description = [{
    Allocate a new object on the heap.
    Returns control, memory, and pointer to new object.

    JMM Support:
    - jmm_final_field: Mark object as having final fields requiring freeze
    - jmm_visibility: Thread-local or shared allocation
  }];
  let arguments = (ins
    AnyType:$ctrl,
    AnyType:$mem,
    TypeAttr:$allocated_type,
    OptionalAttr<SON_JMMFinalFieldAttr>:$jmm_final_field,
    OptionalAttr<SON_JMMVisibilityAttr>:$jmm_visibility
  );
  let results = (outs AnyType:$out_ctrl, AnyType:$out_mem, AnyType:$ptr);
  let assemblyFormat = "$ctrl `:` type($ctrl) `,` $mem `:` type($mem) `:` $allocated_type attr-dict `->` type($out_ctrl) `,` type($out_mem) `,` type($ptr)";
}

def Cpp2SON_MemMergeOp : Cpp2SON_Op<"mem_merge"> {
  let summary = "Memory merge node (MemMergeNode.java)";
  let description = [{
    Merges multiple memory aliases after a call or split.
    Used for memory SSA representation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
}

def Cpp2SON_SplitOp : Cpp2SON_Op<"split"> {
  let summary = "Memory split node (SplitNode.java)";
  let description = [{
    Splits memory into separate aliases for parallel analysis.
  }];
  let arguments = (ins AnyType:$mem, I32Attr:$alias);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$mem `alias` $alias attr-dict `:` type($mem) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Projection Nodes
//===----------------------------------------------------------------------===//

def Cpp2SON_ProjOp : Cpp2SON_Op<"proj", [Pure]> {
  let summary = "Projection node - extract from tuple (ProjNode.java)";
  let description = [{
    Extracts a single value from a multi-output node.
    Used for Start, New, Call results.
  }];
  let arguments = (ins AnyType:$input, I32Attr:$index, OptionalAttr<StrAttr>:$label);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `[` $index `]` (`(` $label^ `)`)? attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Cast and Type Conversion
//===----------------------------------------------------------------------===//

def Cpp2SON_CastOp : Cpp2SON_Op<"cast", [Pure]> {
  let summary = "Type cast node (CastNode.java)";
  let description = [{
    Cast a value to a different type.
    Used for type assertions and narrowing.
  }];
  let arguments = (ins AnyType:$input, TypeAttr:$target_type);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `to` $target_type attr-dict `:` type($input) `->` type($result)";
}

def Cpp2SON_ReadOnlyOp : Cpp2SON_Op<"readonly", [Pure]> {
  let summary = "Mark memory as read-only (ReadOnlyNode.java)";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Scope and Variable Tracking
//===----------------------------------------------------------------------===//

def Cpp2SON_ScopeOp : Cpp2SON_Op<"scope"> {
  let summary = "Scope node - variable tracking (ScopeNode.java)";
  let description = [{
    Tracks variables in a lexical scope.
    Maintains a stack of scopes for nested blocks.
  }];
  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "($name^)? attr-dict-with-keyword $body";
}

//===----------------------------------------------------------------------===//
// Struct and Field Access
//===----------------------------------------------------------------------===//

def Cpp2SON_StructOp : Cpp2SON_Op<"struct"> {
  let summary = "Struct definition (StructNode.java)";
  let description = [{
    Defines a struct type with named fields.
  }];
  let arguments = (ins StrAttr:$name, ArrayAttr:$fields);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$name `{` $fields `}` attr-dict-with-keyword $body";
}

def Cpp2SON_ConFldOffOp : Cpp2SON_Op<"confldoff", [Pure]> {
  let summary = "Constant field offset (ConFldOffNode.java)";
  let arguments = (ins StrAttr:$field_name, I64Attr:$offset);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$field_name `offset` $offset attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Callee-Save and Machine-Level Support
//===----------------------------------------------------------------------===//

def Cpp2SON_CalleeSaveOp : Cpp2SON_Op<"callee_save"> {
  let summary = "Callee-save register marker (CalleeSaveNode.java)";
  let arguments = (ins AnyType:$input, I32Attr:$reg);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `reg` $reg attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Constructor and Object Lifecycle
//===----------------------------------------------------------------------===//

def Cpp2SON_ConstructorEndOp : Cpp2SON_Op<"constructor_end"> {
  let summary = "Constructor completion boundary - final field freeze point";
  let description = [{
    Marks the end of a constructor, establishing the freeze point for final fields.
    Per JMM, a release fence must be emitted after constructor completion
    to ensure final fields are visible to other threads before the object
    reference becomes accessible.

    JMM Support:
    - jmm_final_field: Required - specifies which constructor completed
    - jmm_happens_before: Establishes ordering with object publication
  }];
  let arguments = (ins
    AnyType:$ctrl,
    AnyType:$mem,
    AnyType:$object_ptr,
    SON_JMMFinalFieldAttr:$jmm_final_field,
    OptionalAttr<SON_JMMHappensBeforeAttr>:$jmm_happens_before
  );
  let results = (outs AnyType:$out_ctrl, AnyType:$out_mem);
  let assemblyFormat = "$object_ptr `ctrl` $ctrl `mem` $mem attr-dict `:` type($ctrl) `,` type($mem) `,` type($object_ptr) `->` type($out_ctrl) `,` type($out_mem)";
}

//===----------------------------------------------------------------------===//
// Concurrency Operations (Channel, Spawn, Await)
//===----------------------------------------------------------------------===//

def Cpp2SON_SendOp : Cpp2SON_Op<"send"> {
  let summary = "Channel send operation - transfer ownership";
  let description = [{
    Send a value through a channel, transferring ownership.
    Implements Kotlin-style channel semantics for Cpp2.

    JMM Support:
    - jmm_happens_before: Establish happens-before edge with receive
    - jmm_visibility: Always shared (channel operations are concurrent)
  }];
  let arguments = (ins
    AnyType:$ctrl,
    AnyType:$mem,
    AnyType:$channel,
    AnyType:$value,
    OptionalAttr<SON_JMMHappensBeforeAttr>:$jmm_happens_before,
    OptionalAttr<SON_JMMVisibilityAttr>:$jmm_visibility
  );
  let results = (outs AnyType:$out_ctrl, AnyType:$out_mem);
  let assemblyFormat = "`send` $channel `,` $value `ctrl` $ctrl `mem` $mem attr-dict `:` type($ctrl) `,` type($mem) `,` type($channel) `,` type($value) `->` type($out_ctrl) `,` type($out_mem)";
}

def Cpp2SON_RecvOp : Cpp2SON_Op<"recv"> {
  let summary = "Channel receive operation - acquire ownership";
  let description = [{
    Receive a value from a channel, acquiring ownership.
    Blocks until a value is available or channel is closed.

    JMM Support:
    - jmm_happens_before: Establish happens-before edge with send
    - jmm_visibility: Always shared (channel operations are concurrent)
  }];
  let arguments = (ins
    AnyType:$ctrl,
    AnyType:$mem,
    AnyType:$channel,
    OptionalAttr<SON_JMMHappensBeforeAttr>:$jmm_happens_before,
    OptionalAttr<SON_JMMVisibilityAttr>:$jmm_visibility
  );
  let results = (outs AnyType:$out_ctrl, AnyType:$out_mem, AnyType:$value);
  let assemblyFormat = "`recv` $channel `ctrl` $ctrl `mem` $mem attr-dict `:` type($ctrl) `,` type($mem) `,` type($channel) `->` type($out_ctrl) `,` type($out_mem) `,` type($value)";
}

def Cpp2SON_SpawnOp : Cpp2SON_Op<"spawn"> {
  let summary = "Spawn concurrent task - create new thread/coroutine";
  let description = [{
    Spawn a new concurrent task (thread or coroutine).
    Returns a handle for synchronization (join/await).

    JMM Support:
    - jmm_happens_before: Parent spawns, then child starts
    - jmm_visibility: Shared (task runs concurrently)
  }];
  let arguments = (ins
    AnyType:$ctrl,
    AnyType:$mem,
    AnyType:$task_fn,
    Variadic<AnyType>:$args,
    OptionalAttr<SON_JMMHappensBeforeAttr>:$jmm_happens_before,
    OptionalAttr<SON_JMMVisibilityAttr>:$jmm_visibility
  );
  let results = (outs AnyType:$out_ctrl, AnyType:$out_mem, AnyType:$task_handle);
  let assemblyFormat = "$task_fn `(` $args `)` `ctrl` $ctrl `mem` $mem attr-dict `:` type($ctrl) `,` type($mem) `,` type($task_fn) `,` type($args) `->` type($out_ctrl) `,` type($out_mem) `,` type($task_handle)";
}

def Cpp2SON_AwaitOp : Cpp2SON_Op<"await"> {
  let summary = "Await task completion - synchronization point";
  let description = [{
    Wait for a spawned task to complete and retrieve its result.
    Establishes happens-before relationship with task completion.

    JMM Support:
    - jmm_happens_before: Task completes before await returns
    - jmm_visibility: Shared (synchronizes with spawned task)
  }];
  let arguments = (ins
    AnyType:$ctrl,
    AnyType:$mem,
    AnyType:$task_handle,
    OptionalAttr<SON_JMMHappensBeforeAttr>:$jmm_happens_before,
    OptionalAttr<SON_JMMVisibilityAttr>:$jmm_visibility
  );
  let results = (outs AnyType:$out_ctrl, AnyType:$out_mem, AnyType:$result);
  let assemblyFormat = "$task_handle `ctrl` $ctrl `mem` $mem attr-dict `:` type($ctrl) `,` type($mem) `,` type($task_handle) `->` type($out_ctrl) `,` type($out_mem) `,` type($result)";
}

//===----------------------------------------------------------------------===//
// Indexed/Series/Tensor Operations (cppfort canonical types)
//===----------------------------------------------------------------------===//

def Cpp2SON_IndexedOp : Cpp2SON_Op<"indexed", [Pure]> {
  let summary = "Indexed function from domain I to value type (coordinate-based indexing)";
  let description = [{
    Represents a function from domain I to value type T.
    The 'at' callable takes an index and returns a value.
    
    Canonical cppfort type: indexed<I, F> where F is a callable I -> T
    Domain types: i1, i8, i16, i32, i64, i128, isize
  }];
  let arguments = (ins
    TypeAttr:$domain_type,  // I: the domain type (i32, i64, etc.)
    AnyType:$at            // F: the callable function
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$domain_type `at` $at attr-dict `:` type($result)";
}

def Cpp2SON_SeriesOp : Cpp2SON_Op<"series", [Pure]> {
  let summary = "Indexed sequence (alias for indexed<int, F>)";
  let description = [{
    Represents an indexed sequence of values.
    This is an alias for indexed<int, F> where F is a callable int -> T.
    
    Canonical cppfort type: series<F> = indexed<int, F>
  }];
  let arguments = (ins
    AnyType:$size,          // The size/length of the series
    AnyType:$at            // The callable for element access
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$size `at` $at attr-dict `:` type($result)";
}

def Cpp2SON_TensorOp : Cpp2SON_Op<"tensor", [Pure]> {
  let summary = "Sparse tensor with coordinate keys";
  let description = [{
    Represents a sparse tensor with coordinate-based keys.
    The axes describe the coordinate space, and 'at' maps coordinates to values.
    
    Canonical cppfort type: tensor<K, F> where K is key type, F is coord -> T
  }];
  let arguments = (ins
    AnyType:$axes,         // Series of axis descriptors
    AnyType:$at            // Callable: coord -> value
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$axes `at` $at attr-dict `:` type($result)";
}

def Cpp2SON_DenseTensorOp : Cpp2SON_Op<"dense_tensor", [Pure, MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Dense array with semantic shape + lowered strides";
  let description = [{
    Represents a dense tensor with semantic shape and computed strides.
    The data is a contiguous memory buffer with offset computed from strides.
    
    Canonical cppfort type: dense_tensor<K, T>
    - axes: semantic shape
    - strides: lowered memory layout
    - data: the actual contiguous storage
  }];
  let arguments = (ins
    AnyType:$axes,         // Semantic shape
    AnyType:$strides,      // Computed strides
    AnyType:$data         // The data buffer
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$axes `,` $strides `,` $data attr-dict `:` type($axes) `,` type($strides) `,` type($data) `->` type($result)";
}

def Cpp2SON_AtlasOp : Cpp2SON_Op<"atlas", [Pure]> {
  let summary = "Chart management for manifolds";
  let description = [{
    Represents a collection of charts that cover a manifold.
    Each chart provides a coordinate mapping from the manifold to Euclidean space.
    
    Canonical cppfort type: atlas<C, Chart> where C is coord type, Chart is the chart function
  }];
  let arguments = (ins
    AnyType:$charts,       // Indexed collection of charts
    AnyType:$coord         // Coordinate type
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$charts `coord` $coord attr-dict `:` type($charts) `,` type($coord) `->` type($result)";
}

def Cpp2SON_ManifoldOp : Cpp2SON_Op<"manifold", [Pure]> {
  let summary = "Smooth coordinate space (alias for atlas)";
  let description = [{
    Represents a smooth manifold with coordinate charts.
    This is an alias for atlas<C, Chart>.
    
    Canonical cppfort type: manifold<C, Chart> = atlas<C, Chart>
  }];
  let arguments = (ins
    AnyType:$charts,       // Chart collection
    AnyType:$coord         // Coordinate type
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$charts `coord` $coord attr-dict `:` type($charts) `,` type($coord) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Automatic Differentiation Operations
//===----------------------------------------------------------------------===//

def Cpp2SON_GradDiffOp : Cpp2SON_Op<"grad_diff", [Pure]> {
  let summary = "Automatic differentiation - gradient computation";
  let description = [{
    Computes the gradient of an expression with respect to a variable.
    Implements reverse-mode automatic differentiation.
    
    Canonical cppfort: grad_backend protocol
    - expr: the expression to differentiate
    - variable: the variable to differentiate with respect to
  }];
  let arguments = (ins
    AnyType:$expr,         // Expression to differentiate
    AnyType:$variable      // Variable to differentiate w.r.t.
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$expr `,` $variable attr-dict `:` type($expr) `,` type($variable) `->` type($result)";
}

def Cpp2SON_JacobianOp : Cpp2SON_Op<"jacobian", [Pure]> {
  let summary = "Jacobian matrix computation";
  let description = [{
    Computes the Jacobian matrix of a vector-valued function.
    The Jacobian contains all first-order partial derivatives.
    
    Used in chain rule for manifold differentiation:
    grad(manifold) = jacobian * inner_grad
  }];
  let arguments = (ins
    AnyType:$expr,         // Vector-valued expression
    AnyType:$variables     // List of variables
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$expr `,` $variables attr-dict `:` type($expr) `,` type($variables) `->` type($result)";
}

def Cpp2SON_MatrixMulOp : Cpp2SON_Op<"matrix_mul", [Pure]> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication of two matrices/vectors.
    Used for Jacobian-vector products in AD and general linear algebra.
    
    Represents: result = lhs * rhs
  }];
  let arguments = (ins
    AnyType:$lhs,          // Left-hand matrix/vector
    AnyType:$rhs          // Right-hand matrix/vector
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}
