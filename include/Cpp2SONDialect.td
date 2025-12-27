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
  }];
  let arguments = (ins 
    AnyType:$mem,
    AnyType:$ptr,
    Optional<Index>:$offset,
    StrAttr:$name,
    I32Attr:$alias,
    TypeAttr:$declared_type
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
  }];
  let arguments = (ins 
    AnyType:$mem,
    AnyType:$ptr,
    Optional<Index>:$offset,
    AnyType:$value,
    StrAttr:$name,
    I32Attr:$alias,
    TypeAttr:$declared_type,
    UnitAttr:$init
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$mem `[` $ptr (`+` $offset^)? `]` `as` $name `=` $value attr-dict `:` type($mem) `,` type($ptr) `,` type($value) `->` type($result)";
}

def Cpp2SON_NewOp : Cpp2SON_Op<"new"> {
  let summary = "New node - heap allocation (NewNode.java)";
  let description = [{
    Allocate a new object on the heap.
    Returns control, memory, and pointer to new object.
  }];
  let arguments = (ins 
    AnyType:$ctrl,
    AnyType:$mem,
    TypeAttr:$allocated_type
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
