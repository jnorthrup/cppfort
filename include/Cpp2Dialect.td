 
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Cpp2_Dialect : Dialect {
  let name = "cpp2";
  let summary = "Cpp2 Sea of Nodes MLIR Dialect";
  let description = [{
    MLIR dialect for Cpp2 language compilation using Sea of Nodes IR.
    Implements control flow, data flow, and memory operations with
    equivalence-class aliasing from Cliff Click's Simple chapters.
  }];
  let cppNamespace = "::mlir::cpp2";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Cpp2_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cpp2_Dialect, mnemonic, traits>;

// Type definitions
def Cpp2_IntType : TypeDef<Cpp2_Dialect, "Int"> {
  let mnemonic = "int";
  let summary = "Integer type";
}

def Cpp2_PtrType : TypeDef<Cpp2_Dialect, "Ptr"> {
  let summary = "Pointer type with nullable flag";
  let parameters = (ins "Type":$pointee, "bool":$nullable);
  let typeName = "cpp2.ptr";
}

def Cpp2_StructType : TypeDef<Cpp2_Dialect, "Struct"> {
  let mnemonic = "struct";
  let summary = "Struct type";
  let parameters = (ins StringRefParameter<>:$name);
  let assemblyFormat = "`<` $name `>`";
}

def Cpp2_MemType : TypeDef<Cpp2_Dialect, "Mem"> {
  let mnemonic = "mem";
  let summary = "Memory slice type for alias class";
  let parameters = (ins "uint64_t":$aliasClass);
  let assemblyFormat = "`<` $aliasClass `>`";
}

// Control flow operations (Chapter 5-9)
def Cpp2_StartOp : Cpp2_Op<"start"> {
  let summary = "Start node producing initial control and memory";
  let description = [{
    Produces initial control flow and memory slices for each alias class.
    All memory originates from Start.
  }];
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "attr-dict `:` type($results)";
}

def Cpp2_IfOp : Cpp2_Op<"if", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "If operation with true/false branches";
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$condition $thenRegion `else` $elseRegion attr-dict `:` type($results)";
}

def Cpp2_RegionOp : Cpp2_Op<"region"> {
  let summary = "Control flow merge point (Phi for control)";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
}

def Cpp2_LoopOp : Cpp2_Op<"loop", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Loop operation with back-edge";
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$body attr-dict `:` type($results)";
}

// Data flow operations (Chapter 2-4)
def Cpp2_ConstantOp : Cpp2_Op<"constant", [Pure]> {
  let summary = "Constant value";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let hasFolder = 1;
}

def Cpp2_AddOp : Cpp2_Op<"add", [Pure, Commutative]> {
  let summary = "Integer addition";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Cpp2_SubOp : Cpp2_Op<"sub", [Pure]> {
  let summary = "Integer subtraction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasFolder = 1;
}

def Cpp2_MulOp : Cpp2_Op<"mul", [Pure, Commutative]> {
  let summary = "Integer multiplication";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasFolder = 1;
}

def Cpp2_DivOp : Cpp2_Op<"div", [Pure]> {
  let summary = "Integer division";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  // Include operand types and result type explicitly so type inference is unambiguous
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2_PhiOp : Cpp2_Op<"phi", [Pure]> {
  let summary = "SSA Phi node for control flow merge";
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$values attr-dict `:` type($values) `->` type($result)";
  let hasCanonicalizer = 1;
}

// Memory operations (Chapter 10)
def Cpp2_NewOp : Cpp2_Op<"new", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate new struct instance";
  let description = [{
    Creates a new instance of a struct type, returning a pointer.
    Takes control input to ensure proper ordering.
  }];
  let arguments = (ins AnyType:$ctrl, TypeAttr:$structType);
  let results = (outs Cpp2_PtrType:$ptr);
  let assemblyFormat = "$ctrl `,` $structType attr-dict `:` type($ctrl) `,` type($ptr)";
}

def Cpp2_LoadOp : Cpp2_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load value from struct field";
  let description = [{
    Loads from a specific field of a struct, consuming memory slice
    for that field's alias class. Returns both loaded value and
    updated memory slice.
  }];
  let arguments = (ins Cpp2_MemType:$mem, Cpp2_PtrType:$ptr,
                       StrAttr:$field, UI64Attr:$aliasClass);
  let results = (outs AnyType:$value, Cpp2_MemType:$outMem);
  let assemblyFormat = "$mem `[` $ptr `,` $field `]` attr-dict `:` type($mem) `,` type($ptr) `,` type($value) `,` type($outMem)";
}

def Cpp2_StoreOp : Cpp2_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store value to struct field";
  let description = [{
    Stores value to a specific field, consuming and producing memory
    slice for that field's alias class. Enables load-after-store
    optimization via peephole.
  }];
  let arguments = (ins Cpp2_MemType:$mem, Cpp2_PtrType:$ptr,
                       StrAttr:$field, AnyType:$value,
                       UI64Attr:$aliasClass);
  let results = (outs Cpp2_MemType:$outMem);
  let assemblyFormat = "$value `->` $mem `[` $ptr `,` $field `]` attr-dict `:` type($mem) `,` type($ptr) `,` type($value) `,` type($outMem)";
}

def Cpp2_CastOp : Cpp2_Op<"cast", [Pure]> {
  let summary = "Type cast (upcast for structs)";
  let arguments = (ins AnyType:$input, TypeAttr:$targetType);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `to` $targetType attr-dict `:` type($input) `->` type($result)";
}

// Cpp2-specific operations
def Cpp2_UFCSCallOp : Cpp2_Op<"ufcs_call"> {
  let summary = "Unified Function Call Syntax invocation";
  let description = [{
    Represents method call via UFCS: obj.method(args) becomes method(obj, args).
    First argument is implicit 'this'.
  }];
  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $args `)` attr-dict `:` functional-type($args, $results)";
}

def Cpp2_ContractOp : Cpp2_Op<"contract"> {
  let summary = "Contract assertion (pre/post/assert)";
  let arguments = (ins I1:$condition, StrAttr:$kind, OptionalAttr<StrAttr>:$message);
  let assemblyFormat = "$kind $condition (`,` $message^)? attr-dict";
}

def Cpp2_MetafunctionOp : Cpp2_Op<"metafunction"> {
  let summary = "Metafunction application for code generation";
  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$name `(` $args `)` attr-dict `:` functional-type($args, $results)";
}

def Cpp2_ReturnOp : Cpp2_Op<"return", [Terminator]> {
  let summary = "Return operation";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// Additional ops for AST mapping coverage
def Cpp2_FuncOp : Cpp2_Op<"func"> {
  let summary = "Function definition";
  let description = [{
    Represents a function with name, arguments, and body region.
    Maps from Clang FunctionDecl AST nodes.
  }];
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$name $body attr-dict `:` type($results)";
}

def Cpp2_VarOp : Cpp2_Op<"var"> {
  let summary = "Variable declaration with optional initializer";
  let description = [{
    Declares a variable with optional initialization expression.
    Maps from Clang VarDecl AST nodes.
  }];
  let arguments = (ins StrAttr:$name, OptionalAttr<AnyAttr>:$init);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name (`,` $init^)? attr-dict `:` type($result)";
}

def Cpp2_CallOp : Cpp2_Op<"call"> {
  let summary = "Function call operation";
  let description = [{
    Calls a function with given arguments.
    Maps from Clang CallExpr AST nodes.
  }];
  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $args `)` attr-dict `:` functional-type($args, $results)";
}

def Cpp2_BinOpOp : Cpp2_Op<"binop", [Pure]> {
  let summary = "Generic binary operator";
  let description = [{
    Represents binary operations with operator kind.
    Maps from Clang BinaryOperator AST nodes.
    Operator kinds: add, sub, mul, div, mod, eq, ne, lt, le, gt, ge, etc.
  }];
  let arguments = (ins StrAttr:$op, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$op $lhs `,` $rhs attr-dict `:` type($result)";
  let hasFolder = 1;
}

def Cpp2_ForOp : Cpp2_Op<"for", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "For loop with init, condition, increment";
  let description = [{
    Represents C-style for loop: for(init; cond; inc) { body }
    Maps from Clang ForStmt AST nodes.
  }];
  let arguments = (ins Optional<AnyType>:$init, I1:$cond, Optional<AnyType>:$inc);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "(`init` $init^)? `cond` $cond (`inc` $inc^)? $body attr-dict `:` type($results)";
}

def Cpp2_WhileOp : Cpp2_Op<"while", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "While loop with condition";
  let description = [{
    Represents while loop: while(cond) { body }
    Maps from Clang WhileStmt AST nodes.
  }];
  let arguments = (ins I1:$cond);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$cond $body attr-dict `:` type($results)";
}

 