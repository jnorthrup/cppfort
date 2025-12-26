include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Cpp2FIR_Dialect : Dialect {
  let name = "cpp2fir";
  let summary = "Cpp2 Front-IR MLIR Dialect";
  let description = [{
    MLIR dialect for Cpp2 language frontend representation.
    Faithfully represents Cpp2 language constructs including types,
    functions, unified function calls, and contracts.
  }];
  let cppNamespace = "::mlir::cpp2fir";
}

class Cpp2FIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cpp2FIR_Dialect, mnemonic, traits>;

// Basic Operations

def Cpp2FIR_FuncOp : Cpp2FIR_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "Cpp2 function definition";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<ArrayAttr>:$arg_attrs,
                       OptionalAttr<ArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);
  
  let assemblyFormat = [{
    $sym_name `:` $function_type attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    // FunctionOpInterface methods
    ::mlir::Region *getCallableRegion() { return &getBody(); }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
    void setFunctionType(::mlir::Type type) { setFunctionTypeAttr(::mlir::TypeAttr::get(type)); }
  }];
}

def Cpp2FIR_ConstantOp : Cpp2FIR_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Constant value";
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

def Cpp2FIR_ReturnOp : Cpp2FIR_Op<"return", [Terminator, ReturnLike]> {
  let summary = "Return operation";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// Sea-of-Nodes-oriented expression ops (Chapter 24 compass)

def Cpp2FIR_AddOp : Cpp2FIR_Op<"add", [Pure]> {
  let summary = "Addition";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_SubOp : Cpp2FIR_Op<"sub", [Pure]> {
  let summary = "Subtraction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_MulOp : Cpp2FIR_Op<"mul", [Pure]> {
  let summary = "Multiplication";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_DivOp : Cpp2FIR_Op<"div", [Pure]> {
  let summary = "Division";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_AndOp : Cpp2FIR_Op<"and", [Pure]> {
  let summary = "Boolean AND";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_OrOp : Cpp2FIR_Op<"or", [Pure]> {
  let summary = "Boolean OR";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_NotOp : Cpp2FIR_Op<"not", [Pure]> {
  let summary = "Boolean NOT";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Cpp2FIR_CmpOp : Cpp2FIR_Op<"cmp", [Pure]> {
  let summary = "Comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_PhiOp : Cpp2FIR_Op<"phi", [Pure]> {
  let summary = "Phi";
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$values attr-dict `:` type($values) `->` type($result)";
}

// Contract Operations (Cpp2 safety contracts)

def Cpp2FIR_AssertOp : Cpp2FIR_Op<"assert"> {
  let summary = "Runtime assertion contract";
  let description = [{
    Asserts that a condition is true at runtime. Can include optional message,
    contract categories (type_safety, bounds_safety, etc.), and audit flag.
  }];
  let arguments = (ins AnyType:$condition,
                       OptionalAttr<StrAttr>:$message,
                       OptionalAttr<ArrayAttr>:$categories,
                       UnitAttr:$audit);
  let assemblyFormat = "$condition attr-dict `:` type($condition) (`:` $message^)?";
}

def Cpp2FIR_PreconditionOp : Cpp2FIR_Op<"precondition"> {
  let summary = "Function precondition contract";
  let description = [{
    Specifies a precondition that must be true when a function is called.
    Checked at function entry (or may be checked statically).
  }];
  let arguments = (ins AnyType:$condition,
                       OptionalAttr<StrAttr>:$message,
                       OptionalAttr<ArrayAttr>:$categories,
                       UnitAttr:$audit);
  let assemblyFormat = "$condition attr-dict `:` type($condition) (`:` $message^)?";
}

def Cpp2FIR_PostconditionOp : Cpp2FIR_Op<"postcondition"> {
  let summary = "Function postcondition contract";
  let description = [{
    Specifies a postcondition that must be true when a function returns.
    May reference the return value via special names.
  }];
  let arguments = (ins AnyType:$condition,
                       OptionalAttr<StrAttr>:$message,
                       OptionalAttr<ArrayAttr>:$categories,
                       UnitAttr:$audit);
  let assemblyFormat = "$condition attr-dict `:` type($condition) (`:` $message^)?";
}
