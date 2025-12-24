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
      build($_builder, $_state, type, value);
    }]>
  ];
}

def Cpp2FIR_ReturnOp : Cpp2FIR_Op<"return", [Terminator, ReturnLike]> {
  let summary = "Return operation";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
