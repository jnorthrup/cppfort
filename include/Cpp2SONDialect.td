include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Cpp2SON_Dialect : Dialect {
  let name = "sond";
  let summary = "Sea of Nodes MLIR Dialect";
  let description = [{
    MLIR dialect for Sea of Nodes intermediate representation.
    Represents computation as a directed acyclic graph where nodes
    are operations and edges represent data/control dependencies.
  }];
  let cppNamespace = "::mlir::sond";
}

class Cpp2SON_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cpp2SON_Dialect, mnemonic, traits>;

// Basic Operations

def Cpp2SON_ConstantOp : Cpp2SON_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Constant node in Sea of Nodes graph";
  let description = [{
    Represents a constant value as a node in the computation graph.
    This is a pure data node with no dependencies.
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

def Cpp2SON_AddOp : Cpp2SON_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Addition node";
  let description = [{
    Represents binary addition in the Sea of Nodes graph.
    Takes two value dependencies and produces their sum.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
