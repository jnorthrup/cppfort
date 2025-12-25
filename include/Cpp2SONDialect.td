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

def Cpp2SON_AddOp : Cpp2SON_Op<"add", [Pure]> {
  let summary = "Addition node";
  let description = [{
    Represents binary addition in the Sea of Nodes graph.
    Takes two value dependencies and produces their sum.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_SubOp : Cpp2SON_Op<"sub", [Pure]> {
  let summary = "Subtraction node";
  let description = [{
    Represents binary subtraction in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_MulOp : Cpp2SON_Op<"mul", [Pure]> {
  let summary = "Multiplication node";
  let description = [{
    Represents binary multiplication in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_DivOp : Cpp2SON_Op<"div", [Pure]> {
  let summary = "Division node";
  let description = [{
    Represents binary division in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_AndOp : Cpp2SON_Op<"and", [Pure]> {
  let summary = "Boolean AND node";
  let description = [{
    Represents boolean AND in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_OrOp : Cpp2SON_Op<"or", [Pure]> {
  let summary = "Boolean OR node";
  let description = [{
    Represents boolean OR in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_NotOp : Cpp2SON_Op<"not", [Pure]> {
  let summary = "Boolean NOT node";
  let description = [{
    Represents boolean NOT in the Sea of Nodes graph.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_CmpOp : Cpp2SON_Op<"cmp", [Pure]> {
  let summary = "Integer comparison node";
  let description = [{
    Represents a comparison (e.g. lt/le/gt/ge/eq/ne). The predicate is a string.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasFolder = 1;
}

def Cpp2SON_PhiOp : Cpp2SON_Op<"phi", [Pure]> {
  let summary = "Phi node";
  let description = [{
    Represents an SSA merge. For Sea-of-Nodes this can be used for both data and control merges.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$values attr-dict `:` type($values) `->` type($result)";
}
