//===----------------------------------------------------------------------===//
// Cpp2 Sea-of-Nodes MLIR Dialect Definition
// TrikeShed Math-Based SoN Compiler
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Cpp2SON_Dialect : Dialect {
  let name = "cpp2";
  let cppNamespace = "::cpp2";
  let description = [{
    Cpp2 Sea-of-Nodes dialect for math-based compilation.
    Implements TrikeShed semantic objects with zero-cost abstractions.
  }];
}

//===----------------------------------------------------------------------===//
// Minimal IndexedOp for testing
//===----------------------------------------------------------------------===//

def IndexedOp : Op<Cpp2SON_Dialect, "indexed"> {
  let summary = "Coordinate-based indexing operation";
  let description = [{
    Represents indexed<I, F> - function from domain I to value type F.
  }];

  let arguments = (ins
    AnyType:$domain,
    AnyType:$accessor
  );

  let results = (outs
    AnyType:$result
  );
}
