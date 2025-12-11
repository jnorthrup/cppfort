// Minimal TableGen dialect definition for Cpp2 Sea-of-Nodes dialect

include "mlir/IR/Dialect.td"

def ODDialect : Dialect {
  let name = "cpp2";
  let cppNamespace = "::cppfort::mlir_son";
};
