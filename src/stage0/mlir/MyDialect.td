#ifndef MY_DIALECT_TD
#define MY_DIALECT_TD

include "mlir/IR/OpBase.td"

def MyDialect : Dialect {
  let name = "my";
  let cppNamespace = "::my";
}

class My_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyDialect, mnemonic, traits>;

def MyAddOp : My_Op<"add"> {
  let summary = "My custom add operation";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$res);
}

#endif // MY_DIALECT_TD
