//===----------------------------------------------------------------------===//
// Cpp2 Sea-of-Nodes Operations TableGen
// TrikeShed Math-Based SoN Compiler
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "Cpp2SONDialect.td"

// Generic SoN value type (placeholder for future typed ops
def SonValue : AnyType;

// Coordinate-based indexing operation
// Kotlin: `n j (i: int) => expr`
def IndexedOp : Op<Cpp2SON_Dialect, "indexed", [Pure]> {
  let arguments = (ins SonValue:$domain, SonValue:$accessor);
  let results = (outs SonValue:$result);
  let assemblyFormat = [{
    `(` $domain `,` $accessor `)` attr-dict `:` functional-type(operands, results)
  }];
}

// Series operation
def SeriesOp : Op<Cpp2SON_Dialect, "series", [Pure]> {
  let arguments = (ins Variadic<SonValue>);
  let results = (outs SonValue:$result);
}

// Tensor operations
def TensorOp : Op<Cpp2SON_Dialect, "tensor", [Pure]> {
  let arguments = (ins Variadic<SonValue>);
  let results = (outs SonValue:$result);
}

def DenseTensorOp : Op<Cpp2SON_Dialect, "dense_tensor", [Pure]> {
  let arguments = (ins SonValue:$input);
  let results = (outs SonValue:$result);
}

// Atlas: collection of charts
def AtlasOp : Op<Cpp2SON_Dialect, "atlas", [Pure]> {
  let arguments = (ins Variadic<SonValue>);
  let results = (outs SonValue:$result);
}

// Manifold operation
def ManifoldOp : Op<Cpp2SON_Dialect, "manifold", [Pure]> {
  let arguments = (ins Variadic<SonValue>);
  let results = (outs SonValue:$result);
}

// Gradient/differentiation
def GradDiffOp : Op<Cpp2SON_Dialect, "grad_diff", [Pure]> {
  let arguments = (ins SonValue:$expr, SonValue:$wrt);
  let results = (outs SonValue:$grad);
}

// Jacobian computation
def JacobianOp : Op<Cpp2SON_Dialect, "jacobian", [Pure]> {
  let arguments = (ins SonValue:$expr, SonValue:$wrt);
  let results = (outs SonValue:$jacobian);
}

// Matrix multiplication
def MatrixMulOp : Op<Cpp2SON_Dialect, "matrix_mul", [Pure]> {
  let arguments = (ins SonValue:$lhs, SonValue:$rhs);
  let results = (outs SonValue:$result);
}

// Semantic coordinates in chart domain
def CoordinatesOp : Op<Cpp2SON_Dialect, "coordinates", [Pure]> {
  let arguments = (ins Variadic<SonValue>);
  let results = (outs SonValue:$result);
}

// Chart projection: point to local coordinates
def ChartProjectOp : Op<Cpp2SON_Dialect, "chart_project", [Pure]> {
  let arguments = (ins SonValue:$chart, SonValue:$point);
  let results = (outs SonValue:$result);
}

// Chart embedding: local coordinates to point
def ChartEmbedOp : Op<Cpp2SON_Dialect, "chart_embed", [Pure]> {
  let arguments = (ins SonValue:$chart, SonValue:$coords);
  let results = (outs SonValue:$result);
}

// Transition operation
def TransitionOp : Op<Cpp2SON_Dialect, "transition", [Pure]> {
  let arguments = (ins SonValue:$from, SonValue:$to, SonValue:$params);
  let results = (outs SonValue:$result);
}

// Join/product pair operation
// Kotlin: `a j b` - join or product pair
def JoinOp : Op<Cpp2SON_Dialect, "join", [Pure]> {
  let arguments = (ins SonValue:$lhs, SonValue:$rhs);
  let results = (outs SonValue:$result);
}

// Lower to dense/materialized view
def LowerDenseOp : Op<Cpp2SON_Dialect, "lower_dense", [Pure]> {
  let arguments = (ins SonValue:$input);
  let results = (outs SonValue:$result);
}
