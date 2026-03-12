//===----------------------------------------------------------------------===//
// Cpp2 Sea-of-Nodes Operations TableGen
// TrikeShed Math-Based SoN Compiler
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "Cpp2SONDialect.td"

// Import operations from dialect file
def IndexedOp : Op<Cpp2SON_Dialect, "indexed", [Pure]>;
def SeriesOp : Op<Cpp2SON_Dialect, "series", [Pure]>;
def TensorOp : Op<Cpp2SON_Dialect, "tensor", [Pure]>;
def DenseTensorOp : Op<Cpp2SON_Dialect, "dense_tensor", [Pure]>;
def AtlasOp : Op<Cpp2SON_Dialect, "atlas", [Pure]>;
def ManifoldOp : Op<Cpp2SON_Dialect, "manifold", [Pure]>;
def GradDiffOp : Op<Cpp2SON_Dialect, "grad_diff", [Pure]>;
def JacobianOp : Op<Cpp2SON_Dialect, "jacobian", [Pure]>;
def MatrixMulOp : Op<Cpp2SON_Dialect, "matrix_mul", [Pure]>;
def CoordinatesOp : Op<Cpp2SON_Dialect, "coordinates", [Pure]>;
def ChartProjectOp : Op<Cpp2SON_Dialect, "chart_project", [Pure]>;
def ChartEmbedOp : Op<Cpp2SON_Dialect, "chart_embed", [Pure]>;
def TransitionOp : Op<Cpp2SON_Dialect, "transition", [Pure]>;
