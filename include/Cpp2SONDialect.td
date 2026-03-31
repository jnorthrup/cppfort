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
// Canonical Semantic Operations
// Based on expanded_cpp2_spec.md Kotlin-to-cpp2 mapping
//===----------------------------------------------------------------------===//

def IndexedOp : Op<Cpp2SON_Dialect, "indexed"> {
  let summary = "Coordinate-based indexing operation";
  let description = [{
    Represents indexed<I, F> - function from domain I to value type F.
    Kotlin: `n j (i: int) => expr`
  }];

  let arguments = (ins
    AnyType:$domain,
    AnyType:$accessor
  );

  let results = (outs
    AnyType:$result
  );
}

def JoinOp : Op<Cpp2SON_Dialect, "join"> {
  let summary = "Join/product pair operation";
  let description = [{
    Kotlin: `a j b` - join or product pair
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );

  let results = (outs
    AnyType:$result
  );
}

def CoordinatesOp : Op<Cpp2SON_Dialect, "coordinates"> {
  let summary = "Semantic coordinates in chart domain";
  let description = [{
    Kotlin: `coords[1.0, 2.0]` - semantic coordinate literal
  }];

  let arguments = (ins
    AnyType:$values  // list of coordinate values
  );

  let results = (outs
    AnyType:$result  // semantic coordinate type
  );
}

def ChartProjectOp : Op<Cpp2SON_Dialect, "chart_project"> {
  let summary = "Chart projection: point to local coordinates";
  let description = [{
    Kotlin: `chart.project(point)` - project point to local coordinates
  }];

  let arguments = (ins
    AnyType:$chart,
    AnyType:$point
  );

  let results = (outs
    AnyType:$local_coords  // local coordinate type
  );
}

def ChartEmbedOp : Op<Cpp2SON_Dialect, "chart_embed"> {
  let summary = "Chart embedding: local coordinates to point";
  let description = [{
    Kotlin: `chart.embed(coords)` - embed local coords to point
  }];

  let arguments = (ins
    AnyType:$chart,
    AnyType:$local_coords
  );

  let results = (outs
    AnyType:$point
  );
}

def AtlasOp : Op<Cpp2SON_Dialect, "atlas"> {
  let summary = "Atlas: collection of charts";
  let description = [{
    Kotlin: `atlas[shifted, identity]` - collection of charts
  }];

  let arguments = (ins
    AnyType:$charts  // indexed collection of charts
  );

  let results = (outs
    AnyType:$result  // atlas type
  );
}

def ManifoldOp : Op<Cpp2SON_Dialect, "manifold"> {
  let summary = "Smooth coordinate space (manifold)";
  let description = [{
    Kotlin: `manifold line = atlas[shifted, identity]` - smooth coordinate space
  }];

  let arguments = (ins
    AnyType:$atlas  // atlas of charts
  );

  let results = (outs
    AnyType:$result  // manifold type
  );
}

def TransitionOp : Op<Cpp2SON_Dialect, "transition"> {
  let summary = "Chart reprojection through semantic point";
  let description = [{
    Kotlin: `manifold.transition(from, to, coords[17.0])` - reproject between charts
  }];

  let arguments = (ins
    AnyType:$manifold,
    AnyType:$from_chart,
    AnyType:$to_chart,
    AnyType:$coords
  );

  let results = (outs
    AnyType:$result_coords  // reprojected coordinates
  );
}

def LowerDenseOp : Op<Cpp2SON_Dialect, "lower_dense"> {
  let summary = "Explicit dense/materialized view";
  let description = [{
    Kotlin: `coords.lowered()` - explicit dense view of coordinates
  }];

  let arguments = (ins
    AnyType:$coords  // semantic coordinates
  );

  let results = (outs
    AnyType:$result  // dense coordinate type
  );
}
