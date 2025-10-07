// ============================================================================
// cpp2_transform.td - CPP2 → Sea of Nodes Transformation Patterns
// ============================================================================
//
// Minimal isomorphic CPP2 grammar as TableGen patterns.
// These 5 core patterns handle 80% of CPP2 → Sea of Nodes transformations.
//
// Based on Simple compiler Ch 19-23 architecture.

// ============================================================================
// Pattern 1: Function Declaration (THE critical isomorphism)
// ============================================================================

def CPP2FunctionToSoN : Pattern<
  // CPP2: name: (params) -> return = body
  (CPP2_FuncDecl $name, $params, $ret, $body),

  // Sea of Nodes:
  // START → PARM nodes → body graph → RETURN → STOP
  [
    (StartNode),
    (ParmNodes $params),
    (BuildDataFlow $body),
    (ReturnNode (DataFlowResult $body)),
    (StopNode)
  ]
> {
  string Description = "CPP2 function to SoN control/data flow graph";
  int Priority = 100;  // Highest priority
}

// ============================================================================
// Pattern 2: Type Annotation Swap (syntax transformation)
// ============================================================================

def CPP2TypeAnnotationToSoN : Pattern<
  // CPP2: name: type
  (CPP2_VarDecl $name, $type),

  // Sea of Nodes: Variable node with type metadata
  (VarNode $name, (TypeFromCPP2 $type))
> {
  string Description = "CPP2 postfix type to SoN type metadata";

  // Type mappings
  list<string> TypeMappings = [
    "int -> TypeInt32",
    "i64 -> TypeInt64",
    "f32 -> TypeFloat32",
    "* -> TypePtr",
    "_ -> TypeInferred"
  ];
}

// ============================================================================
// Pattern 3: Parameter Modes (semantic annotations)
// ============================================================================

class CPP2ParamMode<string mode, string sonAnnotation> : Pattern<
  (CPP2_Param $name, $type, mode),
  (ParmNode $name, (TypeFromCPP2 $type), (ParamKind sonAnnotation))
>;

def CPP2InParam : CPP2ParamMode<"in", "VALUE"> {
  string CXXEmit = "const T&";
  string Description = "CPP2 'in' becomes immutable value in SoN";
}

def CPP2OutParam : CPP2ParamMode<"out", "OUT"> {
  string CXXEmit = "T&";
  string Description = "CPP2 'out' becomes mutable reference in SoN";
}

def CPP2MoveParam : CPP2ParamMode<"move", "MOVE"> {
  string CXXEmit = "T&&";
  string Description = "CPP2 'move' becomes rvalue in SoN";
}

def CPP2ForwardParam : CPP2ParamMode<"forward", "FORWARD"> {
  string CXXEmit = "T&&";
  string Description = "CPP2 'forward' becomes perfect forwarding in SoN";
}

// ============================================================================
// Pattern 4: UFCS (Uniform Function Call Syntax)
// ============================================================================

def CPP2UFCSToSoN : Pattern<
  // CPP2: x.f(y) - member call syntax
  (CPP2_MemberCall $x, $f, $args),

  // Sea of Nodes: CallNode with x as first argument
  (CallNode $f, (Cons $x, $args))
> {
  string Description = "UFCS x.f(y) becomes f(x, y) in SoN";

  // Reverse pattern for emission
  string EmitCXX = "call function with args";
  string EmitCPP2 = "first_arg.function(rest_args)";

  bit IsSyntaxOnly = 1;  // No semantic change
}

// ============================================================================
// Pattern 5: Wildcard Type Inference
// ============================================================================

def CPP2WildcardToSoN : Pattern<
  // CPP2: _ (wildcard type)
  (CPP2_WildcardType),

  // Sea of Nodes: Type inference marker
  (TypeInferred)
> {
  string Description = "CPP2 _ wildcard triggers SoN type inference";

  // Type inference algorithm selection
  string InferenceAlgorithm = "LATTICE_BASED";  // Ch 22 lifetime lattice

  // Reverse emission
  string EmitC = "/* inferred */ auto";
  string EmitCXX = "auto";
  string EmitCPP2 = "_";
}

// ============================================================================
// Core Expression Patterns (isomorphic across C/C++/CPP2)
// ============================================================================

class BinaryOpToSoN<string op, string nodeKind> : Pattern<
  (CPP2_BinOp op, $lhs, $rhs),
  (nodeKind $lhs, $rhs)
>;

def CPP2Add : BinaryOpToSoN<"+", "AddNode">;
def CPP2Sub : BinaryOpToSoN<"-", "SubNode">;
def CPP2Mul : BinaryOpToSoN<"*", "MulNode">;
def CPP2Div : BinaryOpToSoN<"/", "DivNode">;
def CPP2Mod : BinaryOpToSoN<"%", "ModNode">;

def CPP2Eq  : BinaryOpToSoN<"==", "EqNode">;
def CPP2Ne  : BinaryOpToSoN<"!=", "NeNode">;
def CPP2Lt  : BinaryOpToSoN<"<", "LtNode">;
def CPP2Le  : BinaryOpToSoN<"<=", "LeNode">;
def CPP2Gt  : BinaryOpToSoN<">", "GtNode">;
def CPP2Ge  : BinaryOpToSoN<">=", "GeNode">;

// ============================================================================
// Control Flow Patterns (isomorphic)
// ============================================================================

def CPP2IfToSoN : Pattern<
  // CPP2: if (cond) then_branch else else_branch
  (CPP2_If $cond, $then, $else),

  // Sea of Nodes: If → CProj(T/F) → Region merge
  [
    (IfNode $cond),
    (CProjNode "T", (BuildDataFlow $then)),
    (CProjNode "F", (BuildDataFlow $else)),
    (RegionNode)
  ]
> {
  string Description = "CPP2 if/else to SoN control flow graph";
}

def CPP2WhileToSoN : Pattern<
  // CPP2: while (cond) body
  (CPP2_While $cond, $body),

  // Sea of Nodes: Loop → If → backedge
  [
    (LoopNode),
    (PhiNode),
    (IfNode $cond),
    (CProjNode "T", (BuildDataFlow $body)),
    (CProjNode "F", "exit")
  ]
> {
  string Description = "CPP2 while loop to SoN loop graph with backedge";
}

// ============================================================================
// Memory Patterns
// ============================================================================

def CPP2NewToSoN : Pattern<
  // CPP2: new<T>
  (CPP2_New $type),

  // Sea of Nodes: AllocNode
  (AllocNode (TypeFromCPP2 $type))
> {
  string Description = "CPP2 new to SoN heap allocation";
  string EmitC = "malloc(sizeof(T))";
  string EmitCXX = "new T";
  string EmitCPP2 = "new<T>";
}

def CPP2AssignToSoN : Pattern<
  // CPP2: x = value
  (CPP2_Assign $lhs, $rhs),

  // Sea of Nodes: StoreNode
  (StoreNode (AddressOf $lhs), (EvalExpr $rhs))
> {
  string Description = "CPP2 assignment to SoN store operation";
}

// ============================================================================
// Optimization Patterns (peephole in SoN)
// ============================================================================

def ConstantFolding : Pattern<
  (AddNode (ConstantNode $c1), (ConstantNode $c2)),
  (ConstantNode (Add $c1, $c2))
> {
  string Description = "Constant fold addition in SoN";
  int OptLevel = 1;  // O1+
}

def IdentityElimination : Pattern<
  (AddNode $x, (ConstantNode 0)),
  $x
> {
  string Description = "x + 0 = x";
  int OptLevel = 0;  // Always apply
}

// ============================================================================
// Integration Metadata
// ============================================================================

def CPP2TransformPipeline {
  string Description = "CPP2 → Sea of Nodes transformation pipeline";

  list<string> Phases = [
    "Parse CPP2 to AST",
    "Apply transform patterns (this file)",
    "Build Sea of Nodes graph",
    "Run peephole optimizations",
    "Apply GCM scheduling (gcm_patterns.td)",
    "Emit to target (C/C++/CPP2/MLIR)"
  ];

  list<string> Dependencies = [
    "src/stage0/gcm_patterns.td",      // Scheduling
    "src/stage0/node.h",               // Node classes
    "src/stage0/type.h"                // Type system
  ];
}

// ============================================================================
// Verification Patterns
// ============================================================================

def VerifyTransform {
  list<string> Checks = [
    "All CPP2 functions produce START/STOP nodes",
    "Parameter modes map to correct SoN kinds",
    "UFCS transforms maintain argument order",
    "Type inference produces valid types",
    "Control flow has valid dominance"
  ];
}
