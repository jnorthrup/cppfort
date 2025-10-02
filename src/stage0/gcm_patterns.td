// ============================================================================
// Band 3: Chapter 11 - Global Code Motion Patterns
// ============================================================================
//
// TableGen specification for Sea of Nodes scheduling patterns.
// This defines the transformation from unordered graph to scheduled code.
//
// THE critical convergence point where graph becomes executable.

// Base scheduling constraint classes
class ScheduleConstraint<string name> {
  string Name = name;
  bit IsPinned = 0;
  bit IsEarly = 0;
  bit IsLate = 0;
}

// Control flow nodes are immovable anchors
class CFGConstraint<string name> : ScheduleConstraint<name> {
  bit IsPinned = 1;
  bit IsImmovable = 1;
}

// Data nodes can float between early and late positions
class DataConstraint<string name> : ScheduleConstraint<name> {
  bit CanFloat = 1;
  bit PreferLate = 1;  // Minimize live ranges
}

// Memory operations need special handling
class MemConstraint<string name> : ScheduleConstraint<name> {
  bit NeedsAntiDep = 1;
  bit PreservesOrder = 1;
}

// ============================================================================
// Basic Block Identification Patterns
// ============================================================================

def BBStart : CFGConstraint<"BBStart"> {
  list<string> NodeTypes = ["Start", "CProj", "Region", "Loop", "Stop"];
}

def BBEnd : CFGConstraint<"BBEnd"> {
  list<string> NodeTypes = ["If", "Return", "Stop"];
}

// ============================================================================
// Dominator Patterns
// ============================================================================

class DominatorPattern<string name> {
  string Name = name;
}

def ImmediateDom : DominatorPattern<"ImmediateDom"> {
  // For most nodes, idom is control input
  string DefaultRule = "in(0)";

  // Special cases
  list<string> SpecialCases = [
    "Region: LCA of all inputs",
    "Loop: entry control",
    "Start: null",
    "Stop: null"
  ];
}

def DominatorDepth : DominatorPattern<"DominatorDepth"> {
  string CacheField = "_idepth";
  string ComputeRule = "idom().idepth() + 1";
}

def LoopNestDepth : DominatorPattern<"LoopNestDepth"> {
  string CacheField = "_loopDepth";
  string ComputeRule = "cfg(0).loopDepth()";

  // Loop nodes increment depth
  string LoopRule = "entry().loopDepth() + 1";
}

// ============================================================================
// Early Schedule Patterns
// ============================================================================

class EarlyScheduleRule<string name> {
  string Name = name;
  string Description = "";
}

def ScheduleAtInputs : EarlyScheduleRule<"ScheduleAtInputs"> {
  string Description = "Schedule at deepest (latest) input";
  string Rule = "max(input.idepth() for input in inputs)";
}

def PinnedNodes : EarlyScheduleRule<"PinnedNodes"> {
  string Description = "Pinned nodes stay at original control";
  list<string> PinnedTypes = [
    "CFGNode",     // All control nodes
    "PhiNode",     // Pinned to regions
    "NewNode",     // Has control input
    "StoreNode"    // Has side effects
  ];
}

def FloatingNodes : EarlyScheduleRule<"FloatingNodes"> {
  string Description = "Floating nodes can move";
  list<string> FloatingTypes = [
    "AddNode",
    "SubNode",
    "MulNode",
    "DivNode",
    "LoadNode",    // Can float but needs anti-deps
    "ConstantNode"
  ];
}

// ============================================================================
// Late Schedule Patterns
// ============================================================================

class LateScheduleRule<string name> {
  string Name = name;
  string Description = "";
}

def ScheduleAtUses : LateScheduleRule<"ScheduleAtUses"> {
  string Description = "Schedule at LCA of uses";
  string Rule = "LCA(use.block for use in outputs)";
}

def PreferShallowLoop : LateScheduleRule<"PreferShallowLoop"> {
  string Description = "Prefer placement in shallower loop";
  int Priority = 1;
  string Rule = "min(block.loopDepth)";
}

def PreferDeepControl : LateScheduleRule<"PreferDeepControl"> {
  string Description = "Prefer more control-dependent placement";
  int Priority = 2;
  string Rule = "max(block.idepth) at same loop depth";
}

def AvoidIfNodes : LateScheduleRule<"AvoidIfNodes"> {
  string Description = "Don't schedule in If nodes";
  string Rule = "if block is If then block.idom()";
}

// ============================================================================
// Anti-Dependency Patterns
// ============================================================================

class AntiDepPattern<string name> {
  string Name = name;
  bit IsMemoryOrdering = 1;
}

def LoadStoreOrder : AntiDepPattern<"LoadStoreOrder"> {
  string Description = "Loads must execute before conflicting stores";
  string Condition = "load.alias == store.alias";
  string Action = "add edge from load to store";
}

def StoreStoreOrder : AntiDepPattern<"StoreStoreOrder"> {
  string Description = "Stores to same location must preserve order";
  string Condition = "store1.alias == store2.alias";
  string Action = "preserve program order";
}

def MemPhiOrder : AntiDepPattern<"MemPhiOrder"> {
  string Description = "Memory Phi nodes serialize memory operations";
  string Action = "raise LCA but no edge";
}

// ============================================================================
// Infinite Loop Handling
// ============================================================================

def InfiniteLoopPattern {
  string Description = "Force exit from infinite loops";
  string Detection = "Loop with no CProj in dominator chain";
  string Fix = "Insert NeverNode with dummy edge to Stop";
}

// ============================================================================
// Scheduling Priorities (for code generation)
// ============================================================================

class SchedulePriority<string name, int pri> {
  string NodeType = name;
  int Priority = pri;  // Higher = schedule earlier within block
}

def MemoryPriorities {
  list<SchedulePriority> Priorities = [
    SchedulePriority<"PhiNode", 100>,      // Phis first
    SchedulePriority<"LoadNode", 80>,      // Loads early
    SchedulePriority<"ArithNode", 50>,     // Arithmetic middle
    SchedulePriority<"StoreNode", 20>,     // Stores late
    SchedulePriority<"ReturnNode", 0>      // Returns last
  ];
}

// ============================================================================
// Cross-Language Normalization
// ============================================================================

def CScheduleMapping {
  string Description = "Map scheduled graph to C code";
  list<string> Rules = [
    "Region -> label:",
    "If -> if statement",
    "Loop -> while/for loop",
    "Load -> variable access",
    "Store -> assignment",
    "Arithmetic -> infix expression"
  ];
}

def CPPScheduleMapping {
  string Description = "Map scheduled graph to C++ code";
  list<string> Rules = [
    "Region -> scope block",
    "NewNode -> new expression",
    "LoadNode -> member access",
    "StoreNode -> member assignment",
    "CastNode -> static_cast"
  ];
}

def MLIRScheduleMapping {
  string Description = "Map scheduled graph to MLIR ops";
  list<string> Rules = [
    "Region -> cf.br or scf.if",
    "Loop -> scf.while",
    "Load -> memref.load",
    "Store -> memref.store",
    "PhiNode -> block arguments"
  ];
}

// ============================================================================
// Verification Patterns
// ============================================================================

class VerifyPattern<string name> {
  string Name = name;
  string Check = "";
}

def VerifyDominance : VerifyPattern<"VerifyDominance"> {
  string Check = "All uses dominated by defs";
}

def VerifyAntiDeps : VerifyPattern<"VerifyAntiDeps"> {
  string Check = "Memory operations properly ordered";
}

def VerifySchedule : VerifyPattern<"VerifySchedule"> {
  string Check = "All nodes assigned to blocks";
}

def VerifyLoops : VerifyPattern<"VerifyLoops"> {
  string Check = "All loops have exits (no infinite loops without Never)";
}

// ============================================================================
// Meta-Patterns for Stage0 Integration
// ============================================================================

def GCMIntegration {
  string Description = "Integration points for stage0 meta-transpiler";

  list<string> Phases = [
    "Parse -> Unordered SoN graph",
    "Peephole -> Optimized graph",
    "GCM Early -> Nodes scheduled to earliest blocks",
    "GCM Late -> Nodes scheduled to optimal blocks",
    "Anti-Deps -> Memory ordering enforced",
    "Emit -> Target language code"
  ];

  list<string> Invariants = [
    "CFG nodes never move",
    "Data flows respect dependencies",
    "Memory operations preserve semantics",
    "Loop-invariant code hoisted",
    "Dead code eliminated"
  ];
}