include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Cpp2FIR_Dialect : Dialect {
  let name = "cpp2fir";
  let summary = "Cpp2 Front-IR MLIR Dialect";
  let description = [{
    MLIR dialect for Cpp2 language frontend representation.
    Faithfully represents Cpp2 language constructs including types,
    functions, unified function calls, and contracts.
  }];
  let cppNamespace = "::mlir::cpp2fir";
}

class Cpp2FIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cpp2FIR_Dialect, mnemonic, traits>;

// ============================================================================
// Escape Analysis Attributes (Phase 3: External Memory Integration)
// ============================================================================

// Escape kind enum for escape analysis annotations
def Cpp2FIR_EscapeKindAttr : StrEnumAttr<"EscapeKind", 
    "Escape analysis result for SSA values", [
  StrEnumAttrCase<"no_escape">,   // Value stays local (stack) - transfer can be eliminated
  StrEnumAttrCase<"heap">,        // Stored in heap-allocated object
  StrEnumAttrCase<"return">,      // Returned from function
  StrEnumAttrCase<"param">,       // Stored via pointer/reference parameter  
  StrEnumAttrCase<"global">,      // Stored in global variable
  StrEnumAttrCase<"channel">,     // Sent through channel
  StrEnumAttrCase<"gpu">,         // Transferred to GPU memory - transfer required
  StrEnumAttrCase<"dma">          // Transferred via DMA buffer - transfer required
]> {
  let cppNamespace = "::mlir::cpp2fir";
}

// Attribute for annotating operations with escape information
// Usage: cpp2fir.add %a, %b {escape = "no_escape"} : i32 -> i32
// The transfer elimination pass uses this to determine if transfers are needed

// Basic Operations

def Cpp2FIR_FuncOp : Cpp2FIR_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "Cpp2 function definition";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<ArrayAttr>:$arg_attrs,
                       OptionalAttr<ArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);
  
  let assemblyFormat = [{
    $sym_name `:` $function_type attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    // FunctionOpInterface methods
    ::mlir::Region *getCallableRegion() { return &getBody(); }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
    void setFunctionType(::mlir::Type type) { setFunctionTypeAttr(::mlir::TypeAttr::get(type)); }
  }];
}

def Cpp2FIR_ConstantOp : Cpp2FIR_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Constant value";
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

def Cpp2FIR_ReturnOp : Cpp2FIR_Op<"return", [Terminator, ReturnLike]> {
  let summary = "Return operation";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// Sea-of-Nodes-oriented expression ops (Chapter 24 compass)

def Cpp2FIR_AddOp : Cpp2FIR_Op<"add", [Pure]> {
  let summary = "Addition";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_SubOp : Cpp2FIR_Op<"sub", [Pure]> {
  let summary = "Subtraction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_MulOp : Cpp2FIR_Op<"mul", [Pure]> {
  let summary = "Multiplication";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_DivOp : Cpp2FIR_Op<"div", [Pure]> {
  let summary = "Division";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_AndOp : Cpp2FIR_Op<"and", [Pure]> {
  let summary = "Boolean AND";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_OrOp : Cpp2FIR_Op<"or", [Pure]> {
  let summary = "Boolean OR";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_NotOp : Cpp2FIR_Op<"not", [Pure]> {
  let summary = "Boolean NOT";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Cpp2FIR_CmpOp : Cpp2FIR_Op<"cmp", [Pure]> {
  let summary = "Comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Cpp2FIR_PhiOp : Cpp2FIR_Op<"phi", [Pure]> {
  let summary = "Phi";
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$values attr-dict `:` type($values) `->` type($result)";
}

// Contract Operations (Cpp2 safety contracts)

def Cpp2FIR_AssertOp : Cpp2FIR_Op<"assert"> {
  let summary = "Runtime assertion contract";
  let description = [{
    Asserts that a condition is true at runtime. Can include optional message,
    contract categories (type_safety, bounds_safety, etc.), and audit flag.
  }];
  let arguments = (ins AnyType:$condition,
                       OptionalAttr<StrAttr>:$message,
                       OptionalAttr<ArrayAttr>:$categories,
                       UnitAttr:$audit);
  let assemblyFormat = "$condition attr-dict `:` type($condition) (`:` $message^)?";
}

def Cpp2FIR_PreconditionOp : Cpp2FIR_Op<"precondition"> {
  let summary = "Function precondition contract";
  let description = [{
    Specifies a precondition that must be true when a function is called.
    Checked at function entry (or may be checked statically).
  }];
  let arguments = (ins AnyType:$condition,
                       OptionalAttr<StrAttr>:$message,
                       OptionalAttr<ArrayAttr>:$categories,
                       UnitAttr:$audit);
  let assemblyFormat = "$condition attr-dict `:` type($condition) (`:` $message^)?";
}

def Cpp2FIR_PostconditionOp : Cpp2FIR_Op<"postcondition"> {
  let summary = "Function postcondition contract";
  let description = [{
    Specifies a postcondition that must be true when a function returns.
    May reference the return value via special names.
  }];
  let arguments = (ins AnyType:$condition,
                       OptionalAttr<StrAttr>:$message,
                       OptionalAttr<ArrayAttr>:$categories,
                       UnitAttr:$audit);
  let assemblyFormat = "$condition attr-dict `:` type($condition) (`:` $message^)?";
}

// UFCS (Unified Function Call Syntax) Operation

def Cpp2FIR_UfcsCallOp : Cpp2FIR_Op<"ufcs_call"> {
  let summary = "Unified Function Call Syntax operation";
  let description = [{
    Represents a UFCS call where method(obj, args...) is called as
    if it were obj.method(args...). The first argument is the object
    being operated on, followed by the method arguments.
  }];
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$args attr-dict `:` type($args) `->` type($result)";
}

// ============================================================================
// Concurrency Operations (Kotlin-style + GPU Kernel)
// ============================================================================

// GPU kernel function definition
def Cpp2FIR_KernelOp : Cpp2FIR_Op<"kernel", []> {
  let summary = "GPU kernel function definition with launch config";
  let description = [{
    Marks a function as a GPU kernel candidate with launch configuration
    and memory policy annotations. The body contains the kernel code that
    will be executed on the GPU.

    Memory policy options for external memory pipeline optimization:
      - "coherent": Cache-coherent access (default)
      - "streaming": Sequential access pattern (DMA-friendly)
      - "private": Per-thread private memory (register file)
  }];
  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$function_type,
    OptionalAttr<StrAttr>:$launch_config,   // e.g., "grid(256,256) block(32)"
    OptionalAttr<StrAttr>:$memory_policy     // Memory access pattern optimization
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name $function_type (`launch` $launch_config^)? (`memory` $memory_policy^)? $body attr-dict";
}

// Spawn async task
def Cpp2FIR_SpawnOp : Cpp2FIR_Op<"spawn"> {
  let summary = "Launch an async task in a coroutine scope";
  let description = [{
    Spawn an asynchronous task that executes concurrently. The task
    can be a function call or a lambda. Returns a handle/future for awaiting.
  }];
  let arguments = (ins AnyType:$task, OptionalAttr<StrAttr>:$result_channel);
  let results = (outs AnyType:$future);
  let assemblyFormat = "$task attr-dict `:` type($task) `->` type($future) (`->` $result_channel^)?";
}

// Await task completion
def Cpp2FIR_AwaitOp : Cpp2FIR_Op<"await"> {
  let summary = "Suspend until async task completes";
  let description = [{
    Suspend the current coroutine until the awaited future/channel
    has a value ready. This is the Kotlin-style await operation.
  }];
  let arguments = (ins AnyType:$future, OptionalAttr<StrAttr>:$source_channel);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$future attr-dict `:` type($future) `->` type($result) (`from` $source_channel^)?";
}

// Channel declaration
def Cpp2FIR_ChannelOp : Cpp2FIR_Op<"channel"> {
  let summary = "Channel declaration for coroutine communication";
  let description = [{
    Creates a channel for communicating between coroutines. Channels can
    be buffered (buffer_size > 0) or unbuffered rendezvous channels (buffer_size = 0).
  }];
  let arguments = (ins TypeAttr:$element_type, OptionalAttr<I32Attr>:$buffer_size);
  let results = (outs AnyType:$channel);
  let assemblyFormat = "$element_type attr-dict (`buffer` $buffer_size^)? `->` type($channel)";
}

// Channel send operation
def Cpp2FIR_SendOp : Cpp2FIR_Op<"send"> {
  let summary = "Send value to channel (suspends if buffer full)";
  let description = [{
    Send a value to a channel. If the channel buffer is full (or unbuffered
    and no receiver is waiting), the sender suspends until space is available.
  }];
  let arguments = (ins AnyType:$channel, AnyType:$value);
  let assemblyFormat = "$channel `,` $value attr-dict `:` type($channel) `,` type($value)";
}

// Channel receive operation
def Cpp2FIR_RecvOp : Cpp2FIR_Op<"recv"> {
  let summary = "Receive value from channel (suspends if buffer empty)";
  let description = [{
    Receive a value from a channel. If the channel is empty (or unbuffered
    and no sender is waiting), the receiver suspends until a value arrives.
  }];
  let arguments = (ins AnyType:$channel, UnitAttr:$non_blocking);
  let results = (outs AnyType:$value, AnyType:$success);
  let assemblyFormat = "$channel `:` type($channel) attr-dict (`nonblocking` $non_blocking^)? `->` type($value) `,` type($success)";
}

// Parallel for loop (for GPU kernel conversion)
def Cpp2FIR_ParallelForOp : Cpp2FIR_Op<"parallel_for"> {
  let summary = "Parallel loop for GPU kernel conversion";
  let description = [{
    A parallel loop that can be converted to a GPU kernel launch. The mapping
    attribute specifies which GPU dimension to map to (global_x, global_y, etc.).
  }];
  let arguments = (ins
    Index:$lower_bound,
    Index:$upper_bound,
    Index:$step,
    StrAttr:$mapping  // "global_x", "global_y", "global_z", "local_x", etc.
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$lower_bound `to` $upper_bound `step` $step `mapping` $mapping $body attr-dict";
}

// Memory region annotation for external memory pipeline
def Cpp2FIR_MemRegionOp : Cpp2FIR_Op<"mem_region"> {
  let summary = "Memory region annotation for external memory pipeline";
  let description = [{
    Annotates a memory region with specific characteristics for external
    memory pipelines. Different regions have different access patterns,
    latency characteristics, and DMA capabilities.

    Region types (external memory pipeline):
      - "cpu_main": Standard CPU-accessible memory
      - "gpu_global": GPU global memory (high latency)
      - "gpu_shared": GPU shared memory (per-block, low latency)
      - "gpu_constant": GPU constant memory (read-only)
      - "dma_buffer": DMA-capable buffer for async transfers
      - "local_compute": Compute-unit local memory (fast, private)

    Supports DMA transfers between regions with lifecycle-based
    super-optimization for SSA state escape avoidance.
  }];
  let arguments = (ins
    StrAttr:$region_type,
    I64Attr:$size_bytes,
    UnitAttr:$dma_enabled   // Enable DMA transfers
  );
  let results = (outs AnyType:$handle);
  let assemblyFormat = "$region_type `size` $size_bytes attr-dict (`dma` $dma_enabled^)? `->` type($handle)";
}

// Coroutine scope (structured concurrency)
def Cpp2FIR_CoroutineScopeOp : Cpp2FIR_Op<"coroutine_scope"> {
  let summary = "Structured coroutine scope - all children complete before exit";
  let description = [{
    A coroutine scope ensures all spawned tasks complete before the scope exits.
    This provides structured concurrency guarantees and prevents task leaks.
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$body attr-dict";
}
