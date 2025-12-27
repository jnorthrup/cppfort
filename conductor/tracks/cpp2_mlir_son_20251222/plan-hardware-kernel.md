## Plan: FIR to Hardware Kernel Pipeline with Kotlin-Style Concurrency

**Objective**: Transform Cpp2 FIR (Front-IR) through a hardware-biased lowering pipeline that converts loops to GPU kernels, implements Kotlin-style Channel-based concurrency, and targets external memory process pipelines using MLIR hardware dialects. Focus on lifecycle-based super-optimization for SSA state escape avoidance.

---

### Architecture Overview

```
Cpp2 Source
    ↓
Parser (with suspend/async/await)
    ↓
AST (with AsyncFunctionDecl, AwaitExpr, ChannelDecl)
    ↓
FIR Dialect (Cpp2FIR) ← bias toward hardware-friendly patterns
    ↓
Hardware-Kernel Transform ← loop-to-kernel conversion
    ↓
Concurrency Dialect (Cpp2Concurrency) ← Channel-based communication
    ↓
MLIR Hardware Dialects (GPU, Vector, LLVM, SPIRV)
    ↓
C++20 Code with GPU Kernels + Coroutine Runtime
```

---

### Phase 1: Extend AST with Concurrency Primitives

**Files**: `include/ast.hpp`, `src/parser.cpp`

**Add to AST**:
```cpp
// Async function declaration
struct AsyncFunctionDecl : FunctionDeclaration {
    bool is_suspend = false;  // Uses suspend semantics
    std::vector<std::string> channels;  // Channels this function uses
};

// Await expression - suspend until value ready
struct AwaitExpr : Expression {
    std::unique_ptr<Expression> value;
    std::string channel;  // Optional: await from specific channel
};

// Spawn expression - launch async task
struct SpawnExpr : Expression {
    std::unique_ptr<Expression> task;
    std::string channel;  // Result channel
};

// Channel declaration
struct ChannelDecl : Statement {
    std::string name;
    std::unique_ptr<Type> element_type;
    size_t buffer_size = 0;  // 0 = unbuffered (rendezvous)
};

// Channel send/receive
struct ChannelSendExpr : Expression {
    std::string channel;
    std::unique_ptr<Expression> value;
};

struct ChannelRecvExpr : Expression {
    std::string channel;
    bool non_blocking = false;
};

// Kernel annotation - marks function as GPU kernel candidate
struct KernelAnnotation {
    bool is_kernel = false;
    std::string launch_config;  // e.g., "grid(256,256) block(32)"
};
```

**Parser Extensions**:
- `async` keyword → `AsyncFunctionDecl`
- `await <expr>` → `AwaitExpr`
- `spawn <expr>` → `SpawnExpr`
- `channel<T>(<name>[<size>])` → `ChannelDecl`
- `<chan> <- <value>` → `ChannelSendExpr`
- `<chan> <-?` → `ChannelRecvExpr` (non-blocking)
- `@kernel` annotation → `KernelAnnotation`

---

### Phase 2: Hardware-Biased FIR Dialect Extensions

**File**: `include/Cpp2FIRDialect.td`

**Add Operations**:

```tablegen
// Async function definition with kernel annotation
def Cpp2FIR_KernelOp : Cpp2FIR_Op<"kernel", []> {
    let summary = "GPU kernel function definition";
    let arguments = (ins
        StrAttr:$sym_name,
        TypeAttr:$function_type,
        OptionalAttr<StrAttr>:$launch_config,  // grid/block dims
        OptionalAttr<StrAttr>:$memory_policy   // PS2-style: "ee_local", "vif_stream"
    );
    let regions = (region AnyRegion:$body);
}

// Spawn async task
def Cpp2FIR_SpawnOp : Cpp2FIR_Op<"spawn"> {
    let arguments = (ins AnyType:$task, OptionalAttr<StrAttr>:$result_channel);
    let results = (outs AnyType:$future);
}

// Await task completion
def Cpp2FIR_AwaitOp : Cpp2FIR_Op<"await"> {
    let arguments = (ins AnyType:$future);
    let results = (outs AnyType:$result);
}

// Channel operations
def Cpp2FIR_ChannelOp : Cpp2FIR_Op<"channel"> {
    let arguments = (ins Type:$element_type, OptionalAttr<I32Attr>:$buffer_size);
    let results = (outs AnyType:$channel);
}

def Cpp2FIR_SendOp : Cpp2FIR_Op<"send"> {
    let arguments = (ins AnyType:$channel, AnyType:$value);
}

def Cpp2FIR_RecvOp : Cpp2FIR_Op<"recv"> {
    let arguments = (ins AnyType:$channel, UnitAttr:$non_blocking);
    let results = (outs AnyType:$value, AnyType:$success);
}

// Loop annotations for kernel conversion
def Cpp2FIR_ParallelForOp : Cpp2FIR_Op<"parallel_for"> {
    let summary = "Parallel loop for GPU kernel conversion";
    let arguments = (ins
        AnyType:$lower_bound,
        AnyType:$upper_bound,
        AnyType:$step,
        StrAttr:$mapping  // "global_x", "global_y", "local_x", etc.
    );
    let regions = (region AnyRegion:$body);
}

// Memory region annotation for PS2-style pipeline
def Cpp2FIR_MemRegionOp : Cpp2FIR_Op<"mem_region"> {
    let arguments = (ins
        StrAttr:$region_type,  // "ee_main", "vif", "gif", "iop", "spu"
        I64Attr:$size_bytes,
        UnitAttr:$dma_enabled   // Enable DMA transfers
    );
    let results = (outs AnyType:$handle);
}
```

---

### Phase 3: Cpp2Concurrency MLIR Dialect

**File**: `include/Cpp2ConcurrencyOps.td`

```tablegen
def Cpp2Concurrency_Dialect : Dialect {
    let name = "cpp2concurrency";
    let summary = "Kotlin-style concurrency primitives for Cpp2";
    let cppNamespace = "::mlir::cpp2concurrency";
}

// Coroutine scope (structured concurrency)
def CoroutineScopeOp : Cpp2Concurrency_Op<"scope"> {
    let summary = "Structured coroutine scope - all children complete before exit";
    let regions = (region AnyRegion:$body);
    let verifier = [{ return verifyCoroutineScope(op); }];
}

// Spawn task in scope
def SpawnTaskOp : Cpp2Concurrency_Op<"spawn_task"> {
    let arguments = (ins AnyType:$task_func, Variadic<AnyType>:$args);
    let results = (outs AnyType:$task_handle);
}

// Suspend coroutine
def SuspendOp : Cpp2Concurrency_Op<"suspend"> {
    let arguments = (ins AnyType:$value);  // Yield value
}

// Resume point
def ResumeOp : Cpp2Concurrency_Op<"resume"> {
    let arguments = (ins AnyType:$coroutine_handle);
    let results = (outs AnyType:$value);
}

// Channel abstraction (Kotlin-style)
def ChannelOp : Cpp2Concurrency_Op<"channel"> {
    let arguments = (ins Type:$element_type, I32Attr:$capacity);
    let results = (outs AnyType:$channel);
}

def ChannelSendOp : Cpp2Concurrency_Op<"send"> {
    let arguments = (ins AnyType:$channel, AnyType:$value, UnitAttr:$suspend);
    let results = (outs AnyType:$send_result);  // success if buffer not full
}

def ChannelRecvOp : Cpp2Concurrency_Op<"recv"> {
    let arguments = (ins AnyType:$channel, UnitAttr:$suspend);
    let results = (outs AnyType:$value, AnyType:$success);
}

// Select statement (Kotlin-style select)
def SelectOp : Cpp2Concurrency_Op<"select"> {
    let regions = (region AnyRegion:$cases);  // Each case: on_send/on_recv + action
    let results = (outs AnyType:$result);
}
```

---

### Phase 4: Loop-to-Kernel Conversion

**File**: `src/LoopToKernel.cpp`

```cpp
// Detect parallelizable loops and convert to GPU kernels
struct ParallelLoopPattern {
    // Identify patterns:
    // 1. Loop with no cross-iteration dependencies
    // 2. Loop body is pure (or has isolated side effects)
    // 3. Loop bounds known at kernel launch
    // 4. Memory access patterns are strided (coalesced access)

    LogicalResult matchForLoop(ForOp forOp, PatternRewriter &rewriter);
    LogicalResult convertToKernelLaunch(ForOp forOp,
                                        gpu::LaunchFuncOp &kernelOp);
};

// Memory access analysis for coalescing
struct MemoryAccessPattern {
    SmallVector<Value> basePointers;  // Identified base pointers
    SmallVector<int64_t> strides;      // Access strides
    bool isCoalesced();                // Can GPU threads coalesce accesses?
};

// PS2-style DMA pipeline modeling
struct DMAPipelineOp {
    // Model EE ←→ VIF ←→ GIF pipeline
    // Model SPU local memory + DMA transfers

    Value createDMAChannel(OpBuilder &builder, Type elementType);
    void scheduleDMATransfer(Value src, Value dst, int64_t size);
};
```

---

### Phase 5: Hardware Dialect Lowering

**Target Dialects**:
1. **GPU Dialect** (`gpu.func`, `gpu.launch`, `gpu.barrier`)
2. **Vector Dialect** (`vector.transfer_read`, `vector.transfer_write`)
3. **SPIRV Dialect** (for Vulkan/OpenCL targets)
4. **LLVM Dialect** (for NVPTX/AMDGPU backend)

**Lowering Pass Structure**:

```
FIR (Cpp2FIR)
  ↓ [LowerConcurrencyPass]
Concurrency (Cpp2Concurrency)
  ↓ [ConvertAsyncToGPUPass]
GPU + Vector (MLIR)
  ↓ [ConvertVectorToLLVMPass]
LLVM (MLIR)
  ↓ [LLVMIRToNative]
C++20 + GPU Kernels
```

---

### Phase 6: Runtime Library

**File**: `include/cpp2/concurrency.hpp`, `src/cpp2/concurrency.cpp`

```cpp
namespace cpp2::concurrency {

// Channel implementation (Kotlin-style)
template<typename T>
class Channel {
public:
    Channel(size_t capacity = 0);

    // Suspend if buffer full
    void send(T value);

    // Suspend if buffer empty
    T recv();

    // Non-blocking variants
    std::optional<T> try_recv();
    bool try_send(T value);

    // Close channel
    void close();

private:
    std::queue<T> buffer_;
    std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    bool closed_ = false;
};

// Coroutine scope
class CoroutineScope {
public:
    template<typename Func>
    void spawn(Func&& task);

    ~CoroutineScope();  // Joins all spawned tasks

private:
    std::vector<std::future<void>> tasks_;
};

// GPU kernel launcher
namespace gpu {
    template<typename KernelFunc>
    void launch(const GridConfig& grid,
                const BlockConfig& block,
                KernelFunc&& kernel);

    // Memory regions (PS2-style)
    enum class MemRegion { EE_MAIN, VIF, GIF, IOP, SPU };

    void* allocate(MemRegion region, size_t size);
    void dma_transfer(void* src, void* dst, size_t size);
}

} // namespace cpp2::concurrency
```

---

### Phase 7: Tests

**File**: `tests/test_hardware_kernel.cpp`

```cpp
// Test kernel annotation
void test_kernel_annotation();

// Test loop-to-kernel conversion
void test_parallel_for_kernel();

// Test channel operations
void test_channel_send_recv();
void test_channel_select();

// Test coroutine scope
void test_coroutine_scope_join();

// Test PS2-style DMA pipeline
void test_dma_pipeline();

// Test mixed CPU/GPU execution
void test_mixed_execution();
```

---

### Implementation Order

1. AST extensions (parser + AST nodes)
2. FIR dialect extensions (kernel, spawn, await, channel ops)
3. Concurrency dialect definition
4. Runtime library (Channel, CoroutineScope, GPU launcher)
5. AST → FIR lowering with concurrency
6. FIR → Concurrency lowering
7. Loop-to-kernel pattern detection and conversion
8. Hardware dialect lowering (GPU, Vector)
9. C++20 code generation
10. Tests and verification
