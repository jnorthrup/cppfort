// Cpp2ConcurrencyOps.td
// Kotlin-style Concurrency Dialect for Cpp2
// Target: C++20 coroutines + C++11/14 stdlib (no experimental features)

#ifndef CPP2_CONCURRENCY_OPS
#define CPP2_CONCURRENCY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Cpp2Concurrency_Dialect : Dialect {
  let name = "cpp2concurrency";
  let summary = "Kotlin-style concurrency primitives for Cpp2";
  let description = [{
    Dialect for implementing Kotlin-style structured concurrency in Cpp2.
    Targets C++20 coroutines (co_await, co_return, std::coroutine_handle)
    with channel-based communication and coroutine scopes.

    Uses only published C++ standards:
    - C++20 coroutines (co_await, co_yield, co_return)
    - std::coroutine_handle, std::suspend_always
    - std::future, std::promise (C++11)
    - std::mutex, std::condition_variable (C++11)
    - std::thread, std::jthread (C++11/20)
  }];
  let cppNamespace = "::mlir::cpp2concurrency";
}

class Cpp2Concurrency_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cpp2Concurrency_Dialect, mnemonic, traits>;

// ============================================================================
// Coroutine Scope (Structured Concurrency)
// ============================================================================

def CoroutineScopeOp : Cpp2Concurrency_Op<"scope", []> {
  let summary = "Structured coroutine scope - all children complete before exit";
  let description = [{
    A coroutine scope ensures all spawned tasks complete before the scope exits.
    This provides structured concurrency guarantees similar to Kotlin's
    coroutineScope.

    Implementation: Uses RAII pattern that joins all std::futures in destructor.
    Equivalent to:
      {
        std::vector<std::future<void>> tasks;
        // ... spawn tasks ...
        // destructor: for (auto& t : tasks) t.wait();
      }
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = $regions attr-dict;
  let verifier = [{ return $0.getName() == "cpp2concurrency.scope" ? success() : failure(); }];
}

// ============================================================================
// Task Spawning and Awaiting
// ============================================================================

def SpawnTaskOp : Cpp2Concurrency_Op<"spawn", []> {
  let summary = "Spawn an async task within a coroutine scope";
  let description = [{
    Spawn an asynchronous task that executes concurrently. The task is
    represented as a function or lambda. Returns a std::future for awaiting.

    Implementation: std::async(std::launch::async, task_func, args...)
  }];
  let arguments = (ins AnyType:$task_func, Variadic<AnyType>:$args);
  let results = (outs AnyType:$task_handle);  // std::future<T>
  let assemblyFormat = $task_func `(` $args `)` attr-dict `:` functional-type(inputs, results);
}

def SuspendOp : Cpp2Concurrency_Op<"suspend"> {
  let summary = "Suspend coroutine and yield value (co_yield)";
  let description = [{
    Suspend the current coroutine, yielding a value to the awaiting coroutine.
    This maps to C++20's co_yield.

    Implementation: co_yield value in generator coroutine
  }];
  let arguments = (ins AnyType:$value);
  let assemblyFormat = $value attr-dict `:` type($value);
}

def ResumeOp : Cpp2Concurrency_Op<"resume"> {
  let summary = "Resume a suspended coroutine";
  let description = [{
    Resume a coroutine that was previously suspended. This maps to
    calling coroutine_handle::resume() in C++20.
  }];
  let arguments = (ins AnyType:$coroutine_handle);
  let results = (outs AnyType:$result);
  let assemblyFormat = $coroutine_handle attr-dict `->` type($result);
}

// ============================================================================
// Channel Abstraction (Kotlin-style)
// ============================================================================

def ChannelOp : Cpp2Concurrency_Op<"channel"> {
  let summary = "Channel declaration for coroutine communication";
  let description = [{
    Creates a channel for communicating between coroutines. Channels can
    be buffered (capacity > 0) or unbuffered rendezvous channels (capacity = 0).

    Implementation: Thread-safe queue with std::mutex and std::condition_variable.
    - Buffered channel: std::queue with bounded capacity
    - Unbuffered: Rendezvous point (both sender and receiver must be ready)

    Kotlin Channel<T> equivalent:
      - Channel() - rendezvous channel
      - Channel(capacity) - buffered channel
  }];
  let arguments = (ins Type:$element_type, I32Attr:$capacity);
  let results = (outs AnyType:$channel);
  let assemblyFormat = $element_type (`capacity` $capacity^)? attr-dict `->` type($channel);
}

def ChannelSendOp : Cpp2Concurrency_Op<"send"> {
  let summary = "Send value to channel (suspend if buffer full)";
  let description = [{
    Send a value to a channel. If the channel buffer is full (or unbuffered
    and no receiver is waiting), the operation can optionally suspend.

    Kotlin equivalent: channel.send(value)
    - Suspend mode: co_await channel.send(value)
    - Non-blocking mode: channel.try_send(value).is_success()

    Implementation:
      - Buffered: push to queue, wait if full (suspend=true)
      - Unbuffered: wait for matching receiver (suspend=true)
  }];
  let arguments = (ins
    AnyType:$channel,
    AnyType:$value,
    UnitAttr:$suspend  // If true, suspend when channel full
  );
  let results = (outs AnyType:$success);  // true if sent immediately
  let assemblyFormat = $channel `,` $value attr-dict (`suspend` $suspend^)? `->` type($success);
}

def ChannelRecvOp : Cpp2Concurrency_Op<"recv"> {
  let summary = "Receive value from channel (suspend if buffer empty)";
  let description = [{
    Receive a value from a channel. If the channel is empty (or unbuffered
    and no sender is waiting), the operation can optionally suspend.

    Kotlin equivalent: val x = channel.receive()
    - Suspend mode: val x = co_await channel.receive()
    - Non-blocking mode: val x = channel.receiveOrNull()

    Implementation:
      - Buffered: pop from queue, wait if empty (suspend=true)
      - Unbuffered: wait for matching sender (suspend=true)
  }];
  let arguments = (ins
    AnyType:$channel,
    UnitAttr:$suspend,  // If true, suspend when channel empty
    UnitAttr:$nullable  // If true, returns std::optional<T> instead of T
  );
  let results = (outs AnyType:$value, AnyType:$success);
  let assemblyFormat = $channel attr-dict (`suspend` $suspend^)? (`nullable` $nullable^)? `->` type($value) `,` type($success);
}

def ChannelCloseOp : Cpp2Concurrency_Op<"close"> {
  let summary = "Close a channel";
  let description = [{
    Close a channel, preventing further sends. Receivers will get
    end-of-stream indication (optional becomes nullopt).

    Kotlin equivalent: channel.close()
  }];
  let arguments = (ins AnyType:$channel);
  let assemblyFormat = $channel attr-dict `:` type($channel);
}

// ============================================================================
// Select Statement (Kotlin-style)
// ============================================================================

def SelectOp : Cpp2Concurrency_Op<"select"> {
  let summary = "Select statement - wait for multiple channel operations";
  let description = [{
    Wait for multiple channel operations simultaneously. Execute the first
    ready case. This is similar to Go's select and Kotlin's select.

    Kotlin equivalent:
      select {
        onSend(ch1) { v -> send v }
        onRecv(ch2) { v -> process(v) }
        default -> do_other_work()
      }

    Implementation: Uses condition_variable::wait_for() with timeouts
    to poll multiple channels.
  }];
  let arguments = (ins Variadic<AnyType>:$channels);  // Channels to select on
  let regions = (region AnyRegion:$cases);  // Each case: on_send/on_recv + action
  let results = (outs AnyType:$result);
  let assemblyFormat = $channels $regions attr-dict `->` type($result);
}

// ============================================================================
// Parallel For (Loop-to-Kernel Conversion)
// ============================================================================

def ParallelForOp : Cpp2Concurrency_Op<"parallel_for", []> {
  let summary = "Parallel for loop - convertible to GPU kernel";
  let description = [{
    A parallel for loop that can be converted to a GPU kernel launch.
    The mapping attribute specifies which GPU/thread dimension to use.

    GPU mappings: "global_x", "global_y", "global_z", "local_x", "local_y", "local_z"
    CPU thread mappings: "thread_id" (std::thread::hardware_concurrency())

    Implementation:
      - GPU: gpu.launch_func with appropriate thread/block config
      - CPU: std::thread or thread pool
  }];
  let arguments = (ins
    AnyType:$lower_bound,
    AnyType:$upper_bound,
    AnyType:$step,
    StrAttr:$mapping,     // "global_x", "thread_id", etc.
    OptionalAttr<I64Attr>$tile_size  // For GPU tiling
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = $lower_bound `..` $upper_bound `step` $step `mapping` $mapping (`tile` $tile_size^)? $regions attr-dict;
}

// ============================================================================
// Memory Pipeline (PS2-style External Memory Architecture)
// ============================================================================

def DMATransferOp : Cpp2Concurrency_Op<"dma_transfer"> {
  let summary = "DMA transfer between memory regions (PS2-style pipeline)";
  let description = [{
    Perform asynchronous DMA transfer between memory regions. Models
    PS2-style external memory architecture with dedicated transfer paths.

    PS2 Architecture regions:
      - EE_MAIN: Emotion Engine main memory (32MB)
      - VIF: Video Interface (for geometry data)
      - GIF: Graphics Interface (for texture data)
      - IOP: I/O Processor (2MB)
      - SPU: Sound Processing Unit (2MB x 2)

    Implementation: std::async with memcpy, or GPU cudaMemcpyAsync
  }];
  let arguments = (ins
    AnyType:$src_handle,
    AnyType:$dst_handle,
    I64Attr:$size_bytes,
    StrAttr:$transfer_kind  // "memcpy", "gpu_memcpy_async", "vif_packet"
  );
  let results = (outs AnyType:$completion_handle);
  let assemblyFormat = $src_handle `->` $dst_handle `size` $size_bytes `kind` $transfer_kind attr-dict `->` type($completion_handle);
}

def MemRegionOp : Cpp2Concurrency_Op<"mem_region"> {
  let summary = "Allocate specialized memory region";
  let description = [{
    Allocate a memory region with specific characteristics. Different regions
    can have different access patterns and DMA capabilities.

    Region types:
      - "cpu_main": Standard CPU memory (malloc/new)
      - "gpu_global": GPU global memory
      - "gpu_shared": GPU shared memory (per-block)
      - "gpu_constant": GPU constant memory
      - "dma_buffer": DMA-capable buffer (for transfers)
  }];
  let arguments = (ins
    StrAttr:$region_type,
    I64Attr:$size_bytes,
    UnitAttr:$dma_capable
  );
  let results = (outs AnyType:$handle);
  let assemblyFormat = $region_type `size` $size_bytes attr-dict (`dma` $dma_capable^)? `->` type($handle);
}

// ============================================================================
// Synchronization Primitives
// ============================================================================

def MutexLockOp : Cpp2Concurrency_Op<"mutex_lock"> {
  let summary = "Lock a mutex";
  let description = [{
    Acquire a mutex lock. For C++ implementation, uses std::lock_guard
    or std::unique_lock RAII patterns.

    Kotlin equivalent: synchronized(lock) { ... }
  }];
  let arguments = (ins AnyType:$mutex_handle);
  let regions = (region AnyRegion:$critical_section);
  let assemblyFormat = $mutex_handle $regions attr-dict `:` type($mutex_handle);
}

def AtomicOp : Cpp2Concurrency_Op<"atomic"> {
  let summary = "Atomic operation on memory location";
  let description = [{
    Perform an atomic operation. Maps to std::atomic<T> operations.

    Operations: "load", "store", "add", "sub", "exchange", "compare_exchange"
  }];
  let arguments = (ins
    AnyType:$ptr,
    AnyType:$value,
    StrAttr:$operation,  // "load", "store", "add", "sub", "exchange", "cmpxchg"
    OptionalAttr<AnyType>:$expected  // For cmpxchg
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = $ptr `,` $value `op` $operation attr-dict `->` type($result);
}

#endif // CPP2_CONCURRENCY_OPS
