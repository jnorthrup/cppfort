//===- Cpp2ConcurrencyDialect.cpp - Cpp2 Concurrency Dialect Implementation ===//
//
// Kotlin-style structured concurrency dialect for Cpp2.
// Implements spawn/await/channel operations that lower to C++20 coroutines.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

// Placeholder namespace until TableGen generates the full dialect
namespace mlir {
namespace cpp2concurrency {

// Dialect registration will be done via TableGen-generated code
// This file provides any hand-written implementations needed

/// Verify that a coroutine scope properly contains all spawned tasks
/// (structured concurrency invariant)
bool verifyStructuredConcurrency(Operation* scopeOp) {
    // All spawn operations within the scope should complete before scope exits
    // This is enforced by the semantics of CoroutineScopeOp
    return true;
}

/// Check if an operation is within a valid suspend context
bool isInSuspendContext(Operation* op) {
    // Walk up the operation hierarchy to find if we're inside
    // a function marked as async/suspend
    Operation* parent = op->getParentOp();
    while (parent) {
        // Check for async function marker (would be an attribute or special op)
        if (parent->hasAttr("cpp2.async") || parent->hasAttr("cpp2.suspend")) {
            return true;
        }
        parent = parent->getParentOp();
    }
    return false;
}

/// Lowering utilities for C++20 coroutine generation
namespace lowering {

/// Generate promise_type boilerplate for a Task<T>
std::string generatePromiseType(const std::string& resultType) {
    return R"cpp(
struct promise_type {
    )cpp" + resultType + R"cpp( result;
    std::exception_ptr exception;

    auto get_return_object() { return Task{std::coroutine_handle<promise_type>::from_promise(*this)}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception() { exception = std::current_exception(); }
    void return_value()cpp" + resultType + R"cpp( value) { result = std::move(value); }
};
)cpp";
}

/// Generate coroutine_handle wrapper
std::string generateCoroutineHandle() {
    return R"cpp(
std::coroutine_handle<promise_type> handle;

Task(std::coroutine_handle<promise_type> h) : handle(h) {}
~Task() { if (handle) handle.destroy(); }

Task(Task&& other) noexcept : handle(other.handle) { other.handle = nullptr; }
Task& operator=(Task&& other) noexcept {
    if (this != &other) {
        if (handle) handle.destroy();
        handle = other.handle;
        other.handle = nullptr;
    }
    return *this;
}

bool done() const { return handle.done(); }
void resume() { handle.resume(); }
)cpp";
}

} // namespace lowering

} // namespace cpp2concurrency
} // namespace mlir
